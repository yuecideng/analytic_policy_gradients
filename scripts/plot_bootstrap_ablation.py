"""Ablation study: MC bootstrap vs critic bootstrap for segmented APG.

Auto-discovers runs from runs/ whose exp_name contains 'mc' or 'critic'.
Groups by env_id and bootstrap mode; produces mean ± std curves.

Metrics plotted (two x-axes):
  - eval/episodic_return   vs global_step  → figures/ablation_bootstrap_return.{pdf,png}
  - eval/success_rate      vs global_step  → figures/ablation_bootstrap_success.{pdf,png}
  - eval/episodic_return   vs grad_step    → figures/ablation_bootstrap_return_gradsteps.{pdf,png}

Usage:
    conda run -n py311 python scripts/plot_bootstrap_ablation.py
"""

import os
import re
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Run directory name: {env_id}__{exp_name}__{seed}__{timestamp}
RUN_DIR_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")

MC_COLOR = "#4C72B0"  # blue
CRITIC_COLOR = "#DD8452"  # orange

BOOTSTRAP_STYLES = {
    "mc": {"color": MC_COLOR, "label": "MC Bootstrap", "ls": "-"},
    "critic": {"color": CRITIC_COLOR, "label": "Critic Bootstrap", "ls": "--"},
}

# (tb_key, by-grad-steps key or None, y-label, figure stem, title suffix)
METRICS = [
    (
        "eval/episodic_return",
        "by_grad_steps/eval_return",
        "Episodic Return",
        "ablation_bootstrap_return",
        "Episodic Return — MC vs Critic Bootstrap",
    ),
    (
        "eval/success_rate",
        "by_grad_steps/eval_success_rate",
        "Success Rate",
        "ablation_bootstrap_success",
        "Success Rate — MC vs Critic Bootstrap",
    ),
]

FALLBACK_METRIC = "charts/episodic_return"


def detect_bootstrap(exp_name: str) -> str | None:
    """Return 'mc', 'critic', or None if the run is not a bootstrap ablation."""
    lower = exp_name.lower()
    if "critic" in lower:
        return "critic"
    if "_mc" in lower or lower.endswith("mc"):
        return "mc"
    return None


def discover_runs() -> dict[str, dict[str, list[str]]]:
    """Return {env_id: {bootstrap_mode: [run_dir, ...]}}."""
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for entry in sorted(os.listdir(RUNS_DIR)):
        full = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(full):
            continue
        m = RUN_DIR_RE.match(entry)
        if not m:
            continue
        env_id, exp_name, _ = m.groups()
        mode = detect_bootstrap(exp_name)
        if mode is None:
            continue
        grouped[env_id][mode].append(full)
    return grouped


def load_metric(
    run_dirs: list[str], metric: str, fallback: str | None = FALLBACK_METRIC
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load scalar metric from multiple runs; interpolate onto a common step grid.

    Returns (common_steps, values_matrix) with shape (n_seeds, n_pts), or None.
    """
    per_run: list[tuple[np.ndarray, np.ndarray]] = []
    for run_dir in run_dirs:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        key = (
            metric
            if metric in available
            else (fallback if fallback and fallback in available else None)
        )
        if key is None:
            continue
        scalars = ea.Scalars(key)
        steps = np.array([s.step for s in scalars])
        values = np.array([s.value for s in scalars])
        per_run.append((steps, values))

    if not per_run:
        return None

    min_last = min(s[-1] for s, _ in per_run)
    max_first = max(s[0] for s, _ in per_run)
    n_pts = min(len(s) for s, _ in per_run)
    common_steps = np.linspace(max_first, min_last, n_pts)
    all_values = [np.interp(common_steps, s, v) for s, v in per_run]
    return common_steps, np.array(all_values)


def plot_ax(ax, grouped_env, tb_key, grad_key, ylabel, title, use_grad_steps=False):
    """Fill a single axes with mean ± std curves for each bootstrap mode."""
    any_data = False
    summary_rows = []

    for mode in ("mc", "critic"):
        run_dirs = grouped_env.get(mode, [])
        if not run_dirs:
            continue
        metric = grad_key if (use_grad_steps and grad_key) else tb_key
        result = load_metric(run_dirs, metric)
        if result is None:
            print(f"    [{mode}] no data for {metric}")
            continue

        steps, values = result
        any_data = True
        n = values.shape[0]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        style = BOOTSTRAP_STYLES[mode]
        label = f"{style['label']} (n={n})"
        ax.plot(
            steps,
            mean,
            label=label,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=2,
        )
        ax.fill_between(steps, mean - std, mean + std, color=style["color"], alpha=0.2)
        summary_rows.append((mode, n, mean[-1], std[-1]))
        print(
            f"    [{mode}] {n} seeds, {len(steps)} pts, "
            f"final={mean[-1]:.4f} ± {std[-1]:.4f}"
        )

    ax.set_title(title, fontsize=11)
    xlabel = "Gradient Steps" if use_grad_steps else "Global Steps"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if any_data:
        ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    return summary_rows


# ── main ──────────────────────────────────────────────────────────────────────

grouped = discover_runs()

if not grouped:
    print("No bootstrap ablation runs found in", RUNS_DIR)
    raise SystemExit(1)

env_ids = sorted(grouped.keys())
print(f"Environments with bootstrap ablation runs: {env_ids}")

for tb_key, grad_key, ylabel, fig_stem, title_suffix in METRICS:
    for x_axis in ("steps", "gradsteps"):
        use_grad = x_axis == "gradsteps"
        n_envs = len(env_ids)
        fig, axes = plt.subplots(
            1, max(n_envs, 1), figsize=(5 * max(n_envs, 1), 4), squeeze=False
        )
        axes = axes[0]
        fig.suptitle(f"APG Segmentation Ablation: {title_suffix}", fontsize=12)

        all_summary = {}
        for ax, env_id in zip(axes, env_ids):
            print(f"\n── {env_id} | {tb_key} | x={x_axis} ──")
            env_title = (
                f"{env_id.replace('-', ' ')} "
                f"({'Grad Steps' if use_grad else 'Global Steps'})"
            )
            rows = plot_ax(
                ax,
                grouped[env_id],
                tb_key,
                grad_key,
                ylabel,
                env_title,
                use_grad_steps=use_grad,
            )
            all_summary[env_id] = rows

        for ax in axes[n_envs:]:
            ax.set_visible(False)

        plt.tight_layout()
        suffix = "_gradsteps" if use_grad else ""
        out_pdf = os.path.join(OUT_DIR, f"{fig_stem}{suffix}.pdf")
        out_png = os.path.join(OUT_DIR, f"{fig_stem}{suffix}.png")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_pdf}")
        print(f"  Saved: {out_png}")

# ── summary table ──────────────────────────────────────────────────────────────
print("\n── Bootstrap Ablation: Final Return Summary ──")
COL = 24
hdr = f"{'Environment':<24}  {'MC (mean ± std)':>{COL}}  {'Critic (mean ± std)':>{COL}}"
sep = "-" * len(hdr)
print(sep)
print(hdr)
print(sep)
for env_id in env_ids:

    def fmt(mode):
        result = load_metric(grouped[env_id].get(mode, []), "eval/episodic_return")
        if result is None:
            return "N/A"
        _, v = result
        return f"{v[:, -1].mean():.2f} ± {v[:, -1].std():.2f}"

    print(f"{env_id:<24}  {fmt('mc'):>{COL}}  {fmt('critic'):>{COL}}")
print(sep)
