"""Plot evaluation curves (mean ± std) for PPO vs APG across available environments.

Auto-discovers runs from runs/ directory — no hardcoded env or experiment names.
Groups by env_id; within each env, groups by algorithm (ppo/apg) based on exp_name.

Produces one figure per metric:
  - eval/episodic_return   → figures/eval_return.{pdf,png}
  - eval/episodic_length   → figures/eval_length.{pdf,png}
  - eval/success_rate      → figures/eval_success_rate.{pdf,png}

Falls back to charts/episodic_return when eval metrics are absent.

Usage:
    conda run -n py311 python scripts/plot_return_curves.py
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

# Each entry: (tb_metric_key, y_axis_label, figure_stem, title_suffix)
METRICS: list[tuple[str, str, str, str]] = [
    (
        "eval/episodic_return",
        "Episodic Return",
        "eval_return",
        "Episodic Return (mean ± std)",
    ),
    (
        "eval/episodic_length",
        "Episode Length",
        "eval_length",
        "Episode Length (mean ± std)",
    ),
    (
        "eval/success_rate",
        "Success Rate",
        "eval_success_rate",
        "Success Rate (mean ± std)",
    ),
]
# Fallback metric when eval metrics are absent
FALLBACK_METRIC = "charts/episodic_return"

PPO_COLOR = "#4C72B0"
APG_COLOR = "#DD8452"

# Run directory name format: {env_id}__{exp_name}__{seed}__{timestamp}
RUN_DIR_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")


def detect_algorithm(exp_name: str) -> str:
    lower = exp_name.lower()
    if "ppo" in lower:
        return "ppo"
    if "apg" in lower:
        return "apg"
    return exp_name


def discover_runs() -> dict[str, dict[str, list[str]]]:
    """Return {env_id: {algo_label: [run_dir, ...]}}."""
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for entry in sorted(os.listdir(RUNS_DIR)):
        full = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(full):
            continue
        m = RUN_DIR_RE.match(entry)
        if not m:
            print(f"  [skip] unrecognised directory name: {entry}")
            continue
        env_id, exp_name, _ = m.groups()
        algo = detect_algorithm(exp_name)
        grouped[env_id][algo].append(full)
    return grouped


def load_metric(
    run_dirs: list[str], metric: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a single metric from multiple run dirs and interpolate onto a common grid.

    Returns (common_steps, values_matrix) where values_matrix shape is
    (n_seeds, n_steps), or None if no data found.
    """
    per_run: list[tuple[np.ndarray, np.ndarray]] = []

    for run_dir in run_dirs:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        key = (
            metric
            if metric in available
            else (FALLBACK_METRIC if FALLBACK_METRIC in available else None)
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

    all_values = [np.interp(common_steps, steps, values) for steps, values in per_run]
    return common_steps, np.array(all_values)


# ── main ─────────────────────────────────────────────────────────────────────

grouped = discover_runs()

if not grouped:
    print("No runs found in", RUNS_DIR)
    raise SystemExit(1)

env_ids = sorted(grouped.keys())
n_envs = len(env_ids)

ALGO_STYLE: dict[str, dict] = {
    "ppo": {"color": PPO_COLOR, "label": "PPO"},
    "apg": {"color": APG_COLOR, "label": "APG"},
}

for tb_metric, ylabel, fig_stem, title_suffix in METRICS:
    print(f"\n── Plotting {tb_metric} ──")
    fig, axes = plt.subplots(
        1, max(n_envs, 1), figsize=(5 * max(n_envs, 1), 4), squeeze=False
    )
    axes = axes[0]
    fig.suptitle(f"APG vs PPO: {title_suffix}", fontsize=13)

    for ax, env_id in zip(axes, env_ids):
        print(f"  {env_id}")
        algos = grouped[env_id]
        any_data = False

        for algo, run_dirs in sorted(algos.items()):
            style = ALGO_STYLE.get(algo, {"color": "gray", "label": algo.upper()})
            result = load_metric(run_dirs, tb_metric)
            if result is None:
                print(f"    [{algo}] no data for {tb_metric}")
                continue
            steps, values = result
            any_data = True
            n_seeds = values.shape[0]
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            label = f"{style['label']} (n={n_seeds})"
            ax.plot(steps, mean, label=label, color=style["color"], linewidth=2)
            ax.fill_between(
                steps, mean - std, mean + std, color=style["color"], alpha=0.2
            )
            print(f"    [{algo}] {n_seeds} seeds, {len(steps)} points")

        ax.set_title(env_id.replace("-", " "), fontsize=11)
        ax.set_xlabel("Global Steps", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if any_data:
            ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[n_envs:]:
        ax.set_visible(False)

    plt.tight_layout()

    out_pdf = os.path.join(OUT_DIR, f"{fig_stem}.pdf")
    out_png = os.path.join(OUT_DIR, f"{fig_stem}.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")

# ── final return table ────────────────────────────────────────────────────────
print("\n── Final Return Summary ──")
COL_W = 22
header = (
    f"{'Environment':<20}  {'PPO (mean ± std)':>{COL_W}}  {'APG (mean ± std)':>{COL_W}}"
)
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for env_id in env_ids:
    algos = grouped[env_id]

    def fmt(algo: str) -> str:
        result = load_metric(algos.get(algo, []), "eval/episodic_return")
        if result is None:
            return "TBD"
        _, values = result
        final_vals = values[:, -1]
        return f"{final_vals.mean():.2f} ± {final_vals.std():.2f}"

    print(f"{env_id:<20}  {fmt('ppo'):>{COL_W}}  {fmt('apg'):>{COL_W}}")
print(sep)
