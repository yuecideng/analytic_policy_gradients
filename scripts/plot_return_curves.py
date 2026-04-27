"""Plot return curves (mean ± std) for PPO vs APG across available environments.

Auto-discovers runs from runs/ directory — no hardcoded env or experiment names.
Groups by env_id; within each env, groups by algorithm (ppo/apg) based on exp_name.

Tries metrics in order of preference:
  by_grad_steps/eval_return → eval/episodic_return → by_grad_steps/episodic_return
  → charts/episodic_return

Usage:
    conda run -n py311 python scripts/plot_return_curves.py
"""

import os
import glob
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

# Metrics tried in order; the first one present in a run is used.
METRIC_PREFERENCE = [
    "eval/episodic_return",
    "charts/episodic_return",
    "by_grad_steps/eval_return",
    "by_grad_steps/episodic_return",
]

PPO_COLOR = "#4C72B0"
APG_COLOR = "#DD8452"

# Run directory name format: {env_id}__{exp_name}__{seed}__{timestamp}
RUN_DIR_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")


def detect_algorithm(exp_name: str) -> str:
    """Return 'ppo' or 'apg' based on exp_name, else the raw exp_name."""
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


def pick_metric(ea: EventAccumulator) -> str | None:
    available = set(ea.Tags().get("scalars", []))
    for m in METRIC_PREFERENCE:
        if m in available:
            return m
    return None


def load_runs(run_dirs: list[str]) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Return (steps, returns_matrix, metric_used).

    returns_matrix shape: (n_seeds, n_steps).
    """
    all_returns = []
    all_steps = None
    metric_used = None

    for run_dir in run_dirs:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        metric = pick_metric(ea)
        if metric is None:
            print(f"  [warn] no usable metric in {os.path.basename(run_dir)}")
            continue
        metric_used = metric
        scalars = ea.Scalars(metric)
        steps = np.array([s.step for s in scalars])
        values = np.array([s.value for s in scalars])
        if all_steps is None:
            all_steps = steps
        all_returns.append(values)

    if not all_returns:
        return np.array([]), np.array([]), None
    return all_steps, np.array(all_returns), metric_used


def smooth(values: np.ndarray, window: int = 1) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ── main ─────────────────────────────────────────────────────────────────────

grouped = discover_runs()

if not grouped:
    print("No runs found in", RUNS_DIR)
    raise SystemExit(1)

env_ids = sorted(grouped.keys())
n = len(env_ids)
fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 4), squeeze=False)
axes = axes[0]

fig.suptitle("APG vs PPO: Episodic Return (mean ± std)", fontsize=13)

ALGO_STYLE: dict[str, dict] = {
    "ppo": {"color": PPO_COLOR, "label": "PPO"},
    "apg": {"color": APG_COLOR, "label": "APG"},
}

# {env_id: {algo: (final_mean, final_std)}}
final_stats: dict[str, dict[str, tuple[float, float]]] = {}

for ax, env_id in zip(axes, env_ids):
    print(f"\nProcessing {env_id}...")
    algos = grouped[env_id]
    any_data = False
    final_stats[env_id] = {}

    for algo, run_dirs in sorted(algos.items()):
        style = ALGO_STYLE.get(algo, {"color": "gray", "label": algo.upper()})
        steps, returns, metric = load_runs(run_dirs)
        if steps.size == 0:
            print(f"  [{algo}] no data")
            continue
        any_data = True
        n_seeds = returns.shape[0]
        mean = returns.mean(axis=0)
        std = returns.std(axis=0)
        # Final return: mean across seeds of their last recorded value
        final_vals = returns[:, -1]
        final_stats[env_id][algo] = (float(final_vals.mean()), float(final_vals.std()))
        label = f"{style['label']} (n={n_seeds})"
        ax.plot(steps, mean, label=label, color=style["color"], linewidth=2)
        ax.fill_between(
            steps, mean - std, mean + std, color=style["color"], alpha=0.2
        )
        print(f"  [{algo}] {n_seeds} seeds, metric={metric}, {len(steps)} points")

    ax.set_title(env_id.replace("-", " "), fontsize=11)
    xlabel = "x-axis: steps"
    if any_data:
        # Infer axis label from metric name
        for m in METRIC_PREFERENCE:
            if "grad_steps" in m:
                xlabel = "Total Gradient Steps"
                break
            if "episodic" in m or "eval" in m:
                xlabel = "Global Steps"
                break
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Episodic Return", fontsize=9)
    if any_data:
        ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

# Hide unused axes (if any)
for ax in axes[n:]:
    ax.set_visible(False)

plt.tight_layout()

out_pdf = os.path.join(OUT_DIR, "return_curves.pdf")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")

out_png = os.path.join(OUT_DIR, "return_curves.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")

# ── final return table ────────────────────────────────────────────────────────
COL_W = 22
header = f"{'Environment':<20}  {'PPO (mean ± std)':>{COL_W}}  {'APG (mean ± std)':>{COL_W}}"
sep = "-" * len(header)
print(f"\n{sep}")
print(header)
print(sep)
for env_id in env_ids:
    stats = final_stats.get(env_id, {})

    def fmt(algo: str) -> str:
        if algo not in stats:
            return "TBD"
        m, s = stats[algo]
        return f"{m:.2f} ± {s:.2f}"

    print(f"{env_id:<20}  {fmt('ppo'):>{COL_W}}  {fmt('apg'):>{COL_W}}")
print(sep)
