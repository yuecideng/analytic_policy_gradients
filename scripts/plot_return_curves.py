"""Plot return curves (mean ± std) for PPO vs APG across three environments.

Reads TensorBoard event files from runs/ and saves figures to figures/.

Usage:
    conda run -n py311 python scripts/plot_return_curves.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC = "by_grad_steps/eval_return"

ENVS = [
    ("PointMassNavigate-v0", "ppo_pm",    "apg_pm",    "PointMass Navigate"),
    ("PushT-v0",             "ppo_pusht", "apg_pusht", "Push-T"),
    ("FrankaReach-v0",       "ppo_franka","apg_franka","Franka Reach"),
]


def load_runs(env_id: str, exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, returns_matrix) where returns_matrix is (n_seeds, n_steps)."""
    pattern = os.path.join(RUNS_DIR, f"{env_id}__{exp_name}__*")
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"  [warn] no runs found for {env_id}/{exp_name}")
        return np.array([]), np.array([])

    all_steps = None
    all_returns = []
    for run_dir in run_dirs:
        ea = EventAccumulator(run_dir)
        ea.Reload()
        if METRIC not in ea.Tags().get("scalars", []):
            print(f"  [warn] metric missing in {os.path.basename(run_dir)}")
            continue
        scalars = ea.Scalars(METRIC)
        steps = np.array([s.step for s in scalars])
        values = np.array([s.value for s in scalars])
        if all_steps is None:
            all_steps = steps
        all_returns.append(values)

    if not all_returns:
        return np.array([]), np.array([])
    return all_steps, np.array(all_returns)


def smooth(values: np.ndarray, window: int = 1) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("APG vs PPO: Episodic Return (mean ± std, 5 seeds)", fontsize=13)

PPO_COLOR = "#4C72B0"
APG_COLOR = "#DD8452"

for ax, (env_id, ppo_exp, apg_exp, title) in zip(axes, ENVS):
    print(f"Processing {title}...")

    ppo_steps, ppo_returns = load_runs(env_id, ppo_exp)
    apg_steps, apg_returns = load_runs(env_id, apg_exp)

    for label, steps, returns, color in [
        ("PPO", ppo_steps, ppo_returns, PPO_COLOR),
        ("APG", apg_steps, apg_returns, APG_COLOR),
    ]:
        if steps.size == 0:
            continue
        mean = returns.mean(axis=0)
        std = returns.std(axis=0)
        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Total Gradient Steps (×10²)", fontsize=9)
    ax.set_ylabel("Episodic Return", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    # Format x-axis ticks: steps are multiples of 192, show as ×100
    xticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(x)}" for x in xticks], fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "return_curves.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# Also save PNG for quick preview
out_png = os.path.join(OUT_DIR, "return_curves.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
