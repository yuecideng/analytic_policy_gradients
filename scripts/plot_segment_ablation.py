"""Ablation study: segment length × bootstrap mode for segmented APG.

Discovers runs whose exp_name matches apg_point_mass_{mc|critic}_{10|25|50}.
Produces mean curves (single seed per condition) for:
  - eval/episodic_return vs global_step
  - eval/success_rate   vs global_step

Output figures:
  figures/ablation_segment_return.{pdf,png}
  figures/ablation_segment_success.{pdf,png}

Usage:
    conda run -n py311 python scripts/plot_segment_ablation.py
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

RUN_DIR_RE = re.compile(r"^(.+?)__(.+?)__(\d+)__\d+$")
SEGMENT_RE = re.compile(r"_(mc|critic)_(\d+)$")

# Colors per bootstrap mode; line styles per segment length
MODE_COLORS = {"mc": "#4C72B0", "critic": "#DD8452"}
SEG_STYLES = {10: "-", 25: "--", 50: ":"}
SEG_LENGTHS = [10, 25, 50]
MODES = ["mc", "critic"]


def discover_runs():
    """Return {(mode, seg_len): run_dir_path}."""
    mapping = {}
    for entry in sorted(os.listdir(RUNS_DIR)):
        full = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(full):
            continue
        m = RUN_DIR_RE.match(entry)
        if not m:
            continue
        _, exp_name, _ = m.groups()
        sm = SEGMENT_RE.search(exp_name)
        if not sm:
            continue
        mode, seg_str = sm.group(1), sm.group(2)
        seg_len = int(seg_str)
        key = (mode, seg_len)
        if key not in mapping:
            mapping[key] = full
    return mapping


def load_scalar(run_dir, key):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    if key not in tags:
        return None, None
    scalars = ea.Scalars(key)
    steps = np.array([s.step for s in scalars])
    vals = np.array([s.value for s in scalars])
    return steps, vals


def make_figure(runs, tb_key, ylabel, fig_stem, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.suptitle(title, fontsize=12)

    for ax, mode in zip(axes, MODES):
        ax.set_title(f"{mode.upper()} Bootstrap", fontsize=11)
        for seg in SEG_LENGTHS:
            run_dir = runs.get((mode, seg))
            if run_dir is None:
                continue
            steps, vals = load_scalar(run_dir, tb_key)
            if steps is None:
                print(f"  [{mode}, L={seg}] missing {tb_key}")
                continue
            color = MODE_COLORS[mode]
            # Vary lightness for different segment lengths
            alpha = {10: 0.55, 25: 0.8, 50: 1.0}[seg]
            ax.plot(
                steps,
                vals,
                label=f"L={seg}",
                color=color,
                linestyle=SEG_STYLES[seg],
                linewidth=2,
                alpha=alpha,
            )
            print(f"  [{mode}, L={seg}] final={vals[-1]:.4f}")
        ax.set_xlabel("Global Steps", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"{fig_stem}.{ext}")
        kwargs = (
            {"bbox_inches": "tight"}
            if ext == "pdf"
            else {"dpi": 150, "bbox_inches": "tight"}
        )
        plt.savefig(out, **kwargs)
        print(f"  Saved: {out}")
    plt.close(fig)


runs = discover_runs()
if not runs:
    print("No segment ablation runs found.")
    raise SystemExit(1)

print(f"Found {len(runs)} (mode, seg_len) pairs: {sorted(runs.keys())}")

make_figure(
    runs,
    "eval/episodic_return",
    "Episodic Return",
    "ablation_segment_return",
    "Segment Length Ablation: Episodic Return",
)

make_figure(
    runs,
    "eval/success_rate",
    "Success Rate",
    "ablation_segment_success",
    "Segment Length Ablation: Success Rate",
)

# Print summary table
print("\n── Segment Length Ablation: Final Metric Summary ──")
header = f"{'Mode':<8} {'L':>4}  {'Return':>12}  {'Success':>10}"
print("-" * len(header))
print(header)
print("-" * len(header))
for mode in MODES:
    for seg in SEG_LENGTHS:
        run_dir = runs.get((mode, seg))
        if run_dir is None:
            print(f"{mode:<8} {seg:>4}  {'N/A':>12}  {'N/A':>10}")
            continue
        _, r_vals = load_scalar(run_dir, "eval/episodic_return")
        _, s_vals = load_scalar(run_dir, "eval/success_rate")
        r_str = f"{r_vals[-1]:.2f}" if r_vals is not None else "N/A"
        s_str = f"{s_vals[-1]:.2f}" if s_vals is not None else "N/A"
        print(f"{mode:<8} {seg:>4}  {r_str:>12}  {s_str:>10}")
print("-" * len(header))
