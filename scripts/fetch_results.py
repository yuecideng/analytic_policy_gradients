# scripts/fetch_results.py
"""Fetch experiment results from local TensorBoard logs and print LaTeX table rows.

Uses streaming event parsing (faster than EventAccumulator.Reload on large runs).

Usage:
    conda run -n py311 python scripts/fetch_results.py
"""

import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto.summary_pb2 import Summary

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")

TABLE1_GROUPS = [
    ("PointMassNavigate-v0", "ppo_pm", "apg_pm"),
    ("PushT-v0", "ppo_pusht", "apg_pusht"),
    ("FrankaReach-v0", "ppo_franka", "apg_franka"),
]

TABLE4_CONFIGS = [
    ("Full episode", "---", "apg_pm", "apg_pusht"),
    ("10", "MC", "apg_pm_seg10mc", "apg_pusht_seg10mc"),
    ("10", "Critic", "apg_pm_seg10critic", "apg_pusht_seg10critic"),
    ("5", "MC", "apg_pm_seg5mc", "apg_pusht_seg5mc"),
    ("5", "Critic", "apg_pm_seg5critic", "apg_pusht_seg5critic"),
]

THRESHOLDS = {
    "PointMassNavigate-v0": -10.0,
    "PushT-v0": -1.0,
    "FrankaReach-v0": 0.5,
}


def read_scalars(run_dir: str, wanted_tags: set) -> dict:
    """Stream a TensorBoard run directory and return {tag: [(step, value), ...]}."""
    results = {tag: [] for tag in wanted_tags}
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    for ef in event_files:
        try:
            loader = EventFileLoader(ef)
            for event in loader.Load():
                if not event.HasField("summary"):
                    continue
                step = event.step
                for value in event.summary.value:
                    if value.tag not in wanted_tags:
                        continue
                    if value.HasField("simple_value"):
                        results[value.tag].append((step, value.simple_value))
                    elif value.HasField("tensor"):
                        import struct

                        raw = value.tensor.float_val
                        if raw:
                            results[value.tag].append((step, raw[0]))
        except Exception:
            pass
    return results


def get_run_dirs(exp_name: str) -> list:
    return sorted(glob.glob(os.path.join(RUNS_DIR, f"*__{exp_name}__*__*")))


WANTED = {"eval/episodic_return", "by_grad_steps/eval_return"}


def load_all(exp_name: str) -> list:
    """Return list of scalar dicts, one per seed run dir."""
    return [read_scalars(d, WANTED) for d in get_run_dirs(exp_name)]


def get_final_returns(data: list) -> list:
    return [d["eval/episodic_return"][-1][1] for d in data if d["eval/episodic_return"]]


def steps_to_threshold(data: list, threshold: float, tag: str) -> list:
    steps = []
    for d in data:
        crossed = next((s for s, v in d[tag] if v >= threshold), None)
        if crossed is not None:
            steps.append(crossed)
    return steps


def fmt(values: list) -> str:
    if not values:
        return r"\tbd"
    return f"{np.mean(values):.2f} $\\pm$ {np.std(values):.2f}"


def fmt_steps(steps: list) -> str:
    return "N/A" if not steps else f"{int(round(np.mean(steps))):,}"


def fmt_speedup(ppo: list, apg: list) -> str:
    if not ppo or not apg:
        return "N/A"
    return f"{np.mean(ppo) / np.mean(apg):.1f}$\\times$"


# Pre-load all data (one pass per run dir)
print("Loading TensorBoard data...", flush=True)
data_cache = {}
all_exps = [e for _, p, a in TABLE1_GROUPS for e in (p, a)] + [
    e for _, _, pm, pt in TABLE4_CONFIGS for e in (pm, pt)
]
for exp in dict.fromkeys(all_exps):  # deduplicate, preserve order
    data_cache[exp] = load_all(exp)
    print(f"  {exp}: {len(data_cache[exp])} runs", flush=True)

# TABLE 1
print("\n% === TABLE 1: Final episodic return ===")
for env_name, ppo_exp, apg_exp in TABLE1_GROUPS:
    ppo_vals = get_final_returns(data_cache[ppo_exp])
    apg_vals = get_final_returns(data_cache[apg_exp])
    print(f"    {env_name:<25} & {fmt(ppo_vals):<30} & {fmt(apg_vals)} \\\\")


# TABLE 4
print("\n% === TABLE 4: Ablation (episodic return) ===")
for seg_label, boot_label, pm_exp, pusht_exp in TABLE4_CONFIGS:
    pm_vals = get_final_returns(data_cache[pm_exp])
    pusht_vals = get_final_returns(data_cache[pusht_exp])
    print(
        f"    {seg_label:<14} & {boot_label:<7} & {fmt(pm_vals):<30} & {fmt(pusht_vals):<30} & N/A \\\\"
    )
