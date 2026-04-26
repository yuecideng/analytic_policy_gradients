# scripts/fetch_results.py
"""Fetch final eval/episodic_return from wandb and print LaTeX table rows.

Usage:
    conda run -n py311 python scripts/fetch_results.py
"""
import numpy as np
import wandb

api = wandb.Api()
PROJECT = "cleanRL"

# --- Table 1: final return per env/algo ---
TABLE1_GROUPS = [
    ("PointMassNavigate-v0", "ppo_pm",    "apg_pm"),
    ("PushT-v0",             "ppo_pusht", "apg_pusht"),
    ("FrankaReach-v0",       "ppo_franka","apg_franka"),
]

# --- Table 4: ablation (episodic return per config) ---
TABLE4_CONFIGS = [
    # (seg_label, bootstrap_label, pm_exp_name,         pusht_exp_name)
    ("Full episode", "---",    "apg_pm",           "apg_pusht"),
    ("10",           "MC",     "apg_pm_seg10mc",   "apg_pusht_seg10mc"),
    ("10",           "Critic", "apg_pm_seg10critic","apg_pusht_seg10critic"),
    ("5",            "MC",     "apg_pm_seg5mc",    "apg_pusht_seg5mc"),
    ("5",            "Critic", "apg_pm_seg5critic", "apg_pusht_seg5critic"),
]


def get_final_returns(exp_name: str) -> list[float]:
    """Return final eval/episodic_return for all finished runs with this exp_name."""
    runs = api.runs(
        PROJECT,
        filters={"config.exp_name": exp_name, "state": "finished"},
    )
    returns = []
    for run in runs:
        history = run.history(keys=["eval/episodic_return"], pandas=False)
        values = [row["eval/episodic_return"] for row in history
                  if row.get("eval/episodic_return") is not None]
        if values:
            returns.append(values[-1])  # final checkpoint
    return returns


def fmt(values: list[float]) -> str:
    if not values:
        return r"\tbd"
    return f"{np.mean(values):.2f} $\\pm$ {np.std(values):.2f}"


# ---- Table 1 output ----
print("\n% === TABLE 1: Final episodic return ===")
for env_name, ppo_exp, apg_exp in TABLE1_GROUPS:
    ppo_vals = get_final_returns(ppo_exp)
    apg_vals = get_final_returns(apg_exp)
    print(f"    {env_name:<25} & {fmt(ppo_vals):<30} & {fmt(apg_vals)} \\\\")

# ---- Table 4 output ----
print("\n% === TABLE 4: Ablation (episodic return) ===")
for seg_label, boot_label, pm_exp, pusht_exp in TABLE4_CONFIGS:
    pm_vals    = get_final_returns(pm_exp)
    pusht_vals = get_final_returns(pusht_exp)
    # FrankaReach excluded from ablation
    print(f"    {seg_label:<14} & {boot_label:<7} & {fmt(pm_vals):<30} & {fmt(pusht_vals):<30} & \\tbd \\\\")
