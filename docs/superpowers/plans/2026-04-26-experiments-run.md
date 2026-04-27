# Experiments: APG vs PPO Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all PPO and APG experiments across three environments, then fetch wandb results and fill Tables 1 and 4 in `report/report.tex` (Tables 2 & 3 thresholds set by user after viewing results).

**Architecture:** Sequential experiment execution inside `conda env py311`. Group A (main, 5 seeds) runs first; Group B (ablation, 3 seeds) runs second. Results are fetched via the wandb Python API and written as LaTeX table rows directly into `report/report.tex`.

**Tech Stack:** Python, PyTorch, wandb Python API (`wandb.Api`), conda (py311), LaTeX (`report/report.tex`)

---

## File Map

| File | Role |
|---|---|
| `rl.py` | Training entry point — no changes needed |
| `report/report.tex` | Target: replace `\tbd` placeholders in Tables 1 and 4 |
| `scripts/fetch_results.py` | New: wandb API script to compute mean ± std per exp_name |

---

## Task 1: Verify Setup

**Files:**
- Read: `rl.py` (already done)

- [ ] **Step 1: Confirm conda env and core imports work**

```bash
conda run -n py311 python -c "import torch, wandb, gymnasium; print('OK', torch.__version__)"
```

Expected output: `OK <version>` with no import errors.

- [ ] **Step 2: Smoke-test PPO on PointMassNavigate-v0 (1 seed, 500 steps)**

```bash
conda run -n py311 python rl.py \
  --algorithm ppo --env_id PointMassNavigate-v0 \
  --num_envs 4 --total_timesteps 500 --max_episode_steps 100 \
  --num_seeds 1 --track False
```

Expected: training loop runs to completion without error.

- [ ] **Step 3: Smoke-test APG on PointMassNavigate-v0 (1 seed, 500 steps)**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 \
  --num_envs 4 --total_timesteps 500 --max_episode_steps 100 \
  --num_seeds 1 --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track False
```

Expected: APG training loop runs to completion without error.

- [ ] **Step 4: Smoke-test FrankaReach-v0 PPO (1 seed, 200 steps)**

```bash
conda run -n py311 python rl.py \
  --algorithm ppo --env_id FrankaReach-v0 \
  --num_envs 4 --total_timesteps 200 --max_episode_steps 30 \
  --num_seeds 1 --track False
```

Expected: no Warp/Newton import errors, training loop completes.

---

## Task 2: Run Group A — PointMassNavigate-v0

**Files:**
- Execute: `rl.py`
- Results logged to wandb project `cleanRL` under `exp_name` = `ppo_pm` and `apg_pm`

- [ ] **Step 1: Run PPO on PointMassNavigate-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm ppo --env_id PointMassNavigate-v0 --exp_name ppo_pm \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 5 --track
```

Expected: 5 wandb runs created under project `cleanRL` with names like `ppo_pm__1__<timestamp>`.

- [ ] **Step 2: Run APG on PointMassNavigate-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

Expected: 5 wandb runs created under `exp_name=apg_pm`.

---

## Task 3: Run Group A — PushT-v0

**Files:**
- Execute: `rl.py`
- Results logged under `exp_name` = `ppo_pusht` and `apg_pusht`

- [ ] **Step 1: Run PPO on PushT-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm ppo --env_id PushT-v0 --exp_name ppo_pusht \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 5 --track
```

- [ ] **Step 2: Run APG on PushT-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PushT-v0 --exp_name apg_pusht \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

---

## Task 4: Run Group A — FrankaReach-v0

**Files:**
- Execute: `rl.py`
- Results logged under `exp_name` = `ppo_franka` and `apg_franka`

- [ ] **Step 1: Run PPO on FrankaReach-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm ppo --env_id FrankaReach-v0 --exp_name ppo_franka \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 30 \
  --num_seeds 5 --track
```

- [ ] **Step 2: Run APG on FrankaReach-v0 (5 seeds)**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id FrankaReach-v0 --exp_name apg_franka \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 30 \
  --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

---

## Task 5: Run Group B — PointMassNavigate-v0 Ablations

**Files:**
- Execute: `rl.py`
- Results logged under `exp_name` = `apg_pm_seg10mc`, `apg_pm_seg10critic`, `apg_pm_seg5mc`, `apg_pm_seg5critic`

Full-episode row in Table 4 reuses `apg_pm` results from Task 2 — no new run needed.

- [ ] **Step 1: seg=10, bootstrap=mc**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg10mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

- [ ] **Step 2: seg=10, bootstrap=critic**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg10critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap critic --equalize_grad_steps \
  --track
```

- [ ] **Step 3: seg=5, bootstrap=mc**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg5mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

- [ ] **Step 4: seg=5, bootstrap=critic**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg5critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap critic --equalize_grad_steps \
  --track
```

---

## Task 6: Run Group B — PushT-v0 Ablations

**Files:**
- Execute: `rl.py`
- Results logged under `exp_name` = `apg_pusht_seg10mc`, `apg_pusht_seg10critic`, `apg_pusht_seg5mc`, `apg_pusht_seg5critic`

- [ ] **Step 1: seg=10, bootstrap=mc**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg10mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

- [ ] **Step 2: seg=10, bootstrap=critic**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg10critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap critic --equalize_grad_steps \
  --track
```

- [ ] **Step 3: seg=5, bootstrap=mc**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg5mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

- [ ] **Step 4: seg=5, bootstrap=critic**

```bash
conda run -n py311 python rl.py \
  --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg5critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 \
  --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap critic --equalize_grad_steps \
  --track
```

---

## Task 7: Write wandb Result-Fetch Script

**Files:**
- Create: `scripts/fetch_results.py`

This script queries wandb, computes mean ± std of the final `eval/episodic_return` across seeds for each `exp_name`, and prints LaTeX-ready rows.

- [ ] **Step 1: Create `scripts/` directory**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write the fetch script**

```python
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
```

- [ ] **Step 3: Run the script to verify it connects and prints rows**

```bash
conda run -n py311 python scripts/fetch_results.py
```

Expected: prints LaTeX rows to stdout. Any `\tbd` entries indicate runs not yet finished — that is expected if experiments are still running.

---

## Task 8: Fill Table 1 in report/report.tex

After all Group A runs finish (Task 2–4 complete), re-run the fetch script and paste results.

**Files:**
- Modify: `report/report.tex` lines ~538–543 (Table 1 `\tbd` rows)

- [ ] **Step 1: Run fetch script and capture Table 1 output**

```bash
conda run -n py311 python scripts/fetch_results.py 2>/dev/null | grep -A 5 "TABLE 1"
```

Example output (values will differ):
```
    PointMassNavigate-v0      & 12.34 $\pm$ 1.20          & 18.56 $\pm$ 0.98 \\
    PushT-v0                  & -8.12 $\pm$ 2.34          & -5.67 $\pm$ 1.11 \\
    FrankaReach-v0            & -2.10 $\pm$ 0.45          & -1.23 $\pm$ 0.30 \\
```

- [ ] **Step 2: Replace Table 1 `\tbd` rows in report/report.tex**

Locate the three `\tbd` rows inside the `tab:final_return` table (around line 539) and replace with the printed values. Example (use your actual values):

```latex
    PointMassNavigate-v0 & 12.34 $\pm$ 1.20 & 18.56 $\pm$ 0.98 \\
    PushT-v0             & -8.12 $\pm$ 2.34 & -5.67 $\pm$ 1.11 \\
    FrankaReach-v0       & -2.10 $\pm$ 0.45 & -1.23 $\pm$ 0.30 \\
```

- [ ] **Step 3: Verify the LaTeX compiles**

```bash
cd report && pdflatex report.tex 2>&1 | tail -5
```

Expected: `Output written on report.pdf` with no undefined reference errors for Table 1.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_results.py report/report.tex
git commit -m "results: fill Table 1 final episodic return from wandb"
```

---

## Task 9: Fill Table 4 in report/report.tex

After all Group B ablation runs finish (Tasks 5–6 complete).

**Files:**
- Modify: `report/report.tex` lines ~587–593 (Table 4 `\tbd` rows)

- [ ] **Step 1: Run fetch script and capture Table 4 output**

```bash
conda run -n py311 python scripts/fetch_results.py 2>/dev/null | grep -A 8 "TABLE 4"
```

Example output:
```
    Full episode   & ---     & 18.56 $\pm$ 0.98          & -5.67 $\pm$ 1.11          & \tbd \\
    10             & MC      & 17.20 $\pm$ 1.45          & -6.10 $\pm$ 0.88          & \tbd \\
    10             & Critic  & 16.80 $\pm$ 2.10          & -5.90 $\pm$ 1.20          & \tbd \\
    5              & MC      & 15.60 $\pm$ 1.90          & -7.30 $\pm$ 1.50          & \tbd \\
    5              & Critic  & 14.90 $\pm$ 2.30          & -8.10 $\pm$ 1.80          & \tbd \\
```

Note: FrankaReach column stays `\tbd` — the user will fill it, or leave it as N/A.

- [ ] **Step 2: Replace Table 4 rows in report/report.tex**

Locate the five `\tbd`-filled rows inside the `tab:ablation_seg` table (around line 588). Replace with the printed values. The FrankaReach column (`& \tbd \\` at the end of each row) can be changed to `& N/A \\` since it was excluded from ablations.

```latex
    Full episode & ---    & <pm_value>  & <pusht_value>  & N/A \\
    10           & MC     & <pm_value>  & <pusht_value>  & N/A \\
    10           & Critic & <pm_value>  & <pusht_value>  & N/A \\
    5            & MC     & <pm_value>  & <pusht_value>  & N/A \\
    5            & Critic & <pm_value>  & <pusht_value>  & N/A \\
```

- [ ] **Step 3: Verify the LaTeX compiles**

```bash
cd report && pdflatex report.tex 2>&1 | tail -5
```

Expected: `Output written on report.pdf` with no errors.

- [ ] **Step 4: Commit**

```bash
git add report/report.tex
git commit -m "results: fill Table 4 ablation results from wandb"
```

---

## Notes

- **Tables 2 & 3** (sample/compute efficiency) are intentionally left as `\tbd` — the user will review the results and specify performance thresholds before these are filled.
- The fetch script uses `state: finished` filter — re-run it as experiments complete to get partial results.
- If `wandb.Api()` requires login: run `conda run -n py311 wandb login` first.
