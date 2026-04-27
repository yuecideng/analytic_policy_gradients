# Experiment Design: APG vs PPO Comparison

**Date:** 2026-04-26  
**Purpose:** Fill Tables 1–4 in Section 5 of `report/report.tex` — final return, sample efficiency, compute efficiency, and segmentation ablation.

---

## Overview

Two experiment groups:

| Group | Tables | Envs | Seeds | Total runs |
|---|---|---|---|---|
| A. Main | 1, 2, 3 | PointMass, PushT, Franka | 5 | 6 (3 envs × 2 algos) |
| B. Ablation | 4 | PointMass, PushT only | 3 | 24 (2 envs × 4 new APG configs) |

All tasks use `conda env py311`.

---

## Group A — Main Experiments

### Shared Hyperparameters

| Param | Value |
|---|---|
| `learning_rate` | `2.5e-4` |
| `gamma` | `0.99` |
| `max_grad_norm` | `0.5` |
| `num_envs` | `32` |
| `total_timesteps` | `500000` |
| `max_episode_steps` | `100` (PointMass, PushT) / `30` (Franka) |
| `num_seeds` | `5` |
| `track` | `True` |

### PPO Hyperparameters

| Param | Value |
|---|---|
| `num_steps` | `128` |
| `num_minibatches` | `4` |
| `update_epochs` | `4` |
| `gae_lambda` | `0.95` |
| `clip_coef` | `0.2` |
| `vf_coef` | `0.5` |
| `ent_coef` | `0.01` |

### APG Hyperparameters (Baseline)

| Param | Value |
|---|---|
| `apg_num_grad_steps` | `8` |
| `apg_per_param_clip` | `1.0` |
| `apg_segment_length` | `0` (full episode, no segmentation) |
| `apg_bootstrap` | `mc` |
| `equalize_grad_steps` | `True` |

### Commands

```bash
# PointMassNavigate-v0
python rl.py --algorithm ppo --env_id PointMassNavigate-v0 --exp_name ppo_pm \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 5 \
  --track

python rl.py --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track

# PushT-v0
python rl.py --algorithm ppo --env_id PushT-v0 --exp_name ppo_pusht \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 5 \
  --track

python rl.py --algorithm apg --env_id PushT-v0 --exp_name apg_pusht \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track

# FrankaReach-v0 
conda run -n py311 python rl.py --algorithm ppo --env_id FrankaReach-v0 --exp_name ppo_franka \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 30 --num_seeds 5 \
  --track

conda run -n py311 python rl.py --algorithm apg --env_id FrankaReach-v0 --exp_name apg_franka \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 30 --num_seeds 5 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 0 --apg_bootstrap mc --equalize_grad_steps \
  --track
```

---

## Group B — Ablation (Table 4)

PointMassNavigate-v0 and PushT-v0 only. FrankaReach excluded (short horizon, 30 steps — segmentation less meaningful).  
Full-episode row in Table 4 reuses Group A results (no new runs needed).

### Ablation Configs

| Config | `apg_segment_length` | `apg_bootstrap` | `exp_name` (pm / pusht) |
|---|---|---|---|
| Full episode (reuse A) | `0` | `mc` | `apg_pm` / `apg_pusht` |
| seg10-mc | `10` | `mc` | `apg_pm_seg10mc` / `apg_pusht_seg10mc` |
| seg10-critic | `10` | `critic` | `apg_pm_seg10critic` / `apg_pusht_seg10critic` |
| seg5-mc | `5` | `mc` | `apg_pm_seg5mc` / `apg_pusht_seg5mc` |
| seg5-critic | `5` | `critic` | `apg_pm_seg5critic` / `apg_pusht_seg5critic` |

### Commands

```bash
# PointMassNavigate-v0 ablations
python rl.py --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg10mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap mc --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg10critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap critic --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg5mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap mc --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PointMassNavigate-v0 --exp_name apg_pm_seg5critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap critic --equalize_grad_steps --track

# PushT-v0 ablations
python rl.py --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg10mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap mc --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg10critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 10 --apg_bootstrap critic --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg5mc \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap mc --equalize_grad_steps --track

python rl.py --algorithm apg --env_id PushT-v0 --exp_name apg_pusht_seg5critic \
  --num_envs 32 --total_timesteps 500000 --max_episode_steps 100 --num_seeds 3 \
  --apg_num_grad_steps 8 --apg_per_param_clip 1.0 \
  --apg_segment_length 5 --apg_bootstrap critic --equalize_grad_steps --track
```

---

## Wandb Fetch Plan

After all runs complete, fetch from wandb project `cleanRL` (entity: default):

- **Table 1**: Final episodic return — query last eval checkpoint per run, group by `exp_name`, compute mean ± std over seeds.
- **Table 2**: Env steps to threshold — scan eval return curve per seed, find first step where return ≥ threshold; threshold values to be determined after viewing results.
- **Table 3**: Grad steps to threshold — same but logged under `by_grad_steps/` axis.
- **Table 4**: Final episodic return grouped by `exp_name` ablation configs.
