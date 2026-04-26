# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project comparing **PPO** (black-box) vs **APG** (Analytic Policy Gradient — backprop through differentiable env dynamics) for learning efficiency and policy training quality. Final project for DDA6129 Reinforcement Learning at CUHKSZ.

## Running Experiments

```bash
# PPO on any gymnasium env
python rl.py --algorithm ppo --env_id CartPole-v1

# APG on a differentiable env
python rl.py --algorithm apg --env_id PointMassNavigate-v0
python rl.py --algorithm apg --env_id PushT-v0
python rl.py --algorithm apg --env_id FrankaReach-v0  # requires newton physics

# Multi-seed sweep with evaluation
python rl.py --algorithm apg --env_id PointMassNavigate-v0 --num_seeds 5 --eval_freq 10

# With wandb tracking
python rl.py --algorithm apg --env_id PointMassNavigate-v0 --track

# Equalize total gradient steps between PPO and APG for fair comparison
python rl.py --algorithm apg --env_id PointMassNavigate-v0 --equalize_grad_steps
```

CLI args use `tyro` — supports both `--key value` and legacy `key value` style. All hyperparameters are in the `Args` dataclass at the top of `rl.py`.

## Architecture

### Core Entry Point: `rl.py`

Single file containing both PPO and APG training loops, selected via `--algorithm`. Key classes:

- **`Args`** (dataclass): All hyperparameters. PPO-specific params have no prefix; APG-specific params are prefixed `apg_`.
- **`Agent`** (nn.Module): Shared actor-critic network. Supports discrete (Gumbel-Softmax for APG) and continuous (reparameterized Gaussian) action spaces. Uses `actor_mean` + `actor_log_std` for continuous, `actor` for discrete.
- **`RunningObsNormalizer`**: Welford-style online obs normalization shared by both algorithms.
- **`_run_training(args, seed)`**: Core training function. Branches into PPO or APG loop based on `args.algorithm`.

### Environment Registry: `env_registry.py`

Pluggable factory pattern. Each env registers an `EnvSpec(ppo_factory, apg_factory)`. Standard gymnasium envs are auto-wrapped via `TorchWrapperEnv`. Custom differentiable envs provide their own APG factory.

### Custom Environments (each file has both VecEnv and APGEnv classes)

| File | Env ID | Domain | Key Dependency |
|---|---|---|---|
| `point_mass_env.py` | `PointMassNavigate-v0` | 2D point-mass with obstacles | Pure PyTorch |
| `push_t_env.py` | `PushT-v0` | 2D rigid-body T-block pushing | Pure PyTorch |
| `franka_reach_env.py` | `FrankaReach-v0` | 7-DOF robot arm reaching | Warp + Newton Physics |

Each env file contains:
- **`*VecEnv`**: Black-box PPO variant (no gradient flow through dynamics)
- **`*APGEnv`**: Differentiable APG variant (full autograd through dynamics)

### Supporting Files

- **`torch_wrapper_env.py`**: Bridges standard gymnasium (numpy) envs to torch tensors. Used as fallback for any unregistered gymnasium env. Detaches gradients (not truly differentiable).
- **`utils.py`**: `set_seed()` — seeds Python, NumPy, PyTorch, Warp, and CUDA for reproducibility.

### APG Algorithm Design

APG unrolls the full episode (or segments) through the differentiable env, computes discounted returns, and calls `loss.backward()` to propagate gradients through rewards, env dynamics, and back into the policy. Key APG-specific features:

- **Segmented rollouts** (`--apg_segment_length`): Breaks long episodes into shorter segments with gradient detachment at boundaries. Supports `'mc'` and `'critic'` bootstrap modes.
- **Stateful training**: Env state carries across gradient steps (reset once, not per step).
- **Per-param gradient clipping** (`--apg_per_param_clip`) before optimizer step.
- **Gumbel-Softmax temperature annealing** for discrete action spaces.
- **Multi-axis logging** (by `global_step`, `total_grad_steps`, `wall_time`) for fair PPO vs APG comparison.

### Logging

TensorBoard logs go to `runs/`. Key metrics are logged under `charts/`, `losses/`, `apg/`, `by_grad_steps/`, and `by_wall_time/` namespaces. View with `tensorboard --logdir runs/`.

## Key Conventions

- All custom envs implement `reset()` → `(obs_tensor, info)` and `step(action_tensor)` → `(obs, reward, terminated, truncated, info)` with `obs` shape `[num_envs, *obs_shape]`.
- APG envs must preserve computation graphs — `step()` returns tensors with `grad_fn` attached.
- The `detach_state()` method on APG envs is called at segment boundaries to break gradient chains.
- Physics envs (FrankaReach) use Warp kernels with tape-based autograd bridged to PyTorch.

## Dependencies

Core: `torch`, `gymnasium`, `numpy`, `tyro`, `tensorboard`, `wandb` (optional)
FrankaReach only: `warp`, `newton` (NVIDIA physics)
Visualization: `matplotlib` (for env rendering)
