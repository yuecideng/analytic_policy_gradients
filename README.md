# analytic_policy_gradients
Final project for DDA6129 Reinforcement Learning in CUHKSZ

A unified PPO / APG training loop. Supports any gymnasium environment out of the box (PPO) and differentiable environments via a pluggable registry (APG).

## Quick Start

```bash
# PPO on CartPole (default)
python rl.py

# PPO on any gymnasium env
python rl.py --algorithm ppo --env_id Pendulum-v1

# APG on a differentiable env
python rl.py --algorithm apg --env_id FrankaReach-v0

# Run with wandb logging (optional)
python rl.py --algorithm ppo --env_id FrankaReach --track
```


## How to Implement a Custom Environment

### For PPO (Black-Box Env)

Any standard [gymnasium](https://gymnasium.farama.org/) environment works automatically — it is wrapped by `TorchWrapperEnv` at runtime. No registration needed.

### For APG (Differentiable Env)

APG backpropagates through the environment dynamics, so your env must be written entirely in PyTorch (or provide a gradient bridge). Your env class must implement:

```python
class MyAPGEnv:
    # Required attributes
    single_observation_space: gym.spaces.Box | gym.spaces.Discrete
    single_action_space:      gym.spaces.Box | gym.spaces.Discrete
    num_envs: int

    def reset(self, *, seed=None, options=None):
        """Return (obs: Tensor[float], info: dict).
        obs shape: [num_envs, *obs_shape]"""
        ...

    def step(self, action: torch.Tensor):
        """Return (obs, reward, terminated, truncated, info).
        All tensors must preserve the computation graph so that
        loss.backward() can flow gradients back through the dynamics."""
        ...

    def close(self):
        ...
```

Key difference from a standard gymnasium env: **tensors returned by `step()` must carry `grad_fn`** so that the policy gradient can flow through the environment transition.

### Registering Your Environment

Add your env to `IMPLEMENTED_ENVS` in `env_registry.py`:

```python
from my_env import MyPPGEnv, MyAPGEnv

IMPLEMENTED_ENVS["MyEnv-v0"] = EnvSpec(
    ppo_factory=lambda **kw: MyPPGEnv(**kw),
    apg_factory=lambda **kw: MyAPGEnv(**kw),   # set to None if not differentiable
)
```

The factory receives these keyword arguments from `rl.py`:

| Argument | Type | Description |
|---|---|---|
| `num_envs` | `int` | Number of parallel environments |
| `device` | `str` | `"cpu"` for PPO, e.g. `"cuda"` for APG |
| `headless` | `bool` | Suppress viewer windows |

Then run:

```bash
python rl.py --algorithm apg --env_id MyEnv-v0
```

### Architecture Overview

```
rl.py
 ├── make_custom_vec_env()  ──→  env_registry.get_env_spec(env_id)
 │                                    │
 │                            ┌───────┴────────┐
 │                       registered?        not registered
 │                            │                  │
 │                   EnvSpec factory      gym.vector.SyncVectorEnv
 │                   (PPO or APG)               │
 │                                        TorchWrapperEnv
 │                                     (numpy ↔ torch bridge)
 │
 ├── PPO loop  ──  sample → env.step() → buffer → surrogate loss
 └── APG loop  ──  differentiable action → env.step() → discounted return → backward()
```

- **PPO** treats the env as a black box. Works with any gymnasium env via `TorchWrapperEnv`.
- **APG** requires a fully differentiable env. Gradients flow from the loss through the return, the reward, the environment dynamics, and back into the policy network.
