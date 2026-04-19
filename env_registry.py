from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gymnasium as gym

from torch_wrapper_env import TorchWrapperEnv


@dataclass(frozen=True)
class EnvSpec:
    """Specification for implemented custom envs."""

    ppo_factory: Optional[Callable[..., Any]] = None
    apg_factory: Optional[Callable[..., Any]] = None


def _make_torch_wrapped_env(env_id: str, num_envs: int, **kwargs):
    """Create a TorchWrapperEnv around a standard gymnasium SyncVectorEnv."""
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id) for _ in range(num_envs)],
    )
    return TorchWrapperEnv(envs)


# Try importing optional heavy dependencies so they don't block core usage.
try:
    from franka_reach_env import FrankaReachAPGEnv, FrankaReachVecEnv

    _franka_spec = EnvSpec(
        ppo_factory=lambda **kw: FrankaReachVecEnv(**kw),
        apg_factory=lambda **kw: FrankaReachAPGEnv(**kw),
    )
except ImportError:
    _franka_spec = None


IMPLEMENTED_ENVS: Dict[str, EnvSpec] = {}

# Register FrankaReach only if the dependency is available.
if _franka_spec is not None:
    IMPLEMENTED_ENVS["FrankaReach-v0"] = _franka_spec


def _register_gym_envs():
    """Auto-register every gymnasium env with a TorchWrapperEnv factory."""
    try:
        all_envs = gym.envs.registry.keys()
    except Exception:
        return
    for env_id in all_envs:
        if env_id in IMPLEMENTED_ENVS:
            continue
        IMPLEMENTED_ENVS[env_id] = EnvSpec(
            ppo_factory=lambda env_id=env_id, **kw: _make_torch_wrapped_env(
                env_id,
                num_envs=kw.get("num_envs", 4),
            ),
            apg_factory=lambda env_id=env_id, **kw: _make_torch_wrapped_env(
                env_id,
                num_envs=kw.get("num_envs", 4),
            ),
        )


_register_gym_envs()


def get_env_spec(env_id: str) -> Optional[EnvSpec]:
    return IMPLEMENTED_ENVS.get(env_id)
