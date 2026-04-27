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

try:
    from push_t_env import PushTAPGEnv, PushTVecEnv

    _pusht_spec = EnvSpec(
        ppo_factory=lambda **kw: PushTVecEnv(**kw),
        apg_factory=lambda **kw: PushTAPGEnv(**kw),
    )
except ImportError:
    _pusht_spec = None

try:
    from point_mass_env import PointMassAPGEnv, PointMassVecEnv

    _pointmass_spec = EnvSpec(
        ppo_factory=lambda **kw: PointMassVecEnv(**kw),
        apg_factory=lambda **kw: PointMassAPGEnv(**kw),
    )
except ImportError:
    _pointmass_spec = None

try:
    from point_mass_simple_env import PointMassSimpleEnv

    _pointmass_simple_spec = EnvSpec(
        ppo_factory=lambda **kw: PointMassSimpleEnv(**kw),
        apg_factory=lambda **kw: PointMassSimpleEnv(**kw),
    )
except ImportError:
    _pointmass_simple_spec = None

try:
    from mountaincar_continuous_diff_env import (
        MountainCarContinuousAPGEnv,
        MountainCarContinuousVecEnv,
    )

    _diff_mountaincar_spec = EnvSpec(
        ppo_factory=lambda **kw: MountainCarContinuousVecEnv(**kw),
        apg_factory=lambda **kw: MountainCarContinuousAPGEnv(**kw),
    )
except ImportError:
    _diff_mountaincar_spec = None


IMPLEMENTED_ENVS: Dict[str, EnvSpec] = {}

# Register FrankaReach only if the dependency is available.
if _franka_spec is not None:
    IMPLEMENTED_ENVS["FrankaReach-v0"] = _franka_spec

# Register PushT (pure PyTorch, no heavy deps — always available).
if _pusht_spec is not None:
    IMPLEMENTED_ENVS["PushT-v0"] = _pusht_spec

# Register PointMassNavigate (pure PyTorch, no heavy deps — always available).
if _pointmass_spec is not None:
    IMPLEMENTED_ENVS["PointMassNavigate-v0"] = _pointmass_spec

# Register simple 1D PointMass target-reaching env.
if _pointmass_simple_spec is not None:
    IMPLEMENTED_ENVS["PointMassSimple-v0"] = _pointmass_simple_spec

# Register DiffMountainCarContinuous
if _diff_mountaincar_spec is not None:
    IMPLEMENTED_ENVS["DiffMountainCarContinuous-v0"] = _diff_mountaincar_spec


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
