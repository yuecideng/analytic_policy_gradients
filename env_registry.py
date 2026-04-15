from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from franka_reach_env import FrankaReachAPGEnv, FrankaReachEnv, FrankaReachVecEnv


@dataclass(frozen=True)
class EnvSpec:
    """Specification for implemented custom envs."""

    ppo_factory: Optional[Callable[..., Any]] = None
    apg_factory: Optional[Callable[..., Any]] = None


IMPLEMENTED_ENVS: Dict[str, EnvSpec] = {
    "FrankaReach-v0": EnvSpec(
        ppo_factory=lambda **kwargs: FrankaReachVecEnv(**kwargs),
        apg_factory=lambda **kwargs: FrankaReachAPGEnv(**kwargs),
    ),
}


def get_env_spec(env_id: str) -> Optional[EnvSpec]:
    return IMPLEMENTED_ENVS.get(env_id)
