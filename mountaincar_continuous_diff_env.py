import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

MIN_ACTION = -1.0
MAX_ACTION = 1.0
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.45
GOAL_VELOCITY = 0.0
POWER = 0.0015
GRAVITY = 0.0025
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_REWARD_MODE = "smooth"


def _set_local_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _format_action(action: torch.Tensor | np.ndarray | list, device: torch.device) -> torch.Tensor:
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device)
    if action_t.ndim == 1:
        action_t = action_t.unsqueeze(-1)
    return torch.clamp(action_t, MIN_ACTION, MAX_ACTION)


def _check_success(
    position: torch.Tensor,
    velocity: torch.Tensor,
    goal_position: float = GOAL_POSITION,
    goal_velocity: float = GOAL_VELOCITY,
) -> torch.Tensor:
    return (position >= goal_position) & (velocity >= goal_velocity)


def _transition(
    position: torch.Tensor,
    velocity: torch.Tensor,
    action: torch.Tensor,
    power: float = POWER,
) -> tuple[torch.Tensor, torch.Tensor]:
    force = action.squeeze(-1)
    next_velocity = velocity + force * power - GRAVITY * torch.cos(3.0 * position)
    next_velocity = torch.clamp(next_velocity, -MAX_SPEED, MAX_SPEED)

    next_position = position + next_velocity
    next_position = torch.clamp(next_position, MIN_POSITION, MAX_POSITION)

    hit_left_wall = (next_position <= MIN_POSITION) & (next_velocity < 0.0)
    next_velocity = torch.where(hit_left_wall, torch.zeros_like(next_velocity), next_velocity)
    return next_position, next_velocity


def _compute_reward(
    position: torch.Tensor,
    velocity: torch.Tensor,
    action: torch.Tensor,
    reward_mode: str = DEFAULT_REWARD_MODE,
) -> torch.Tensor:
    force = action.squeeze(-1)
    action_penalty = 0.1 * force.square()

    if reward_mode == "gym":
        success = _check_success(position, velocity)
        return success.to(position.dtype) * 100.0 - action_penalty

    if reward_mode == "smooth":
        normalized_position = (position - MIN_POSITION) / (GOAL_POSITION - MIN_POSITION)
        normalized_position = torch.clamp(normalized_position, 0.0, 1.2)
        goal_bonus = torch.sigmoid((position - GOAL_POSITION) / 0.02)
        velocity_term = velocity / MAX_SPEED
        return 2.0 * normalized_position + 4.0 * goal_bonus + 0.5 * velocity_term - action_penalty

    raise ValueError(f"Unsupported reward_mode: {reward_mode}")


class _MountainCarContinuousBaseEnv:
    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
        reward_mode: str = DEFAULT_REWARD_MODE,
    ):
        self._num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.headless = headless
        self.reward_mode = reward_mode

        self.single_observation_space = gym.spaces.Box(
            low=np.array([MIN_POSITION, -MAX_SPEED], dtype=np.float32),
            high=np.array([MAX_POSITION, MAX_SPEED], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=np.array([MIN_ACTION], dtype=np.float32),
            high=np.array([MAX_ACTION], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        self.position = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.velocity = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        self.last_action = torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self.device)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def _normalize_env_ids(self, env_ids=None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._num_envs, device=self.device, dtype=torch.long)
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        if env_ids_t.dtype == torch.bool:
            return torch.nonzero(env_ids_t, as_tuple=False).squeeze(-1).to(torch.long)
        return env_ids_t.to(torch.long).reshape(-1)

    def _sample_reset_state(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = -0.6 + 0.2 * torch.rand(n, device=self.device, dtype=torch.float32)
        velocity = torch.zeros(n, device=self.device, dtype=torch.float32)
        return position, velocity

    def _get_obs(
        self,
        position: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position = self.position if position is None else position
        velocity = self.velocity if velocity is None else velocity
        return torch.stack((position, velocity), dim=-1)

    def _reset_selected(self, env_ids_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position, velocity = self._sample_reset_state(env_ids_t.numel())
        self.position[env_ids_t] = position
        self.velocity[env_ids_t] = velocity
        self.step_count[env_ids_t] = 0
        self.last_action[env_ids_t] = 0.0
        return position, velocity

    def reset(self, env_ids=None, seed=None, options=None):
        del options
        if seed is not None:
            _set_local_seed(seed)

        env_ids_t = self._normalize_env_ids(env_ids)
        if env_ids_t.numel() > 0:
            self._reset_selected(env_ids_t)

        return self._get_obs(), {}

    def _step_impl(self, action, preserve_grad: bool):
        action_t = _format_action(action, self.device)
        next_position, next_velocity = _transition(self.position, self.velocity, action_t)
        reward = _compute_reward(
            next_position,
            next_velocity,
            action_t,
            reward_mode=self.reward_mode,
        )
        terminated = _check_success(next_position, next_velocity)
        next_step_count = self.step_count + 1
        truncated = next_step_count >= self.max_episode_steps
        done = terminated | truncated

        if preserve_grad:
            stored_position = next_position
            stored_velocity = next_velocity
            stored_last_action = action_t
        else:
            stored_position = next_position.detach()
            stored_velocity = next_velocity.detach()
            stored_last_action = action_t.detach()

        self.position = stored_position
        self.velocity = stored_velocity
        self.step_count = next_step_count
        self.last_action = stored_last_action

        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            reset_position, reset_velocity = self._sample_reset_state(done_ids.numel())
            self.position = self.position.clone()
            self.velocity = self.velocity.clone()
            self.last_action = self.last_action.clone()
            self.step_count = self.step_count.clone()
            self.position[done_ids] = reset_position
            self.velocity[done_ids] = reset_velocity
            self.last_action[done_ids] = 0.0
            self.step_count[done_ids] = 0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def step(self, action):
        return self._step_impl(action, preserve_grad=False)

    def detach_state(self):
        self.position = self.position.detach()
        self.velocity = self.velocity.detach()
        self.last_action = self.last_action.detach()

    def close(self):
        return None


class MountainCarContinuousVecEnv(_MountainCarContinuousBaseEnv):
    """Torch-native batched MountainCarContinuous environment for PPO-style use."""


class MountainCarContinuousAPGEnv(_MountainCarContinuousBaseEnv):
    """Differentiable batched MountainCarContinuous environment for APG."""

    def step(self, action):
        return self._step_impl(action, preserve_grad=True)
