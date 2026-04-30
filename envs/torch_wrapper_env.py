"""Wraps standard gymnasium (numpy) vector envs into a torch-compatible interface.

This allows both PPO and APG code paths in rl.py to work with any gymnasium
environment (e.g. CartPole-v1).  For APG the env dynamics are not truly
differentiable — the wrapper detaches inputs before stepping the underlying
numpy env and returns fresh tensors — but the plumbing (torch tensors,
matching spaces) is correct.
"""

import gymnasium as gym
import numpy as np
import torch


class TorchWrapperEnv:
    """Thin wrapper around a gymnasium VectorEnv that speaks torch tensors.

    Attributes:
        single_observation_space: copied from the underlying env
        single_action_space:      copied from the underlying env
        num_envs:                 number of parallel environments
    """

    def __init__(self, envs: gym.vector.VectorEnv):
        self._envs = envs
        self.single_observation_space = envs.single_observation_space
        self.single_action_space = envs.single_action_space
        self.num_envs = envs.num_envs

    # ------------------------------------------------------------------
    # gymnasium-compatible API (torch tensors)
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self._envs.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32), info

    def step(self, action):
        # Convert torch action -> numpy for the underlying env
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        obs, reward, terminated, truncated, info = self._envs.step(action_np)

        obs = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, dtype=torch.bool)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._envs.close()

    def __getattr__(self, name):
        """Forward unknown attributes to the underlying env."""
        return getattr(self._envs, name)
