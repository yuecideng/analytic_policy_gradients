import torch
import numpy as np
import gymnasium as gym


class PointMassSimpleEnv(gym.Env):
    def __init__(self, num_envs=1, max_episode_steps=50, device="cpu", **kwargs):
        """
        额外参数 (如 headless) 会被 kwargs 吸收，避免 TypeError。
        """
        super().__init__()
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.device = device

        # 动力学参数
        self.dt = 0.1
        self.damping = 0.5
        self.max_force = 1.0

        # 状态变量（在 reset 中初始化）
        self.pos = torch.zeros(num_envs, device=device)
        self.vel = torch.zeros(num_envs, device=device)
        self.target = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # 标准 gym 空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(self, seed=None, **kwargs):
        # 正确处理随机种子
        if seed is not None:
            super().reset(seed=seed)
            torch.manual_seed(seed)
        else:
            super().reset()

        # 随机初始化
        self.pos = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0
        self.vel = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.5
        self.target = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0
        self.step_count.zero_()
        return self._get_obs(), {}

    def step(self, action):
        if action.dim() == 2:
            action = action.squeeze(-1)
        force = torch.clamp(action, -1.0, 1.0) * self.max_force

        acc = force - self.damping * self.vel
        self.vel = self.vel + acc * self.dt
        self.pos = self.pos + self.vel * self.dt

        error = self.pos - self.target
        reward = -(error**2)

        self.step_count = self.step_count + 1
        truncated = self.step_count >= self.max_episode_steps
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        done = truncated | terminated

        # 记录 episode 信息（用于日志）
        infos = {"final_info": [None] * self.num_envs}
        for i in range(self.num_envs):
            if done[i]:
                infos["final_info"][i] = {
                    "episode": {"r": reward[i].item(), "l": self.step_count[i].item()}
                }
        if done.any():
            self._auto_reset(done)

        obs = self._get_obs().detach()
        return obs, reward, terminated, truncated, infos

    def _get_obs(self):
        return torch.stack([self.pos, self.vel, self.target], dim=-1)

    def _auto_reset(self, done_mask):
        with torch.no_grad():
            idx = done_mask.nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                return
            self.pos = self.pos.detach().clone()
            self.vel = self.vel.detach().clone()
            self.target = self.target.detach().clone()
            self.pos[idx] = (torch.rand(len(idx), device=self.device) - 0.5) * 2.0
            self.vel[idx] = (torch.rand(len(idx), device=self.device) - 0.5) * 0.5
            self.target[idx] = (torch.rand(len(idx), device=self.device) - 0.5) * 2.0
            self.step_count[idx] = 0

    def detach_state(self):
        self.pos = self.pos.detach()
        self.vel = self.vel.detach()
        self.target = self.target.detach()

    def close(self):
        pass
