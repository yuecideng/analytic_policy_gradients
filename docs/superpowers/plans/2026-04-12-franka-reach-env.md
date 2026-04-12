# Franka Reach Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 6D end-effector reaching task for Franka FR3 using Newton physics, trainable with both PPO and APG from `rl.py`.

**Architecture:** Two environment classes sharing Newton simulation logic. `FrankaReachEnv` (gymnasium.Env) for PPO uses standard gym interface. `FrankaReachAPGEnv` for APG uses Warp tape autodiff bridged to PyTorch via a custom `torch.autograd.Function`. Both load the same Franka URDF, compute EEF pose via `newton.eval_fk()`, and use negative L2 distance as reward.

**Tech Stack:** Newton Physics (Warp), PyTorch, Gymnasium, NumPy

---

## File Structure

| File | Responsibility |
|------|---------------|
| `franka_reach_env.py` | Newton sim setup, FrankaReachEnv (PPO), FrankaReachAPGEnv (APG), NewtonStepFunction (gradient bridge) |
| `rl.py` | Minimal edits: import envs, replace NotImplementedError with APG env instantiation, add Franka option for PPO |

---

### Task 1: Create Newton simulation helpers and FrankaReachEnv skeleton

**Files:**
- Create: `franka_reach_env.py`

This task sets up the Newton simulation infrastructure and creates the PPO-compatible gymnasium.Env.

- [ ] **Step 1: Create franka_reach_env.py with imports and constants**

```python
franka_reach_env.py
```

```python
"""Franka FR3 6D end-effector reaching task using Newton Physics.

Supports two modes:
  - PPO: Standard gymnasium.Env (black-box, no gradients)
  - APG: Differentiable env with Warp tape → PyTorch gradient bridge
"""

import math
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import newton
import newton.utils

# Franka FR3 arm has 7 actuated joints + 2 gripper joints = 9 total
FRANKA_NUM_ARM_JOINTS = 7
FRANKA_NUM_JOINTS = 9  # 7 arm + 2 gripper
FRANKA_EE_BODY_NAME = "fr3_hand"
DEFAULT_ACTION_SCALE = 0.1  # radians
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_SUBSTEPS = 4
DEFAULT_DT = 1.0 / 100.0
DEFAULT_W_ROT = 1.0

# Default initial joint configuration for Franka (home position)
DEFAULT_JOINT_Q = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.05, 0.05]

# Target sampling workspace (in front of robot, reachable region)
TARGET_POS_RANGE = {
    "x": (0.3, 0.8),
    "y": (-0.4, 0.4),
    "z": (0.1, 0.8),
}


def _build_franka_model(
    num_envs: int = 1,
    requires_grad: bool = False,
    device: str = "cpu",
):
    """Build a Newton model with Franka FR3 robot(s).

    Returns (model, ee_body_index, articulation_start_indices).
    """
    wp.init()

    if num_envs == 1:
        # Single world
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        # Set initial joint configuration
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(builder.joint_q):
                builder.joint_q[i] = q
                builder.joint_target[i] = q
        model = builder.finalize(device=device, requires_grad=requires_grad)

        # Find EE body index by name
        ee_index = None
        for i, label in enumerate(builder.body_label):
            if FRANKA_EE_BODY_NAME in label:
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10  # fallback: hardcoded hand TCP
        return model, ee_index, None
    else:
        # Multi-world using builder replication
        robot_builder = newton.ModelBuilder()
        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        robot_builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        # Set initial joint configuration
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(robot_builder.joint_q):
                robot_builder.joint_q[i] = q
                robot_builder.joint_target[i] = q

        # Find EE body index in the single-world builder
        ee_index = None
        for i, label in enumerate(robot_builder.body_label):
            if FRANKA_EE_BODY_NAME in label:
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10

        # Build multi-world model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(robot_builder, world_count=num_envs)
        model = builder.finalize(device=device, requires_grad=requires_grad)

        return model, ee_index, None


def _compute_obs(joint_q_np, joint_qd_np, eef_pos, eef_quat, target_pos, target_quat, num_envs=1):
    """Compute observation vector: [joint_pos(7), joint_vel(7), eef_pos(3), eef_quat(4), target_pos(3), target_quat(4)]"""
    obs_list = []
    for i in range(num_envs):
        offset = i * FRANKA_NUM_JOINTS
        jpos = joint_q_np[offset : offset + FRANKA_NUM_ARM_JOINTS]
        jvel = joint_qd_np[offset : offset + FRANKA_NUM_ARM_JOINTS]
        obs_list.append(np.concatenate([jpos, jvel, eef_pos[i], eef_quat[i], target_pos[i], target_quat[i]]))
    return np.stack(obs_list)


def _sample_target(num_envs, rng):
    """Sample random target poses within reachable workspace."""
    target_pos = np.stack([
        np.array([
            rng.uniform(*TARGET_POS_RANGE["x"]),
            rng.uniform(*TARGET_POS_RANGE["y"]),
            rng.uniform(*TARGET_POS_RANGE["z"]),
        ])
        for _ in range(num_envs)
    ])
    # Fixed upright orientation as target (gripper pointing down)
    target_quat = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (num_envs, 1))
    return target_pos.astype(np.float32), target_quat.astype(np.float32)


def _quat_distance(q1, q2):
    """Compute quaternion distance handling double cover."""
    # q and -q represent the same rotation
    d1 = np.sum((q1 - q2) ** 2, axis=-1)
    d2 = np.sum((q1 + q2) ** 2, axis=-1)
    return np.minimum(d1, d2)


def _compute_reward(eef_pos, eef_quat, target_pos, target_quat, w_rot=DEFAULT_W_ROT):
    """Compute reward as negative L2 distance."""
    pos_dist = np.linalg.norm(eef_pos - target_pos, axis=-1)
    rot_dist = _quat_distance(eef_quat, target_quat)
    return (-pos_dist - w_rot * rot_dist).astype(np.float32)
```

- [ ] **Step 2: Implement FrankaReachEnv (gymnasium.Env for PPO)**

Add to `franka_reach_env.py`:

```python
class FrankaReachEnv(gym.Env):
    """Single Franka reach environment compatible with gymnasium (PPO mode).

    Observations: [joint_pos(7), joint_vel(7), eef_pos(3), eef_quat(4), target_pos(3), target_quat(4)] = 28-dim
    Actions: delta joint positions (7-dim), clipped to [-1, 1]
    Reward: negative L2 pose distance
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        substeps: int = DEFAULT_SUBSTEPS,
        dt: float = DEFAULT_DT,
        w_rot: float = DEFAULT_W_ROT,
        device: str = "cpu",
    ):
        super().__init__()

        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.dt = dt
        self.w_rot = w_rot
        self.device = device

        # Build single-world Newton model
        self.model, self.ee_body_index, _ = _build_franka_model(
            num_envs=1, requires_grad=False, device=device
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # State buffers (double-buffered for physics stepping)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Compute initial FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Gymnasium spaces
        obs_dim = FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4 + 3 + 4  # 28
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(FRANKA_NUM_ARM_JOINTS,), dtype=np.float32
        )

        # Episode state
        self.step_count = 0
        self.target_pos = None
        self.target_quat = None
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # Reset joint state to default
        joint_q_np = np.array(DEFAULT_JOINT_Q, dtype=np.float32)
        joint_qd_np = np.zeros(FRANKA_NUM_JOINTS, dtype=np.float32)

        wpjoint_q = wp.array(joint_q_np, dtype=wp.float32, device=self.model.device)
        wpjoint_qd = wp.array(joint_qd_np, dtype=wp.float32, device=self.model.device)

        wp.copy(self.state_0.joint_q, wpjoint_q)
        wp.copy(self.state_0.joint_qd, wpjoint_qd)

        # Reset control targets
        wp.copy(self.control.joint_target_pos, wpjoint_q)

        # Recompute FK
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Sample new target
        self.target_pos, self.target_quat = _sample_target(1, self._rng)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # Convert delta action to absolute joint targets
        current_q = self.state_0.joint_q.numpy()
        new_target = current_q[:FRANKA_NUM_ARM_JOINTS] + action * self.action_scale
        # Keep gripper joints at default
        full_target = np.array(DEFAULT_JOINT_Q, dtype=np.float32)
        full_target[:FRANKA_NUM_ARM_JOINTS] = new_target

        wp_target = wp.array(full_target, dtype=wp.float32, device=self.model.device)
        wp.copy(self.control.joint_target_pos, wp_target)

        # Step physics
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Compute observation and reward
        obs = self._get_obs()
        reward = self._get_reward()

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {}
        if truncated:
            eef_pos = self._get_eef_pos()
            pos_dist = np.linalg.norm(eef_pos[0] - self.target_pos[0])
            info["episode"] = {"r": reward, "l": self.step_count, "success": pos_dist < 0.05}

        return obs, float(reward), terminated, truncated, info

    def _get_eef_pos_quat(self):
        """Get end-effector position and quaternion from current state."""
        body_q = self.state_0.body_q.numpy()
        ee_transform = body_q[self.ee_body_index]
        eef_pos = ee_transform[:3].reshape(1, 3)
        eef_quat = ee_transform[3:7].reshape(1, 4)
        return eef_pos, eef_quat

    def _get_eef_pos(self):
        pos, _ = self._get_eef_pos_quat()
        return pos

    def _get_obs(self):
        joint_q_np = self.state_0.joint_q.numpy()
        joint_qd_np = self.state_0.joint_qd.numpy()
        eef_pos, eef_quat = self._get_eef_pos_quat()
        obs = _compute_obs(
            joint_q_np, joint_qd_np,
            eef_pos, eef_quat,
            self.target_pos, self.target_quat,
            num_envs=1,
        )
        return obs[0]  # single env, return 1D

    def _get_reward(self):
        eef_pos, eef_quat = self._get_eef_pos_quat()
        return _compute_reward(eef_pos, eef_quat, self.target_pos, self.target_quat, self.w_rot)[0]
```

- [ ] **Step 3: Commit FrankaReachEnv skeleton**

```bash
git add franka_reach_env.py
git commit -m "feat: add FrankaReachEnv with Newton simulation (PPO mode)"
```

---

### Task 2: Implement FrankaReachAPGEnv with gradient bridge

**Files:**
- Modify: `franka_reach_env.py` (append APG classes)

This task adds the APG-compatible environment with the Warp tape → PyTorch gradient bridge.

- [ ] **Step 1: Implement NewtonStepFunction (custom torch.autograd.Function)**

Append to `franka_reach_env.py`:

```python
class _NewtonStepFunc(torch.autograd.Function):
    """Bridges Warp tape autodiff to PyTorch autograd.

    Forward: Runs one Newton sim step inside a Warp tape, returns reward as detached tensor.
    Backward: Uses stored Warp tape to compute d(reward)/d(action), returns as PyTorch gradient.
    """

    @staticmethod
    def forward(ctx, action_torch, sim_state):
        """Run one env step in Warp, return reward.

        Args:
            action_torch: [num_envs, 7] action tensor from policy
            sim_state: dict with keys 'model', 'solver', 'control', 'contacts',
                       'state_0', 'state_1', 'substeps', 'dt',
                       'ee_body_index', 'target_pos', 'target_quat', 'w_rot',
                       'num_envs', 'body_world_start_arr'
        Returns:
            reward_torch: [num_envs] detached reward tensor
            obs_torch: [num_envs, 28] detached obs tensor
            terminated_torch: [num_envs] bool tensor
        """
        device_str = str(action_torch.device)
        wp_device = wp.get_device(device_str)

        num_envs = sim_state["num_envs"]
        substeps = sim_state["substeps"]
        dt = sim_state["dt"]
        model = sim_state["model"]
        solver = sim_state["solver"]
        control = sim_state["control"]
        contacts = sim_state["contacts"]
        state_0 = sim_state["state_0"]
        state_1 = sim_state["state_1"]
        ee_idx = sim_state["ee_body_index"]
        target_pos_np = sim_state["target_pos"]
        target_quat_np = sim_state["target_quat"]
        w_rot = sim_state["w_rot"]
        body_world_start = sim_state["body_world_start_arr"]

        # Convert action to Warp array with gradient tracking
        action_wp = wp.from_torch(
            action_torch.detach().clone().contiguous(),
            dtype=wp.float32,
            requires_grad=True,
        )

        # Set joint targets from action (delta joint positions)
        current_q = wp.to_torch(state_0.joint_q).clone()
        action_np = action_torch.detach().cpu().numpy()

        for env_idx in range(num_envs):
            body_start = body_world_start[env_idx]
            ee_global = body_start + ee_idx

            # Get current joint positions for this env
            q_start = env_idx * FRANKA_NUM_JOINTS
            current_jpos = current_q[q_start : q_start + FRANKA_NUM_ARM_JOINTS].numpy()
            new_target = current_jpos + action_np[env_idx] * DEFAULT_ACTION_SCALE
            full_target = np.array(DEFAULT_JOINT_Q, dtype=np.float32)
            full_target[:FRANKA_NUM_ARM_JOINTS] = new_target

            wp_target = wp.array(full_target, dtype=wp.float32, device=model.device)
            # Set control targets for this world's joints
            ctrl_start = env_idx * FRANKA_NUM_JOINTS
            wp.copy(
                control.joint_target_pos[ctrl_start : ctrl_start + FRANKA_NUM_JOINTS],
                wp_target,
            )

        # Run forward in Warp tape
        tape = wp.Tape()
        with tape:
            for _ in range(substeps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            # Compute FK to get updated body poses
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

            # Compute reward as Warp array
            body_q_torch = wp.to_torch(state_0.body_q)
            reward_t = torch.zeros(num_envs, dtype=torch.float32, device=device_str)
            obs_list = []

            for env_idx in range(num_envs):
                body_start = body_world_start[env_idx]
                ee_global = body_start + ee_idx

                ee_transform = body_q_torch[ee_global]
                eef_pos = ee_transform[:3]
                eef_quat = ee_transform[3:7]

                target_pos_t = torch.tensor(target_pos_np[env_idx], dtype=torch.float32, device=device_str)
                target_quat_t = torch.tensor(target_quat_np[env_idx], dtype=torch.float32, device=device_str)

                pos_dist = torch.norm(eef_pos - target_pos_t)
                rot_dist = torch.min(
                    torch.sum((eef_quat - target_quat_t) ** 2),
                    torch.sum((eef_quat + target_quat_t) ** 2),
                )
                reward_t[env_idx] = -pos_dist - w_rot * rot_dist

                # Build obs for this env
                q_start = env_idx * FRANKA_NUM_JOINTS
                joint_q_t = body_q_torch  # not used directly; use state joint_q
                joint_q_all = wp.to_torch(state_0.joint_q)
                joint_qd_all = wp.to_torch(state_0.joint_qd)
                jpos = joint_q_all[q_start : q_start + FRANKA_NUM_ARM_JOINTS]
                jvel = joint_qd_all[q_start : q_start + FRANKA_NUM_ARM_JOINTS]
                obs_i = torch.cat([jpos, jvel, eef_pos, eef_quat, target_pos_t, target_quat_t])
                obs_list.append(obs_i)

        obs_torch = torch.stack(obs_list)

        # Compute total reward as scalar for backward
        reward_sum = reward_t.sum()

        # Save for backward
        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.reward_sum_wp = wp.from_torch(
            reward_sum.detach().clone().contiguous(),
            dtype=wp.float32,
            requires_grad=True,
        )

        return reward_t.detach(), obs_torch.detach(), torch.zeros(num_envs, dtype=torch.bool, device=device_str)

    @staticmethod
    def backward(ctx, grad_reward, grad_obs, grad_terminated):
        """Compute d(reward)/d(action) via Warp tape and return to PyTorch."""
        # Set the gradient of the loss w.r.t. the output
        ctx.reward_sum_wp.grad.fill_(1.0)

        # Backward through Warp tape
        ctx.tape.backward()

        # Extract action gradient from Warp
        action_grad = wp.to_torch(ctx.action_wp.grad)

        # Clean up tape
        ctx.tape.zero()

        return action_grad, None
```

- [ ] **Step 2: Implement FrankaReachAPGEnv**

Append to `franka_reach_env.py`:

```python
class FrankaReachAPGEnv:
    """Batched Franka reach environment for APG with differentiable physics.

    Returns PyTorch tensors connected to the computation graph via the
    Warp tape bridge (_NewtonStepFunc). When loss.backward() is called,
    gradients flow from reward through the physics simulation back to actions.

    Interface matches what rl.py APG loop expects:
      - reset() -> (obs [num_envs, 28], info)
      - step(action [num_envs, 7]) -> (obs, reward, terminated, truncated, info)
      - attributes: single_observation_space, single_action_space, num_envs
    """

    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        substeps: int = DEFAULT_SUBSTEPS,
        dt: float = DEFAULT_DT,
        w_rot: float = DEFAULT_W_ROT,
        device: str = "cpu",
    ):
        self._num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.dt = dt
        self.w_rot = w_rot
        self.device_str = device

        # Build multi-world Newton model
        self.model, self.ee_body_index, _ = _build_franka_model(
            num_envs=num_envs, requires_grad=True, device=device
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Compute body_world_start for indexing
        body_world_start = self.model.body_world_start.numpy()
        self.body_world_start_arr = body_world_start

        # Initial FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Gymnasium spaces (for single env)
        obs_dim = FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4 + 3 + 4  # 28
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(FRANKA_NUM_ARM_JOINTS,), dtype=np.float32
        )

        # Episode state
        self.step_count = 0
        self.target_pos = None
        self.target_quat = None
        self._rng = np.random.default_rng()

    @property
    def num_envs(self):
        return self._num_envs

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # Reset all worlds to default joint configuration
        joint_q_np = np.tile(DEFAULT_JOINT_Q, self._num_envs).astype(np.float32)
        joint_qd_np = np.zeros(self._num_envs * FRANKA_NUM_JOINTS, dtype=np.float32)

        wpjoint_q = wp.array(joint_q_np, dtype=wp.float32, device=self.model.device)
        wpjoint_qd = wp.array(joint_qd_np, dtype=wp.float32, device=self.model.device)

        wp.copy(self.state_0.joint_q, wpjoint_q)
        wp.copy(self.state_0.joint_qd, wpjoint_qd)
        wp.copy(self.control.joint_target_pos, wpjoint_q)

        # Recompute FK
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Sample new targets
        self.target_pos, self.target_quat = _sample_target(self._num_envs, self._rng)

        obs = self._get_obs_detached()
        info = {}
        return obs, info

    def step(self, action):
        """Step all envs. action: [num_envs, 7] torch tensor (with grad from policy)."""
        self.step_count += 1

        # Package sim state for the autograd function
        sim_state = {
            "model": self.model,
            "solver": self.solver,
            "control": self.control,
            "contacts": self.contacts,
            "state_0": self.state_0,
            "state_1": self.state_1,
            "substeps": self.substeps,
            "dt": self.dt,
            "ee_body_index": self.ee_body_index,
            "target_pos": self.target_pos,
            "target_quat": self.target_quat,
            "w_rot": self.w_rot,
            "num_envs": self._num_envs,
            "body_world_start_arr": self.body_world_start_arr,
        }

        reward, obs, terminated = _NewtonStepFunc.apply(action, sim_state)

        truncated = torch.full((self._num_envs,), self.step_count >= self.max_episode_steps,
                               dtype=torch.bool, device=action.device)

        infos = {}
        if truncated.any():
            infos["final_info"] = [
                {"episode": {"r": reward[i].item(), "l": self.step_count}}
                for i in range(self._num_envs)
                if truncated[i]
            ]

        return obs, reward, terminated, truncated, infos

    def _get_obs_detached(self):
        """Get observation as detached tensor (for reset, no gradient needed)."""
        body_q_torch = wp.to_torch(self.state_0.body_q)
        joint_q_torch = wp.to_torch(self.state_0.joint_q)
        joint_qd_torch = wp.to_torch(self.state_0.joint_qd)

        obs_list = []
        for i in range(self._num_envs):
            body_start = self.body_world_start_arr[i]
            ee_global = body_start + self.ee_body_index

            ee_transform = body_q_torch[ee_global]
            eef_pos = ee_transform[:3]
            eef_quat = ee_transform[3:7]

            q_start = i * FRANKA_NUM_JOINTS
            jpos = joint_q_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]
            jvel = joint_qd_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]

            target_pos_t = torch.tensor(self.target_pos[i], dtype=torch.float32, device=self.device_str)
            target_quat_t = torch.tensor(self.target_quat[i], dtype=torch.float32, device=self.device_str)

            obs_i = torch.cat([jpos, jvel, eef_pos, eef_quat, target_pos_t, target_quat_t])
            obs_list.append(obs_i)

        return torch.stack(obs_list)
```

- [ ] **Step 3: Commit APG env**

```bash
git add franka_reach_env.py
git commit -m "feat: add FrankaReachAPGEnv with Warp tape gradient bridge"
```

---

### Task 3: Integrate with rl.py

**Files:**
- Modify: `rl.py`

Minimal changes to support the Franka reach task in both PPO and APG modes.

- [ ] **Step 1: Add Franka env import and factory function**

At the top of `rl.py` (after existing imports), add:

```python
from franka_reach_env import FrankaReachEnv, FrankaReachAPGEnv
```

- [ ] **Step 2: Replace PPO env setup to support Franka**

In the PPO branch of the `__main__` block, replace lines 280-284:

```python
        if args.algorithm == "ppo":
            if args.env_id == "FrankaReach-v0":
                gym_envs = gym.vector.SyncVectorEnv(
                    [lambda i=i: FrankaReachEnv(device="cpu") for i in range(args.num_envs)],
                )
            else:
                gym_envs = gym.vector.SyncVectorEnv(
                    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
                )
            envs = TorchSyncVecEnv(gym_envs)
```

- [ ] **Step 3: Replace APG NotImplementedError with FrankaReachAPGEnv**

Replace lines 285-291:

```python
        else:
            envs = FrankaReachAPGEnv(
                num_envs=args.num_envs,
                max_episode_steps=200,
                device=str(device),
            )
```

- [ ] **Step 4: Commit rl.py integration**

```bash
git add rl.py
git commit -m "feat: integrate FrankaReachEnv with PPO and APG training loops"
```

---

### Task 4: Smoke test

- [ ] **Step 1: Test PPO env creation and step**

```bash
cd /home/dex/workspace/sources/analytic_policy_gradients
python -c "
from franka_reach_env import FrankaReachEnv
env = FrankaReachEnv()
obs, info = env.reset(seed=42)
print('Obs shape:', obs.shape)
print('Obs:', obs)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print('Reward:', reward)
print('Step obs shape:', obs.shape)
print('PPO env OK')
"
```

Expected: Obs shape (28,), reasonable reward value, no errors.

- [ ] **Step 2: Test APG env creation and step**

```bash
python -c "
import torch
from franka_reach_env import FrankaReachAPGEnv
env = FrankaReachAPGEnv(num_envs=2, device='cpu')
obs, info = env.reset(seed=42)
print('Obs shape:', obs.shape)
action = torch.randn(2, 7, requires_grad=True)
obs, reward, terminated, truncated, infos = env.step(action)
print('Reward shape:', reward.shape)
print('Reward:', reward)
print('APG env OK')
"
```

Expected: Obs shape (2, 28), reward shape (2,), no errors.

- [ ] **Step 3: Test APG gradient flow**

```bash
python -c "
import torch
from franka_reach_env import FrankaReachAPGEnv
env = FrankaReachAPGEnv(num_envs=2, device='cpu')
obs, info = env.reset(seed=42)
action = torch.randn(2, 7, requires_grad=True)
obs, reward, terminated, truncated, infos = env.step(action)
loss = -reward.sum()
loss.backward()
print('Action grad shape:', action.grad.shape)
print('Action grad norm:', action.grad.norm().item())
print('Gradient flow OK')
"
```

Expected: action.grad is not None, has shape (2, 7), non-zero norm.

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -A && git commit -m "fix: address smoke test issues"
```
