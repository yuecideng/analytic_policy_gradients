"""Franka FR3 6D end-effector reaching task using Newton Physics.

Supports two modes:
  - PPO: Standard gymnasium.Env (black-box, no gradients)
  - APG: Differentiable env with Warp tape -> PyTorch gradient bridge
"""

import math
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import newton
import newton.utils
import newton.viewer as newton_viewer

# Franka FR3 arm has 7 actuated arm joints.
FRANKA_NUM_ARM_JOINTS = 7
FRANKA_EE_BODY_NAME = "fr3_link8"  # last link (flange) in fr3.urdf
DEFAULT_ACTION_SCALE = 0.1  # radians
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_SUBSTEPS = 4
DEFAULT_DT = 1.0 / 100.0
DEFAULT_W_ROT = 1.0
DEFAULT_ARM_JOINT_LIMIT = 2.9

# Default initial joint configuration for Franka (home position):
# 7 arm joints + 2 finger joints.
DEFAULT_ARM_JOINT_Q = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
DEFAULT_FINGER_Q = [0.0, 0.0]
DEFAULT_JOINT_Q = DEFAULT_ARM_JOINT_Q + DEFAULT_FINGER_Q
FRANKA_NUM_JOINTS = len(DEFAULT_JOINT_Q)

# Target sampling workspace (in front of robot, reachable region)
TARGET_POS_RANGE = {
    "x": (0.3, 0.8),
    "y": (-0.4, 0.4),
    "z": (0.1, 0.8),
}


def _build_viewer(model, headless: bool):
    """Create Newton viewer with explicit headless control."""
    if headless:
        viewer = newton_viewer.ViewerNull()
    else:
        viewer = newton_viewer.ViewerGL(headless=False)
    viewer.set_model(model)
    return viewer


def _resolve_urdf_path():
    """Resolve Franka URDF path, trying multiple locations."""
    # 1. Try Newton asset cache (requires network for first download)
    try:
        import os

        if os.environ.get("_NEWTON_SKIP_DOWNLOAD"):
            raise RuntimeError("skip")
        urdf = (
            newton.utils.download_asset("franka_emika_panda")
            / "urdf/fr3_franka_hand.urdf"
        )
        if urdf.exists():
            return str(urdf)
    except Exception:
        pass

    # 2. Try local fallback paths
    candidates = [
        Path.home() / "Downloads" / "robot_challenge_model" / "franka_fr3" / "fr3.urdf",
        Path("/home/dex/Downloads/robot_challenge_model/franka_fr3/fr3.urdf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Franka URDF not found. Either set up network access for Newton asset download, "
        "or place the FR3 URDF at one of: " + ", ".join(str(c) for c in candidates)
    )


def _build_franka_model(num_envs=1, requires_grad=False, device="cpu", urdf_path=None):
    """Build a Newton model with Franka FR3 robot(s).

    Returns (model, ee_body_index).
    """
    cache_dir = os.environ.get("WARP_CACHE_DIR")
    if cache_dir:
        wp.config.kernel_cache_dir = cache_dir

    wp.init()

    if urdf_path is None:
        urdf_path = _resolve_urdf_path()

    if num_envs == 1:
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(builder.joint_q):
                builder.joint_q[i] = q
                builder.joint_target_pos[i] = q
        model = builder.finalize(device=device, requires_grad=requires_grad)

        ee_index = None
        for i, key in enumerate(builder.body_label):
            if FRANKA_EE_BODY_NAME in str(key):
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10
        return model, ee_index
    else:
        robot_builder = newton.ModelBuilder()
        robot_builder.add_urdf(
            str(urdf_path),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        for i, q in enumerate(DEFAULT_JOINT_Q):
            if i < len(robot_builder.joint_q):
                robot_builder.joint_q[i] = q
                robot_builder.joint_target_pos[i] = q

        ee_index = None
        for i, key in enumerate(robot_builder.body_label):
            if FRANKA_EE_BODY_NAME in str(key):
                ee_index = i
                break
        if ee_index is None:
            ee_index = 10

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(robot_builder, world_count=num_envs)
        model = builder.finalize(device=device, requires_grad=requires_grad)

        return model, ee_index


def _sample_target(num_envs, rng):
    """Sample random target poses within reachable workspace."""
    target_pos = np.stack(
        [
            np.array(
                [
                    rng.uniform(*TARGET_POS_RANGE["x"]),
                    rng.uniform(*TARGET_POS_RANGE["y"]),
                    rng.uniform(*TARGET_POS_RANGE["z"]),
                ]
            )
            for _ in range(num_envs)
        ]
    )
    # Fixed upright orientation as target
    target_quat = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (num_envs, 1))
    return target_pos.astype(np.float32), target_quat.astype(np.float32)


def _quat_distance(q1, q2):
    """Quaternion distance handling double cover (q and -q = same rotation)."""
    d1 = np.sum((q1 - q2) ** 2, axis=-1)
    d2 = np.sum((q1 + q2) ** 2, axis=-1)
    return np.minimum(d1, d2)


def _compute_reward(eef_pos, eef_quat, target_pos, target_quat, w_rot=DEFAULT_W_ROT):
    """Negative L2 pose distance reward."""
    pos_dist = np.linalg.norm(eef_pos - target_pos, axis=-1)
    rot_dist = _quat_distance(eef_quat, target_quat)
    return (-pos_dist - w_rot * rot_dist).astype(np.float32)


# ---------------------------------------------------------------------------
# PPO Environment (single gymnasium.Env)
# ---------------------------------------------------------------------------


class FrankaReachEnv(gym.Env):
    """Single Franka reach environment compatible with gymnasium (PPO mode).

    Observations: [joint_pos(7), joint_vel(7), eef_pos(3), eef_quat(4),
                   target_pos(3), target_quat(4)] = 28-dim
    Actions: delta joint positions (7-dim), clipped to [-1, 1]
    Reward: negative L2 pose distance
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        substeps: int = DEFAULT_SUBSTEPS,
        dt: float = DEFAULT_DT,
        w_rot: float = DEFAULT_W_ROT,
        device: str = "cpu",
        headless: bool = True,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.dt = dt
        self.w_rot = w_rot
        self.device = device
        self.headless = headless

        # Build single-world Newton model
        self.model, self.ee_body_index = _build_franka_model(
            num_envs=1,
            requires_grad=False,
            device=device,
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer = _build_viewer(self.model, headless=self.headless)
        self._sim_time = 0.0

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        self._render_current_state()

        obs_dim = FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4 + 3 + 4  # 28
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FRANKA_NUM_ARM_JOINTS,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.target_pos = None
        self.target_quat = None
        self._rng = np.random.default_rng()

    def _set_default_state(self):
        joint_q_np = np.array(DEFAULT_JOINT_Q, dtype=np.float32)
        joint_qd_np = np.zeros(FRANKA_NUM_JOINTS, dtype=np.float32)

        wp.copy(
            self.state_0.joint_q,
            wp.array(joint_q_np, dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.state_0.joint_qd,
            wp.array(joint_qd_np, dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.control.joint_target_pos,
            wp.array(joint_q_np, dtype=wp.float32, device=self.model.device),
        )
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0

        # Reset to default joint state
        self._set_default_state()

        self.target_pos, self.target_quat = _sample_target(1, self._rng)
        self._sim_time = 0.0
        self._render_current_state()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Delta action -> absolute joint target
        current_q = self.state_0.joint_q.numpy()
        new_target = current_q[:FRANKA_NUM_ARM_JOINTS] + action * self.action_scale
        new_target = np.clip(
            new_target, -DEFAULT_ARM_JOINT_LIMIT, DEFAULT_ARM_JOINT_LIMIT
        )
        full_target = np.array(DEFAULT_JOINT_Q, dtype=np.float32)
        full_target[:FRANKA_NUM_ARM_JOINTS] = new_target

        wp.copy(
            self.control.joint_target_pos,
            wp.array(full_target, dtype=wp.float32, device=self.model.device),
        )

        # Physics substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._sim_time += self.dt

        obs = self._get_obs()
        self._render_current_state()
        if not np.isfinite(obs).all():
            # Recover from occasional solver divergence and end the episode.
            self._set_default_state()
            self.target_pos, self.target_quat = _sample_target(1, self._rng)
            obs = self._get_obs()
            return obs, -100.0, True, True, {"nan_recovery": True}
        reward = self._get_reward()

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {}
        if truncated:
            eef_pos = self._get_eef_pos()
            pos_dist = np.linalg.norm(eef_pos - self.target_pos[0])
            info["episode"] = {"r": float(reward), "l": self.step_count}

        return obs, float(reward), terminated, truncated, info

    def _render_current_state(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render(self):
        self._render_current_state()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_eef_pose(self):
        body_q = self.state_0.body_q.numpy()
        ee = body_q[self.ee_body_index]
        return ee[:3], ee[3:7]

    def _get_eef_pos(self):
        pos, _ = self._get_eef_pose()
        return pos

    def _get_obs(self):
        joint_q_np = self.state_0.joint_q.numpy()
        joint_qd_np = self.state_0.joint_qd.numpy()
        eef_pos, eef_quat = self._get_eef_pose()
        return np.concatenate(
            [
                joint_q_np[:FRANKA_NUM_ARM_JOINTS],
                joint_qd_np[:FRANKA_NUM_ARM_JOINTS],
                eef_pos,
                eef_quat,
                self.target_pos[0],
                self.target_quat[0],
            ]
        ).astype(np.float32)

    def _get_reward(self):
        eef_pos, eef_quat = self._get_eef_pose()
        return _compute_reward(
            eef_pos.reshape(1, 3),
            eef_quat.reshape(1, 4),
            self.target_pos,
            self.target_quat,
            self.w_rot,
        )[0]


# ---------------------------------------------------------------------------
# APG Environment (batched, differentiable)
# ---------------------------------------------------------------------------

# Warp kernels for differentiable action-to-control and reward computation


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),  # [num_envs * 7] flat action
    current_q: wp.array(dtype=wp.float32),  # joint_q from state
    target_pos: wp.array(dtype=wp.float32),  # control.joint_target
    action_scale: wp.float32,
    num_joints_per_env: wp.int32,
    num_arm_joints: wp.int32,
    total_dims: wp.int32,  # num_envs * num_arm_joints
):
    """Compute joint targets from delta actions: target = current + action * scale."""
    tid = wp.tid()
    if tid < total_dims:
        env_idx = tid / num_arm_joints
        j = tid % num_arm_joints
        q_offset = env_idx * num_joints_per_env + j
        target_pos[q_offset] = current_q[q_offset] + action[tid] * action_scale


@wp.kernel
def _compute_reward_kernel(
    body_q: wp.array(dtype=wp.transformf),
    ee_body_indices: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3f),
    w_rot: wp.float32,
    reward_out: wp.array(dtype=wp.float32),
):
    """Compute reward = -||eef_pos - target_pos||^2 (squared distance, fully differentiable)."""
    env_idx = wp.tid()
    ee_global = ee_body_indices[env_idx]

    ee_transform = body_q[ee_global]
    eef_pos = wp.transform_get_translation(ee_transform)

    t_pos = target_pos[env_idx]

    # Squared position distance (differentiable, no sqrt)
    diff = eef_pos - t_pos
    pos_dist_sq = wp.dot(diff, diff)

    reward_out[env_idx] = -pos_dist_sq


class _NewtonStepFunc(torch.autograd.Function):
    """Bridges Warp tape autodiff to PyTorch autograd.

    For APG, we bypass the dynamics solver and compute reward directly from FK.
    This is because the Featherstone solver doesn't propagate gradients through
    the control input. The differentiable path is:
        action → joint_q → FK → body_q → EEF pose → reward

    Forward: runs FK inside Warp tape, returns reward + obs.
    Backward: uses stored Warp tape to compute d(reward)/d(action).
    """

    @staticmethod
    def forward(ctx, action_torch, sim_state):
        model = sim_state["model"]
        ee_indices_wp = sim_state["ee_body_indices_wp"]
        target_pos_np = sim_state["target_pos"]
        num_envs = sim_state["num_envs"]

        device_str = str(action_torch.device)

        # Convert action to Warp with gradient tracking
        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        # Create joint_q from action: new_q = current_q + action * scale
        # We need the state's joint_q to compute the target, but the gradient
        # should flow through action_wp. We compute the new joint_q in a kernel.
        num_joints_per_env = FRANKA_NUM_JOINTS
        new_joint_q = wp.zeros(
            num_envs * num_joints_per_env,
            dtype=wp.float32,
            device=model.device,
            requires_grad=True,
        )

        # Target and reward arrays
        target_pos_wp = wp.array(
            target_pos_np.tolist(), dtype=wp.vec3f, device=model.device
        )
        reward_wp = wp.zeros(
            num_envs, dtype=wp.float32, device=model.device, requires_grad=True
        )

        # Create temporary state for FK
        fk_state = model.state()

        tape = wp.Tape()
        with tape:
            # Compute new joint_q = current + action * scale
            wp.launch(
                _set_joint_targets_kernel,
                dim=num_envs * FRANKA_NUM_ARM_JOINTS,
                inputs=[
                    action_wp,
                    model.joint_q,  # current joint positions (constant, no grad needed)
                    new_joint_q,  # output: new joint positions (differentiable)
                    wp.float32(DEFAULT_ACTION_SCALE),
                    wp.int32(num_joints_per_env),
                    wp.int32(FRANKA_NUM_ARM_JOINTS),
                    wp.int32(num_envs * FRANKA_NUM_ARM_JOINTS),
                ],
                device=model.device,
            )

            # Set zero velocities
            wp.copy(fk_state.joint_qd, model.joint_qd)

            # Compute FK to get EEF pose from new joint positions
            newton.eval_fk(model, new_joint_q, fk_state.joint_qd, fk_state)

            # Compute reward in Warp
            wp.launch(
                _compute_reward_kernel,
                dim=num_envs,
                inputs=[
                    fk_state.body_q,
                    ee_indices_wp,
                    target_pos_wp,
                    wp.float32(1.0),
                ],
                outputs=[reward_wp],
                device=model.device,
            )

        # Extract results
        reward_t = wp.to_torch(reward_wp).detach().clone()

        # Build observation (detached)
        body_q_torch = wp.to_torch(fk_state.body_q)
        joint_q_torch = wp.to_torch(new_joint_q)
        joint_qd_torch = wp.to_torch(model.joint_qd)  # zero velocities
        ee_indices_np = wp.to_torch(ee_indices_wp).numpy()

        obs_list = []
        for i in range(num_envs):
            ee_global = int(ee_indices_np[i])
            ee_transform = body_q_torch[ee_global]
            eef_pos = ee_transform[:3]
            eef_quat = ee_transform[3:7]

            q_start = i * FRANKA_NUM_JOINTS
            jpos = joint_q_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]
            jvel = joint_qd_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]

            target_pos_t = torch.tensor(
                target_pos_np[i], dtype=torch.float32, device=device_str
            )
            target_quat_t = torch.zeros(4, dtype=torch.float32, device=device_str)
            obs_list.append(
                torch.cat([jpos, jvel, eef_pos, eef_quat, target_pos_t, target_quat_t])
            )

        obs_torch = torch.stack(obs_list).detach()

        # Save for backward
        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.reward_wp = reward_wp

        return (
            reward_t,
            obs_torch,
            torch.zeros(num_envs, dtype=torch.bool, device=device_str),
        )

    @staticmethod
    def backward(ctx, grad_reward, grad_obs, grad_terminated):
        # Set incoming gradient on the reward Warp array
        grad_reward_wp = wp.from_torch(
            grad_reward.detach().clone().contiguous(), dtype=wp.float32
        )
        wp.copy(ctx.reward_wp.grad, grad_reward_wp)

        # Backward through Warp tape
        ctx.tape.backward()

        # Extract action gradient and reshape
        action_grad = (
            wp.to_torch(ctx.action_wp.grad).clone().reshape(grad_reward.shape[0], -1)
        )
        ctx.tape.zero()

        return action_grad, None


class FrankaReachAPGEnv:
    """Batched Franka reach environment for APG with differentiable physics.

    Returns PyTorch tensors connected to the computation graph via the
    Warp tape bridge (_NewtonStepFunc).

    Interface matches rl.py APG loop:
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
        headless: bool = True,
    ):
        self._num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.dt = dt
        self.w_rot = w_rot
        self.device_str = device
        self.headless = headless

        # Build multi-world Newton model with gradient support
        self.model, self.ee_body_index = _build_franka_model(
            num_envs=num_envs,
            requires_grad=True,
            device=device,
        )
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer = _build_viewer(self.model, headless=self.headless)
        self._sim_time = 0.0

        # Global EEF body index per replicated environment.
        ee_indices = [
            i
            for i, key in enumerate(self.model.body_label)
            if FRANKA_EE_BODY_NAME in str(key)
        ]
        if len(ee_indices) != self._num_envs:
            raise RuntimeError(
                f"Expected {self._num_envs} '{FRANKA_EE_BODY_NAME}' bodies, found {len(ee_indices)}."
            )
        self.ee_body_indices = np.asarray(ee_indices, dtype=np.int32)
        self.ee_body_indices_wp = wp.array(
            self.ee_body_indices, dtype=wp.int32, device=self.model.device
        )

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        self._render_current_state()

        obs_dim = FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4 + 3 + 4  # 28
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(FRANKA_NUM_ARM_JOINTS,),
            dtype=np.float32,
        )

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

        joint_q_np = np.tile(DEFAULT_JOINT_Q, self._num_envs).astype(np.float32)
        joint_qd_np = np.zeros(self._num_envs * FRANKA_NUM_JOINTS, dtype=np.float32)

        wp.copy(
            self.state_0.joint_q,
            wp.array(joint_q_np, dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.state_0.joint_qd,
            wp.array(joint_qd_np, dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.control.joint_target_pos,
            wp.array(joint_q_np, dtype=wp.float32, device=self.model.device),
        )

        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        self.target_pos, self.target_quat = _sample_target(self._num_envs, self._rng)
        self._sim_time = 0.0
        self._render_current_state()
        return self._get_obs_detached(), {}

    def step(self, action):
        """Step all envs. action: [num_envs, 7] torch tensor (with grad)."""
        self.step_count += 1
        action = torch.clamp(action, -1.0, 1.0)

        sim_state = {
            "model": self.model,
            "ee_body_indices_wp": self.ee_body_indices_wp,
            "target_pos": self.target_pos,
            "num_envs": self._num_envs,
        }

        reward, obs, terminated = _NewtonStepFunc.apply(action, sim_state)
        self._sim_time += self.dt
        self._render_current_state()

        truncated = torch.full(
            (self._num_envs,),
            self.step_count >= self.max_episode_steps,
            dtype=torch.bool,
            device=action.device,
        )

        infos = {}
        if truncated.any():
            infos["final_info"] = [
                {"episode": {"r": reward[i].item(), "l": self.step_count}}
                for i in range(self._num_envs)
                if truncated[i]
            ]

        return obs, reward, terminated, truncated, infos

    def _render_current_state(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs_detached(self):
        """Get observation as detached tensor (for reset)."""
        body_q_torch = wp.to_torch(self.state_0.body_q)
        joint_q_torch = wp.to_torch(self.state_0.joint_q)
        joint_qd_torch = wp.to_torch(self.state_0.joint_qd)

        obs_list = []
        for i in range(self._num_envs):
            ee_global = int(self.ee_body_indices[i])
            ee_transform = body_q_torch[ee_global]
            eef_pos = ee_transform[:3]
            eef_quat = ee_transform[3:7]

            q_start = i * FRANKA_NUM_JOINTS
            jpos = joint_q_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]
            jvel = joint_qd_torch[q_start : q_start + FRANKA_NUM_ARM_JOINTS]

            target_pos_t = torch.tensor(
                self.target_pos[i], dtype=torch.float32, device=self.device_str
            )
            target_quat_t = torch.tensor(
                self.target_quat[i], dtype=torch.float32, device=self.device_str
            )

            obs_list.append(
                torch.cat([jpos, jvel, eef_pos, eef_quat, target_pos_t, target_quat_t])
            )

        return torch.stack(obs_list)
