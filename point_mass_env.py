"""Point-mass navigation with obstacle avoidance.

Two modes matching the project's PPO/APG pattern:
  - PPO: PointMassVecEnv  — black-box, no gradients through dynamics
  - APG: PointMassAPGEnv  — differentiable, full autograd through dynamics

Both use the same simplified 2D point-mass dynamics (Euler integration).
State is (x, y, vx, vy). Action is a 2D force vector.
"""

import math
import os

import gymnasium as gym
import matplotlib
import matplotlib.patches as patches
import numpy as np
import torch

from utils import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_HALF = 1.0  # workspace is [-1, 1] x [-1, 1] metres

POINT_MASS = 1.0  # kg
POINT_RADIUS = 0.03  # metres (visual + collision radius)
FORCE_SCALE = 5.0  # N (max force per axis)
LINEAR_DAMPING = 5.0  # velocity damping coefficient

DT = 1.0 / 60.0
DEFAULT_MAX_EPISODE_STEPS = 100
SUCCESS_POS_THRESHOLD = 0.01  # 1 cm

# Obstacle configuration
NUM_OBSTACLES = 2
OBSTACLE_RADIUS_RANGE = (0.08, 0.15)  # metres
OBSTACLE_MIN_DIST_FROM_GOAL = 0.25  # metres

# Observation: [pos(2), vel(2), goal_pos(2), last_action(2),
#               obs1_pos(2), obs1_r(1), obs2_pos(2), obs2_r(1)] = 14
OBS_DIM = 2 + 2 + 2 + 2 + NUM_OBSTACLES * 3
ACTION_DIM = 2  # [force_x, force_y]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_reward(
    pos: torch.Tensor,
    goal_pos: torch.Tensor,
    vel: torch.Tensor,
    obstacle_pos: torch.Tensor,
    obstacle_radii: torch.Tensor,
    action: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute reward (works with or without grad).

    Terms:
      - position_tracking:             -1.0  * ||pos - goal||
      - position_tracking_fine_grained: +0.5 * (1 - tanh(dist / 0.05))
      - obstacle_penalty:              -2.0  * sum(max(0, r - d)^2) per obstacle
      - action_rate:                   -0.001 * ||action - prev_action||^2
      - speed_penalty:                 -0.01 * ||vel||^2
    """
    pos_dist = (pos - goal_pos).norm(dim=-1)
    reward = (
        -1.0 * pos_dist
        + 0.5 * (1.0 - torch.tanh(pos_dist / 0.05))
    )

    # Obstacle penalty: smooth repulsion when inside obstacle radius
    for i in range(NUM_OBSTACLES):
        diff = pos - obstacle_pos[:, i * 2 : i * 2 + 2]
        dist_to_center = diff.norm(dim=-1)
        penetration = torch.clamp(obstacle_radii[:, i] - dist_to_center, min=0.0)
        reward = reward - 2.0 * penetration ** 2

    # Speed penalty
    reward = reward - 0.01 * (vel ** 2).sum(dim=-1)

    if action is not None and last_action is not None:
        action_rate = ((action - last_action) ** 2).sum(dim=-1)
        reward = reward - 0.001 * action_rate
    return reward


def _check_success(
    pos: torch.Tensor,
    goal_pos: torch.Tensor,
    pos_threshold: float = SUCCESS_POS_THRESHOLD,
) -> torch.Tensor:
    pos_dist = (pos - goal_pos).norm(dim=-1)
    return pos_dist < pos_threshold


def _sample_goal(num_envs: int, device: str = "cpu"):
    """Sample random goal positions within the workspace."""
    goal_pos = (torch.rand(num_envs, 2, device=device) - 0.5) * 2 * WORKSPACE_HALF * 0.7
    return goal_pos


def _sample_obstacles(num_envs: int, goal_pos: torch.Tensor, device: str = "cpu"):
    """Sample random circular obstacles, ensuring distance from goal.

    Returns:
        obstacle_pos: [num_envs, NUM_OBSTACLES * 2]
        obstacle_radii: [num_envs, NUM_OBSTACLES]
    """
    obstacle_pos = torch.zeros(num_envs, NUM_OBSTACLES * 2, device=device)
    obstacle_radii = torch.zeros(num_envs, NUM_OBSTACLES, device=device)

    for i in range(NUM_OBSTACLES):
        obstacle_radii[:, i] = (
            OBSTACLE_RADIUS_RANGE[0]
            + torch.rand(num_envs, device=device)
            * (OBSTACLE_RADIUS_RANGE[1] - OBSTACLE_RADIUS_RANGE[0])
        )
        # Rejection-sample positions far enough from goal
        for env_idx in range(num_envs):
            for _ in range(50):  # max attempts
                candidate = (
                    (torch.rand(2, device=device) - 0.5) * 2 * WORKSPACE_HALF * 0.8
                )
                dist_to_goal = (candidate - goal_pos[env_idx]).norm()
                if dist_to_goal > OBSTACLE_MIN_DIST_FROM_GOAL:
                    obstacle_pos[env_idx, i * 2 : i * 2 + 2] = candidate
                    break
            else:
                # Fallback: place at workspace edge opposite to goal
                obstacle_pos[env_idx, i * 2 : i * 2 + 2] = -goal_pos[env_idx] * 0.5

    return obstacle_pos, obstacle_radii


def _sample_start(num_envs: int, goal_pos: torch.Tensor, device: str = "cpu"):
    """Sample random starting positions, ensuring distance from goal."""
    starts = (torch.rand(num_envs, 2, device=device) - 0.5) * 2 * WORKSPACE_HALF * 0.6
    return starts


# ---------------------------------------------------------------------------
# Matplotlib 2-D rendering
# ---------------------------------------------------------------------------


def _render_frame(
    pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacle_pos: np.ndarray,
    obstacle_radii: np.ndarray,
    action: np.ndarray | None = None,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Render a single 2-D frame. Returns an (H, W, 3) uint8 array."""
    import io

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(-WORKSPACE_HALF, WORKSPACE_HALF)
    ax.set_ylim(-WORKSPACE_HALF, WORKSPACE_HALF)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # Workspace boundary
    ax.add_patch(
        patches.Rectangle(
            (-WORKSPACE_HALF, -WORKSPACE_HALF),
            2 * WORKSPACE_HALF,
            2 * WORKSPACE_HALF,
            linewidth=2,
            edgecolor="gray",
            facecolor="none",
        )
    )

    # Obstacles (red circles)
    for i in range(NUM_OBSTACLES):
        cx = obstacle_pos[i * 2]
        cy = obstacle_pos[i * 2 + 1]
        r = obstacle_radii[i]
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                r,
                linewidth=1,
                edgecolor="darkred",
                facecolor="salmon",
                alpha=0.5,
            )
        )

    # Goal (green star)
    ax.plot(
        goal_pos[0], goal_pos[1], marker="*", color="green",
        markersize=15, zorder=5,
    )

    # Point mass (blue circle)
    ax.add_patch(
        patches.Circle(
            (pos[0], pos[1]),
            POINT_RADIUS,
            linewidth=1,
            edgecolor="darkblue",
            facecolor="royalblue",
            zorder=5,
        )
    )

    # Force arrow
    if action is not None:
        fx, fy = action[:2] * FORCE_SCALE
        arrow_scale = 0.04
        ax.arrow(
            pos[0], pos[1],
            fx * arrow_scale, fy * arrow_scale,
            head_width=0.03, head_length=0.02,
            fc="red", ec="red",
        )

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=100)
    buf.seek(0)
    raw = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()[::-1]
    img = raw.reshape(h, w, 4)[:, :, :3].copy()  # RGBA -> RGB
    plt.close(fig)
    buf.close()
    return img


# ---------------------------------------------------------------------------
# PPO Environment (black-box, no gradients)
# ---------------------------------------------------------------------------


class PointMassVecEnv:
    """Batched Point Mass navigation environment for vectorized PPO.

    All dynamics are pure PyTorch (Euler integration) with no gradient
    tracking.  Returns torch tensors matching the rl.py PPO interface.

    Attributes:
        single_observation_space: Box(14,)
        single_action_space:      Box(2,)
        num_envs:                 number of parallel environments
    """

    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
        capture_video: bool = False,
        video_dir: str = "videos",
    ):
        self._num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.headless = headless

        # State tensors (no grad needed for PPO)
        self.pos = torch.zeros(num_envs, 2, device=device)
        self.vel = torch.zeros(num_envs, 2, device=device)

        # Goal
        self.goal_pos = torch.zeros(num_envs, 2, device=device)

        # Obstacles
        self.obstacle_pos = torch.zeros(num_envs, NUM_OBSTACLES * 2, device=device)
        self.obstacle_radii = torch.zeros(num_envs, NUM_OBSTACLES, device=device)

        # Bookkeeping
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.last_action = torch.zeros(num_envs, ACTION_DIM, device=device)

        # Spaces
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # Video recording
        if capture_video:
            self.video_recorder = VideoRecorder(self, video_dir=video_dir)
        else:
            self.video_recorder = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, env_ids=None, seed=None):
        if seed is not None:
            set_seed(seed)

        ids = env_ids if env_ids is not None else torch.arange(self._num_envs, device=self.device)
        n = len(ids)

        goal_pos = _sample_goal(n, device=self.device)
        obstacle_pos, obstacle_radii = _sample_obstacles(n, goal_pos, device=self.device)
        starts = _sample_start(n, goal_pos, device=self.device)

        with torch.no_grad():
            self.pos[ids] = starts
            self.vel[ids] = 0.0
            self.goal_pos[ids] = goal_pos
            self.obstacle_pos[ids] = obstacle_pos
            self.obstacle_radii[ids] = obstacle_radii
            self.step_count[ids] = 0
            self.last_action[ids] = 0.0

        # Capture initial frame on full reset (new episode start)
        if env_ids is None and self.video_recorder is not None:
            self.video_recorder.capture_frame()

        return self._get_obs(), {}

    def step(self, action: torch.Tensor):
        self.step_count += 1

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.clamp(-1.0, 1.0).to(self.device)

        with torch.no_grad():
            self._integrate(action)

        obs = self._get_obs()
        reward = self._get_reward(action)

        self.last_action = action.detach().clone()

        terminated = self._check_success()
        truncated = self.step_count >= self.max_episode_steps
        done_mask = terminated | truncated

        # Video recording: capture frame, save on episode end
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
            if done_mask[0]:
                self.video_recorder.on_episode_end()

        infos = {}
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return obs, reward, terminated, truncated, infos

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _integrate(self, action: torch.Tensor):
        """Euler-step the point mass. Must be called inside no_grad."""
        force = action[:, :2] * FORCE_SCALE

        # Damping
        self.vel *= 1.0 - LINEAR_DAMPING * DT

        # Apply force
        self.vel += force / POINT_MASS * DT

        # Integrate position
        self.pos += self.vel * DT

        # Clamp to workspace
        self.pos = self.pos.clamp(-WORKSPACE_HALF, WORKSPACE_HALF)

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat(
            [
                self.pos,
                self.vel,
                self.goal_pos,
                self.last_action,
                self.obstacle_pos,
                self.obstacle_radii,
            ],
            dim=-1,
        )
        return obs.detach()

    def _get_reward(self, action: torch.Tensor | None = None) -> torch.Tensor:
        return _compute_reward(
            self.pos,
            self.goal_pos,
            self.vel,
            self.obstacle_pos,
            self.obstacle_radii,
            action=action,
            last_action=self.last_action,
        ).detach()

    def _check_success(self) -> torch.Tensor:
        return _check_success(self.pos, self.goal_pos)

    def render(self, env_idx: int = 0) -> np.ndarray:
        """Render a single env as an RGB image (H, W, 3) uint8."""
        obs_np = self.obstacle_pos[env_idx].detach().cpu().numpy()
        return _render_frame(
            pos=self.pos[env_idx].detach().cpu().numpy(),
            goal_pos=self.goal_pos[env_idx].detach().cpu().numpy(),
            obstacle_pos=obs_np,
            obstacle_radii=self.obstacle_radii[env_idx].detach().cpu().numpy(),
            action=self.last_action[env_idx].detach().cpu().numpy(),
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# APG Environment (differentiable)
# ---------------------------------------------------------------------------


class PointMassAPGEnv(PointMassVecEnv):
    """Batched Point Mass navigation environment for APG with differentiable dynamics.

    Inherits shared logic (reset, obs, reward, rendering, auto-reset).
    Overrides ``step()`` so that the computation graph is preserved:
    reward has ``grad_fn`` linking back through the force/integration chain,
    allowing ``loss.backward()`` to compute analytic policy gradients.

    Interface matches rl.py APG loop:
      - reset() -> (obs [num_envs, 14], info)
      - step(action [num_envs, 2]) -> (obs, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            device=device,
            headless=headless,
        )

    def detach_state(self):
        """Detach all state tensors to break the computation graph between grad steps."""
        self.pos = self.pos.detach()
        self.vel = self.vel.detach()
        self.last_action = self.last_action.detach()

    def reset(self, env_ids=None, seed=None):
        # Detach state tensors to break computation graph from previous iteration.
        self.pos = self.pos.detach().clone()
        self.vel = self.vel.detach().clone()
        return super().reset(env_ids, seed)

    def step(self, action: torch.Tensor):
        """Step all envs with differentiable dynamics.

        The computation graph is preserved so that:
            action -> force -> accel -> vel -> pos -> reward
        all have grad_fn.
        """
        self.step_count += 1

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.clamp(-1.0, 1.0).to(self.device)

        # --- differentiable integration (no torch.no_grad!) ---
        force = action[:, :2] * FORCE_SCALE

        # Damping (in-place on leaf tensors is fine — they have no grad history)
        self.vel = self.vel * (1.0 - LINEAR_DAMPING * DT)

        # Apply force (builds computation graph)
        self.vel = self.vel + force / POINT_MASS * DT

        # Integrate position
        self.pos = self.pos + self.vel * DT

        # Clamp — differentiable (clamp has well-defined grad)
        self.pos = self.pos.clamp(-WORKSPACE_HALF, WORKSPACE_HALF)

        # Observation (detached — policy doesn't need grad through obs)
        obs = self._get_obs()

        # Reward (preserves grad through pos, vel, action)
        reward = _compute_reward(
            self.pos,
            self.goal_pos,
            self.vel,
            self.obstacle_pos,
            self.obstacle_radii,
            action=action,
            last_action=self.last_action,
        )

        # Update last_action (detached — only used as constant in next step)
        self.last_action = action.detach().clone()

        terminated = self._check_success()
        truncated = self.step_count >= self.max_episode_steps
        done_mask = terminated | truncated

        infos = {}
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return obs, reward, terminated, truncated, infos


# ---------------------------------------------------------------------------
# Episode recording & video saving
# ---------------------------------------------------------------------------


def save_video(frames: list[np.ndarray], path: str, fps: int = 60) -> None:
    """Save a list of frames as a gif or mp4 using matplotlib."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 / fps, blit=True,
    )
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "gif":
        ani.save(path, writer="pillow", fps=fps)
    elif ext == "mp4":
        ani.save(path, writer="ffmpeg", fps=fps)
    else:
        raise ValueError(f"Unsupported video format '.{ext}'. Use .gif or .mp4")
    plt.close(fig)


class VideoRecorder:
    """Records rendered frames during ``step()`` and auto-saves on episode end."""

    def __init__(
        self,
        env,
        video_dir: str = "videos",
        env_idx: int = 0,
        fps: int = 60,
    ):
        self.env = env
        self.video_dir = video_dir
        self.env_idx = env_idx
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._episode_count = 0

        os.makedirs(video_dir, exist_ok=True)

    def capture_frame(self) -> None:
        """Render and append one frame (call after each ``step()``)."""
        self._frames.append(self.env.render(env_idx=self.env_idx))

    def on_episode_end(self, global_step: int | None = None) -> str | None:
        """Save buffered frames as a gif and clear the buffer."""
        if not self._frames:
            return None
        self._episode_count += 1
        tag = f"ep{self._episode_count}"
        if global_step is not None:
            tag = f"step{global_step}_{tag}"
        path = os.path.join(self.video_dir, f"point_mass_{tag}.gif")
        save_video(self._frames, path, fps=self.fps)
        self._frames = []
        return path


def record_episode(env, env_idx: int = 0, max_steps: int | None = None):
    """Run a random episode and collect rendered frames."""
    frames = []
    if max_steps is None:
        max_steps = env.max_episode_steps
    obs, _ = env.reset()
    frames.append(env.render(env_idx=env_idx))
    for _ in range(max_steps):
        action = torch.zeros(env._num_envs, ACTION_DIM)
        obs, reward, terminated, truncated, infos = env.step(action)
        frames.append(env.render(env_idx=env_idx))
    return frames


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PPO (black-box) ===")
    ppo_env = PointMassVecEnv(num_envs=4, headless=True)
    obs, _ = ppo_env.reset(seed=42)
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dim: {OBS_DIM} (expected {obs.shape[-1]})")
    for i in range(5):
        action = torch.randn(ppo_env.num_envs, ACTION_DIM)
        obs, reward, terminated, truncated, infos = ppo_env.step(action)
        print(f"  step {i}: reward={reward.mean().item():.4f}")
    ppo_env.close()

    print("\n=== APG (differentiable) ===")
    apg_env = PointMassAPGEnv(num_envs=4, headless=True)
    obs, _ = apg_env.reset(seed=42)

    # Verify gradient flow
    action = torch.randn(apg_env.num_envs, ACTION_DIM, requires_grad=True)
    obs, reward, terminated, truncated, infos = apg_env.step(action)
    loss = reward.sum()
    loss.backward()
    print(f"  reward grad_fn: {reward.grad_fn}")
    print(f"  action.grad is not None: {action.grad is not None}")
    print(f"  action.grad norm: {action.grad.norm().item():.6f}")
    print(f"  reward mean: {reward.mean().item():.4f}")

    # Verify render
    img = apg_env.render(env_idx=0)
    print(f"\n  render shape: {img.shape}, dtype: {img.dtype}")
    apg_env.close()

    print("\nAll checks passed.")
