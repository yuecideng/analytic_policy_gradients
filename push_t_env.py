"""Push T environment: push a T-shaped block to a goal pose using external forces.

Two modes matching the project's PPO/APG pattern:
  - PPO: PushTVecEnv  — black-box, no gradients through dynamics
  - APG: PushTAPGEnv  — differentiable, full autograd through dynamics

Both use the same simplified 2D rigid body dynamics (Euler integration).
The T-block state is (x, y, θ, vx, vy, ω). The action is a 3D force/torque.
"""

import math
import os

import gymnasium as gym
import matplotlib
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
import torch

from utils import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_HALF = 0.25  # workspace is [-0.25, 0.25] x [-0.25, 0.25] metres

# T-shape geometry (metres), matching gym-pusht proportions (scale=30, 1px≈1mm)
T_SCALE = 0.03
T_TOP_W = 4 * T_SCALE  # 0.12 m  (top bar width)
T_TOP_H = T_SCALE  # 0.03 m
T_STEM_W = T_SCALE  # 0.03 m
T_STEM_H = 3 * T_SCALE  # 0.09 m

# Approximate centre-of-mass offset from the geometric origin (bottom-left).
# Original: CoM ≈ (0, 45) px → in metres, shifted so CoM is at body origin.
_T_COM_Y = 0.045  # metres from bottom of top bar

# Top-bar centre relative to CoM: y = T_TOP_H/2 - _T_COM_Y
T_TOP_CY = T_TOP_H / 2 - _T_COM_Y  # ≈ −0.03
# Stem centre relative to CoM: y = T_TOP_H + T_STEM_H/2 - _T_COM_Y
T_STEM_CY = T_TOP_H + T_STEM_H / 2 - _T_COM_Y  # ≈ +0.03

BLOCK_MASS = 0.1  # kg
BLOCK_INERTIA = 1.95e-4  # kg⋅m² (computed analytically from T geometry)

FORCE_SCALE = 5.0  # N  (max force per axis)
TORQUE_SCALE = 0.2  # Nm (max torque)
LINEAR_DAMPING = 5.0  # velocity damping coefficient
ANGULAR_DAMPING = 3.0

DT = 1.0 / 60.0
DEFAULT_MAX_EPISODE_STEPS = 100
SUCCESS_POS_THRESHOLD = 0.01  # 1 cm
SUCCESS_ANGLE_THRESHOLD = 0.1  # ≈ 5.7°

# Observation: [block_x, block_y, block_θ, goal_x, goal_y, goal_θ, vx, vy, ω]
OBS_DIM = 9
ACTION_DIM = 3  # [force_x, force_y, torque_z]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wrap_to_pi_np(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _compute_reward(
    block_pos: torch.Tensor,
    block_angle: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_angle: torch.Tensor,
    action: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute reward (works with or without grad).

    Terms:
      - position_tracking:             −1.0  * ||pos − goal||
      - position_tracking_fine_grained: +0.5 * exp(-dist^2 / (2*0.02^2))
      - orientation_tracking:          −0.5  * |wrap(θ − goal_θ)|
      - action_rate:                   −0.001 * ||action − prev_action||²
    """
    pos_dist = (block_pos - goal_pos).norm(dim=-1)
    angle_diff = _wrap_to_pi(block_angle - goal_angle).squeeze(-1).abs()
    reward = (
        -1.0 * pos_dist
        + 0.5 * torch.exp(-(pos_dist**2) / (2 * 0.02**2))
        - 0.5 * angle_diff
    )
    if action is not None and last_action is not None:
        action_rate = ((action - last_action) ** 2).sum(dim=-1)
        reward = reward - 0.001 * action_rate
    return reward


def _check_success(
    block_pos: torch.Tensor,
    block_angle: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_angle: torch.Tensor,
    pos_threshold: float = SUCCESS_POS_THRESHOLD,
    angle_threshold: float = SUCCESS_ANGLE_THRESHOLD,
) -> torch.Tensor:
    pos_dist = (block_pos - goal_pos).norm(dim=-1)
    angle_dist = _wrap_to_pi(block_angle - goal_angle).squeeze(-1).abs()
    return (pos_dist < pos_threshold) & (angle_dist < angle_threshold)


def _sample_goal(num_envs: int, device: str = "cpu"):
    """Sample random goal poses within the workspace."""
    goal_pos = (torch.rand(num_envs, 2, device=device) - 0.5) * 2 * WORKSPACE_HALF * 0.6
    goal_angle = (torch.rand(num_envs, 1, device=device) - 0.5) * 2 * math.pi
    return goal_pos, goal_angle


def _sample_block_start(num_envs: int, device: str = "cpu"):
    """Sample random initial block poses."""
    pos = (torch.rand(num_envs, 2, device=device) - 0.5) * 2 * WORKSPACE_HALF * 0.6
    angle = (torch.rand(num_envs, 1, device=device) - 0.5) * 2 * math.pi
    return pos, angle


# ---------------------------------------------------------------------------
# Matplotlib 2-D rendering
# ---------------------------------------------------------------------------


def _make_t_patches(cx, cy, angle, color, ax, alpha=1.0):
    """Return two Rectangle patches forming a T centred at (cx, cy).

    The transform rotates around the CoM, translates to (cx, cy),
    and composes with ``ax.transData`` so patches render in axes coords.
    """
    top = patches.Rectangle(
        (-T_TOP_W / 2, T_TOP_CY - T_TOP_H / 2),
        T_TOP_W,
        T_TOP_H,
        linewidth=1,
        edgecolor="black",
        facecolor=color,
        alpha=alpha,
    )
    stem = patches.Rectangle(
        (-T_STEM_W / 2, T_STEM_CY - T_STEM_H / 2),
        T_STEM_W,
        T_STEM_H,
        linewidth=1,
        edgecolor="black",
        facecolor=color,
        alpha=alpha,
    )
    # Rotate around CoM → translate → map to display coordinates.
    t = mtransforms.Affine2D().rotate(angle).translate(cx, cy) + ax.transData
    top.set_transform(t)
    stem.set_transform(t)
    return top, stem


def _render_frame(
    block_pos: np.ndarray,
    block_angle: float,
    goal_pos: np.ndarray,
    goal_angle: float,
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

    # Goal T (light green, semi-transparent)
    gt, gs = _make_t_patches(
        goal_pos[0],
        goal_pos[1],
        goal_angle,
        color="lightgreen",
        ax=ax,
        alpha=0.4,
    )
    ax.add_patch(gt)
    ax.add_patch(gs)

    # Current T (slate gray)
    ct, cs = _make_t_patches(
        block_pos[0],
        block_pos[1],
        block_angle,
        color="slategray",
        ax=ax,
        alpha=0.9,
    )
    ax.add_patch(ct)
    ax.add_patch(cs)

    # Force arrow
    if action is not None:
        fx, fy = action[:2] * FORCE_SCALE
        arrow_scale = 0.02
        ax.arrow(
            block_pos[0],
            block_pos[1],
            fx * arrow_scale,
            fy * arrow_scale,
            head_width=0.008,
            head_length=0.005,
            fc="red",
            ec="red",
        )

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=100)
    buf.seek(0)
    raw = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()[::-1]
    img = raw.reshape(h, w, 4)[:, :, :3].copy()  # RGBA → RGB
    plt.close(fig)
    buf.close()
    return img


# ---------------------------------------------------------------------------
# PPO Environment (black-box, no gradients)
# ---------------------------------------------------------------------------


class PushTVecEnv:
    """Batched Push T environment for vectorized PPO.

    All dynamics are pure PyTorch (Euler integration) with no gradient
    tracking.  Returns torch tensors matching the rl.py PPO interface.

    Attributes:
        single_observation_space: Box(9,)
        single_action_space:      Box(3,)
        num_envs:                 number of parallel environments
    """

    def __init__(
        self,
        num_envs: int = 4,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cpu",
        headless: bool = True,
        capture_video: bool = True,
        video_dir: str = "videos",
    ):
        self._num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.headless = headless

        # State tensors (no grad needed for PPO)
        self.block_pos = torch.zeros(num_envs, 2, device=device)
        self.block_angle = torch.zeros(num_envs, 1, device=device)
        self.block_vel = torch.zeros(num_envs, 2, device=device)
        self.block_ang_vel = torch.zeros(num_envs, 1, device=device)

        # Goal
        self.goal_pos = torch.zeros(num_envs, 2, device=device)
        self.goal_angle = torch.zeros(num_envs, 1, device=device)

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

        ids = (
            env_ids
            if env_ids is not None
            else torch.arange(self._num_envs, device=self.device)
        )
        n = len(ids)

        pos, angle = _sample_block_start(n, device=self.device)
        goal_pos, goal_angle = _sample_goal(n, device=self.device)

        with torch.no_grad():
            self.block_pos[ids] = pos
            self.block_angle[ids] = angle
            self.block_vel[ids] = 0.0
            self.block_ang_vel[ids] = 0.0
            self.goal_pos[ids] = goal_pos
            self.goal_angle[ids] = goal_angle
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

        infos = {
            "final_distance": (self.block_pos - self.goal_pos).norm(dim=-1).detach(),
            "success": terminated.detach(),
        }
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return obs, reward, terminated, truncated, infos

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _integrate(self, action: torch.Tensor):
        """Euler-step the 2D rigid body. Must be called inside no_grad."""
        force = action[:, :2] * FORCE_SCALE
        torque = action[:, 2:3] * TORQUE_SCALE

        # Damping
        self.block_vel *= 1.0 - LINEAR_DAMPING * DT
        self.block_ang_vel *= 1.0 - ANGULAR_DAMPING * DT

        # Apply force / torque
        self.block_vel += force / BLOCK_MASS * DT
        self.block_ang_vel += torque / BLOCK_INERTIA * DT

        # Integrate position
        self.block_pos += self.block_vel * DT
        self.block_angle += self.block_ang_vel * DT

        # Clamp to workspace
        self.block_pos = self.block_pos.clamp(-WORKSPACE_HALF, WORKSPACE_HALF)

        # Wrap angle
        self.block_angle = _wrap_to_pi(self.block_angle)

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat(
            [
                self.block_pos,
                self.block_angle,
                self.goal_pos,
                self.goal_angle,
                self.block_vel,
                self.block_ang_vel,
            ],
            dim=-1,
        )
        return obs.detach()

    def _get_reward(self, action: torch.Tensor | None = None) -> torch.Tensor:
        return _compute_reward(
            self.block_pos,
            self.block_angle,
            self.goal_pos,
            self.goal_angle,
            action=action,
            last_action=self.last_action,
        ).detach()

    def _check_success(self) -> torch.Tensor:
        return _check_success(
            self.block_pos,
            self.block_angle,
            self.goal_pos,
            self.goal_angle,
        )

    def render(self, env_idx: int = 0) -> np.ndarray:
        """Render a single env as an RGB image (H, W, 3) uint8."""
        return _render_frame(
            block_pos=self.block_pos[env_idx].detach().cpu().numpy(),
            block_angle=float(self.block_angle[env_idx].detach().cpu()),
            goal_pos=self.goal_pos[env_idx].detach().cpu().numpy(),
            goal_angle=float(self.goal_angle[env_idx].detach().cpu()),
            action=self.last_action[env_idx].detach().cpu().numpy(),
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# APG Environment (differentiable)
# ---------------------------------------------------------------------------


class PushTAPGEnv(PushTVecEnv):
    """Batched Push T environment for APG with differentiable dynamics.

    Inherits shared logic (reset, obs, reward, rendering, auto-reset).
    Overrides ``step()`` so that the computation graph is preserved:
    reward has ``grad_fn`` linking back through the force/integration chain,
    allowing ``loss.backward()`` to compute analytic policy gradients.

    Interface matches rl.py APG loop:
      - reset() -> (obs [num_envs, 9], info)
      - step(action [num_envs, 3]) -> (obs, reward, terminated, truncated, info)
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

    def reset(self, env_ids=None, seed=None):
        # Detach state tensors to break computation graph from previous iteration.
        # Without this, in-place writes to tensors that still have grad_fn
        # cause "backward through the graph a second time" errors.
        self.block_pos = self.block_pos.detach().clone()
        self.block_angle = self.block_angle.detach().clone()
        self.block_vel = self.block_vel.detach().clone()
        self.block_ang_vel = self.block_ang_vel.detach().clone()
        return super().reset(env_ids, seed)

    def step(self, action: torch.Tensor):
        """Step all envs with differentiable dynamics.

        The computation graph is preserved so that:
            action → force → accel → vel → pos → reward
        all have grad_fn.
        """
        self.step_count += 1

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.clamp(-1.0, 1.0).to(self.device)

        # --- differentiable integration (no torch.no_grad!) ---
        force = action[:, :2] * FORCE_SCALE
        torque = action[:, 2:3] * TORQUE_SCALE

        # Damping (in-place on leaf tensors is fine — they have no grad history)
        self.block_vel = self.block_vel * (1.0 - LINEAR_DAMPING * DT)
        self.block_ang_vel = self.block_ang_vel * (1.0 - ANGULAR_DAMPING * DT)

        # Apply force / torque (builds computation graph)
        self.block_vel = self.block_vel + force / BLOCK_MASS * DT
        self.block_ang_vel = self.block_ang_vel + torque / BLOCK_INERTIA * DT

        # Integrate position
        self.block_pos = self.block_pos + self.block_vel * DT
        self.block_angle = self.block_angle + self.block_ang_vel * DT

        # Clamp / wrap — these are differentiable (clamp has well-defined grad)
        self.block_pos = self.block_pos.clamp(-WORKSPACE_HALF, WORKSPACE_HALF)
        self.block_angle = _wrap_to_pi(self.block_angle)

        # Observation (detached — policy doesn't need grad through obs)
        obs = self._get_obs()

        # Reward (preserves grad through block_pos, block_angle, action)
        reward = _compute_reward(
            self.block_pos,
            self.block_angle,
            self.goal_pos,
            self.goal_angle,
            action=action,
            last_action=self.last_action,
        )

        # Update last_action (detached — only used as constant in next step)
        self.last_action = action.detach().clone()

        terminated = self._check_success()
        truncated = self.step_count >= self.max_episode_steps
        done_mask = terminated | truncated

        infos = {
            "final_distance": (self.block_pos - self.goal_pos).norm(dim=-1).detach(),
            "success": terminated.detach(),
        }
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            obs, _ = self.reset(reset_ids)

        return obs, reward, terminated, truncated, infos


# ---------------------------------------------------------------------------
# Episode recording & video saving
# ---------------------------------------------------------------------------


def save_video(frames: list[np.ndarray], path: str, fps: int = 60) -> None:
    """Save a list of frames as a gif or mp4 using matplotlib.

    Format is inferred from *path* extension (``.gif`` or ``.mp4``).
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True,
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
    """Records rendered frames during ``step()`` and auto-saves on episode end.

    Attach to a ``PushTVecEnv`` (or subclass) by passing
    ``capture_video=True, video_dir=...`` to the env constructor.

    The recorder watches env_idx 0 only.  Every time env 0 finishes an
    episode (terminated or truncated), the accumulated frames are saved as
    a gif and the buffer is cleared for the next episode.
    """

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
        """Save buffered frames as a gif and clear the buffer.

        Returns the saved file path, or *None* if no frames were captured.
        """
        if not self._frames:
            return None
        self._episode_count += 1
        tag = f"ep{self._episode_count}"
        if global_step is not None:
            tag = f"step{global_step}_{tag}"
        path = os.path.join(self.video_dir, f"push_t_{tag}.gif")
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
    ppo_env = PushTVecEnv(num_envs=4, headless=True)
    obs, _ = ppo_env.reset(seed=42)
    print(f"  obs shape: {obs.shape}")
    for i in range(5):
        action = torch.randn(ppo_env.num_envs, ACTION_DIM)
        obs, reward, terminated, truncated, infos = ppo_env.step(action)
        print(f"  step {i}: reward={reward.mean().item():.4f}")
    ppo_env.close()

    print("\n=== APG (differentiable) ===")
    apg_env = PushTAPGEnv(num_envs=4, headless=True)
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

    # Verify visualization (record + save)
    print("\n=== Visualization ===")
    vis_env = PushTVecEnv(num_envs=1, headless=True)
    frames = record_episode(vis_env, env_idx=0)
    print(f"  recorded {len(frames)} frames")
    gif_path = "push_t_episode.gif"
    save_video(frames, gif_path)
    print(f"  saved to {gif_path}")
    vis_env.close()

    print("\nAll checks passed.")
