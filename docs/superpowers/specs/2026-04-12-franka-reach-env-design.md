# Franka Reach Environment Design

## Goal

Implement a 6D end-effector reaching task using the Franka FR3 robot in Newton physics, trainable with both PPO (black-box) and APG (backprop-through-env) from `rl.py`.

## Environment: `FrankaReachEnv`

**File**: `franka_reach_env.py`

### Simulation Setup

- Load Franka FR3 URDF via `newton.utils.download_asset()` + `builder.add_urdf()`
- Add ground plane via `builder.add_ground_plane()`
- Finalize model with `requires_grad=True` for APG, `False` for PPO
- Use `SolverFeatherstone` for articulated body dynamics
- Multi-env via Newton's batched state (single model, `num_envs` parallel worlds)
- Sub-stepping: configurable (default 4 substeps per env step)

### Observation Space (28-dim, continuous Box)

| Component | Dim | Description |
|-----------|-----|-------------|
| joint_pos | 7 | Franka arm joint positions |
| joint_vel | 7 | Joint velocities |
| eef_pos | 3 | End-effector position (x,y,z) |
| eef_quat | 4 | EEF orientation (qx,qy,qz,qw) |
| target_pos | 3 | Target position |
| target_quat | 4 | Target orientation |

Target is sampled at `reset()` within a reachable workspace region in front of the robot.

### Action Space (7-dim, continuous Box [-1, 1])

Delta joint positions: `new_joint_pos = current_joint_pos + action * action_scale`

Where `action_scale` is a configurable parameter (default 0.1 rad).

### Reward

```
r = -||eef_pos - target_pos||_2 - w_rot * quaternion_distance(eef_quat, target_quat)
```

- Position L2 distance (3D Euclidean)
- Quaternion distance using L2 on quaternion components (with sign handling for double cover)
- `w_rot` = weight for rotation term (default 1.0)

### Termination

- Episode truncation after `max_episode_length` steps (default 200)
- Success threshold: total pose distance < 0.05 (for logging only, no early termination)

## PPO Mode

- Environment runs Newton sim without gradient tracking
- Returns detached torch tensors (standard gym-style)
- Compatible with existing `SyncVectorEnv` + `TorchSyncVecEnv` wrapper in `rl.py`
- Uses gymnasium `Env` interface

## APG Mode

### Gradient Bridge: Custom `torch.autograd.Function`

The core challenge is that Newton uses Warp's tape-based autodiff, not PyTorch autograd. We bridge this with a custom autograd function.

```
Forward pass:
  action_torch → wp.from_torch() → Newton sim (wp.Tape records) → reward (Warp scalar)
  Return: detached reward tensor (saved for backward)

Backward pass:
  Use stored Warp tape → tape.backward(reward_grad)
  Extract action.grad from Warp → wp.to_torch() → return as PyTorch gradient
```

### `NewtonStepFunction(torch.autograd.Function)`

- `forward(ctx, action_torch, sim_state)`: Converts action to Warp, runs sim step inside `wp.Tape()`, returns detached reward
- `backward(ctx, grad_output)`: Runs `tape.backward()` on stored tape, extracts action gradient, returns as torch tensor

This allows `rl.py`'s APG loop to call `loss.backward()` and have gradients flow through the env naturally.

### APG Trajectory

For APG, the env tracks trajectory-level returns:
- Each `step()` accumulates reward with discount
- After `num_steps` steps, returns total discounted return
- Loss = `-traj_return / num_envs` (matches existing `rl.py` APG loop)

## Integration with `rl.py`

Minimal changes:
1. Import `FrankaReachEnv` and helper function `make_franka_env()`
2. PPO branch: use `make_franka_env()` to create gymnasium envs
3. APG branch: replace `NotImplementedError` with `FrankaReachEnv(algorithm="apg", ...)`

## File Structure

```
analytic_policy_gradients/
├── rl.py                    # existing (minor edits)
├── franka_reach_env.py      # new - env implementation
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_envs | 4 | Parallel environments |
| max_episode_length | 200 | Steps before truncation |
| action_scale | 0.1 | Delta joint position scaling |
| w_rot | 1.0 | Rotation distance weight |
| substeps | 4 | Physics substeps per env step |
| dt | 1/100 | Physics timestep |
| target_reach_radius | [0.3, 0.8] | Distance range for target sampling |
