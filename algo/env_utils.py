import gymnasium as gym

from envs.env_registry import get_env_spec


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_custom_vec_env(
    env_id,
    num_envs,
    algorithm,
    device,
    headless,
    max_episode_steps=200,
    capture_video=False,
    video_dir="videos",
):
    env_spec = get_env_spec(env_id)
    if env_spec is None:
        return None

    factory = env_spec.ppo_factory if algorithm == "ppo" else env_spec.apg_factory
    if factory is None:
        mode = "PPO" if algorithm == "ppo" else "APG"
        raise ValueError(f"Environment '{env_id}' does not implement {mode} mode.")

    kw = dict(
        num_envs=num_envs,
        device=str(device),
        headless=headless,
        max_episode_steps=max_episode_steps,
    )

    # Pass capture_video / video_dir; if the env constructor doesn't accept
    # them (e.g. FrankaReachVecEnv), fall back without them.
    try:
        return factory(**kw, capture_video=capture_video, video_dir=video_dir)
    except TypeError:
        return factory(**kw)


def _create_eval_envs(args, device, run_name=""):
    """Create a separate eval environment (black-box, no gradients needed)."""
    video_dir = f"videos/{run_name}" if args.capture_video else "videos"
    eval_envs = make_custom_vec_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        algorithm="ppo",  # Always use black-box factory for eval
        device=device,
        headless=args.headless,
        max_episode_steps=args.max_episode_steps,
        capture_video=args.capture_video,
        video_dir=video_dir,
    )
    if eval_envs is not None:
        return eval_envs
    # Fallback for standard gymnasium envs
    from envs.torch_wrapper_env import TorchWrapperEnv

    gym_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, "") for i in range(args.num_envs)],
    )
    return TorchWrapperEnv(gym_envs)
