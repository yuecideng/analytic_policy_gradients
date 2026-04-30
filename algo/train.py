import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algo.agent import Agent, RunningObsNormalizer
from algo.env_utils import _create_eval_envs, make_custom_vec_env, make_env
from algo.ppo import run_ppo
from algo.apg import run_apg
from utils import set_seed


def _run_training(args, seed):
    """Run a single training experiment with the given seed."""
    args.seed = seed
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    set_seed(args.seed, deterministic=args.torch_deterministic)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # env setup
    custom_envs = make_custom_vec_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        algorithm=args.algorithm,
        device=device,
        headless=args.headless,
        max_episode_steps=args.max_episode_steps,
        capture_video=False,
        video_dir=f"videos/{run_name}",
    )
    if custom_envs is not None:
        envs = custom_envs
    elif args.algorithm == "ppo":
        gym_envs = gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, i, args.capture_video, run_name)
                for i in range(args.num_envs)
            ],
        )
    else:
        raise ValueError(
            f"APG requires an implemented differentiable env. "
            f"'{args.env_id}' is not in IMPLEMENTED_ENVS."
        )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only box (continuous) action spaces are supported"

    agent = Agent(envs, use_layernorm=False).to(device)

    # Eval env setup (separate from training env)
    eval_envs = (
        _create_eval_envs(args, device, run_name) if args.eval_freq > 0 else None
    )

    # optimizer setup
    if args.algorithm == "ppo":
        optimizer = optim.Adam(
            agent.actor_parameters() + list(agent.critic.parameters()),
            lr=args.learning_rate,
            eps=1e-5,
        )
    else:
        # Determine if segmented APG with critic bootstrap is active
        effective_seg = (
            args.apg_segment_length
            if args.apg_segment_length > 0
            else args.max_episode_steps
        )
        use_critic = (
            effective_seg < args.max_episode_steps and args.apg_bootstrap == "critic"
        )
        if use_critic:
            critic_lr = (
                args.apg_critic_lr
                if args.apg_critic_lr is not None
                else args.learning_rate
            )
            optimizer = optim.Adam(
                [
                    {"params": agent.actor_parameters(), "lr": args.learning_rate},
                    {"params": list(agent.critic.parameters()), "lr": critic_lr},
                ],
                eps=1e-5,
            )
        else:
            optimizer = optim.Adam(
                agent.actor_parameters(),
                lr=args.learning_rate,
                eps=1e-5,
            )

    # Shared obs normalizer
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    obs_normalizer = RunningObsNormalizer(obs_dim, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    should_print_episodes = args.print_every_n_episodes > 0

    if args.algorithm == "ppo":
        global_step, eval_success_rates = run_ppo(
            args,
            agent,
            envs,
            optimizer,
            obs_normalizer,
            eval_envs,
            writer,
            device,
            global_step,
            start_time,
            should_print_episodes,
        )
    else:
        global_step, eval_success_rates = run_apg(
            args,
            agent,
            envs,
            optimizer,
            obs_normalizer,
            eval_envs,
            writer,
            device,
            global_step,
            start_time,
            should_print_episodes,
            use_critic=use_critic,
        )

    # Compute AUC of success rate curve
    if eval_success_rates:
        gs = np.array([x[0] for x in eval_success_rates])
        sr = np.array([x[1] for x in eval_success_rates])
        if gs[-1] > 0:
            auc = np.trapezoid(sr, gs / gs[-1])
            writer.add_scalar("summary/auc_success_rate", auc, 0)
            print(f"  AUC (success rate): {auc:.6f}")

    print(f"(global_step={global_step})")

    if eval_envs is not None:
        eval_envs.close()
    envs.close()
    writer.close()
