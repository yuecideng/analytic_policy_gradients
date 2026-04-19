# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
#
# Supports two algorithms:
#   PPO: Proximal Policy Optimization (standard, black-box env)
#   APG: Analytic Policy Gradient (backprop through differentiable env)
#
# Usage:
#   PPO (discrete):   python ppo.py --algorithm ppo --env_id CartPole-v1
#   PPO (continuous):  python ppo.py --algorithm ppo --env_id Pendulum-v1
#   APG:               python ppo.py --algorithm apg --env_id <your-diff-env>
import os
import random
import sys
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from env_registry import get_env_spec


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    headless: bool = True
    """whether to run custom environments without an interactive viewer window"""

    # Algorithm selection
    algorithm: str = "ppo"
    """the algorithm to use: 'ppo' or 'apg'"""

    # Environment
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""

    # Shared hyperparameters
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # PPO-specific
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # APG-specific
    apg_gumbel_temp_init: float = 1.0
    """initial temperature for Gumbel-Softmax (discrete APG)"""
    apg_gumbel_temp_min: float = 0.1
    """minimum temperature for Gumbel-Softmax"""
    apg_anneal_temp: bool = True
    """whether to anneal Gumbel-Softmax temperature over training"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def _normalize_cli_args(argv: list[str]) -> list[str]:
    """Allow legacy key-value CLI style without leading dashes.

    Examples:
      python rl.py env_id FrankaReach-v0
      python rl.py env_id=FrankaReach-v0
    """
    valid_keys = {f.name for f in dataclass_fields(Args)}
    normalized: list[str] = []

    for token in argv:
        if token.startswith("-"):
            normalized.append(token)
            continue

        if "=" in token:
            key, value = token.split("=", 1)
            if key in valid_keys:
                normalized.extend([f"--{key}", value])
                continue

        if token in valid_keys:
            normalized.append(f"--{token}")
        else:
            normalized.append(token)

    return normalized


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


def make_custom_vec_env(env_id, num_envs, algorithm, device, headless):
    env_spec = get_env_spec(env_id)
    if env_spec is None:
        return None

    if algorithm == "ppo":
        if env_spec.ppo_factory is None:
            raise ValueError(f"Environment '{env_id}' does not implement PPO mode.")
        return env_spec.ppo_factory(num_envs=num_envs, device="cpu", headless=headless)

    if env_spec.apg_factory is None:
        raise ValueError(f"Environment '{env_id}' does not implement APG mode.")
    return env_spec.apg_factory(
        num_envs=num_envs,
        max_episode_steps=200,
        device=str(device),
        headless=headless,
    )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)

        if self.discrete:
            n_actions = envs.single_action_space.n
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, n_actions), std=0.01),
            )
        else:
            action_dim = np.array(envs.single_action_space.shape).prod()
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_dim), std=0.01),
            )
            # Smaller initial std for stability in physics sim (exp(-2.0) ≈ 0.135)
            self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -2.0))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.discrete:
            logits = self.actor(x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_log_std.expand_as(mean))
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            logprob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)

    def get_apg_action(self, x, temp=1.0):
        """Get differentiable action for APG.

        Discrete: Gumbel-Softmax relaxation (soft one-hot vector).
        Continuous: Reparameterized Gaussian (mean + std * noise).
        """
        if self.discrete:
            logits = self.actor(x)
            return F.gumbel_softmax(logits, tau=temp, hard=False)
        else:
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_log_std.expand_as(mean))
            return mean + std * torch.randn_like(mean)

    def actor_parameters(self):
        """Return actor parameters regardless of discrete/continuous."""
        if self.discrete:
            return list(self.actor.parameters())
        else:
            return list(self.actor_mean.parameters()) + [self.actor_log_std]


if __name__ == "__main__":
    args = tyro.cli(Args, args=_normalize_cli_args(sys.argv[1:]))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # For PPO: wraps gymnasium envs in a torch-compatible interface.
    # For APG: replace with your differentiable torch environment that implements:
    #   reset() -> (obs: Tensor, info)          (obs shape: [num_envs, *obs_shape])
    #   step(action: Tensor) -> (obs, reward, terminated, truncated, info)
    #     where all tensors preserve gradients through the dynamics.
    #   Must also have: single_observation_space, single_action_space, num_envs
    custom_envs = make_custom_vec_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        algorithm=args.algorithm,
        device=device,
        headless=args.headless,
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

    assert isinstance(envs.single_action_space, gym.spaces.Discrete) or isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only discrete and box action spaces are supported"

    agent = Agent(envs).to(device)

    # optimizer setup
    if args.algorithm == "ppo":
        optimizer = optim.Adam(
            agent.actor_parameters() + list(agent.critic.parameters()),
            lr=args.learning_rate,
            eps=1e-5,
        )
    else:
        optimizer = optim.Adam(
            agent.actor_parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    if args.algorithm == "ppo":
        # ========== PPO Training Loop ==========

        # ALGO Logic: Storage setup
        obs_buf = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(device)
        actions_buf = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = next_obs.to(device)
        next_done = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values_buf[step] = value.flatten()
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, infos = envs.step(action)
                next_done = (terminated | truncated).float()
                rewards_buf[step] = reward.to(device).view(-1)
                next_obs = next_obs.to(device)
                next_done = next_done.to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buf).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues = values_buf[t + 1]
                    delta = (
                        rewards_buf[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values_buf[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values_buf

            # flatten the batch
            b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    b_act = (
                        b_actions.long()[mb_inds]
                        if agent.discrete
                        else b_actions[mb_inds]
                    )
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_act
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

    elif args.algorithm == "apg":
        # ========== APG Training Loop ==========
        # Analytic Policy Gradient: backpropagates through a differentiable environment.
        # No value function, no GAE, no clipping. The loss is simply the negative
        # discounted return, and gradients flow through both the policy and the env.

        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = next_obs.to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            # Temperature annealing for Gumbel-Softmax (discrete actions)
            if args.apg_anneal_temp and agent.discrete:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                temp = args.apg_gumbel_temp_min + frac * (
                    args.apg_gumbel_temp_init - args.apg_gumbel_temp_min
                )
            else:
                temp = args.apg_gumbel_temp_init

            optimizer.zero_grad()

            obs, _ = envs.reset()
            obs = obs.to(device)
            done = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
            traj_return = torch.tensor(0.0, device=device)

            for step in range(0, args.num_steps):
                global_step += args.num_envs

                # Differentiable action — gradients flow through into the environment
                action = agent.get_apg_action(obs, temp=temp)

                # Step through differentiable environment (computation graph preserved)
                obs, reward, terminated, truncated, infos = envs.step(action)
                obs = obs.to(device)
                reward = reward.to(device)
                terminated = terminated.to(device)
                truncated = truncated.to(device)
                done = (terminated | truncated).float()

                # Accumulate discounted return
                traj_return = traj_return + (args.gamma**step) * reward.sum()

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # Loss: minimize negative discounted return (maximize return)
            loss = -traj_return / envs.num_envs

            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor_parameters(), args.max_grad_norm)
            optimizer.step()

            # Logging
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar(
                "apg/trajectory_return", traj_return.item() / envs.num_envs, global_step
            )
            writer.add_scalar("apg/total_loss", loss.item(), global_step)
            if agent.discrete:
                writer.add_scalar("apg/gumbel_temperature", temp, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

    envs.close()
    writer.close()
