from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import tyro


@dataclass
class Args:
    exp_name: str = "rl"
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
    print_every_n_episodes: int = 10
    """print episodic return to console every N finished episodes (<=0 disables prints)"""

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
    apg_num_grad_steps: int = 8
    """number of gradient steps per iteration"""
    apg_ent_coef: float = 0.0
    """coefficient of the entropy bonus for APG (0 = disabled, 0.01 = match PPO default)"""
    apg_segment_length: int = 0
    """segment length for value bootstrapping (0 = max_episode_steps, backward compat)"""
    apg_bootstrap: str = "critic"
    """bootstrap mode for segmented APG: 'mc' uses collected future rewards, 'critic' uses learned V(s)"""
    apg_critic_coef: float = 0.5
    """coefficient of the critic loss for segmented APG (only used with --apg_bootstrap critic)"""
    apg_critic_lr: Optional[float] = None
    """separate learning rate for critic (None = use policy LR, only used with --apg_bootstrap critic)"""

    # Comparison
    max_episode_steps: int = 30
    """max episode steps for custom environments"""
    equalize_grad_steps: bool = False
    """scale APG iterations so total gradient steps match PPO"""

    # Seed sweep & evaluation
    num_seeds: int = 1
    """number of seeds to sweep (runs training N times with seeds 1..N)"""
    eval_grad_interval: int = 0
    """evaluate every N gradient/optimizer steps (0 = auto: ~20 evals across training)"""
    eval_episodes: int = 10
    """number of episodes per deterministic evaluation"""

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
