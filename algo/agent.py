import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningObsNormalizer:
    """Welford-style running mean/std tracker for observation normalization."""

    def __init__(self, obs_dim, device):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 1e-4

    def update(self, obs_batch):
        # Online Welford update on detached obs
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0, unbiased=False)
        batch_count = obs_batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        self.var = m2 / total
        self.count = total

    def normalize(self, obs):
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


class Agent(nn.Module):
    def __init__(self, envs, use_layernorm=False):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.array(envs.single_action_space.shape).prod()

        layers = [
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(256))
        layers += [
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(256))
        layers.append(layer_init(nn.Linear(256, action_dim), std=0.01))
        self.actor_mean = nn.Sequential(*layers)
        # Smaller initial std for stability in physics sim (exp(-2.0) ≈ 0.135)
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -2.0))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def _get_dist(self, x):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std.expand_as(mean))
        return Normal(mean, std)

    def get_action_and_value(self, x, action=None):
        dist = self._get_dist(x)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)

    def get_apg_action(self, x, temp=1.0):
        """Get differentiable action for APG (reparameterized Gaussian sample)."""
        return self._get_dist(x).rsample()

    def actor_parameters(self):
        return list(self.actor_mean.parameters()) + [self.actor_log_std]
