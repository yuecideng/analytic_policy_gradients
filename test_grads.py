"""Verify critic bootstrap now gives actor gradients after the env fix."""
import torch, math, numpy as np
import torch.nn.functional as F
from rl import Agent, RunningObsNormalizer
from env_registry import get_env_spec
from utils import set_seed

set_seed(1)
factory = get_env_spec('PointMassNavigate-v0').apg_factory
envs = factory(num_envs=4, device='cpu', headless=True, max_episode_steps=100)
agent = Agent(envs)
obs_dim = int(np.prod(envs.single_observation_space.shape))
obs_normalizer = RunningObsNormalizer(obs_dim, 'cpu')
gamma, effective_seg = 0.99, 25
num_segments = math.ceil(100 / effective_seg)

obs, _ = envs.reset(seed=1)
seg_norm_obs_end, seg_rewards_all, seg_steps_list = [], [], []
for si in range(num_segments):
    steps = min((si+1)*effective_seg, 100) - si*effective_seg
    seg_steps_list.append(steps)
    rewards = []
    for _ in range(steps):
        action = agent.get_apg_action(obs_normalizer.normalize(obs), temp=1.0)
        obs, r, _, _, _ = envs.step(action)
        rewards.append(r.view(-1))
    seg_rewards_all.append(rewards)
    seg_norm_obs_end.append(obs_normalizer.normalize(obs))  # no detach
    obs = obs.detach()
    if hasattr(envs, 'detach_state'):
        envs.detach_state()

print("=== After fix: env obs grad_fn ===")
# re-run one step to get fresh obs with grad
obs2, _ = envs.reset(seed=1)
act = agent.get_apg_action(obs_normalizer.normalize(obs2), temp=1.0)
obs3, r3, _, _, _ = envs.step(act)
print(f"obs.requires_grad={obs3.requires_grad}, obs.grad_fn={obs3.grad_fn is not None}")
print(f"reward.requires_grad={r3.requires_grad}")

print("\n=== Gradient from critic bootstrap to actor ===")
agent.zero_grad()
loss = torch.zeros(1)
for si in range(num_segments - 1):
    ss = seg_steps_list[si]
    bv = (gamma**ss) * agent.get_value(seg_norm_obs_end[si]).squeeze(-1)
    loss = loss - bv.mean()
loss.backward()
ag = [(n, p.grad.norm().item()) for n, p in agent.named_parameters() if p.grad is not None and p.grad.norm() > 0]
print("Params with nonzero grad:", [n for n,_ in ag])
actor_has_grad = any('actor' in n or 'actor_mean' in n for n, _ in ag)
critic_has_grad = any('critic' in n for n, _ in ag)
print(f"Actor gets gradient from V(obs_end): {actor_has_grad}")
print(f"Critic gets gradient from V(obs_end): {critic_has_grad}")

print("\n=== Loss comparison: MC vs Critic bootstrap ===")
def compute_policy_loss(bootstrap_mode):
    agent.zero_grad()
    policy_loss = torch.zeros(1)
    mc_future = [None] * num_segments
    if bootstrap_mode == 'mc':
        running_future = torch.zeros(4)
        for si in reversed(range(num_segments)):
            mc_future[si] = running_future.clone()
            ss = seg_steps_list[si]
            running_future = running_future * (gamma**ss)
            for t, r in enumerate(seg_rewards_all[si]):
                running_future = running_future + r.detach() * (gamma**t)
    for si in range(num_segments):
        ss = seg_steps_list[si]
        seg_r = torch.stack(seg_rewards_all[si])
        disc = gamma ** torch.arange(ss, dtype=torch.float32)
        seg_ret = (seg_r * disc.unsqueeze(1)).sum(0)
        is_last = si == num_segments - 1
        if not is_last:
            if bootstrap_mode == 'mc':
                bv = (gamma**ss) * mc_future[si]
            else:
                bv = (gamma**ss) * agent.get_value(seg_norm_obs_end[si]).squeeze(-1)
        else:
            bv = torch.zeros(4)
        policy_loss = policy_loss - (seg_ret + bv).mean()
    return policy_loss

pl_mc = compute_policy_loss('mc')
pl_critic = compute_policy_loss('critic')
print(f"MC policy_loss:     {pl_mc.item():.4f}")
print(f"Critic policy_loss: {pl_critic.item():.4f}")
print(f"Different: {abs(pl_mc.item() - pl_critic.item()) > 1e-6}")
