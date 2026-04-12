# Analytic Policy Gradient (APG) Design

## Goal

Add an analytic policy gradient algorithm alongside PPO in `ppo.py`, switchable via `--algorithm {ppo,apg}`. APG exploits differentiable environments by backpropagating through the environment dynamics to compute exact policy gradients. The purpose is to compare learning efficiency of APG vs PPO on differentiable environments.

## Environment Interface

Both PPO and APG use torch-based environments:

- `env.reset()` returns `torch.Tensor` observations
- `env.step(action)` accepts `torch.Tensor` actions, returns `(obs, reward, terminated, truncated, info)` as torch tensors
- For APG, `env.step()` must preserve the computation graph (gradients flow through)
- For PPO, the rollout wraps steps in `torch.no_grad()` so gradient support is irrelevant

No numpy/torch conversions are needed anywhere.

## Args

Add to the existing `Args` dataclass:

```python
algorithm: str = "ppo"  # choices: "ppo", "apg"

# APG-specific
apg_traj_length: int = 128       # trajectory length per rollout
apg_num_trajectories: int = 4    # parallel trajectories per iteration
apg_gumbel_temp_init: float = 1.0  # initial Gumbel-Softmax temperature (discrete)
apg_gumbel_temp_min: float = 0.1   # minimum temperature
apg_anneal_temp: bool = True       # anneal temperature over training
```

PPO-specific args (`clip_coef`, `clip_vloss`, `target_kl`, `gae_lambda`, `norm_adv`, `vf_coef`, `ent_coef`) remain and are only used when `algorithm="ppo"`.

## Agent Class

Single `Agent` class with both heads. Parameters passed to the optimizer are filtered by algorithm.

```
Agent:
  actor:  obs → 64 → Tanh → 64 → Tanh → n_actions  (policy logits, both modes)
  critic: obs → 64 → Tanh → 64 → Tanh → 1           (V(s), PPO only)
```

**Optimizer filtering:**

- PPO: `params = actor.parameters() + critic.parameters()`
- APG: `params = actor.parameters()` only

### Action computation

- **Discrete actions:**
  - PPO: `Categorical(logits=logits).sample()` (no gradients needed)
  - APG: `F.gumbel_softmax(logits, tau=temp, hard=False)` (differentiable relaxation)

- **Continuous actions:**
  - PPO: `Normal(mean, std).sample()` (no gradients needed)
  - APG: `mean + std * torch.randn_like(std)` (reparameterization trick, gradients flow)

## Training Loop

### PPO mode (unchanged logic, adapted for torch env)

```
for iteration:
    anneal_lr()
    collect_rollout():           # torch.no_grad() wrapper
        for step in num_steps:
            action = actor.sample()
            obs, reward, term, trunc, info = env.step(action)
            store(obs, action, logprob, reward, done, value)
    compute_gae()
    for epoch in update_epochs:
        for minibatch:
            ratio = exp(new_logprob - old_logprob)
            pg_loss = clipped_surrogate(ratio, advantage)
            v_loss = clipped_value_loss()
            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
            loss.backward()
            optimizer.step()
    log_metrics()
```

### APG mode (new)

```
for iteration:
    anneal_lr()
    if anneal_temp:
        temp = compute_annealed_temp()

    optimizer.zero_grad()
    for traj_idx in num_trajectories:
        obs = env.reset()
        traj_return = 0.0
        for t in traj_length:
            logits = actor(obs)
            action = differentiable_action(logits, temp)  # Gumbel-Softmax or reparam
            obs, reward, term, trunc, info = env.step(action)  # gradients flow through
            traj_return = traj_return + (gamma ** t) * reward.sum()
            # If episode ends mid-trajectory, break and log. The partial
            # trajectory's return is still valid for gradient computation.
            if term or trunc:
                log_episode(info)
                obs = env.reset()
        traj_loss = -traj_return / num_trajectories
        traj_loss.backward()   # accumulate gradients across trajectories

    clip_grad_norm_(actor.parameters(), max_grad_norm)
    optimizer.step()
    log_metrics()
```

Key differences from PPO:
- No `torch.no_grad()` during rollout — the entire trajectory stays in the computation graph
- No value function, no GAE — reward signal is backpropagated directly through the environment
- No importance sampling ratio, no clipping
- Loss is simply negative discounted return
- Gradient accumulation across multiple trajectories before an optimizer step

## Gumbel-Softmax Temperature Annealing

For discrete action spaces, APG uses Gumbel-Softmax relaxation. Temperature controls the relaxation:

- High temp (1.0): smooth, well-behaved gradients, but actions are "soft" (not truly discrete)
- Low temp (0.1): near-one-hot actions, but gradients can be poorly conditioned

Annealing schedule: linear decay from `apg_gumbel_temp_init` to `apg_gumbel_temp_min` over training.

## Logging

Both modes log to TensorBoard:

| Metric | PPO | APG |
|--------|-----|-----|
| `charts/episodic_return` | yes | yes |
| `charts/episodic_length` | yes | yes |
| `charts/learning_rate` | yes | yes |
| `charts/SPS` | yes | yes |
| `losses/policy_loss` | yes | yes |
| `losses/value_loss` | yes | no |
| `losses/entropy` | yes | no |
| `losses/approx_kl` | yes | no |
| `losses/clipfrac` | yes | no |
| `losses/explained_variance` | yes | no |
| `apg/trajectory_return` | no | yes |
| `apg/gumbel_temperature` | no | yes (discrete only) |
| `apg/total_loss` | no | yes |

## File Structure

Everything stays in `ppo.py`. The structure:

1. `Args` dataclass (with `algorithm` field and APG-specific args)
2. `layer_init()` helper (unchanged)
3. `Agent` class (actor + critic, APG only uses actor)
4. `run_ppo_update()` — extracted PPO update logic
5. `run_apg_update()` — APG update logic
6. `__main__` block — env setup, algorithm dispatch, training loop

## Continuous Action Space

For `Box` (continuous) action spaces:

**PPO:**
- Actor outputs `mean` (linear layer) + learnable `log_std` parameter
- Uses `Normal(mean, std)` distribution
- Clipped surrogate with Gaussian policy

**APG:**
- Actor outputs `mean` + learnable `log_std`
- Uses reparameterization: `action = mean + std * epsilon` where `epsilon ~ N(0,1)`
- Gradients flow through `mean` and `std` into the environment

## Out of Scope

- Q-network or value network for APG (gradients come from the environment)
- Target networks
- Multiple differentiable environment libraries
- Mixed PPO+APG training
