import time

import numpy as np
import torch
import torch.nn as nn

from algo.evaluate import deterministic_eval


def run_ppo(
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
):
    """Run the PPO training loop. Returns (global_step, eval_success_rates)."""
    eval_success_rates = []
    episode_count = 0
    total_grad_steps = 0
    total_optim_steps = 0

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

    current_ep_ret = torch.zeros(args.num_envs, dtype=torch.float32).to(device)
    current_ep_len = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

    for iteration in range(1, args.num_iterations + 1):
        iter_start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            obs_normalizer.update(next_obs.detach())
            norm_obs = obs_normalizer.normalize(next_obs)
            obs_buf[step] = norm_obs
            dones_buf[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(norm_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action)
            next_done = (terminated | truncated).float()
            rewards_buf[step] = reward.to(device).view(-1)
            next_obs = next_obs.to(device)
            next_done = next_done.to(device)

            current_ep_ret += reward.to(device).view(-1)
            current_ep_len += 1

            done_mask = next_done.bool()
            num_done = done_mask.sum().item()
            if num_done > 0:
                episode_count += num_done
                avg_ret = current_ep_ret[done_mask].mean().item()
                avg_len = current_ep_len[done_mask].mean().item()
                writer.add_scalar("charts/episodic_return", avg_ret, global_step)
                writer.add_scalar("charts/episodic_length", avg_len, global_step)
                writer.add_scalar(
                    "charts/success_rate",
                    terminated[done_mask].bool().float().mean().item(),
                    global_step,
                )
                # Mean time to success
                success_mask = terminated[done_mask].bool()
                if success_mask.any():
                    writer.add_scalar(
                        "charts/mean_time_to_success",
                        current_ep_len[done_mask][success_mask].mean().item(),
                        global_step,
                    )
                # Mean final distance
                if "final_distance" in infos:
                    writer.add_scalar(
                        "charts/mean_final_distance",
                        infos["final_distance"][done_mask].mean().item(),
                        global_step,
                    )
                if "final_rot_distance" in infos:
                    writer.add_scalar(
                        "charts/mean_final_rot_distance",
                        infos["final_rot_distance"][done_mask].mean().item(),
                        global_step,
                    )
                if (
                    should_print_episodes
                    and episode_count % args.print_every_n_episodes == 0
                ):
                    print(
                        f"global_step={global_step}, total_optim_steps={total_optim_steps}, episodic_return={avg_ret:.5f}, episodic_length={avg_len}"
                    )

            # Reset tracking for done environments
            current_ep_ret[done_mask] = 0.0
            current_ep_len[done_mask] = 0.0

        # bootstrap value if not done
        with torch.no_grad():
            norm_next_obs = obs_normalizer.normalize(next_obs)
            next_value = agent.get_value(norm_next_obs).reshape(1, -1)
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
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
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

                b_act = b_actions[mb_inds]
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
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                total_grad_steps += 1
                total_optim_steps += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging: multiple x-axes for fair comparison
        elapsed = time.time() - start_time
        iter_elapsed = time.time() - iter_start_time
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
        writer.add_scalar("charts/SPS", int(global_step / elapsed), global_step)
        # Multi-axis logging for fair PPO vs APG comparison
        writer.add_scalar("charts/total_grad_steps", total_grad_steps, global_step)
        writer.add_scalar("charts/total_optim_steps", total_optim_steps, global_step)
        writer.add_scalar("charts/wall_time", elapsed, global_step)
        writer.add_scalar("perf/iter_time_sec", iter_elapsed, global_step)
        writer.add_scalar(
            "perf/grad_steps_per_sec", total_grad_steps / elapsed, global_step
        )
        # Log key metrics keyed by grad_steps and wall_time too
        writer.add_scalar(
            "by_grad_steps/learning_rate",
            optimizer.param_groups[0]["lr"],
            total_grad_steps,
        )
        writer.add_scalar("by_grad_steps/total_loss", loss.item(), total_grad_steps)
        writer.add_scalar(
            "by_wall_time/learning_rate", optimizer.param_groups[0]["lr"], elapsed
        )

        # Deterministic evaluation
        if eval_envs is not None and iteration % args.eval_freq == 0:
            eval_result = deterministic_eval(
                agent,
                obs_normalizer,
                args,
                device,
                eval_envs,
                writer,
                global_step,
                total_grad_steps,
            )
            if eval_result["success_rate"] is not None:
                eval_success_rates.append(
                    (total_grad_steps, eval_result["success_rate"])
                )

    return global_step, eval_success_rates
