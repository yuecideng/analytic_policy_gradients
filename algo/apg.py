import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.evaluate import deterministic_eval


def run_apg(
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
    use_critic,
):
    """Run the APG training loop. Returns (global_step, eval_success_rates)."""
    eval_success_rates = []
    episode_count = 0
    total_grad_steps = 0
    total_optim_steps = 0

    # Stateful: reset once, carry env state forward across gradient steps.
    obs, _ = envs.reset(seed=args.seed)
    obs = obs.to(device)

    current_ep_ret = torch.zeros(args.num_envs, dtype=torch.float32).to(device)
    current_ep_len = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

    for iteration in range(1, args.num_iterations + 1):
        iter_start_time = time.time()

        # Linear LR annealing (matching PPO schedule for fair comparison).
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            if use_critic and len(optimizer.param_groups) > 1:
                init_critic_lr = args.apg_critic_lr or args.learning_rate
                optimizer.param_groups[1]["lr"] = frac * init_critic_lr

        for grad_step in range(args.apg_num_grad_steps):
            total_grad_steps += 1
            total_optim_steps += 1

            optimizer.zero_grad()

            effective_seg = (
                args.apg_segment_length
                if args.apg_segment_length > 0
                else args.max_episode_steps
            )
            num_segments = math.ceil(args.max_episode_steps / effective_seg)
            use_seg = effective_seg < args.max_episode_steps

            policy_loss = torch.tensor(0.0, device=device)
            all_obs_for_norm = []
            all_entropies = []

            # Per-segment storage
            seg_rewards_all = []  # list of lists of [num_envs] reward tensors
            seg_norm_obs_end = []  # normalized obs at end of each segment (for critic)
            seg_norm_obs_start = (
                []
            )  # normalized obs at start of each segment (for critic)
            seg_steps_list = []  # actual steps per segment

            # ===== Segmented forward pass =====
            for seg_idx in range(num_segments):
                seg_start = seg_idx * effective_seg
                seg_end = min(seg_start + effective_seg, args.max_episode_steps)
                seg_steps = seg_end - seg_start
                seg_steps_list.append(seg_steps)

                seg_norm_obs_start.append(obs_normalizer.normalize(obs.detach()))

                segment_rewards = []
                for step in range(seg_steps):
                    global_step += args.num_envs

                    norm_obs = obs_normalizer.normalize(obs)
                    all_obs_for_norm.append(obs.detach())

                    action = agent.get_apg_action(norm_obs)

                    if args.apg_ent_coef > 0:
                        dist = agent._get_dist(norm_obs)
                        ent = dist.entropy().sum(-1)
                        all_entropies.append(ent)

                    obs, reward, terminated, truncated, infos = envs.step(action)
                    obs = obs.to(device)
                    reward = reward.to(device)
                    terminated = terminated.to(device)
                    truncated = truncated.to(device)
                    done = (terminated | truncated).float()

                    segment_rewards.append(reward.view(-1))

                    current_ep_ret += reward.detach().view(-1)
                    current_ep_len += 1

                    done_mask = done.bool()
                    num_done = done_mask.sum().item()
                    if num_done > 0:
                        episode_count += num_done
                        avg_ret = current_ep_ret[done_mask].mean().item()
                        avg_len = current_ep_len[done_mask].mean().item()
                        writer.add_scalar(
                            "charts/episodic_return", avg_ret, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", avg_len, global_step
                        )
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
                        if (
                            should_print_episodes
                            and episode_count % args.print_every_n_episodes == 0
                        ):
                            print(
                                f"global_step={global_step}, total_optim_steps={total_optim_steps}, episodic_return={avg_ret:.5f}, episodic_length={avg_len}"
                            )

                    current_ep_ret[done_mask] = 0.0
                    current_ep_len[done_mask] = 0.0

                seg_rewards_all.append(segment_rewards)
                seg_norm_obs_end.append(obs_normalizer.normalize(obs))

                # Detach at segment boundary — limits gradient chain to seg_steps
                obs = obs.detach()
                if hasattr(envs, "detach_state"):
                    envs.detach_state()
                else:
                    for attr in (
                        "block_pos",
                        "block_angle",
                        "block_vel",
                        "block_ang_vel",
                        "last_action",
                    ):
                        t = getattr(envs, attr, None)
                        if isinstance(t, torch.Tensor):
                            setattr(envs, attr, t.detach())

            # Update obs normalizer
            obs_normalizer.update(torch.cat(all_obs_for_norm, dim=0))

            # ===== Compute per-segment returns and bootstrap =====
            critic_values_list = []
            critic_targets_list = []

            # Pre-compute MC future returns (backwards accumulation)
            mc_future = [None] * num_segments
            if use_seg and args.apg_bootstrap == "mc":
                running_future = torch.zeros(args.num_envs, device=device)
                for seg_idx in reversed(range(num_segments)):
                    mc_future[seg_idx] = running_future.clone()
                    seg_steps = seg_steps_list[seg_idx]
                    running_future = running_future * (args.gamma**seg_steps)
                    for t, r in enumerate(seg_rewards_all[seg_idx]):
                        running_future = running_future + r.detach() * (args.gamma**t)

            # Build per-segment losses
            for seg_idx in range(num_segments):
                seg_steps = seg_steps_list[seg_idx]
                seg_rewards_t = torch.stack(seg_rewards_all[seg_idx])
                discounts = args.gamma ** torch.arange(
                    seg_steps, device=device, dtype=torch.float32
                )
                seg_return = (seg_rewards_t * discounts.unsqueeze(1)).sum(dim=0)

                is_last = seg_idx == num_segments - 1
                if not is_last and use_seg:
                    if args.apg_bootstrap == "mc":
                        bootstrap_value = (args.gamma**seg_steps) * mc_future[seg_idx]
                    elif args.apg_bootstrap == "critic":
                        bootstrap_value = (args.gamma**seg_steps) * agent.get_value(
                            seg_norm_obs_end[seg_idx]
                        ).squeeze(-1)
                    else:
                        bootstrap_value = torch.zeros(args.num_envs, device=device)
                else:
                    bootstrap_value = torch.zeros(args.num_envs, device=device)

                policy_loss = policy_loss - (seg_return + bootstrap_value).mean()

                # Collect critic training data
                if use_seg and args.apg_bootstrap == "critic":
                    critic_pred = agent.get_value(seg_norm_obs_start[seg_idx]).squeeze(
                        -1
                    )
                    critic_values_list.append(critic_pred)
                    critic_targets_list.append((seg_return + bootstrap_value).detach())

            # ===== Critic loss =====
            if (
                use_seg
                and args.apg_bootstrap == "critic"
                and len(critic_values_list) > 0
            ):
                critic_values_t = torch.cat(critic_values_list)
                critic_targets_t = torch.cat(critic_targets_list)
                critic_loss = F.mse_loss(critic_values_t, critic_targets_t)
            else:
                critic_loss = torch.tensor(0.0, device=device)

            # ===== Total loss =====
            loss = policy_loss + args.apg_critic_coef * critic_loss

            if args.apg_ent_coef > 0 and all_entropies:
                entropy_loss = torch.stack(all_entropies).mean()
                loss = loss - args.apg_ent_coef * entropy_loss

            loss.backward()

            nn.utils.clip_grad_norm_(agent.actor_parameters(), args.max_grad_norm)
            if use_seg and args.apg_bootstrap == "critic":
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
            optimizer.step()

            # Logging (once per iteration, after all grad steps)
            # Multi-axis logging for fair PPO vs APG comparison
            elapsed = time.time() - start_time
            iter_elapsed = time.time() - iter_start_time

            # Compute total reward across all segments for logging
            total_reward = sum(r.sum().item() for seg in seg_rewards_all for r in seg)
            horizon_return = total_reward / envs.num_envs

            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("apg/horizon_return", horizon_return, global_step)
            writer.add_scalar("apg/total_loss", loss.item(), global_step)
            if args.apg_ent_coef > 0 and all_entropies:
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

            # Segmented APG metrics
            effective_seg_log = (
                args.apg_segment_length
                if args.apg_segment_length > 0
                else args.max_episode_steps
            )
            use_seg_log = effective_seg_log < args.max_episode_steps
            if use_seg_log:
                writer.add_scalar("apg/num_segments", num_segments, global_step)
                writer.add_scalar(
                    "apg/bootstrap_mode",
                    0.0 if args.apg_bootstrap == "mc" else 1.0,
                    global_step,
                )
                if args.apg_bootstrap == "critic":
                    writer.add_scalar(
                        "apg/critic_loss", critic_loss.item(), global_step
                    )
                    if critic_values_list:
                        critic_val_mean = torch.cat(critic_values_list).mean().item()
                        critic_tgt_mean = torch.cat(critic_targets_list).mean().item()
                        writer.add_scalar(
                            "apg/critic_value_mean", critic_val_mean, global_step
                        )
                        writer.add_scalar(
                            "apg/critic_target_mean", critic_tgt_mean, global_step
                        )

            writer.add_scalar("charts/SPS", int(global_step / elapsed), global_step)
            writer.add_scalar("charts/total_grad_steps", total_grad_steps, global_step)
            writer.add_scalar(
                "charts/total_optim_steps", total_optim_steps, global_step
            )
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
                "by_grad_steps/horizon_return", horizon_return, total_grad_steps
            )
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
