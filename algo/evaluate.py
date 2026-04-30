import time

import numpy as np
import torch


def deterministic_eval(
    agent,
    obs_normalizer,
    args,
    device,
    eval_envs,
    writer,
    global_step,
    total_grad_steps,
):
    """Run deterministic evaluation episodes and log results."""
    completed_returns = []
    completed_lengths = []
    completed_successes = []
    completed_final_dists = []
    obs, _ = eval_envs.reset()
    obs = obs.to(device)
    ep_ret = torch.zeros(eval_envs.num_envs, dtype=torch.float32, device=device)
    ep_len = torch.zeros(eval_envs.num_envs, dtype=torch.int64, device=device)

    max_steps = args.max_episode_steps * max(
        3, args.eval_episodes // eval_envs.num_envs + 1
    )

    for _ in range(max_steps):
        with torch.no_grad():
            norm_obs = obs_normalizer.normalize(obs)
            action = agent.actor_mean(norm_obs).clamp(-1.0, 1.0)

        obs, reward, terminated, truncated, infos = eval_envs.step(action)
        time.sleep(0.05)  # Add slight delay for better video rendering (if enabled)
        obs = obs.to(device)
        reward = reward.to(device)
        terminated = terminated.to(device)
        truncated = truncated.to(device)

        ep_ret += reward.view(-1).detach()
        ep_len += 1

        done_mask = (terminated | truncated).bool()
        if done_mask.any():
            indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            for i in indices:
                completed_returns.append(ep_ret[i].item())
                completed_lengths.append(ep_len[i].item())
                completed_successes.append(terminated[i].item())
                if "final_distance" in infos:
                    completed_final_dists.append(
                        infos["final_distance"][i].item()
                        if isinstance(infos["final_distance"], torch.Tensor)
                        else infos["final_distance"][i]
                    )
            ep_ret[done_mask] = 0.0
            ep_len[done_mask] = 0

        if len(completed_returns) >= args.eval_episodes:
            break

    result = {"mean_ret": None, "success_rate": None}
    if completed_returns:
        n = min(len(completed_returns), args.eval_episodes)
        mean_ret = np.mean(completed_returns[:n])
        mean_len = np.mean(completed_lengths[:n])
        success_rate = np.mean(completed_successes[:n])
        writer.add_scalar("eval/episodic_return", mean_ret, global_step)
        writer.add_scalar("eval/episodic_length", mean_len, global_step)
        writer.add_scalar("by_grad_steps/eval_return", mean_ret, total_grad_steps)
        writer.add_scalar("eval/success_rate", success_rate, global_step)
        writer.add_scalar(
            "by_grad_steps/eval_success_rate", success_rate, total_grad_steps
        )
        # Mean time to success (episode length conditioned on success)
        success_lengths = [
            completed_lengths[i] for i in range(n) if completed_successes[i]
        ]
        if success_lengths:
            writer.add_scalar(
                "eval/mean_time_to_success", np.mean(success_lengths), global_step
            )
        # Mean final distance
        if completed_final_dists:
            writer.add_scalar(
                "eval/mean_final_distance",
                np.mean(completed_final_dists[:n]),
                global_step,
            )
        print(
            f"  eval (step={global_step}): return={mean_ret:.5f}, length={mean_len:.1f}, "
            f"success_rate={success_rate:.3f} ({n} episodes)"
        )
        result["mean_ret"] = mean_ret
        result["success_rate"] = success_rate
    # Flush any remaining recorded frames as a video
    if hasattr(eval_envs, "video_recorder") and eval_envs.video_recorder is not None:
        eval_envs.video_recorder.on_episode_end(global_step)
    return result
