import sys

import tyro

from algo.args import Args, _normalize_cli_args
from algo.train import _run_training


def main():
    args = tyro.cli(Args, args=_normalize_cli_args(sys.argv[1:]))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if args.algorithm == "apg":
        apg_batch_size = int(
            args.num_envs * args.max_episode_steps * args.apg_num_grad_steps
        )

        args.num_iterations = args.total_timesteps // apg_batch_size
        if args.equalize_grad_steps:
            ppo_grads_per_iter = args.update_epochs * args.num_minibatches
            apg_grads_per_iter = args.apg_num_grad_steps
            ppo_iters = args.total_timesteps // args.batch_size
            args.num_iterations = ppo_iters * ppo_grads_per_iter // apg_grads_per_iter
            print(
                f"equalize_grad_steps: scaling APG to {args.num_iterations} iterations "
                f"(PPO would have {ppo_iters} iters × {ppo_grads_per_iter} grads = "
                f"{ppo_iters * ppo_grads_per_iter} total grad steps)"
            )
        print(f"APG: batch_size={args.num_envs}, num_iterations={args.num_iterations}")
    else:
        args.num_iterations = args.total_timesteps // args.batch_size
        print(
            f"PPO: minibatch_size={args.minibatch_size}, num_iterations={args.num_iterations}"
        )

    if args.eval_freq == 0 and args.num_iterations > 0:
        args.eval_freq = max(args.num_iterations // 10, 1)

    seeds = list(range(1, args.num_seeds + 1)) if args.num_seeds > 1 else [args.seed]
    for seed_idx, seed in enumerate(seeds):
        if args.num_seeds > 1:
            print(f"\n{'='*60}\nSeed {seed} ({seed_idx+1}/{len(seeds)})\n{'='*60}")
        _run_training(args, seed)


if __name__ == "__main__":
    main()
