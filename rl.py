# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
#
# Supports two algorithms:
#   PPO: Proximal Policy Optimization (standard, black-box env)
#   APG: Analytic Policy Gradient (backprop through differentiable env)
#
# Only continuous (Box) action spaces are supported.
#
# Usage:
#   PPO: python rl.py --algorithm ppo --env_id Pendulum-v1
#   APG: python rl.py --algorithm apg --env_id <your-diff-env>
from algo.__main__ import main

if __name__ == "__main__":
    main()
