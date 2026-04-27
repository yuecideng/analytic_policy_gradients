## PointMass
PPO:
python rl.py --env_id PointMassNavigate-v0 --num_envs 16 --total_timesteps 1000000 --max_episode_steps 100 --num-steps 400  --num-seeds 5 --exp_name ppo_point_mass

## Franka Reach
APG:
python rl.py --env_id FrankaReach-v0 --num_envs 32 --total_timesteps 500000 --max_episode_steps 30 --algorithm apg   --num_seeds 5 --exp_name apg_franka_reach

## Push T
APG:
python rl.py --env_id PushT-v0 --num_envs 16 --total_timesteps 500000 --max_episode_steps 30 --num-steps 400 --algorithm apg  