from pathlib import Path

import torch


def save_best_checkpoint(
    checkpoint_dir,
    agent,
    obs_normalizer,
    args,
    global_step,
    total_grad_steps,
    success_rate,
):
    """Save the best policy checkpoint for later deterministic rollout."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent_state = {
        key: value.detach().cpu() for key, value in agent.state_dict().items()
    }
    payload = {
        "agent": agent_state,
        "obs_normalizer": {
            "mean": obs_normalizer.mean.detach().cpu(),
            "var": obs_normalizer.var.detach().cpu(),
            "count": obs_normalizer.count,
        },
        "args": vars(args).copy(),
        "global_step": global_step,
        "total_grad_steps": total_grad_steps,
        "success_rate": success_rate,
    }
    torch.save(payload, checkpoint_dir / "best.pt")
