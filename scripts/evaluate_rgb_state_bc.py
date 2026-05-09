"""
Evaluate a trained behavior cloning policy in ManiSkill.

Usage:
    # Evaluate the latest checkpoint:
    python scripts/evaluate_rgb_state_bc.py --config configs/pickcube_rgb_state_bc.yaml

    # Evaluate a specific checkpoint:
    python scripts/evaluate_rgb_state_bc.py \\
        --config configs/pickcube_rgb_state_bc.yaml \\
        --checkpoint checkpoints/epoch_0100.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.eval import evaluate
from mini_vla.train import build_policy
from mini_vla.utils import (
    latest_checkpoint,
    load_checkpoint,
    load_config,
    parse_overrides,
    set_seed,
    setup_logging,
    wandb_finish,
    wandb_init,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pickcube_rgb_state_bc.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=None, help="Override eval_episodes")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["eval_episodes"] = args.episodes
    cfg.update(parse_overrides(args.override))

    set_seed(cfg.get("seed", 42))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Find checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.checkpoint or cfg.get("eval_checkpoint")
    if ckpt_path is None:
        ckpt_path = latest_checkpoint(cfg.get("checkpoint_dir", "checkpoints"))
    if ckpt_path is None:
        logger.error("No checkpoint found. Train the policy first.")
        sys.exit(1)
    logger.info("Loading checkpoint: %s", ckpt_path)

    # ── Build and load policy ─────────────────────────────────────────────
    # We need state_dim and action_dim.  Read them from the saved config.
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg  = ckpt_data.get("config", cfg)

    # The dimensions are not saved explicitly; they're embedded in the weight shapes.
    # Extract from the checkpoint's first and last layer.
    sd = ckpt_data["model_state_dict"]
    state_dim  = sd["state_encoder.0.weight"].shape[1]
    action_dim = sd["action_head.2.weight"].shape[0]  # last Linear out_features
    # (fallback: search for the last Linear layer)
    last_key = [k for k in sd if "weight" in k and "action_head" in k][-1]
    action_dim = sd[last_key].shape[0]

    logger.info("Inferred state_dim=%d  action_dim=%d", state_dim, action_dim)

    policy = build_policy(saved_cfg, state_dim, action_dim).to(device)
    load_checkpoint(ckpt_path, policy, device=device)

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb_run = wandb_init(cfg)

    # ── Evaluate ──────────────────────────────────────────────────────────
    result = evaluate(policy, cfg, device, wandb_run=wandb_run)

    print("\n" + "="*50)
    print("  Evaluation Results")
    print("="*50)
    print(f"  Episodes        : {len(result.episode_rewards)}")
    print(f"  Success Rate    : {result.success_rate:.2%}")
    print(f"  Avg Reward      : {result.avg_reward:.3f}")
    print(f"  Avg Ep Length   : {result.avg_episode_length:.1f}")
    print("="*50 + "\n")

    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
