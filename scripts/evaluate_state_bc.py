"""
Evaluate a trained state-only MLP behavior cloning policy in ManiSkill.

Usage:
    # From a wandb run (downloads latest .pt from Files tab):
    python scripts/evaluate_state_bc.py --config configs/pickcube_state_bc.yaml \\
        --wandb-run <entity/project/run_id>

    # From a local checkpoint:
    python scripts/evaluate_state_bc.py --config configs/pickcube_state_bc.yaml \\
        --checkpoint checkpoints/state_bc/epoch_0500.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.eval import evaluate_state
from mini_vla.models import MLPPolicy
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


def _download_checkpoint_from_wandb(run_id: str, cfg: dict) -> Path | None:
    try:
        import wandb
        api = wandb.Api()

        if run_id.count("/") < 2:
            entity = cfg.get("wandb_entity") or api.default_entity
            project = cfg.get("wandb_project", "mini-vla")
            run_path = f"{entity}/{project}/{run_id}"
        else:
            run_path = run_id

        logger.info("Fetching files from wandb run: %s", run_path)
        run = api.run(run_path)

        pt_files = [f for f in run.files() if f.name.endswith(".pt")]
        if not pt_files:
            logger.error("No .pt files found in wandb run %s", run_path)
            return None

        names = [f.name for f in pt_files]
        target = next((f for f in pt_files if "best" in f.name), None) or pt_files[-1]
        logger.info("Available checkpoints: %s", names)
        logger.info("Downloading: %s", target.name)

        download_dir = Path("checkpoints/wandb_download")
        download_dir.mkdir(parents=True, exist_ok=True)
        target.download(root=str(download_dir), replace=True)

        candidates = list(download_dir.rglob("*.pt"))
        return candidates[-1] if candidates else None

    except Exception as e:
        logger.error("Failed to download checkpoint from wandb: %s", e)
        return None


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pickcube_state_bc.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--wandb-run", default=None,
                        help="Download latest checkpoint from a wandb run. "
                             "Format: <run_id> or <entity/project/run_id>")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[])
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

    ckpt_path = args.checkpoint or cfg.get("eval_checkpoint")

    if ckpt_path is None and args.wandb_run:
        ckpt_path = _download_checkpoint_from_wandb(args.wandb_run, cfg)

    if ckpt_path is None:
        ckpt_path = latest_checkpoint(cfg.get("checkpoint_dir", "checkpoints/state_bc"))
    if ckpt_path is None:
        logger.error("No checkpoint found. Pass --checkpoint, --wandb-run, or train first.")
        sys.exit(1)
    logger.info("Loading checkpoint: %s", ckpt_path)

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg  = ckpt_data.get("config", cfg)

    # Infer dims from weight shapes: net is [Linear, ReLU, Linear, ReLU, ..., Linear]
    sd = ckpt_data["model_state_dict"]
    state_dim  = sd["net.0.weight"].shape[1]
    action_dim = sd[[k for k in sd if "weight" in k][-1]].shape[0]
    hidden_dims = saved_cfg.get("hidden_dims", cfg.get("hidden_dims", [256, 256, 256]))

    logger.info("Inferred state_dim=%d  action_dim=%d  hidden_dims=%s",
                state_dim, action_dim, hidden_dims)

    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
    load_checkpoint(ckpt_path, policy, device=device)

    # Use normalisation stats from the checkpoint's saved config
    for key in ("state_mean", "state_std"):
        if key in saved_cfg and key not in cfg:
            cfg[key] = saved_cfg[key]

    wandb_run = wandb_init(cfg)

    result = evaluate_state(policy, cfg, device, wandb_run=wandb_run)

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Episodes        : {len(result.episode_rewards)}")
    print(f"  Success Rate    : {result.success_rate:.2%}")
    print(f"  Avg Reward      : {result.avg_reward:.3f}")
    print(f"  Avg Ep Length   : {result.avg_episode_length:.1f}")
    print("=" * 50 + "\n")

    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
