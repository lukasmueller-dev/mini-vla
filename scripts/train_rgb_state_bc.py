"""
Train the RGB + state behavior cloning policy.

Usage:
    python scripts/train_rgb_state_bc.py --config configs/pickcube_rgb_state_bc.yaml
    python scripts/train_rgb_state_bc.py --config configs/pickcube_rgb_state_bc.yaml \\
        --override num_epochs=50 learning_rate=1e-3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.datasets import BCDataset, load_maniskill_demos
from mini_vla.train import train
from mini_vla.utils import load_config, parse_overrides, set_seed, setup_logging, wandb_finish, wandb_init

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pickcube_rgb_state_bc.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values, e.g. --override num_epochs=50"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.update(parse_overrides(args.override))

    set_seed(cfg.get("seed", 42))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Load demos ────────────────────────────────────────────────────────
    demo_path = cfg.get("demo_path")
    logger.info("Loading demos from %s", demo_path)
    try:
        samples = load_maniskill_demos(
            demo_path=demo_path,
            camera_name=cfg.get("camera_name"),
            image_size=tuple(cfg.get("image_size", [128, 128])),  # type: ignore
            num_demos=cfg.get("num_demos"),
        )
    except NotImplementedError as e:
        logger.error(
            "Demo loading not yet implemented.\n%s\n\n"
            "Once you have loaded demos, convert them to DemoSample dicts and "
            "pass them to BCDataset (see datasets.py).",
            e,
        )
        sys.exit(1)

    logger.info("Loaded %d transitions", len(samples))

    train_ds, val_ds = BCDataset.train_val_split(
        samples,
        train_frac=cfg.get("train_val_split", 0.9),
        seed=cfg.get("seed", 42),
    )

    num_workers: int = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    ) if len(val_ds) > 0 else None

    state_dim  = train_ds.state.shape[1]
    action_dim = train_ds.action.shape[1]
    logger.info("state_dim=%d  action_dim=%d", state_dim, action_dim)

    # Persist normalisation stats so eval can apply the same transform.
    cfg["state_mean"] = train_ds.state_mean.tolist()
    cfg["state_std"]  = train_ds.state_std.tolist()

    # ── WandB ─────────────────────────────────────────────────────────────
    wandb_run = wandb_init(cfg)

    # ── Train ─────────────────────────────────────────────────────────────
    train(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        wandb_run=wandb_run,
    )

    wandb_finish(wandb_run)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
