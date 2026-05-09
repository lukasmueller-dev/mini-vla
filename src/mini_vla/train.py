"""Training loop for the behavior cloning policy."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mini_vla.models import BCPolicy
from mini_vla.utils import (
    save_checkpoint,
    wandb_log,
    wandb_log_checkpoint,
)

logger = logging.getLogger(__name__)


def build_policy(cfg: dict[str, Any], state_dim: int, action_dim: int) -> BCPolicy:
    return BCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        image_channels=cfg.get("image_channels", 3),
        image_size=tuple(cfg.get("image_size", [128, 128])),  # type: ignore[arg-type]
        cnn_channels=cfg.get("cnn_channels", [32, 64, 64]),
        cnn_kernel_sizes=cfg.get("cnn_kernel_sizes", [8, 4, 3]),
        cnn_strides=cfg.get("cnn_strides", [4, 2, 1]),
        image_embed_dim=cfg.get("image_embed_dim", 256),
        state_hidden_dims=cfg.get("state_hidden_dims", [128, 128]),
        state_embed_dim=cfg.get("state_embed_dim", 64),
        action_hidden_dims=cfg.get("action_hidden_dims", [256, 256]),
    )


def train(
    cfg: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    wandb_run: Any = None,
) -> BCPolicy:
    policy = build_policy(cfg, state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    num_epochs: int = cfg.get("num_epochs", 100)
    scheduler = None
    if cfg.get("lr_scheduler") == "cosine":
        warmup = cfg.get("warmup_epochs", 5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup, eta_min=1e-6)

    criterion = nn.MSELoss()
    grad_clip: float = cfg.get("grad_clip", 1.0)
    log_every: int = cfg.get("log_every", 50)
    ckpt_every: int = cfg.get("checkpoint_every", 10)
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        policy.train()
        epoch_loss = 0.0
        for batch in train_loader:
            image  = batch["rgb"].to(device)
            state  = batch["state"].to(device)
            action = batch["action"].to(device)

            pred = policy(image, state)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch %d  step %d  loss %.4f  lr %.2e",
                    epoch, global_step, loss.item(), lr,
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/loss": loss.item(),
                        "train/mse": loss.item(),
                        "learning_rate": lr,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d  avg_train_loss %.4f", epoch, avg_train_loss)

        if scheduler is not None and epoch > cfg.get("warmup_epochs", 5):
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        val_loss = _validate(policy, val_loader, criterion, device)
        if val_loss is not None:
            logger.info("Epoch %d  val_loss %.4f", epoch, val_loss)
            wandb_log(wandb_run, {"val/loss": val_loss, "epoch": epoch}, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % ckpt_every == 0:
            is_best = val_loss is not None and val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            ckpt_path = save_checkpoint(ckpt_dir, epoch, policy, optimizer, cfg, is_best)
            wandb_log_checkpoint(wandb_run, ckpt_path, name=f"checkpoint-epoch-{epoch}")

    # Final checkpoint
    save_checkpoint(ckpt_dir, num_epochs, policy, optimizer, cfg)
    return policy


def _validate(
    policy: BCPolicy,
    loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
) -> float | None:
    if loader is None:
        return None
    policy.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            pred = policy(batch["rgb"].to(device), batch["state"].to(device))
            total_loss += criterion(pred, batch["action"].to(device)).item()
    return total_loss / len(loader)
