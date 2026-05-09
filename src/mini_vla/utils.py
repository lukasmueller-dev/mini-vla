"""Shared utilities: config loading, logging setup, checkpointing, wandb helpers."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ['key=value', ...] into a dict, coercing ints, floats, and bools."""
    out: dict[str, Any] = {}
    for item in overrides:
        k, _, v = item.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)  # type: ignore[assignment]
                break
            except ValueError:
                pass
        if v == "true":
            v = True  # type: ignore[assignment]
        elif v == "false":
            v = False  # type: ignore[assignment]
        out[k.strip()] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    is_best: bool = False,
) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    if is_best:
        shutil.copy(path, ckpt_dir / "best.pt")
    logger.info("Saved checkpoint → %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, state.get("epoch", -1))
    return state


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    ckpts = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


# ─────────────────────────────────────────────────────────────────────────────
# WandB helpers
# ─────────────────────────────────────────────────────────────────────────────

def wandb_init(cfg: dict[str, Any]) -> Any:
    """Initialise a wandb run. Returns the run object or None."""
    if not cfg.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed – skipping. pip install mini-vla[wandb]")
        return None

    run = wandb.init(
        project=cfg.get("wandb_project", "mini-vla-maniskill"),
        entity=cfg.get("wandb_entity") or None,
        name=cfg.get("run_name"),
        config=cfg,
        resume="allow",
    )
    logger.info("WandB run: %s", run.url if run else "disabled")
    return run


def wandb_log(run: Any, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def wandb_log_checkpoint(run: Any, path: str | Path, name: str = "checkpoint") -> None:
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
    except Exception as e:
        logger.warning("Failed to log checkpoint artifact: %s", e)


def wandb_log_video(run: Any, path: str | Path, key: str = "rollout") -> None:
    if run is None:
        return
    try:
        import wandb
        run.log({key: wandb.Video(str(path))})
    except Exception as e:
        logger.warning("Failed to log video: %s", e)


def wandb_finish(run: Any) -> None:
    if run is not None:
        run.finish()
