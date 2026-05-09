"""Dataset utilities for behavior cloning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DemoSample(TypedDict):
    rgb: torch.Tensor    # (C, H, W) float32 in [0, 1]
    state: torch.Tensor  # (state_dim,) float32
    action: torch.Tensor # (action_dim,) float32


def load_maniskill_demos(
    demo_path: str | Path,
    camera_name: str | None = None,
    image_size: tuple[int, int] = (128, 128),
    num_demos: int | None = None,
) -> list[DemoSample]:
    """
    Load ManiSkill3 demonstration trajectories from an HDF5 file.

    Verified HDF5 layout (PickCube-v1, obs_mode=rgb+state, pd_ee_delta_pose):
        traj_N/
            obs/
                sensor_data/base_camera/rgb  – (T+1, 128, 128, 3) uint8
                state                        – (T+1, 42) float32
            actions                          – (T, 7) float32

    obs has T+1 entries (includes terminal obs); actions has T entries.
    We pair obs[t] → action[t] for t in 0..T-1.
    """
    import h5py
    import torchvision.transforms.functional as TF

    path = Path(demo_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")

    samples: list[DemoSample] = []
    H, W = image_size

    with h5py.File(path, "r") as f:
        traj_keys = sorted(f.keys())
        if num_demos is not None:
            traj_keys = traj_keys[:num_demos]

        for key in traj_keys:
            traj = f[key]
            actions = traj["actions"][:]                          # (T, 7)
            T = len(actions)

            # RGB: (T+1, H, W, 3) uint8
            sensor_data = traj["obs/sensor_data"]
            cam = camera_name or next(iter(sensor_data))
            rgb_raw = traj[f"obs/sensor_data/{cam}/rgb"][:T]     # (T, H, W, 3)

            # State: (T+1, 42) float32
            state_raw = traj["obs/state"][:T]                    # (T, 42)

            for t in range(T):
                img = torch.from_numpy(rgb_raw[t]).permute(2, 0, 1).float() / 255.0  # (3, H, W)
                if img.shape[-2:] != (H, W):
                    img = TF.resize(img, [H, W], antialias=True)

                samples.append({
                    "rgb":    img,
                    "state":  torch.from_numpy(state_raw[t].astype(np.float32)),
                    "action": torch.from_numpy(actions[t].astype(np.float32)),
                })

    logger.info("Loaded %d transitions from %d trajectories in %s", len(samples), len(traj_keys), path)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Torch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BCDataset(Dataset[DemoSample]):
    """
    Behavior cloning dataset.

    Accepts a list of DemoSample dicts or a pre-built dict of stacked tensors.
    Each sample is:
        {
            "rgb":    (C, H, W) float32 in [0, 1],
            "state":  (state_dim,) float32,
            "action": (action_dim,) float32,
        }
    """

    def __init__(
        self,
        samples: list[DemoSample],
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
    ) -> None:
        self.rgb     = torch.stack([s["rgb"] for s in samples])
        self.state   = torch.stack([s["state"] for s in samples])
        self.action  = torch.stack([s["action"] for s in samples])

        # Compute or apply state normalisation statistics.
        if state_mean is None:
            self.state_mean = self.state.mean(0)
            self.state_std  = self.state.std(0).clamp(min=1e-6)
        else:
            self.state_mean = state_mean
            self.state_std  = state_std
        self.state = (self.state - self.state_mean) / self.state_std

        logger.info(
            "BCDataset: %d samples  |  rgb %s  |  state %s  |  action %s",
            len(self),
            tuple(self.rgb.shape[1:]),
            tuple(self.state.shape[1:]),
            tuple(self.action.shape[1:]),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, idx: int) -> DemoSample:
        return {
            "rgb":    self.rgb[idx],
            "state":  self.state[idx],
            "action": self.action[idx],
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(
        cls,
        rgb: np.ndarray,    # (N, H, W, C) uint8
        state: np.ndarray,  # (N, state_dim) float32
        action: np.ndarray, # (N, action_dim) float32
    ) -> "BCDataset":
        """Convenience constructor from raw numpy arrays."""
        rgb_t    = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        state_t  = torch.from_numpy(state).float()
        action_t = torch.from_numpy(action).float()
        samples: list[DemoSample] = [
            {"rgb": rgb_t[i], "state": state_t[i], "action": action_t[i]}
            for i in range(len(rgb_t))
        ]
        return cls(samples)

    @staticmethod
    def train_val_split(
        samples: list[DemoSample],
        train_frac: float = 0.9,
        seed: int = 42,
    ) -> tuple["BCDataset", "BCDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))
        cut = int(len(idx) * train_frac)
        train = [samples[i] for i in idx[:cut]]
        val   = [samples[i] for i in idx[cut:]]
        train_ds = BCDataset(train)
        # Val uses train's normalisation stats to avoid data leakage.
        val_ds = BCDataset(val, state_mean=train_ds.state_mean, state_std=train_ds.state_std)
        return train_ds, val_ds
