"""Policy evaluation: rollout in ManiSkill and report success/reward/length."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF

from mini_vla.models import BCPolicy, MLPPolicy

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    success_rate: float
    avg_reward: float
    avg_episode_length: float
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)


def _extract_obs(
    obs: Any,
    camera_name: str | None,
    image_size: tuple[int, int],
    device: torch.device,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (image, state) tensors from a ManiSkill3 observation dict.

    ManiSkill3 layout (obs_mode="rgb+state"):
        obs["sensor_data"][<camera>]["rgb"]  – (1, H, W, 3) uint8
        obs["state"]                         – (1, state_dim) float32
    """
    # ── Image ─────────────────────────────────────────────────────────────
    sensor_data = obs.get("sensor_data", obs.get("image"))
    if sensor_data is None:
        raise ValueError("No 'sensor_data' or 'image' key in obs. Check obs_mode includes 'rgb'.")
    cam = camera_name or next(iter(sensor_data))
    rgb = sensor_data[cam]["rgb"]           # (1, H, W, 3) or (H, W, 3)
    img_np = np.asarray(rgb, dtype=np.uint8).squeeze(0)  # → (H, W, 3)

    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    H, W = image_size
    img_t = TF.resize(img_t, [H, W], antialias=True)
    img_t = img_t.unsqueeze(0).to(device)  # (1, 3, H, W)

    # ── State ─────────────────────────────────────────────────────────────
    if "state" in obs:
        state_np = np.asarray(obs["state"], dtype=np.float32).squeeze(0)  # (state_dim,)
    elif "agent" in obs:
        agent = obs["agent"]
        parts = [np.asarray(agent[k], dtype=np.float32).ravel() for k in ("qpos", "qvel") if k in agent]
        state_np = np.concatenate(parts)
    else:
        raise ValueError("Could not find 'state' or 'agent' key in obs dict.")
    state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)  # (1, state_dim)
    if state_mean is not None and state_std is not None:
        state_t = (state_t - state_mean.to(device)) / state_std.to(device)

    return img_t, state_t


def evaluate(
    policy: BCPolicy,
    cfg: dict[str, Any],
    device: torch.device,
    wandb_run: Any = None,
) -> EvalResult:
    import gymnasium as gym
    try:
        import mani_skill.envs  # noqa: F401 – registers ManiSkill envs
    except ImportError:
        pass

    env_id: str = cfg["env_id"]
    control_mode: str = cfg.get("control_mode", "pd_ee_delta_pose")
    camera_name: str | None = cfg.get("camera_name")
    image_size: tuple[int, int] = tuple(cfg.get("image_size", [128, 128]))  # type: ignore
    n_episodes: int = cfg.get("eval_episodes", 50)
    max_steps: int = cfg.get("eval_max_steps", 200)
    save_video: bool = cfg.get("save_video", False)
    video_dir = Path(cfg.get("video_dir", "videos"))

    video_w: int = cfg.get("video_width", 512)
    video_h: int = cfg.get("video_height", 512)

    env = gym.make(
        env_id,
        obs_mode="rgb+state",
        control_mode=control_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(render_camera=dict(width=video_w, height=video_h)),
        max_episode_steps=max_steps,
    )

    # Load state normalisation stats saved during training.
    state_mean = state_std = None
    if "state_mean" in cfg and "state_std" in cfg:
        state_mean = torch.tensor(cfg["state_mean"], dtype=torch.float32)
        state_std  = torch.tensor(cfg["state_std"],  dtype=torch.float32)
    else:
        logger.warning("No state_mean/state_std in config — state will NOT be normalised. "
                       "Re-train to get a checkpoint with normalisation stats.")

    result = EvalResult(0.0, 0.0, 0.0)
    policy.eval()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_success = False
        frames: list[np.ndarray] = []

        with torch.no_grad():
            while not done and ep_len < max_steps:
                try:
                    img_t, state_t = _extract_obs(obs, camera_name, image_size, device,
                                                  state_mean, state_std)
                except Exception as e:
                    logger.error("Obs extraction failed: %s", e)
                    break

                action: np.ndarray = policy(img_t, state_t).cpu().numpy().squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += float(reward)
                ep_len += 1

                if save_video:
                    frame = env.render()
                    if frame is not None:
                        frame = np.asarray(frame)
                        if frame.ndim == 4:
                            frame = frame[0]  # squeeze batch dim (1, H, W, 3) → (H, W, 3)
                        frames.append(frame)

                if info.get("success", False):
                    ep_success = True

        result.episode_rewards.append(ep_reward)
        result.episode_lengths.append(ep_len)
        result.episode_success.append(ep_success)
        logger.info(
            "Episode %d/%d  reward %.2f  len %d  success %s",
            ep + 1, n_episodes, ep_reward, ep_len, ep_success,
        )

        if save_video and frames:
            _save_video(frames, video_dir, ep, cfg.get("video_fps", 20), wandb_run)

    env.close()

    result.success_rate      = float(np.mean(result.episode_success))
    result.avg_reward        = float(np.mean(result.episode_rewards))
    result.avg_episode_length = float(np.mean(result.episode_lengths))

    logger.info(
        "Eval done  success_rate %.2f  avg_reward %.2f  avg_ep_len %.1f",
        result.success_rate,
        result.avg_reward,
        result.avg_episode_length,
    )

    if wandb_run is not None:
        from mini_vla.utils import wandb_log
        wandb_log(
            wandb_run,
            {
                "eval/success_rate": result.success_rate,
                "eval/avg_reward": result.avg_reward,
                "eval/avg_episode_length": result.avg_episode_length,
            },
        )

    return result


def _extract_state(
    obs: Any,
    device: torch.device,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract flat state tensor. Handles raw tensor obs (obs_mode='state') and dict obs."""
    if isinstance(obs, torch.Tensor):
        state_np = obs.cpu().numpy().astype(np.float32).squeeze(0)
    elif isinstance(obs, np.ndarray):
        state_np = obs.astype(np.float32).squeeze(0)
    elif isinstance(obs, dict):
        if "state" in obs:
            state_np = np.asarray(obs["state"], dtype=np.float32).squeeze(0)
        elif "agent" in obs:
            agent = obs["agent"]
            parts = [np.asarray(agent[k], dtype=np.float32).ravel() for k in ("qpos", "qvel") if k in agent]
            state_np = np.concatenate(parts)
        else:
            raise ValueError("Could not find 'state' or 'agent' key in obs dict.")
    else:
        raise ValueError(f"Unrecognised obs type: {type(obs)}")
    state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)
    if state_mean is not None and state_std is not None:
        state_t = (state_t - state_mean.to(device)) / state_std.to(device)
    return state_t


def evaluate_state(
    policy: MLPPolicy,
    cfg: dict[str, Any],
    device: torch.device,
    wandb_run: Any = None,
) -> EvalResult:
    """Evaluate a state-only MLP policy (no images)."""
    import gymnasium as gym
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError:
        pass

    env_id: str = cfg["env_id"]
    control_mode: str = cfg.get("control_mode", "pd_ee_delta_pose")
    n_episodes: int = cfg.get("eval_episodes", 50)
    max_steps: int = cfg.get("eval_max_steps", 200)
    save_video: bool = cfg.get("save_video", False)
    video_dir = Path(cfg.get("video_dir", "videos"))
    video_w: int = cfg.get("video_width", 512)
    video_h: int = cfg.get("video_height", 512)

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode=control_mode,
        render_mode="rgb_array" if save_video else None,
        human_render_camera_configs=dict(render_camera=dict(width=video_w, height=video_h)),
        max_episode_steps=max_steps,
    )

    state_mean = state_std = None
    if "state_mean" in cfg and "state_std" in cfg:
        state_mean = torch.tensor(cfg["state_mean"], dtype=torch.float32)
        state_std  = torch.tensor(cfg["state_std"],  dtype=torch.float32)
    else:
        logger.warning("No state_mean/state_std in config — state will NOT be normalised.")

    result = EvalResult(0.0, 0.0, 0.0)
    policy.eval()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_success = False
        frames: list[np.ndarray] = []

        with torch.no_grad():
            while not done and ep_len < max_steps:
                try:
                    state_t = _extract_state(obs, device, state_mean, state_std)
                except Exception as e:
                    logger.error("State extraction failed: %s", e)
                    break

                action: np.ndarray = policy(state_t).cpu().numpy().squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                ep_len += 1

                if save_video:
                    frame = env.render()
                    if frame is not None:
                        frame = np.asarray(frame)
                        if frame.ndim == 4:
                            frame = frame[0]
                        frames.append(frame)

                if info.get("success", False):
                    ep_success = True

        result.episode_rewards.append(ep_reward)
        result.episode_lengths.append(ep_len)
        result.episode_success.append(ep_success)
        logger.info("Episode %d/%d  reward %.2f  len %d  success %s",
                    ep + 1, n_episodes, ep_reward, ep_len, ep_success)

        if save_video and frames:
            _save_video(frames, video_dir, ep, cfg.get("video_fps", 20), wandb_run)

    env.close()

    result.success_rate       = float(np.mean(result.episode_success))
    result.avg_reward         = float(np.mean(result.episode_rewards))
    result.avg_episode_length = float(np.mean(result.episode_lengths))

    logger.info("Eval done  success_rate %.2f  avg_reward %.2f  avg_ep_len %.1f",
                result.success_rate, result.avg_reward, result.avg_episode_length)

    if wandb_run is not None:
        from mini_vla.utils import wandb_log
        wandb_log(wandb_run, {
            "eval/success_rate": result.success_rate,
            "eval/avg_reward": result.avg_reward,
            "eval/avg_episode_length": result.avg_episode_length,
        })

    return result


def _save_video(
    frames: list[np.ndarray],
    video_dir: Path,
    episode_idx: int,
    fps: int,
    wandb_run: Any,
) -> None:
    try:
        import imageio
        video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / f"episode_{episode_idx:04d}.mp4"
        imageio.mimwrite(str(path), frames, fps=fps)
        logger.info("Saved rollout video → %s", path)
        if wandb_run is not None:
            from mini_vla.utils import wandb_log_video
            wandb_log_video(wandb_run, path, key=f"rollout/episode_{episode_idx}")
    except Exception as e:
        logger.warning("Could not save video: %s", e)
