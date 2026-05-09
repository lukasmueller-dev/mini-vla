"""
Print a full diagnostic of the ManiSkill environment.

Usage:
    python scripts/inspect_env.py
    python scripts/inspect_env.py --env PickCube-v1 --control-mode pd_ee_delta_pose
"""

import argparse
import pprint

import gymnasium as gym
import numpy as np

try:
    import mani_skill.envs  # noqa: F401
except ImportError:
    print("[warn] mani_skill not found; make sure it is installed.")


def _shape(x: np.ndarray | None) -> tuple | str:
    return tuple(x.shape) if x is not None else "n/a"


def main(env_id: str, control_mode: str, obs_mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"  ManiSkill Environment Inspector")
    print(f"  env_id       : {env_id}")
    print(f"  control_mode : {control_mode}")
    print(f"  obs_mode     : {obs_mode}")
    print(f"{'='*60}\n")

    # Try with the requested obs_mode first; fall back to state-only if the
    # renderer (Vulkan/MoltenVK) is not available.
    render_mode = "rgb_array"
    for attempt_obs_mode, attempt_render in [(obs_mode, render_mode), ("state", None)]:
        try:
            env = gym.make(
                env_id,
                obs_mode=attempt_obs_mode,
                control_mode=control_mode,
                render_mode=attempt_render,
            )
            if attempt_obs_mode != obs_mode:
                print(
                    f"[warn] Could not initialise with obs_mode='{obs_mode}' "
                    f"(Vulkan/MoltenVK unavailable). Falling back to obs_mode='state'.\n"
                    f"       RGB camera info will not be shown.\n"
                )
            break
        except RuntimeError as e:
            if "Vulkan" in str(e) or "IncompatibleDriver" in str(e) or "vk::" in str(e):
                continue
            raise
    else:
        print("ERROR: Could not create environment even in state-only mode.")
        return

    # ── Spaces ────────────────────────────────────────────────────────────
    print("── Observation Space ──────────────────────────────────────")
    pprint.pprint(env.observation_space)

    print("\n── Action Space ────────────────────────────────────────────")
    print(env.action_space)
    print(f"  action_dim : {env.action_space.shape}")

    # ── First observation ─────────────────────────────────────────────────
    obs, info = env.reset()
    print("\n── Observation Keys (top-level) ───────────────────────────")
    if isinstance(obs, dict):
        print(list(obs.keys()))

        # Camera / image — ManiSkill3 uses sensor_data, not image
        print("\n── Camera Keys ─────────────────────────────────────────────")
        sensor_data = obs.get("sensor_data", obs.get("image"))
        if sensor_data is not None:
            cam_keys = list(sensor_data.keys())
            print(f"  cameras : {cam_keys}")
            for cam in cam_keys:
                cam_data = sensor_data[cam]
                print(f"\n  [{cam}] sub-keys: {list(cam_data.keys())}")
                for subkey, val in cam_data.items():
                    arr = np.asarray(val)
                    print(f"    {subkey:12s}: shape={arr.shape}  dtype={arr.dtype}")
        else:
            print("  (no 'sensor_data' or 'image' key in obs)")

        # Robot state — ManiSkill3 exposes a flat 'state' vector
        print("\n── Robot State / Proprioception ────────────────────────────")
        if "state" in obs:
            arr = np.asarray(obs["state"])
            print(f"  obs['state'] : shape={arr.shape}  dtype={arr.dtype}")
            print(f"  state_dim    : {arr.shape[-1]}")
        elif "agent" in obs:
            agent = obs["agent"]
            print(f"  agent sub-keys: {list(agent.keys())}")
            total_state = 0
            for key, val in agent.items():
                arr = np.asarray(val)
                print(f"    {key:16s}: shape={arr.shape}  dtype={arr.dtype}")
                total_state += arr.size
            print(f"\n  total proprioceptive dim: {total_state}")
        else:
            print("  (no 'state' or 'agent' key in obs)")

        # Extra top-level keys
        extra = {k: v for k, v in obs.items() if k not in ("image", "agent")}
        if extra:
            print("\n── Extra Obs Keys ──────────────────────────────────────────")
            for k, v in extra.items():
                arr = np.asarray(v)
                print(f"    {k:20s}: shape={arr.shape}  dtype={arr.dtype}")
    else:
        print(f"  obs type: {type(obs)}")

    # ── Info / reward ─────────────────────────────────────────────────────
    print("\n── Info Dict (after reset) ─────────────────────────────────")
    pprint.pprint(info)

    print("\n── Reward Mode / Control Mode ──────────────────────────────")
    print(f"  control_mode : {getattr(env.unwrapped, 'control_mode', control_mode)}")
    if hasattr(env.unwrapped, "reward_mode"):
        print(f"  reward_mode  : {env.unwrapped.reward_mode}")
    else:
        print("  reward_mode  : (attribute not found)")

    env.close()
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PickCube-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--obs-mode", default="rgb+state")
    args = parser.parse_args()
    main(args.env, args.control_mode, args.obs_mode)
