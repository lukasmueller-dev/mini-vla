"""
Collect or validate ManiSkill demonstration data.

Two modes:
  1. --validate  : load an existing HDF5 demo file and print statistics.
  2. (default)   : print instructions for collecting demos via ManiSkill's
                   built-in tools, since scripted/teleoperated collection
                   requires interactive or motion-planner pipelines.

Usage:
    # Check what demos you have:
    python scripts/collect_or_load_demos.py --validate --demo-path data/demos/demo.h5

    # Print collection instructions:
    python scripts/collect_or_load_demos.py
"""

import argparse
from pathlib import Path


COLLECTION_INSTRUCTIONS = """
─────────────────────────────────────────────────────────────────
  ManiSkill Demo Collection Instructions
─────────────────────────────────────────────────────────────────

Option A – Use ManiSkill's built-in motion-planner demos (recommended):

  python -m mani_skill.trajectory.replay_trajectory \\
      --traj-path <MANISKILL_TRAJ_PATH> \\
      --obs-mode rgb+state \\
      --target-control-mode pd_ee_delta_pose \\
      -o data/demos/PickCube-v1_pd_ee_delta_pose.h5

  ManiSkill ships motion-planner trajectories for PickCube-v1.
  Find the source path with:
      python -c "import mani_skill; print(mani_skill.__file__)"
  then look under assets/demos/.

Option B – Download community demos:
  See https://huggingface.co/datasets/haosulab/ManiSkill-Demos

Option C – Collect via teleoperation (SpaceMouse / keyboard):
  python -m mani_skill.examples.demo_teleop \\
      --env-id PickCube-v1 \\
      --control-mode pd_ee_delta_pose \\
      --save-traj --traj-name data/demos/teleop.h5

─────────────────────────────────────────────────────────────────
  Expected HDF5 layout for load_maniskill_demos():
─────────────────────────────────────────────────────────────────

  /data/
      traj_0/
          obs/
              image/<camera_name>/rgb  – (T, H, W, 3) uint8
              agent/qpos               – (T, qpos_dim) float32
              agent/qvel               – (T, qvel_dim) float32
          actions                      – (T, action_dim) float32
          terminated                   – (T,) bool
          success                      – (T,) bool
      traj_1/ ...

─────────────────────────────────────────────────────────────────
"""


def validate_demos(demo_path: Path) -> None:
    try:
        import h5py
    except ImportError:
        print("h5py is not installed. Run: pip install h5py")
        return

    print(f"\nValidating demo file: {demo_path}")
    with h5py.File(demo_path, "r") as f:
        keys = list(f.get("data", f).keys())
        print(f"  Trajectories found: {len(keys)}")
        total_steps = 0
        for k in keys[:5]:  # inspect first 5
            grp = f.get(f"data/{k}", f.get(k))
            if grp is None:
                continue
            acts = grp.get("actions")
            n = len(acts) if acts is not None else "?"
            total_steps += (n if isinstance(n, int) else 0)
            print(f"  [{k}]: {n} steps")
            # Print obs keys
            obs_grp = grp.get("obs")
            if obs_grp is not None:
                print(f"    obs keys: {list(obs_grp.keys())}")
                img_grp = obs_grp.get("image")
                if img_grp:
                    print(f"    camera keys: {list(img_grp.keys())}")
        if len(keys) > 5:
            print(f"  ... and {len(keys) - 5} more")
        print(f"  Total steps (first 5): {total_steps}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--demo-path", type=str, default=None)
    args = parser.parse_args()

    if args.validate:
        if args.demo_path is None:
            print("--demo-path required with --validate")
            return
        validate_demos(Path(args.demo_path))
    else:
        print(COLLECTION_INSTRUCTIONS)


if __name__ == "__main__":
    main()
