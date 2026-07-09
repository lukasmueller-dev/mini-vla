# Closed-loop rollout metrics over simulated episodes with the trained policy —
# the dial the open-loop probes lack. Python port of closedLoopEval in
# js/eval/main.ts. Same integrator shape as the JS RolloutEngine: proportional
# step toward the latest prediction, re-predict every few frames, learned gripper
# grasp on the rising edge over a block, settle-then-release — using THE shared
# effector_over_block predicate.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import CONFIG
from .geometry import REST, effector_over_block
from .task import block_of_color, random_layout, sample_command
from .trainer import VLATrainer

_PRED_EVERY = 6  # frames between re-predictions ≈ the engine's predictMs
_MAX_FRAMES = 300


@dataclass
class RolloutEval:
    episodes: int
    graspRate: float
    meanGraspFrames: Optional[float]  # mean frames to grasp (successes only)
    reachJitter: Optional[float]  # mean |Δ target| (rad) between reach predictions


def closed_loop_eval(trainer: VLATrainer, episodes: int) -> RolloutEval:
    """Simulate `episodes` closed-loop pick-up attempts against the trained
    policy and return grasp rate + timing/jitter dials."""
    R = CONFIG.rollout
    grip_radius = CONFIG.gripper.radius
    grip_threshold = CONFIG.gripper.threshold

    grasps = 0
    grasp_frames_sum = 0
    jitter_sum = 0.0
    jitter_n = 0

    for _ in range(episodes):
        layout = random_layout()
        cmd = sample_command(layout)
        target = block_of_color(layout, cmd.color)

        a1, a2 = REST
        carry: Optional[int] = None
        pred = (a1, a2)
        prev_pred: Optional[tuple[float, float]] = None
        near = 0
        settle = 0
        pred_grip = 0.0
        saw_open = False

        for f in range(_MAX_FRAMES):
            if f % _PRED_EVERY == 0:
                p = trainer.predict_target(a1, a2, cmd.tokens, layout, carry)
                if p is None:
                    return RolloutEval(0, 0.0, None, None)
                if prev_pred is not None and carry is None and f > _PRED_EVERY:
                    # reach-phase prediction wobble, skipping the first transient
                    jitter_sum += math.hypot(
                        p.target[0] - prev_pred[0], p.target[1] - prev_pred[1]
                    )
                    jitter_n += 1
                prev_pred = pred = p.target
                pred_grip = p.grip
            a1 += (pred[0] - a1) * R.stepGain
            a2 += (pred[1] - a2) * R.stepGain

            if carry is None:
                # learned grasp gate — SAME predicate as training: effector fully
                # over the commanded block AND the gripper closing on the rising
                # edge (a prior open frame required)
                closing = pred_grip >= grip_threshold
                over = effector_over_block(a1, a2, target.as_block(), grip_radius)
                if over and closing and saw_open:
                    near += 1
                    if near >= R.nearFrames:
                        carry = cmd.color
                        grasps += 1
                        grasp_frames_sum += f
                        prev_pred = None  # carry phase — jitter metric stays reach-only
                else:
                    near = 0
                if not closing:
                    saw_open = True
            elif abs(pred[0] - a1) < R.settleEps and abs(pred[1] - a2) < R.settleEps:
                settle += 1
                if settle >= 4:  # carry settled — episode done
                    break
            else:
                settle = 0

    return RolloutEval(
        episodes=episodes,
        graspRate=grasps / episodes,
        meanGraspFrames=(grasp_frames_sum / grasps) if grasps else None,
        reachJitter=(jitter_sum / jitter_n) if jitter_n else None,
    )
