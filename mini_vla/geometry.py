# Shared 2-link arm geometry, in a normalized 1x1 workspace with y UP (unlike
# canvas coords — render.py does the flip). Python port of js/src/geometry.ts.
# The arm base is anchored at (0.5, 0.2); blocks rest on the floor (y = 0) at
# per-scene randomized positions. Everything downstream — the analytical IK
# expert, the training-sample synthesizer, the closed-loop eval and the
# silhouette renderer — works in these units so the vision input and the expert
# labels can never disagree about where things are.

from __future__ import annotations

import math

from .config import CONFIG

L1 = CONFIG.arm.l1
L2 = CONFIG.arm.l2
BASE = CONFIG.arm.base  # (x, y)

# Reference block side length; per-scene blocks randomize in [BLOCK_MIN, BLOCK_MAX].
BLOCK = CONFIG.block.ref
BLOCK_MIN = CONFIG.block.min
BLOCK_MAX = CONFIG.block.max

# Upright rest pose (straight up from the base).
REST: tuple[float, float] = CONFIG.arm.rest

THETA1_RANGE: tuple[float, float] = CONFIG.arm.theta1Range
THETA2_RANGE: tuple[float, float] = CONFIG.arm.theta2Range


def solve_ik(target_x: float, target_y: float, l1: float = L1, l2: float = L2) -> tuple[float, float]:
    """Analytical 2-link inverse kinematics. Target coords are relative to the
    arm base (subtract BASE before calling). Returns the safe fallback (0, 0)
    when the target is geometrically out of reach."""
    d_sq = target_x * target_x + target_y * target_y
    cos_a2 = (d_sq - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(cos_a2) > 1:
        return (0.0, 0.0)

    # elbow-UP branch: which sine sign is "up" depends on the target's side of
    # the base (negative for targets to the right, positive to the left) — a
    # fixed sign choice puts the elbow underground on one side.
    sin_a2 = (-1 if target_x >= 0 else 1) * math.sqrt(1 - cos_a2 * cos_a2)
    theta2 = math.atan2(sin_a2, cos_a2)

    k1 = l1 + l2 * cos_a2
    k2 = l2 * sin_a2
    theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)

    # The atan2 difference can wrap (e.g. -3.93 instead of the identical +2.35
    # for a left-side block). As a regression LABEL the raw value matters:
    # un-normalized it tells the policy to push theta1 the wrong way around,
    # through the joint limit. Wrap into [-pi/2, 3pi/2), the band around the
    # sampled theta1 range.
    while theta1 < -math.pi / 2:
        theta1 += 2 * math.pi
    while theta1 >= (3 * math.pi) / 2:
        theta1 -= 2 * math.pi

    return (theta1, theta2)


def grasp_target(x: float, size: float = BLOCK, rest: float = 0.0) -> dict:
    """Grasp target for a block of side `size` at floor position x: the block
    CENTER (y = rest + size/2). A bigger block is grasped higher, so its size
    feeds the IK; `rest` is the block's bottom height (>0 when it sits on
    another block)."""
    return {"x": x, "y": rest + size / 2}


def ik_to_x(x: float, size: float = BLOCK, rest: float = 0.0) -> tuple[float, float]:
    """IK joint angles that put the end effector at a block's grasp point."""
    t = grasp_target(x, size, rest)
    return solve_ik(t["x"] - BASE[0], t["y"] - BASE[1])


def fk(a1: float, a2: float) -> dict:
    """Forward kinematics: elbow + end-effector positions in workspace units."""
    j1x = BASE[0] + math.cos(a1) * L1
    j1y = BASE[1] + math.sin(a1) * L1
    ex = j1x + math.cos(a1 + a2) * L2
    ey = j1y + math.sin(a1 + a2) * L2
    return {"j1x": j1x, "j1y": j1y, "ex": ex, "ey": ey}


def effector_over_block(a1: float, a2: float, block: dict, grip_radius: float) -> bool:
    """THE grasp predicate: is the effector disk fully inside a block's
    footprint? Center = fk(a1,a2).ex/ey, radius = grip_radius; the block
    footprint is x ∈ [b.x − s/2, b.x + s/2], y ∈ [b.y, b.y + s]. "Fully
    contained" (not just center-in) so the closed gripper genuinely straddles
    the block.

    This is the SINGLE shared "correct-to-close" test used identically by the
    training label (trainer.synth_batch) and the rollout/eval grasp gate —
    keeping the network's supervision and the physical grasp condition provably
    the same fact. `block` is a dict with keys x, size, and optional y."""
    e = fk(a1, a2)
    ex, ey = e["ex"], e["ey"]
    s = block["size"]
    rest = block.get("y") or 0.0
    return (
        ex - grip_radius >= block["x"] - s / 2
        and ex + grip_radius <= block["x"] + s / 2
        and ey - grip_radius >= rest
        and ey + grip_radius <= rest + s
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
