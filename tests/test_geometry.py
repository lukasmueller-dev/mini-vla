"""Unit tests for mini_vla.geometry — the shared 2-link IK expert and THE
grasp predicate (effector_over_block), reused by the training label, the
closed-loop eval, and (per geometry.py's own module docstring) the rollout.
Previously untested: a sign error in solve_ik's elbow-branch selection, or a
boundary bug in effector_over_block, would have shipped silently."""

import math

import pytest

from mini_vla.geometry import (
    BASE,
    BLOCK_MAX,
    BLOCK_MIN,
    L1,
    L2,
    clamp,
    effector_over_block,
    fk,
    ik_to_x,
    solve_ik,
)


def _reachable_targets():
    """A spread of workspace points strictly inside the reachable annulus
    [|L1-L2|, L1+L2] around the base, on both sides — solve_ik picks the
    elbow-branch sign from which side of the base the target is on, so both
    sides must be covered."""
    r_min, r_max = abs(L1 - L2), L1 + L2
    radii = [r_min + 0.1 * (r_max - r_min), 0.5 * (r_min + r_max), r_min + 0.9 * (r_max - r_min)]
    angles = [0.3, 1.2, 2.0, 2.8, -0.5, -1.5]  # both target_x >= 0 and < 0
    return [(r * math.cos(a), r * math.sin(a)) for r in radii for a in angles]


@pytest.mark.parametrize("dx,dy", _reachable_targets())
def test_solve_ik_round_trips_through_fk(dx, dy):
    theta1, theta2 = solve_ik(dx, dy)
    e = fk(theta1, theta2)
    ex, ey = e["ex"] - BASE[0], e["ey"] - BASE[1]
    assert math.isclose(ex, dx, abs_tol=1e-6)
    assert math.isclose(ey, dy, abs_tol=1e-6)


def test_solve_ik_theta1_stays_in_the_sampled_band():
    for dx, dy in _reachable_targets():
        theta1, _ = solve_ik(dx, dy)
        assert -math.pi / 2 <= theta1 < 3 * math.pi / 2


def test_solve_ik_out_of_reach_returns_safe_fallback():
    beyond_reach = (L1 + L2) * 1.5
    assert solve_ik(beyond_reach, 0.0) == (0.0, 0.0)
    inside_dead_zone = abs(L1 - L2) * 0.1
    assert solve_ik(inside_dead_zone, 0.0) == (0.0, 0.0)


def test_ik_to_x_places_effector_on_the_grasp_point():
    size = (BLOCK_MIN + BLOCK_MAX) / 2
    for x in (BASE[0] - 0.3, BASE[0], BASE[0] + 0.3):
        theta1, theta2 = ik_to_x(x, size=size, rest=0.0)
        e = fk(theta1, theta2)
        assert math.isclose(e["ex"], x, abs_tol=1e-6)
        assert math.isclose(e["ey"], size / 2, abs_tol=1e-6)


def _block(x=0.5, size=0.12, y=0.0):
    return {"x": x, "size": size, "y": y}


def test_effector_over_block_true_when_centered_inside():
    block = _block()
    assert effector_over_block(*ik_to_x(block["x"], block["size"], block["y"]), block, grip_radius=0.025)


def test_effector_over_block_false_just_outside_each_edge():
    block = _block(x=0.5, size=0.12, y=0.0)
    grip_radius = 0.025
    cx, cy = block["x"], block["y"] + block["size"] / 2
    half = block["size"] / 2
    # Centered on each edge (not the block center) means the disk of
    # grip_radius pokes out past that edge — must read as NOT fully contained.
    edges = [
        (block["x"] - half, cy),  # left edge
        (block["x"] + half, cy),  # right edge
        (cx, block["y"]),  # bottom edge
        (cx, block["y"] + block["size"]),  # top edge
    ]
    for ex, ey in edges:
        a1, a2 = solve_ik(ex - BASE[0], ey - BASE[1])
        assert not effector_over_block(a1, a2, block, grip_radius)


def test_effector_over_block_false_when_far_away():
    block = _block()
    a1, a2 = solve_ik(block["x"] - BASE[0] + 0.5, block["y"] - BASE[1] + 0.5)
    assert not effector_over_block(a1, a2, block, grip_radius=0.025)


@pytest.mark.parametrize(
    "v,lo,hi,expected",
    [(0.5, 0.0, 1.0, 0.5), (-1.0, 0.0, 1.0, 0.0), (2.0, 0.0, 1.0, 1.0)],
)
def test_clamp(v, lo, hi, expected):
    assert clamp(v, lo, hi) == expected
