"""Unit tests for mini_vla.trainer's batch synthesis — previously untested (see
review notes). synth_batch() builds EVERY training label from scratch: the
mirror-augmentation transform (`mirror_into`) that side-balances each batch by
closed-form reflection instead of a second render, and the bilinear
attention-map label writer (`_write_map_label`). A sign error in the mirror
transform, or an off-by-one in the bilinear weights, would silently corrupt
half of every training batch — showing up only as vaguely-worse convergence in
the browser demo, never as a failing test. That's the gap this file closes.

Strategy: synth_batch does real (cheap, PIL-only) rendering + IK, so tests run
it for real rather than mocking the pipeline, but pin `random` with a seed (or
monkeypatch the scene/command samplers, mirroring test_eval.py's approach) so
assertions are exact rather than "roughly right"."""

import random

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from mini_vla import trainer as T
from mini_vla.geometry import BASE, effector_over_block, fk
from mini_vla.model import ATTN_GRID
from mini_vla.task import BlockPos, sample_sentence
from mini_vla.trainer import GRIP_RADIUS, VLATrainer, _angles_from_circular

G = ATTN_GRID


# ── mirror-augmentation correctness ─────────────────────────────────────────
# synth_batch's docstring claims the mirror twin is the CLOSED-FORM reflection
# (θ1,θ2)→(π−θ1,−θ2), and that this is physically a left-right flip about the
# arm base (x=0.5) because fk() mirrors as ex'=1−ex. These tests verify that
# claim against the actual geometry — not just that mirror_into's arithmetic
# matches itself — so a sign error (e.g. flipping sin1 instead of cos1) fails
# on the FK-reconstructed effector position, not just on a restated formula.


def test_mirror_reflects_effector_position_left_right():
    random.seed(12345)
    trainer = VLATrainer()
    batch = trainer.synth_batch(16)  # 8 mirror pairs
    for k in range(8):
        src, dst = 2 * k, 2 * k + 1
        t1s, t2s = _angles_from_circular(*batch.ysA[src])
        t1d, t2d = _angles_from_circular(*batch.ysA[dst])
        e_src = fk(t1s, t2s)
        e_dst = fk(t1d, t2d)
        # Horizontal mirror about the arm base: ex' = 2*base_x - ex.
        assert e_dst["ex"] == pytest.approx(2 * BASE[0] - e_src["ex"], abs=1e-6)
        # Mirroring is purely left-right: the effector's height is unchanged.
        assert e_dst["ey"] == pytest.approx(e_src["ey"], abs=1e-9)


def test_mirror_maps_rest_pose_to_itself():
    """REST = (pi/2, 0): the docstring calls out REST as a fixed point of the
    mirror transform (straight up is its own reflection). A carry-phase sample
    (mid_carry -> target angles are REST) exercises exactly this branch."""
    random.seed(999)
    trainer = VLATrainer()
    batch = trainer.synth_batch(4, force_carry=True)
    for i in range(4):
        t1, t2 = _angles_from_circular(*batch.ysA[i])
        assert t1 == pytest.approx(T.REST[0], abs=1e-6)
        assert t2 == pytest.approx(T.REST[1], abs=1e-6)


def test_mirror_copies_carry_color_and_gripper_labels_exactly():
    """carry/color/gripper are claimed phase- and side-invariant, so
    mirror_into COPIES them rather than recomputing — verify that copy is
    exact (a stray recompute that occasionally disagrees would slip past a
    tolerance-based check)."""
    random.seed(55)
    trainer = VLATrainer()
    batch = trainer.synth_batch(20)
    for k in range(10):
        src, dst = 2 * k, 2 * k + 1
        assert batch.carryF[dst, 0] == batch.carryF[src, 0]
        assert np.array_equal(batch.ysC[dst], batch.ysC[src])
        assert batch.ysG[dst, 0] == batch.ysG[src, 0]


def test_mirror_flips_the_rendered_image_and_the_attention_map_columns():
    random.seed(2024)
    trainer = VLATrainer()
    batch = trainer.synth_batch(6)
    for k in range(3):
        src, dst = 2 * k, 2 * k + 1
        assert np.array_equal(batch.vis[dst], batch.vis[src, :, ::-1, :])
        assert np.array_equal(
            batch.ysMPick[dst].reshape(G, G),
            batch.ysMPick[src].reshape(G, G)[:, ::-1],
        )


def test_gripper_label_matches_the_shared_grasp_predicate_for_a_controlled_pose(monkeypatch):
    """The gripper label is supposed to be `effector_over_block` (THE shared
    grasp predicate, also used by the closed-loop eval) applied to the actual
    RENDERED pose (a1, a2) — a value the returned Batch does not expose
    (ysA stores the regression TARGET angles, which differ from the rendered
    pose except when the near-target/grasp-now jitter branches fire). So this
    pins random.random()'s exact draw sequence to force the "uniform random
    pose" branch (by zeroing GRASP_FRAC/NEAR_TARGET_FRAC) and to choose that
    pose itself — once exactly on the block's grasp point, once far from it —
    so the expected label is known independently and exactly."""
    layout = [BlockPos(color=0, x=0.3, size=0.1, y=0.0)]
    cmd = sample_sentence(0)
    monkeypatch.setattr(T, "random_layout", lambda: layout)
    monkeypatch.setattr(T, "sample_command", lambda _layout: cmd)
    monkeypatch.setattr(T, "block_of_color", lambda _layout, _color: layout[0])
    monkeypatch.setattr(T, "GRASP_FRAC", 0.0)
    monkeypatch.setattr(T, "NEAR_TARGET_FRAC", 0.0)

    from mini_vla.geometry import THETA1_RANGE, THETA2_RANGE, ik_to_x

    def _draws_for_pose(a1: float, a2: float):
        """random.random() call order inside synth_one, once GRASP_FRAC and
        NEAR_TARGET_FRAC are 0: [grasp_now check (any value fails <0), near
        check (any value fails <0), a1's uniform fraction, a2's uniform
        fraction], then arbitrary values for word-dropout's per-token draws."""
        r1 = (a1 - THETA1_RANGE[0]) / (THETA1_RANGE[1] - THETA1_RANGE[0])
        r2 = (a2 - THETA2_RANGE[0]) / (THETA2_RANGE[1] - THETA2_RANGE[0])
        return iter([0.9, 0.9, r1, r2] + [0.5] * 200)

    trainer = VLATrainer()

    at_grasp = ik_to_x(layout[0].x, layout[0].size, 0.0)
    seq = _draws_for_pose(*at_grasp)
    monkeypatch.setattr(T.random, "random", lambda: next(seq))
    batch_at = trainer.synth_batch(2, force_carry=False)
    assert effector_over_block(*at_grasp, layout[0].as_block(), GRIP_RADIUS)  # sanity: pose really is contained
    assert batch_at.ysG[0, 0] == 1.0

    far_away = (THETA1_RANGE[0], THETA2_RANGE[0])
    seq2 = _draws_for_pose(*far_away)
    monkeypatch.setattr(T.random, "random", lambda: next(seq2))
    batch_far = trainer.synth_batch(2, force_carry=False)
    assert not effector_over_block(*far_away, layout[0].as_block(), GRIP_RADIUS)  # sanity: pose really is clear
    assert batch_far.ysG[0, 0] == 0.0


# ── bilinear attention-map label placement ──────────────────────────────────
# _write_map_label spreads a soft target over the (up to) 4 cells around a
# continuous (u, v) point. An off-by-one in the floor/frac split would shift
# every attention-map label by a cell — the tests below pin down the exact
# cell(s) and weights for a center, an edge, and an exact grid-line point, plus
# the reconstruction identity a correct bilinear split must satisfy.


def _reconstruct_uv(ys_m_flat: np.ndarray, base: int) -> tuple[float, float]:
    """Weighted-average of cell centers under the label -- the same quantity
    the model's soft-argmax head reads out. For an interior point (no
    clipping), a correct bilinear split reconstructs the original (u, v)
    exactly."""
    u_hat = v_hat = 0.0
    for i in range(G):
        for j in range(G):
            w = float(ys_m_flat[base + i * G + j])
            u_hat += w * (j + 0.5) / G
            v_hat += w * (i + 0.5) / G
    return u_hat, v_hat


def test_write_map_label_reconstructs_interior_point_exactly():
    trainer = VLATrainer()
    arr = np.zeros(G * G, dtype=np.float32)
    u, v = 0.37, 0.62
    trainer._write_map_label(arr, 0, u, v)
    assert arr.sum() == pytest.approx(1.0, abs=1e-6)
    u_hat, v_hat = _reconstruct_uv(arr, 0)
    assert u_hat == pytest.approx(u, abs=1e-5)
    assert v_hat == pytest.approx(v, abs=1e-5)


def test_write_map_label_at_exact_cell_center_is_a_one_hot():
    """(u, v) sitting exactly on a cell's center should land ALL weight in
    that single cell -- a soft-label sanity check that a hard center isn't
    smeared across neighbors by an indexing bug."""
    trainer = VLATrainer()
    arr = np.zeros(G * G, dtype=np.float32)
    j, i = 4, 7
    u, v = (j + 0.5) / G, (i + 0.5) / G
    trainer._write_map_label(arr, 0, u, v)
    nonzero = np.flatnonzero(arr)
    assert list(nonzero) == [i * G + j]
    assert arr[i * G + j] == pytest.approx(1.0, abs=1e-6)


def test_write_map_label_on_a_grid_line_splits_evenly_across_four_cells():
    """(u, v) exactly on the boundary between cells in BOTH axes must split
    its weight evenly (0.25) across all four surrounding cells -- the
    canonical boundary case for a bilinear kernel."""
    trainer = VLATrainer()
    arr = np.zeros(G * G, dtype=np.float32)
    j, i = 3, 5
    u, v = (j + 1) / G, (i + 1) / G  # boundary between column j/j+1, row i/i+1
    trainer._write_map_label(arr, 0, u, v)
    expected_cells = {i * G + j: 0.25, i * G + (j + 1): 0.25, (i + 1) * G + j: 0.25, (i + 1) * G + (j + 1): 0.25}
    nonzero = {idx: float(arr[idx]) for idx in np.flatnonzero(arr)}
    assert nonzero.keys() == expected_cells.keys()
    for idx, w in expected_cells.items():
        assert nonzero[idx] == pytest.approx(w, abs=1e-6)


def test_write_map_label_near_image_edge_clips_out_of_bounds_weight():
    """u=0 pushes one bilinear neighbor to grid column -1, which must be
    dropped (not wrapped, not clamped back on) -- so the total on-grid weight
    for an edge point is partial, not the interior's full 1.0."""
    trainer = VLATrainer()
    arr = np.zeros(G * G, dtype=np.float32)
    trainer._write_map_label(arr, 0, 0.0, 0.5)
    assert 0.0 < arr.sum() < 1.0
    # every surviving weight must be inside a valid row/col.
    for idx in np.flatnonzero(arr):
        assert 0 <= idx % G < G
        assert 0 <= idx // G < G


def test_write_map_label_accumulates_across_multiple_calls():
    """ys_m_flat is written in place with `+=`, and synth_batch relies on this
    to lay several samples' labels into one flat array at different `base`
    offsets -- a regression to `=` would make each write clobber, rather than
    add to, its slot when called twice at the same base (not the batch's
    normal path, but the accumulation semantics is exactly what the `+=` is
    for and is otherwise unexercised)."""
    trainer = VLATrainer()
    arr = np.zeros(G * G, dtype=np.float32)
    trainer._write_map_label(arr, 0, 0.5, 0.5)
    once = arr.copy()
    trainer._write_map_label(arr, 0, 0.5, 0.5)
    assert np.allclose(arr, 2 * once)


# ── batch-composition invariants ────────────────────────────────────────────


def test_carry_fraction_matches_config_knob_over_many_batches():
    """CARRY_FRAC is a product knob (config.py) controlling how much of
    training is the carry-home phase vs. reach; a regression that stops
    sampling it (e.g. an inverted condition) would silently starve one phase
    of supervision. Checked over enough samples to be a meaningful statistical
    check, not a coin flip."""
    random.seed(4242)
    trainer = VLATrainer()
    batch = trainer.synth_batch(400)
    frac = float(batch.carryF.mean())
    assert frac == pytest.approx(T.CARRY_FRAC, abs=0.08)


def test_color_label_is_a_one_hot_matching_the_sampled_command(monkeypatch):
    """ysC's one-hot must name the color the sampled sentence actually
    commands -- not just be *some* one-hot row."""
    layout = [BlockPos(color=2, x=0.4, size=0.1, y=0.0)]
    cmd = sample_sentence(2)
    monkeypatch.setattr(T, "random_layout", lambda: layout)
    monkeypatch.setattr(T, "sample_command", lambda _layout: cmd)
    monkeypatch.setattr(T, "block_of_color", lambda _layout, _color: layout[0])

    random.seed(1)
    trainer = VLATrainer()
    batch = trainer.synth_batch(2)
    for idx in (0, 1):
        row = batch.ysC[idx]
        assert row.sum() == pytest.approx(1.0)
        assert int(np.argmax(row)) == 2
