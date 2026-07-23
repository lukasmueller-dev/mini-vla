"""Unit tests for mini_vla.eval's closed-loop grasp state machine — previously
untested (see review notes). The trainer is stubbed to return a scripted,
geometrically-exact IK target + gripper signal, and the layout/command inputs
are pinned via monkeypatch (closed_loop_eval samples its own random layout
internally, so the test fixes it rather than trying to predict it). No TF
forward pass or trained weights are needed: only closed_loop_eval's own
frame-by-frame state machine and its use of the shared effector_over_block
predicate are under test."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorflow")

from mini_vla import eval as E
from mini_vla.geometry import ik_to_x
from mini_vla.task import block_of_color, random_layout, sample_command


def _pin_episode(monkeypatch):
    """Fix closed_loop_eval's internally-sampled layout/command/target to one
    concrete, known scene so the test's scripted trainer can compute the
    exact IK target the eval loop will actually check against."""
    layout = random_layout()
    cmd = sample_command(layout)
    target = block_of_color(layout, cmd.color)
    monkeypatch.setattr(E, "random_layout", lambda: layout)
    monkeypatch.setattr(E, "sample_command", lambda _layout: cmd)
    monkeypatch.setattr(E, "block_of_color", lambda _layout, _color: target)
    return target


def _scripted_trainer(target_block, grip_schedule):
    """A fake VLATrainer: predict_target always returns the exact IK solution
    toward `target_block`'s grasp point (the arm still has to close the gap
    itself via the eval's own stepGain integrator), with `grip` taken from
    `grip_schedule` by PREDICT-CALL index (predict_target is only invoked
    every _PRED_EVERY frames, not every frame), held at its last value once
    exhausted."""
    theta1, theta2 = ik_to_x(target_block.x, target_block.size, target_block.y or 0.0)
    calls = {"n": 0}

    def predict_target(a1, a2, tokens, layout, carry):
        i = min(calls["n"], len(grip_schedule) - 1)
        calls["n"] += 1
        return SimpleNamespace(target=(theta1, theta2), grip=grip_schedule[i], xy=(0.5, 0.5), attn=None)

    return SimpleNamespace(predict_target=predict_target)


def test_grasps_after_opening_then_closing_over_the_block(monkeypatch):
    target = _pin_episode(monkeypatch)
    # 30 open predict-calls (~180 frames at _PRED_EVERY=6) is ample for the
    # stepGain=0.08 integrator to converge onto a static IK target before the
    # gripper closes for the remaining calls, well inside _MAX_FRAMES=300.
    schedule = [0.0] * 30 + [1.0] * 50
    trainer = _scripted_trainer(target, schedule)
    result = E.closed_loop_eval(trainer, episodes=1)
    assert result.graspRate == 1.0
    assert result.meanGraspFrames is not None


def test_never_grasps_if_gripper_never_closes(monkeypatch):
    target = _pin_episode(monkeypatch)
    trainer = _scripted_trainer(target, [0.0] * 100)
    result = E.closed_loop_eval(trainer, episodes=1)
    assert result.graspRate == 0.0
    assert result.meanGraspFrames is None


def test_rising_edge_required_closed_from_frame_zero_never_grasps(monkeypatch):
    """closing=True on every frame (no prior open frame observed) must never
    register a grasp: `saw_open` gates the rising edge specifically so a
    policy that starts closed can't spuriously pass over the target on the
    way through, per eval.py's own "prior open frame required" comment."""
    target = _pin_episode(monkeypatch)
    trainer = _scripted_trainer(target, [1.0] * 100)
    result = E.closed_loop_eval(trainer, episodes=1)
    assert result.graspRate == 0.0
