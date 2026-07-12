"""Unit tests for the language warm-up early-stop knobs — warmupMinBatches /
warmupStopRatio (see mini_vla.trainer.language_warmup).

The language model and batch synth are stubbed so NO TensorFlow graph is built,
but importing mini_vla.trainer still loads TF at import time — so the whole
module is skipped when TF isn't installed (offline / CI without the dep)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorflow")

from mini_vla import trainer as T  # noqa: E402  (import after importorskip guard)


def _trainer_with_losses(loss_seq):
    """A VLATrainer whose language head replays `loss_seq` (held at its last
    value once exhausted) and whose batch synth is a no-op — language_warmup
    only consumes the per-step scalar loss, not the batch tensors. Returns the
    trainer plus a mutable call counter so a test can assert how many warm-up
    steps ran before the early-stop fired."""
    tr = T.VLATrainer()
    calls = {"n": 0}

    def train_on_batch(_lang, _ys):
        loss = loss_seq[min(calls["n"], len(loss_seq) - 1)]
        calls["n"] += 1
        return loss

    tr.models = SimpleNamespace(lang=SimpleNamespace(train_on_batch=train_on_batch))
    tr.synth_lang_batch = lambda _n: (None, None)
    return tr, calls


@pytest.mark.parametrize("floor", [15, 20, 30])
def test_floor_gates_the_stop_point(monkeypatch, floor):
    # Loss plateaus at 5% of initial immediately (well under the 10% ratio), so
    # only the floor holds warm-up back: it must run to exactly `floor` steps,
    # never fewer. This is the knob a follow-up lowers to shorten the browser
    # Loading phase, so the stop point has to track it.
    monkeypatch.setattr(T, "WARMUP_MIN_BATCHES", floor)
    tr, calls = _trainer_with_losses([2.0] + [0.1] * 500)
    tr.language_warmup()
    assert calls["n"] == floor + 1  # steps k = 0..floor inclusive


def test_stop_ratio_gates_convergence(monkeypatch):
    # Tight ratio: the trailing mean must fall under 2% of initial. A loss stuck
    # at 10% of initial never qualifies, so warm-up runs to the cap despite being
    # well past the floor — proving the ratio, not just the floor, is wired.
    monkeypatch.setattr(T, "WARMUP_MIN_BATCHES", 5)
    monkeypatch.setattr(T, "WARMUP_STOP_RATIO", 0.02)
    tr, calls = _trainer_with_losses([2.0] + [0.2] * (T.WARMUP_BATCHES + 50))
    tr.language_warmup()
    assert calls["n"] == T.WARMUP_BATCHES


def test_runs_to_cap_when_never_converges():
    # Flat loss → trailing mean never drops below the ratio → no early stop, so
    # warm-up exhausts the WARMUP_BATCHES hard cap.
    tr, calls = _trainer_with_losses([2.0] * (T.WARMUP_BATCHES + 50))
    tr.language_warmup()
    assert calls["n"] == T.WARMUP_BATCHES
