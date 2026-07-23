"""Unit tests for mini_vla.embeddings' asset-shape guard (see review notes) —
mirrors js/src/embeddings.ts's assertAssetShape. Without it, a mismatched
vocab.txt/embeddings-50d.bin pair whose total element count still happens to
divide evenly reshapes and dequantizes silently, seeding the frozen
text_embedding layer with garbage."""

from pathlib import Path

import numpy as np
import pytest

from mini_vla.embeddings import _assert_asset_shape
from mini_vla.vocab_gen import EMBED_DIM, VOCAB_SIZE

_BASE = Path("assets")


def _raw(n_words: int) -> np.ndarray:
    return np.zeros(n_words * EMBED_DIM, dtype=np.int8)


def _words(n: int) -> list[str]:
    return [f"w{i}" for i in range(n)]


def test_accepts_the_real_shipped_assets():
    n = VOCAB_SIZE - 2
    _assert_asset_shape(_raw(n), _words(n), _BASE)


def test_rejects_word_count_mismatch():
    n = VOCAB_SIZE - 2
    with pytest.raises(ValueError, match="vocab.txt has"):
        _assert_asset_shape(_raw(n), _words(n - 1), _BASE)


def test_rejects_bin_length_mismatch_even_when_word_count_is_right():
    n = VOCAB_SIZE - 2
    truncated = _raw(n)[:-EMBED_DIM]  # one row short, but words list is correct
    with pytest.raises(ValueError, match="embeddings-50d.bin is"):
        _assert_asset_shape(truncated, _words(n), _BASE)
