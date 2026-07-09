# Language + scene-layout space for the VLA tasks. Python port of
# js/src/examples.ts.
#
# Eight named colors (chosen for maximum hue contrast at small silhouette
# sizes), each with synonyms; pick-up commands are generated from a slot grammar
# (filler? verb article color-word noun please?) so hundreds of surface forms
# collapse onto a single color intent. The word inventory lives in
# assets/grammar.json (single source of truth shared with js/ and the embedding
# generator). Which colors/densities are actually SAMPLED is the run config
# (run_config.py).
#
# Token ids index the pretrained GloVe table (vocab_gen.py / assets/): 0 is
# <pad>, 1 is <unk>, then the ~20k-word GloVe vocab. Only the grammar words' ids
# (CORE_VOCAB) are available synchronously; embeddings.load() registers the full
# list so free user text like "gold" resolves to a real pretrained vector.

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .geometry import BLOCK, BLOCK_MAX, BLOCK_MIN
from .run_config import run_config
from .vocab_gen import CORE_VOCAB, VOCAB_SIZE  # noqa: F401  (VOCAB_SIZE re-exported)

_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "assets" / "grammar.json"
_grammar = json.loads(_GRAMMAR_PATH.read_text())

MAX_SEQ_LEN = CONFIG.task.maxSeqLen
PAD = 0
UNK = 1


@dataclass
class ColorDef:
    name: str
    hex: str
    synonyms: list[str]


COLORS: list[ColorDef] = [ColorDef(**c) for c in _grammar["colors"]]

_LIFT_VERBS: list[list[str]] = _grammar["tasks"]["lift"]["verbs"]
_ARTICLES: list[str] = _grammar["articles"]
_NOUNS: list[str] = _grammar["nouns"]
_FILLERS: list[str] = _grammar["fillers"]

# Token ids of every color synonym — exempt from training word-dropout.
COLOR_TOKEN_IDS: set[int] = {
    CORE_VOCAB[s] for c in COLORS for s in c.synonyms if s in CORE_VOCAB
}

# Token ids that CARRY LABEL INFORMATION — the color synonyms plus every verb
# word. Exempt from training word-dropout: the color word carries the whole
# intent, and the verb words are the tokens a terse command leans on.
LABEL_TOKEN_IDS: set[int] = COLOR_TOKEN_IDS | {
    CORE_VOCAB[w] for verb in _LIFT_VERBS for w in verb if w in CORE_VOCAB
}

# full GloVe word list, registered by embeddings.load() once read.
_full_vocab: Optional[dict[str, int]] = None


def register_full_vocab(words: list[str]) -> None:
    """Install the complete vocab (word at index i → token id i+2)."""
    global _full_vocab
    _full_vocab = {w: i + 2 for i, w in enumerate(words)}


# ---- typo tolerance for user-typed words ----
# A misspelling ("puple") is in neither the full GloVe list nor CORE_VOCAB, so
# it tokenizes to <unk>. To recover the intent we fuzzy-map an OOV word to the
# nearest CORE_VOCAB entry (color synonyms + grammar words). Distance is Optimal
# String Alignment (Levenshtein + adjacent transposition at cost 1). This only
# ever runs on the fallback branch — never on the training hot path.

_MAX_EDITS = 2


def _edit_dist(a: str, b: str, max_edits: int) -> int:
    """OSA edit distance, early-exiting once every cell in a row exceeds max."""
    if abs(len(a) - len(b)) > max_edits:
        return max_edits + 1
    prev_prev: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        row_min = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev_prev[j - 2] + 1)  # adjacent transposition
            cur.append(v)
            if v < row_min:
                row_min = v
        if row_min > max_edits:
            return max_edits + 1
        prev_prev = prev
        prev = cur
    return prev[len(b)]


def _correct_word(word: str) -> Optional[int]:
    """Token id of the nearest CORE_VOCAB word within the edit budget, or None.
    Budget scales down for short words so 2-3 letter garbage can't match a
    color."""
    budget = min(_MAX_EDITS, 1 if len(word) <= 3 else _MAX_EDITS)
    best: Optional[str] = None
    best_d = budget + 1
    for cand in CORE_VOCAB:
        d = _edit_dist(word, cand, best_d - 1)  # only care if it strictly beats best
        if d < best_d:
            best_d = d
            best = cand
    return None if best is None else CORE_VOCAB[best]


def tokenize(sentence: str) -> list[int]:
    """Lowercase, strip punctuation, map OOV words to <unk>, pad to MAX_SEQ_LEN."""
    words = [w for w in "".join(
        c if c.isalpha() or c == " " else "" for c in sentence.lower()
    ).split(" ") if w]
    tokens = [PAD] * MAX_SEQ_LEN
    for i in range(min(len(words), MAX_SEQ_LEN)):
        w = words[i]
        tok = None
        if _full_vocab is not None:
            tok = _full_vocab.get(w)
        if tok is None:
            tok = CORE_VOCAB.get(w)
        if tok is None:
            tok = _correct_word(w)
        tokens[i] = UNK if tok is None else tok
    return tokens


@dataclass
class Sentence:
    color: int  # index into COLORS — the block the arm acts ON
    text: str
    words: list[str]
    tokens: list[int]


def _pick(seq):
    return seq[random.randrange(len(seq))]


def sample_sentence(color: int) -> Sentence:
    """A random surface form for a color: filler? verb? article? color noun?
    please?. Verb/article/noun are randomly DROPPED so the language scorer also
    trains on compressed forms ("grab red"). The color word never drops."""
    def keep(drop_prob: float) -> bool:
        return random.random() >= drop_prob

    parts: list[str] = []
    if random.random() < CONFIG.task.fillerProb:
        parts.append(_pick(_FILLERS))
    if keep(CONFIG.task.dropVerbProb):
        parts.extend(_pick(_LIFT_VERBS))
    if keep(CONFIG.task.dropArticleProb):
        parts.append(_pick(_ARTICLES))
    parts.append(_pick(COLORS[color].synonyms))
    if keep(CONFIG.task.dropNounProb):
        parts.append(_pick(_NOUNS))
    if random.random() < CONFIG.task.pleaseProb:
        parts.append("please")
    text = " ".join(parts)
    return Sentence(color=color, text=text, words=parts, tokens=tokenize(text))


def sample_command(layout: "Layout") -> Sentence:
    """A random command EXECUTABLE in the given layout: the acted-on color is
    one that is actually present in the scene."""
    return sample_sentence(present_color(layout))


DEFAULT_SENTENCE = Sentence(
    color=0,
    text="pick up the red block",
    words=["pick", "up", "the", "red", "block"],
    tokens=tokenize("pick up the red block"),
)

# ---- scene layouts ----


@dataclass
class BlockPos:
    x: float  # block center, workspace units
    color: int  # index into COLORS
    size: float  # block side length, workspace units (grasp height = size/2)
    # Rest height of the block's BOTTOM (0 / None = on the floor). Pick-up scenes
    # are always flat, so this is currently always 0.
    y: Optional[float] = None

    def as_block(self) -> dict:
        """geometry.effector_over_block / render take a plain dict."""
        return {"x": self.x, "size": self.size, "y": self.y}


Layout = list[BlockPos]

_PLACE_L: tuple[float, float] = CONFIG.task.placeLeft
_PLACE_R: tuple[float, float] = CONFIG.task.placeRight
_MIN_GAP = CONFIG.task.minBlockGap


def _in_band(band: tuple[float, float]) -> float:
    lo, hi = band
    return lo + random.random() * (hi - lo)


def _rand_size() -> float:
    return BLOCK_MIN + random.random() * (BLOCK_MAX - BLOCK_MIN)


def _sil_half(size: float) -> float:
    """Half-width a block occupies in the MODEL'S-EYE view (silBlockScale boost)
    — the wider of the two views, so clearing this clears the display too."""
    return (size * CONFIG.render.silBlockScale) / 2


def _pick_colors(k: int, pool_size: int) -> list[int]:
    """k distinct color indices from the palette's first `pool_size` entries,
    uniformly without replacement (partial shuffle)."""
    pool = list(range(pool_size))
    for i in range(k):
        j = i + random.randrange(len(pool) - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def _place_side(band: tuple[float, float], colors: list[int]) -> list[BlockPos]:
    """Lay `colors` (1 or 2) into one band, constructing a non-overlapping pair
    rather than reject-sampling. A pair whose required clearance exceeds the
    whole band drops to a single block."""
    lo, hi = band
    if len(colors) == 1:
        return [BlockPos(color=colors[0], x=_in_band(band), size=_rand_size())]
    size_l = _rand_size()
    size_r = _rand_size()
    sep = _sil_half(size_l) + _sil_half(size_r) + _MIN_GAP
    if sep > hi - lo:
        return [BlockPos(color=colors[0], x=_in_band(band), size=size_l)]
    x_l = lo + random.random() * (hi - lo - sep)  # left block, clearance still fits
    x_r = x_l + sep + random.random() * (hi - (x_l + sep))  # right block, past clearance
    return [
        BlockPos(color=colors[0], x=x_l, size=size_l),
        BlockPos(color=colors[1], x=x_r, size=size_r),
    ]


def random_layout() -> Layout:
    """2..min(maxBlocks, numColors) blocks per scene, at most 2 per side band,
    each at a random position across its cleanly-reachable side of the floor.
    Colors are unique per scene."""
    rc = run_config()
    cap = min(rc.maxBlocks, rc.numColors)  # colors are unique, so palette caps count
    n = 2 + random.randrange(cap - 1)  # uniform 2..cap
    # split across the two bands, ≤2 per band: 2 → 1+1, 4 → 2+2, 3 → coin flip
    n_l = 1 if n == 2 else 2 if n == 4 else (1 if random.random() < 0.5 else 2)
    colors = _pick_colors(n, rc.numColors)  # unique across the whole scene
    return [
        *_place_side(_PLACE_L, colors[:n_l]),
        *_place_side(_PLACE_R, colors[n_l:]),
    ]


DEFAULT_LAYOUT: Layout = [
    BlockPos(color=1, x=0.16, size=0.09),
    BlockPos(color=0, x=0.84, size=BLOCK),
]


def block_of_color(layout: Layout, color: int) -> BlockPos:
    for b in layout:
        if b.color == color:
            return b
    return layout[0]


def present_color(layout: Layout) -> int:
    """A random color that IS present in the given layout."""
    return layout[random.randrange(len(layout))].color
