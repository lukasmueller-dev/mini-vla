"""Unit tests for mini_vla.task's typo-tolerance fuzzy matcher (see review
notes): _edit_dist (OSA — Optimal String Alignment, i.e. Levenshtein plus
adjacent-transposition at cost 1) and _correct_word, the OOV-word-to-nearest-
CORE_VOCAB-entry resolver that backs tokenize()'s fallback branch.

The whole point of hand-rolling OSA instead of using a stock Levenshtein
implementation is that a single adjacent-letter swap ("puprle") should cost 1,
not 2 — that is the case a plain Levenshtein distance gets wrong, so it gets
its own test below.
"""

from mini_vla.task import _MAX_EDITS, _correct_word, _edit_dist, tokenize
from mini_vla.vocab_gen import CORE_VOCAB


# ---- _edit_dist ----


def test_edit_dist_exact_match_is_zero():
    assert _edit_dist("purple", "purple", _MAX_EDITS) == 0


def test_edit_dist_single_substitution_is_one():
    assert _edit_dist("purple", "purpla", _MAX_EDITS) == 1


def test_edit_dist_adjacent_transposition_is_one_not_two():
    # "puprle" is "purple" with the 'r' and 'p' swapped: one adjacent
    # transposition. Plain Levenshtein (substitution/insert/delete only) has
    # no single-edit path between these two strings and scores it 2; OSA adds
    # the transposition operation at cost 1, which is exactly why task.py
    # hand-rolls this instead of using a stock Levenshtein distance.
    assert _edit_dist("puprle", "purple", _MAX_EDITS) == 1


def test_edit_dist_length_diff_early_exit_caps_at_budget_plus_one():
    # abs(len(a) - len(b)) = 3 > max_edits = 2: the length-diff early exit
    # fires and returns max_edits + 1 immediately, without running the DP —
    # it never claims an exact distance beyond the budget, only "too far".
    assert _edit_dist("red", "redddd", 2) == 3


def test_edit_dist_row_min_early_exit_caps_at_budget_plus_one():
    # Same length, no character in common at any aligned position: every row
    # minimum climbs past max_edits=1, tripping the row_min > max_edits
    # early-return inside the DP loop (not just the length-diff pre-check).
    assert _edit_dist("purple", "zzzzzq", 1) == 2


# ---- _correct_word ----


def test_correct_word_resolves_real_color_typo():
    # The example from task.py's own comment: a dropped letter in "purple".
    assert _correct_word("puple") == CORE_VOCAB["purple"]


def test_correct_word_resolves_adjacent_transposition_typo():
    assert _correct_word("puprle") == CORE_VOCAB["purple"]


def test_correct_word_rejects_word_too_far_from_any_vocab_entry():
    # No CORE_VOCAB entry is within the (len>3) budget of 2 edits.
    assert _correct_word("gibberish") is None


def test_correct_word_short_word_uses_a_tighter_budget():
    # "xu" is 2 edits from the real short word "up" (both letters differ,
    # neither aligns) — within the general _MAX_EDITS=2 budget, but words of
    # length <= 3 are held to a budget of 1 so 2-3 letter garbage can't
    # spuriously match a short color/verb word. Without that length-scaled
    # budget this would incorrectly resolve to "up".
    assert _edit_dist("xu", "up", _MAX_EDITS) == 2
    assert _correct_word("xu") is None


# ---- end-to-end: tokenize() resolving a real typo to the right vocabulary word ----


def test_tokenize_resolves_color_typo_to_the_right_token_end_to_end():
    tokens = tokenize("puple block")
    assert tokens[0] == CORE_VOCAB["purple"]
    assert tokens[1] == CORE_VOCAB["block"]
