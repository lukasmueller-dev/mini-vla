"""Grammar/vocab staleness detector (see review notes).

assets/grammar.json is the single source of truth for the words the model's
language side needs to know synchronously (color synonyms, verbs, articles,
nouns, fillers) — mini_vla/task.py builds COLORS/verbs/etc. straight from it.
mini_vla/vocab_gen.py's CORE_VOCAB dict (word -> GloVe token id) and
assets/vocab.txt (the generated full vocab list) are both produced by
`npm run gen:embeddings` from that same grammar file.

If someone adds a word to grammar.json and forgets to regenerate, two things
can go stale independently, and neither currently has a test:

1. CORE_VOCAB may simply be missing the new grammar word (regeneration never
   ran), so task.py's word-dropout/label-token bookkeeping and the
   typo-corrector's candidate set silently don't know about it.
2. Even for words CORE_VOCAB does contain, its token id is only correct if it
   still points at the matching line in assets/vocab.txt (id - 2 == line
   index) — a partial/mismatched regeneration (e.g. vocab.txt regenerated
   without vocab_gen.py, or vice versa) would desync the two while each looks
   individually fine, silently seeding the frozen embedding row for that word
   with the wrong vector.

Both are asserted here so a future drift fails loudly, naming the offending
word, rather than shipping a demo where a grammar word silently resolves to
<unk> or to another word's embedding.
"""

import json
from pathlib import Path

from mini_vla.vocab_gen import CORE_VOCAB, VOCAB_SIZE

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _grammar_words() -> set[str]:
    grammar = json.loads((_ASSETS / "grammar.json").read_text())
    words: set[str] = set()
    for verb in grammar["tasks"]["lift"]["verbs"]:
        words.update(verb)
    words.update(grammar["articles"])
    words.update(grammar["nouns"])
    words.update(grammar["fillers"])
    for color in grammar["colors"]:
        words.update(color["synonyms"])
    return words


def test_every_grammar_word_is_registered_in_core_vocab():
    """Catches: a word added to assets/grammar.json without regenerating
    CORE_VOCAB (mini_vla/vocab_gen.py) via `npm run gen:embeddings` — the
    exact staleness scenario the review flagged as uncovered."""
    missing = sorted(w for w in _grammar_words() if w not in CORE_VOCAB)
    assert not missing, (
        "assets/grammar.json references word(s) missing from "
        f"mini_vla.vocab_gen.CORE_VOCAB: {missing!r} — regenerate the "
        "embeddings vocab with `npm run gen:embeddings` (mini_vla/vocab_gen.py "
        "and assets/vocab.txt must be regenerated together with grammar.json)."
    )


def test_core_vocab_token_ids_match_assets_vocab_txt():
    """Catches: CORE_VOCAB and assets/vocab.txt drifting out of sync with each
    other (one regenerated, the other stale) even when neither is missing a
    grammar word individually — token id `tid` must line up with
    vocab.txt's line `tid - 2` naming the same word."""
    words = [w for w in (_ASSETS / "vocab.txt").read_text().split("\n") if w]
    assert len(words) == VOCAB_SIZE - 2, (
        f"assets/vocab.txt has {len(words)} words, expected {VOCAB_SIZE - 2} "
        "(VOCAB_SIZE less <pad>/<unk>) — regenerate with `npm run gen:embeddings`."
    )

    mismatches = []
    for word, tid in CORE_VOCAB.items():
        idx = tid - 2
        actual = words[idx] if 0 <= idx < len(words) else "<out of range>"
        if actual != word:
            mismatches.append((word, tid, actual))

    assert not mismatches, (
        "CORE_VOCAB token id(s) no longer match assets/vocab.txt — "
        "(grammar_word, expected_token_id, word_actually_at_that_id): "
        f"{mismatches!r} — regenerate with `npm run gen:embeddings`."
    )
