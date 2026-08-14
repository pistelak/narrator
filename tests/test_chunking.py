"""Chunker tests.

The headline property is losslessness. A chunker that silently drops a sentence
is indistinguishable, downstream, from a TTS engine that does — and the whole
point of this library is telling those apart. So that gets a property test over
generated text, not a single example.
"""

from __future__ import annotations

import random
import re

import pytest

from narrator.chunking import MIN_WORDS, chunk, split_sentences

CZECH = (
    "Na severním pobřeží stojí maják, kolem kterého nikdy nikdo nepropluje. "
    "Ani kapitán. Ani cizinec. Není zamčený. Byl postaven na útesu, který mapy neznají — "
    "a pobřeží si ho stejně zapsalo. Navždy. Stačí jediný špatný znak v deníku."
)
ENGLISH = (
    "On the northern coast stands a lighthouse that no ship will ever pass. Not the keeper. "
    "Not a stranger. Not any council with any mandate. It isn't shut away — it was"
    "built on a headland that maps do not chart, and the coast kept it anyway, forever."
)


def words(s: str) -> list[str]:
    return re.findall(r"\S+", s)


# --------------------------------------------------------------------- lossless

@pytest.mark.parametrize("text", [ENGLISH, CZECH])
def test_chunking_is_lossless(text: str) -> None:
    assert words(" ".join(chunk(text))) == words(text)


def test_lossless_over_generated_text() -> None:
    """Random punctuation and lengths, because the real inputs are not tidy."""
    rng = random.Random(0)
    vocab = "alpha beta gamma delta epsilon zeta příliš žluťoučký kůň úpěl".split()
    for _ in range(200):
        sentences = []
        for _ in range(rng.randint(1, 12)):
            n = rng.randint(1, 40)
            body = " ".join(rng.choice(vocab) for _ in range(n))
            sentences.append(body + rng.choice([".", "!", "?", " — a pak."]))
        text = " ".join(sentences)
        assert words(" ".join(chunk(text))) == words(text), text[:120]


# ---------------------------------------------------------------------- budget

@pytest.mark.parametrize("limit", [80, 150, 250, 400])
def test_respects_budget(limit: int) -> None:
    long_text = " ".join([ENGLISH, CZECH] * 4)
    for c in chunk(long_text, max_chars=limit):
        # A single unbreakable token may exceed the budget; nothing else may.
        assert len(c) <= limit or len(c.split()) == 1


def test_oversized_sentence_is_split() -> None:
    sentence = "First, it puts a version byte on the front, " + \
               "and second it appends a seal computed from that, " * 6 + "and that is all."
    out = chunk(sentence, max_chars=120)
    assert len(out) > 1
    assert all(len(c) <= 120 for c in out)
    assert words(" ".join(out)) == words(sentence)


# ------------------------------------------------------------------ guarantees

@pytest.mark.parametrize("text", [ENGLISH, CZECH, "  ragged   spacing\n\n here. And more.  "])
def test_no_empty_or_whitespace_leading_chunks(text: str) -> None:
    """Leading whitespace caused first-word crackling and hallucination upstream."""
    for c in chunk(text):
        assert c
        assert c == c.strip()


def test_orphans_are_merged_backwards() -> None:
    out = chunk("A long enough opening sentence to stand on its own here. Yes.")
    assert all(len(c.split()) >= MIN_WORDS for c in out)


def test_orphan_not_merged_when_it_would_break_budget() -> None:
    text = "x" * 240 + ". Yes."
    out = chunk(text, max_chars=250)
    assert words(" ".join(out)) == words(text)


def test_empty_input() -> None:
    assert chunk("") == []
    assert chunk("   \n  ") == []


def test_single_short_input_survives() -> None:
    assert chunk("Just this.") == ["Just this."]


# ------------------------------------------------------------ split point choice

def test_prefers_near_comma_over_distant_full_stop() -> None:
    """Distance decay: a comma at the target beats a full stop far behind it.

    A plain priority ladder always takes the full stop and produces a chunk far
    under budget, wasting the very headroom that makes long-form reliable.
    """
    text = "Short one. " + "w " * 40 + ", and then the clause continues past the limit here."
    out = chunk(text, max_chars=110)
    assert words(" ".join(out)) == words(text)
    assert len(out[0]) > 40, "first chunk collapsed onto the distant full stop"


def test_sentence_splitter_handles_both_languages() -> None:
    assert len(split_sentences(ENGLISH)) == 5
    assert len(split_sentences(CZECH)) == 7


def test_unbreakable_token_is_passed_through_not_dropped() -> None:
    text = "x" * 400
    out = chunk(text, max_chars=100)
    assert "".join(out) == text
