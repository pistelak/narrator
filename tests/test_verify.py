"""Verification tests.

Every case here is a real failure observed in the predecessor pipeline, with the
measured numbers preserved so a regression is recognisable rather than merely red.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pytest

from narrator.types import Audio
from narrator.verify import (
    CoverageVerifier,
    NullVerifier,
    content_words,
    coverage,
    is_numberish,
)

# The real 227-character chunk that exposed the aggregate-similarity problem.
CHUNK = (
    "So Milo goes to file a permit. He copies the harbor's twenty byte code into his filing"
    "line by hand and gets the last character wrong. A D where a C should be. "
    "The seal is valid. He really did file that entry typo included."
)
ASR_NUMERALS = CHUNK.replace("twenty byte", "20 byte")
ASR_DROPPED = CHUNK.replace("A D where a C should be. ", "")


@dataclass
class FakeASR:
    text: str

    def transcribe(self, audio: Audio, lang: str) -> str:
        return self.text


SILENCE: Audio = np.zeros(1000, dtype=np.float32)


# ------------------------------------------------------- trap 1: autojunk

def test_autojunk_would_have_broken_this() -> None:
    """Documents the trap rather than just avoiding it.

    difflib's default junk heuristic scored 98.2%-identical text at 0.231.
    If this ever stops being true, the comment in verify.py is stale.
    """
    a, b = CHUNK.lower(), ASR_NUMERALS.lower()
    with_junk = difflib.SequenceMatcher(None, a, b).ratio()
    without = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    assert with_junk < 0.35, "autojunk no longer catastrophic — recheck verify.py"
    assert without > 0.95
    assert without - with_junk > 0.5


# ------------------------------- trap 2: aggregate cannot separate the cases

def test_aggregate_similarity_cannot_separate_correct_from_dropped() -> None:
    """Why coverage is per-sentence. These two must be told apart."""
    correct = difflib.SequenceMatcher(None, CHUNK.lower(), ASR_NUMERALS.lower(), autojunk=False).ratio()
    dropped = difflib.SequenceMatcher(None, CHUNK.lower(), ASR_DROPPED.lower(), autojunk=False).ratio()
    assert abs(correct - dropped) < 0.06, "aggregate ratios are ~4 points apart"

    correct_cov, _ = coverage(CHUNK, ASR_NUMERALS)
    dropped_cov, sentence = coverage(CHUNK, ASR_DROPPED)
    assert correct_cov > 0.9
    assert dropped_cov == 0.0
    assert "A D where a C should be." in sentence


# ---------------------------------------------------- trap 3: number-blind

@pytest.mark.parametrize("word,expected", [
    ("twenty", True), ("dvacet", True), ("256", True), ("20-byte", True),
    ("padesát", True), ("lantern", False), ("hash", False), ("paluba", False),
])
def test_is_numberish(word: str, expected: bool) -> None:
    assert is_numberish(word.lower()) is expected


def test_spelled_numerals_do_not_count_as_drops() -> None:
    score, _ = coverage(CHUNK, ASR_NUMERALS)
    assert score > 0.9


def test_two_word_sentence_containing_a_numeral_survives() -> None:
    """'Episode one.' scored 0.5 and failed before numbers were excluded."""
    ref = "This is the field guide, chapter three. Episode one."
    hyp = "This is the field guide, chapter 3. Episode 1."
    score, _ = coverage(ref, hyp)
    assert score == 1.0


# ------------------------- short sentences and word-boundary disagreement

def test_czech_word_boundary_merge_is_not_a_drop() -> None:
    """The open false positive in the predecessor: 'Ne znemožní.' -> 'Neznemožní'."""
    ref = "Nová známka na každý dopis. Což třídění ztíží. Ne znemožní."
    hyp = "Nová známka na každý dopis. Což třídění ztíží. Neznemožní."
    score, _ = coverage(ref, hyp)
    assert score == 1.0


def test_short_sentence_genuinely_missing_is_still_caught() -> None:
    """The boundary leniency must not swallow a real drop."""
    ref = "Nová známka na každý dopis. Což třídění ztíží. Ne znemožní."
    hyp = "Nová známka na každý dopis. Což třídění ztíží."
    score, sentence = coverage(ref, hyp)
    assert score == 0.0
    assert "Ne znemožní." in sentence


# ------------------------------------------------------------- other cases

def test_repetition_loop_is_caught() -> None:
    """The observed opening failure: 'Not the keeper' four times."""
    ref = "Not the keeper. Not a stranger. Not any council with any mandate."
    hyp = "Not the keeper. Not the keeper. Not the keeper. Not the keeper."
    score, sentence = coverage(ref, hyp)
    assert score < 0.6
    assert "government" in sentence


def test_empty_transcript_is_total_failure() -> None:
    score, _ = coverage(CHUNK, "")
    assert score == 0.0


def test_empty_reference_is_vacuously_fine() -> None:
    assert coverage("", "anything") == (1.0, "")


def test_content_words_strips_punctuation_and_numerals() -> None:
    """Every numeral goes, which is what lets "SHA two fifty six" match "SHA 256"
    — both sides reduce to the same non-numeric residue."""
    assert content_words("SHA two fifty-six, then CRC thirty-two.") == [
        "sha", "then", "crc",
    ]


# ------------------------------------------------------------- verifiers

def test_coverage_verifier_passes_and_fails() -> None:
    assert CoverageVerifier(FakeASR(ASR_NUMERALS)).verify(SILENCE, CHUNK, "en").ok
    verdict = CoverageVerifier(FakeASR(ASR_DROPPED)).verify(SILENCE, CHUNK, "en")
    assert not verdict.ok
    assert verdict.dropped_sentence
    assert verdict.transcript == ASR_DROPPED


def test_null_verifier_accepts_anything() -> None:
    assert NullVerifier().verify(SILENCE, CHUNK, "en").ok
