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
    ("twenty", True), ("dvacet", True), ("256", True), ("padesát", True),
    ("lantern", False), ("hash", False), ("paluba", False),
    # Alphanumeric domain terms are CONTENT, not numerals. Treating "contains a
    # digit" as numeric deleted these from both sides, leaving the verifier blind
    # to whether the audio said them at all.
    ("utf8", False), ("iso9001", False), ("rfc822", False), ("base64", False),
])
def test_is_numberish(word: str, expected: bool) -> None:
    """Operates on normalized tokens: `normalize` strips punctuation first, so
    "20-byte" arrives as the two tokens "20" and "byte", never as one."""
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
    assert score < 0.9
    assert sentence, "must name what went wrong"


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


# ------------------------------- false passes found by adversarial self-review

def test_dropped_sentence_is_caught_even_when_it_appears_elsewhere() -> None:
    """Containment alone is not presence.

    The short-sentence leniency asked "does this text appear in the transcript",
    which an identical sentence elsewhere in the chunk answers yes to while this
    one is genuinely gone. Occurrences are counted now.
    """
    ref = "Not the keeper. It burns. A stranger cannot douse it. It burns."
    hyp = "Not the keeper. It burns. A stranger cannot douse it."
    score, sentence = coverage(ref, hyp)
    assert score == 0.0
    assert "It burns." in sentence


def test_one_of_several_identical_sentences_dropped_is_caught() -> None:
    assert coverage("Again. Again. Again.", "Again. Again.")[0] == 0.0


def test_repeated_sentences_all_present_still_pass() -> None:
    """The fix must not make legitimate repetition fail."""
    assert coverage("Again. Again.", "Again. Again.")[0] == 1.0


def test_czech_boundary_leniency_survives_the_fix() -> None:
    """The case the leniency exists for must still work."""
    ref = "Což třídění ztíží. Ne znemožní."
    hyp = "Což třídění ztíží. Neznemožní."
    assert coverage(ref, hyp)[0] == 1.0


def test_alphanumeric_terms_are_verified_not_discarded() -> None:
    """utf8 / iso9001 / base64 are content words in this domain."""
    ref = "The file is utf8 encoded. It uses iso9001 forms."
    assert coverage(ref, "The file is utf8 encoded. It uses iso9001 forms.")[0] == 1.0
    assert coverage(ref, "The file is encoded. It uses forms.")[0] < 0.9


def test_all_numeral_sentence_fails_closed() -> None:
    """Unverifiable is not the same as verified.

    "Two fifty six." has no content words after number-blinding, so a text
    round-trip genuinely cannot check it: the script spells the numeral out
    because digits are unspeakable and the ASR writes it back as "256". An
    earlier version recorded these in a list and then never read it, so a dropped
    all-numeral sentence scored a clean 1.0. Documenting a blind spot is not the
    same as refusing to certify what you cannot see.
    """
    ref = "The seal is four bytes. Two fifty six. That catches the typo."
    hyp = "The seal is four bytes. That catches the typo."
    score, sentence = coverage(ref, hyp)
    assert score == 0.0
    assert "unverifiable" in sentence


# ------------------------------ insertion blindness (found by external review)

def test_inserted_negation_is_caught() -> None:
    """The worst corruption possible in teaching material, and it scored 1.00.

    Coverage measured recall only — it marked which REFERENCE words appeared in
    the transcript, so anything the engine ADDED was invisible. Precision over
    the hypothesis is what catches it.
    """
    assert coverage("The key is safe.", "The key is not safe.")[0] < 0.9


def test_hallucinated_extra_sentence_is_caught() -> None:
    ref = "Alpha beta gamma."
    assert coverage(ref, "Alpha beta gamma. And then some invented extra sentence.")[0] < 0.9


def test_partial_short_sentence_no_longer_scores_perfect() -> None:
    """`or score > 0` turned ANY partial coverage into a pass."""
    assert coverage("Alpha beta.", "Alpha.")[0] < 0.9


def test_containment_does_not_match_across_word_boundaries() -> None:
    """Squashed containment was an unanchored substring search: "go now" was
    found inside "under-go now-here" and the dropped sentence passed."""
    ref = "Stop here. Go now."
    assert coverage(ref, "Stop here. We undergo nowhere near that place.")[0] < 0.9


def test_dropped_negation_fails_outright() -> None:
    """A fuzzy threshold cannot protect meaning-critical words.

    At 0.60 this scored 0.857 and passed. Raising the threshold to 0.90 helped,
    but "The keeper can open" -> "cannot open" still scored 0.9167 in a
    longer sentence: meaning is not proportional to word count. Negation and
    modality tokens are now required to match exactly.
    """
    ref = "Never share the master password to anyone."
    score, sentence = coverage(ref, "Share the master password with anyone.")
    assert score == 0.0
    assert "meaning-critical" in sentence


def test_inserted_negation_in_a_long_sentence_fails() -> None:
    """Scored 0.9167 — above the tightened threshold — before token protection."""
    ref = "The visitor can open these doors after presenting the matching front key."
    assert coverage(ref, ref.replace("can open", "cannot open"))[0] == 0.0


def test_closing_quote_does_not_merge_sentences() -> None:
    """Boundaries only fired on `[.!?]` + whitespace, so a closing quote merged
    two sentences and restored the aggregate scoring this replaces."""
    ref = 'He shouted "do not enter!" Then everyone left the room.'
    assert coverage(ref, "He shouted Then everyone left the room.")[0] < 0.9
