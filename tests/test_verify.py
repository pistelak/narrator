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
    CascadeVerifier,
    CoverageVerifier,
    NullVerifier,
    content_words,
    coverage,
    is_numberish,
)

# The real 227-character chunk that exposed the aggregate-similarity problem.
CHUNK = (
    "So Milo goes to file a permit. He copies the harbor's twenty byte code into his filing "
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
    ref = "Never share the master password with anyone."
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


# ---------------------------- isolated numerals (found by the opinion review)

def test_wrong_isolated_number_is_caught() -> None:
    """Number-blinding made the rest of this verifier work, and left the wrong
    number scoring 1.00 — in a pipeline that teaches "four checksum bytes"."""
    assert coverage("The seal is four bytes long.", "The seal is nine bytes long.")[0] == 0.0
    assert coverage("Pečeť má čtyři bajty.", "Pečeť má devět bajtů.")[0] == 0.0


def test_word_to_digit_transcription_still_passes() -> None:
    assert coverage("He copies the twenty byte code.", "He copies the 20 byte code.")[0] == 1.0


def test_compound_numerals_are_skipped_symmetrically() -> None:
    """"two fifty six" is three adjacent numerals and collapses to one "256".
    An asymmetric rule reads a correct transcription as a changed number."""
    assert coverage("It uses SHA two fifty six today.", "It uses SHA 256 today.")[0] == 1.0
    assert coverage("Použije ša dvě stě padesát šest dnes.", "Použije ša 256 dnes.")[0] == 1.0


# ------------------- word-boundary disagreement (found by the acceptance run)

def test_hyphenated_compound_is_not_a_drop() -> None:
    """Measured on real Whisper output at 0.83, on correct audio.

    The script says "coworkers"; Whisper writes "co-worker's", which normalises
    to three tokens matching none of it. Same mechanism as the Czech merge, in
    reverse — a split rather than a join — and it happens in long sentences where
    the short-sentence rescue never applied.
    """
    ref = "The coworkers reaction is not subtle here today."
    assert coverage(ref, "The co-worker's reaction is not subtle here today.")[0] == 1.0


def test_boundary_rescue_still_cannot_resurrect_a_dropped_sentence() -> None:
    """The rescue searches only UNCLAIMED transcript text, which is what keeps a
    genuine drop detectable."""
    assert coverage("Not the keeper. Not a stranger. Not any council.",
                    "Not the keeper. Not a stranger.")[0] == 0.0
    assert coverage("It burns. The sign says it burns brightly.",
                    "The sign says it burns brightly.")[0] == 0.0


# --------------- Czech orthographic folding (measured over 201 real chunks)

@pytest.mark.parametrize("script,heard", [
    ("Zkus tipovat co bude dál.",       "Zkus typovat co bude dál."),            # i/y, same sound
    ("Tak mi odpověz hned teď.",        "Tak mi odpověs hned teď."),             # final devoicing
    ("Zahashovaný soubor leží tady.",  "Zahašovaný soubor leží tady."),        # loanword digraph
    ("Adresy ztíží pozorovateli práci.", "Adresy stíží pozorovateli práci."),    # cluster assimilation
    ("Tak co to opravdu spraví teď?",   "Tak co to opravdu zpraví teď?"),        # cluster assimilation
])
def test_czech_spelling_variants_are_not_drops(script: str, heard: str) -> None:
    """Czech orthography encodes distinctions its phonology does not.

    Across 201 real Czech chunks every single rejection was of this kind, and
    Czech failed at 18.4% against English at 2.4% for this reason alone. The
    audio was correct in all of them; only the spelling differed.
    """
    assert coverage(script, heard, "cs")[0] >= 0.90


def test_assimilated_short_sentences_survive_the_negation_merge() -> None:
    """The exact shape of a real chunk three recognisers all hard-failed at 0.00:
    "Ztíží. Ne znemožní." heard as "Stíží. Neznemožní." — cluster assimilation on
    a short sentence, plus the negation-prefix merge, on correct audio."""
    ref = "Množství. Nová známka na každý dopis. Což třídění ztíží. Ne znemožní. Ztíží."
    hyp = "Množství. Nová známka na každý dopis. Což třídění stíží. Neznemožní. Stíží."
    assert coverage(ref, hyp, "cs")[0] >= 0.90


def test_case_inflection_is_not_folded() -> None:
    """A deliberate limit. "Lisa" -> "Lise" is Czech dative, a different word form
    that SOUNDS different — unlike i/y or final devoicing, which do not. Folding
    case endings would collapse genuinely distinct words, so this class of
    rejection survives and is adjudicated by hand."""
    assert coverage("Napíšeš Lisa e-mail dnes.", "Napíšeš Lise e-mail dnes.", "cs")[0] < 0.90


def test_folding_does_not_apply_to_english() -> None:
    assert coverage("The key is safe here.", "The kay is safe here.", "en")[0] < 0.90


def test_folding_cannot_hide_a_missing_word() -> None:
    """Folding makes two spellings of one word match; it cannot invent a word.

    This is why it is safe: drop detection is unaffected.
    """
    assert coverage("Ztíží to třídění ale neznemožní.", "Ztíží to ale.", "cs")[0] < 0.90
    assert coverage("Ne surový záznam ale souhrn.", "Ne surový záznam.", "cs")[0] < 0.90


# --------------- Cascade: a second opinion consulted on rejection only

@dataclass
class CountingASR(FakeASR):
    calls: int = 0

    def transcribe(self, audio: Audio, lang: str) -> str:
        self.calls += 1
        return super().transcribe(audio, lang)


def test_cascade_accepts_when_any_recogniser_confirms() -> None:
    """A recogniser never sees the script, so a transcript that matches it is
    evidence the audio is right regardless of which model produced it. This is
    the measured ~10% of rejections that were one model's misreading."""
    first = CoverageVerifier(FakeASR(ASR_DROPPED))
    second = CoverageVerifier(FakeASR(CHUNK))
    assert CascadeVerifier([first, second]).verify(SILENCE, CHUNK, "en").ok


def test_cascade_escalates_only_on_rejection() -> None:
    """The fallback's cost must be paid only when the primary rejects — that is
    what makes putting the fast model first nearly free."""
    fallback = CountingASR(CHUNK)
    cascade = CascadeVerifier([CoverageVerifier(FakeASR(CHUNK)), CoverageVerifier(fallback)])
    assert cascade.verify(SILENCE, CHUNK, "en").ok
    assert fallback.calls == 0


def test_cascade_total_failure_returns_best_coverage() -> None:
    """The retry ladder ranks attempts by coverage, so the cascade must surface
    the least-bad verdict, not the first or the last."""
    dropped = CoverageVerifier(FakeASR(ASR_DROPPED))
    silent = CoverageVerifier(FakeASR(""))
    verdict = CascadeVerifier([silent, dropped]).verify(SILENCE, CHUNK, "en")
    assert not verdict.ok
    assert verdict.coverage == pytest.approx(
        CoverageVerifier(FakeASR(ASR_DROPPED)).verify(SILENCE, CHUNK, "en").coverage
    )


class RaisingVerifier:
    def verify(self, audio: Audio, text: str, lang: str):
        raise RuntimeError("model download failed")


def test_cascade_survives_an_erroring_verifier() -> None:
    """find_spec proves the package exists, not that the model loads. A primary
    that raises mid-render must be skipped — aborting would defeat the fallback
    this class exists to provide."""
    cascade = CascadeVerifier([RaisingVerifier(), CoverageVerifier(FakeASR(CHUNK))])
    assert cascade.verify(SILENCE, CHUNK, "en").ok


def test_cascade_raises_only_when_every_verifier_errors() -> None:
    with pytest.raises(RuntimeError):
        CascadeVerifier([RaisingVerifier(), RaisingVerifier()]).verify(SILENCE, CHUNK, "en")


def test_inflected_czech_numerals_are_number_blind() -> None:
    """Czech declines its numerals and prose lives in the oblique cases.

    "dvou tisíc čtyřiceti osmi slov" is 2048 in the genitive; the ASR writes
    digits. With only citation forms in the blind list, four chunks of a real
    render failed at 0.52-0.89 on this alone — orthography, not audio.
    """
    ref = "Každá skupina vybere jedno slovo ze seznamu dvou tisíc čtyřiceti osmi anglických slov."
    hyp = "Každá skupina vybere jedno slovo ze seznamu 2048 anglických slov."
    assert coverage(ref, hyp, "cs")[0] == 1.0
    ref2 = "Shodí kontrolní součet patnáctkrát ze šestnácti pokusů tady."
    hyp2 = "Shodí kontrolní součet patnáctkrát z 16 pokusů tady."
    assert coverage(ref2, hyp2, "cs")[0] >= 0.90


def test_english_loanword_seed_folds_to_czech_transcription() -> None:
    """The ASR hears [si:d] and writes the Czech word it knows: seed -> sít,
    declined seedu -> sídu/sídů. Measured as three chunks of a real render
    failing at 0.80-0.88 on correct audio. "ee" exists in no native Czech word
    and final devoicing is general, so both folds collide with nothing."""
    assert coverage("Stejný seed, stejný strom, pokaždé.", "Stejný sít, stejný strom, pokaždé.", "cs")[0] >= 0.90
    assert coverage("Svazek klíčů vyrostlý z jednoho seedu.", "Svazek klíčů vyrostlý z jednoho sídu.", "cs")[0] >= 0.90
    assert coverage("Se zálohou seedu je ta ztráta nulová.", "Se zálohou sídů je ta stráta nulová.", "cs")[0] >= 0.90


def test_native_sh_prefix_is_not_a_loanword() -> None:
    """The sh->š rule is for loanwords (hash). Native s+h (shodí = s+hodit)
    is pronounced [sx], the ASR correctly writes it "schodí", and folding it
    to š corrupted a correct word on a real render. And the loanword itself is
    pronounced [heš]: hashování comes back as hešování."""
    ref = "Špatné slovo shodí kontrolní součet patnáctkrát ze šestnácti; slovo mimo seznam selže okamžitě."
    hyp = "Špatné slovo schodí kontrolní součet patnáctkrát ze šestnácti. Slovo mimo seznam se lže okamžitě."
    assert coverage(ref, hyp, "cs")[0] >= 0.90
    assert coverage("Pod kapotou je to samé hashování až dolů.",
                    "Pod kapotou je to samé hešování až dolů.", "cs")[0] >= 0.90
