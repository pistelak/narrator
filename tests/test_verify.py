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
    coverage_detail,
    is_numberish,
    normalize,
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


def test_contracted_negation_dropped_is_caught() -> None:
    """The critical-token protection did not fire through a contraction.

    normalize spaced the apostrophe, so "can't" became the tokens "can t" and
    matched no critical token: "can't open" rendered as "can open" scored 0.923
    and PASSED the 0.90 gate — the exact corruption the hard-fail rule exists
    for, already pinned below in its uncontracted form. Contractions now expand
    to their two-word forms, so "not" carries the protection.
    """
    ref = "The visitor can't open these doors after presenting the matching front key."
    score, sentence = coverage(ref, ref.replace("can't", "can"))
    assert score == 0.0
    assert "meaning-critical" in sentence
    assert coverage("Don't share the master password.",
                    "Share the master password.")[0] == 0.0


def test_contraction_and_expansion_are_the_same_words() -> None:
    """The same bug's other face: "don't" against an ASR that writes "do not"
    hard-failed CORRECT audio at 0.0 with [meaning-critical token changed: not].
    Both sides expand, so contracting is a spelling choice, not a content change."""
    assert coverage("Don't share the master password.",
                    "Do not share the master password.")[0] == 1.0
    assert coverage("You cannot enter the vault today.",
                    "You can't enter the vault today.")[0] == 1.0
    assert coverage("It won't open without the key.",
                    "It will not open without the key.")[0] == 1.0
    # The full n't family, not just the common few: "mightn't" was missing from
    # the first expansion table, and its dropped negation passed at 0.9167 —
    # the original bug reproduced one auxiliary over. Each root is pinned:
    # removing any single one from the regex must turn a test red.
    ref = "The visitor mightn't open these doors after presenting the matching front key."
    assert coverage(ref, ref.replace("mightn't", "might not"))[0] == 1.0
    assert coverage(ref, ref.replace("mightn't", "might"))[0] == 0.0
    for contraction, expansion in [
        ("amn't", "am not"), ("aren't", "are not"), ("couldn't", "could not"),
        ("daren't", "dare not"), ("didn't", "did not"), ("doesn't", "does not"),
        ("don't", "do not"), ("hadn't", "had not"), ("hasn't", "has not"),
        ("haven't", "have not"), ("isn't", "is not"), ("mayn't", "may not"),
        ("mightn't", "might not"), ("mustn't", "must not"),
        ("needn't", "need not"), ("oughtn't", "ought not"),
        ("shan't", "shall not"), ("shouldn't", "should not"),
        ("usedn't", "used not"), ("wasn't", "was not"), ("weren't", "were not"),
        ("won't", "will not"), ("wouldn't", "would not"), ("can't", "cannot"),
    ]:
        assert normalize(contraction) == expansion, contraction
        # The apostrophe-less spelling must NOT expand — a bare-token table was
        # tried and opened three false-accept paths (acronyms lowercasing into
        # it, auxiliary identity erased, wont/cant unprotectable). Bare forms
        # stay themselves and are critical tokens instead.
        bare = contraction.replace("'", "")
        assert normalize(bare) == bare, contraction


# ------------- contraction expansion overreach (found by independent review)

def test_apostrophe_is_required_to_expand_wont_and_cant() -> None:
    """Bare "wont" (habit) and "cant" (jargon) are genuine English words. An
    unconditional table read "he was wont to visit" as "was will not to visit",
    and the fabricated negation scored 0.923 — a false accept manufactured by
    the fix itself. Unexpanded, the transcript's inserted "not" hard-fails."""
    assert coverage("He was wont to visit the old harbor every winter before sunrise.",
                    "He was not to visit the old harbor every winter before sunrise.")[0] == 0.0
    assert normalize("wont") == "wont"
    assert normalize("cant") == "cant"


def test_every_apostrophe_codepoint_expands() -> None:
    """The expansion regexes matched only the ASCII and U+2019 apostrophes, so
    the left quotation mark (U+2018) and modifier letter apostrophe (U+02BC) —
    codepoints editors and ASRs genuinely emit, which NFC does not unify —
    slipped past, and the dropped negation passed again at 0.92: the original
    hole reopened through a side door."""
    ref = "The visitor can{0}t open these doors after presenting the matching front key."
    hyp = "The visitor can open these doors after presenting the matching front key."
    for apostrophe in ("'", "’", "‘", "ʼ", "＇"):
        assert coverage(ref.format(apostrophe), hyp)[0] == 0.0, repr(apostrophe)
        assert coverage(ref.format(apostrophe), ref.format("'"))[0] == 1.0, repr(apostrophe)


def test_aint_collapses_to_one_protected_token() -> None:
    """No expansion of ain't is faithful — its auxiliary depends on the subject,
    and mapping it to bare "not" accepted "I not ready" at 1.0. Collapsed to the
    single critical token "aint" instead: a dropped ain't hard-fails (it passed
    at 0.92 unprotected), and ain't against an ASR's "am not" keeps rejecting —
    the fail-closed side, adjudicated by hand like Czech case inflection.
    The bare spelling "aint" lands on the same token; tokenized-apart variants
    ("ain t", "ain 't") are deliberately NOT folded — see the strict-separator
    rationale at the expansion tables — and fail closed."""
    ref = "The keeper ain't ready to open these doors for any visitor before sunrise."
    assert coverage(ref, ref.replace("ain't ", ""))[0] == 0.0
    assert coverage(ref, ref)[0] == 1.0
    assert coverage(ref, ref.replace("ain't", "aint"))[0] == 1.0
    assert coverage(ref, ref.replace("ain't", "am not"))[0] == 0.0  # fail closed


def test_tokenized_apart_contractions_fail_closed() -> None:
    """A deliberate limit, pinned as one. Widening the separator to accept
    "don 't", "don' t" or "don ' t" was tried in three shapes, and every shape
    rewrote real English, because a loosened contraction is the same string as
    quotation: "Don T Harris", the quoted letter in "Can 'T'", and the closing
    quote in "After 'can' T. S. Eliot" all welded into negations. No
    recogniser emits the tokenized-apart spellings, so they stay unexpanded
    and hard-fail against their expanded counterparts — fail closed, like the
    bare misspellings."""
    ref = "The visitors don't open these doors after presenting the matching front key."
    for variant in ("don 't", "don' t", "don ' t"):
        assert coverage(ref, ref.replace("don't", variant))[0] == 0.0, repr(variant)
    ain = "The keeper ain't ready to open these doors for any visitor before sunrise."
    assert coverage(ain, ain.replace("ain't", "ain t"))[0] == 0.0


def test_a_standalone_letter_t_is_not_a_contraction() -> None:
    """Loosened separators were tried and rewrote real English: whitespace-only
    welded "Don T Harris" and "can T-test" into negations — a name and a
    statistics term becoming "will not"/"cannot" in both directions — and the
    apostrophe-tolerant forms welded the quoted letter in "Can 'T' represent".
    The pattern is strict "n't": a space on either side of the apostrophe
    breaks the match, and nothing after the t is guarded."""
    ref = "Don T Harris will inspect the harbor gates with the deputy before sunrise today."
    assert coverage(ref, ref)[0] == 1.0
    assert coverage(ref, ref.replace(" T ", " tee "))[0] >= 0.90
    ref2 = "You can T-test the samples in the laboratory before the sunrise shift today."
    assert coverage(ref2, ref2)[0] == 1.0
    assert coverage(ref2, ref2.replace("can T-test", "cannot test"))[0] == 0.0
    ref3 = "Can 'T' represent temperature in the harbor logs we keep today?"
    assert coverage(ref3, ref3)[0] == 1.0
    assert coverage(ref3, ref3.replace("Can 'T' represent", "Cannot represent"))[0] == 0.0
    # The quotes spaced apart are the same quoted letter: "Can ' T '" welded
    # into "cannot" under a looser separator — wrong audio accepted in one
    # direction, an identical pair hard-failed in the other.
    ref4 = "Can ' T ' represent temperature in the harbor logs we keep today?"
    assert coverage(ref4, ref4)[0] == 1.0
    assert coverage(ref4, ref4.replace("Can ' T ' represent", "Cannot represent"))[0] == 0.0
    # A contraction followed by quotation is ordinary text: trailing-apostrophe
    # guards broke both of these while chasing the quoted-letter cases.
    assert normalize("don't 'cause") == "do not cause"
    quoted = "The guide emphasized the word 'Don't' during the harbor safety briefing today."
    assert coverage(quoted, quoted.replace("'Don't'", "Don't"))[0] == 1.0
    # And apostrophe-then-space is a CLOSING quote, not a contraction:
    # "After 'can' T. S. Eliot" welded into "cannot" under the loosened
    # separator, erasing the very distinction the critical list protects.
    eliot = "After 'can' T. S. Eliot inserts a deliberate pause for emphasis in this reading."
    assert coverage(eliot, eliot)[0] == 1.0
    assert coverage(eliot, eliot.replace("'can' T.", "'cannot'"))[0] == 0.0


def test_sound_alikes_override_the_critical_veto() -> None:
    """A sound_alikes pair is the caller declaring two spellings one sound —
    authoritative vocabulary per the lexicon contract. The critical hard-fail
    compared raw tokens and vetoed it: the AINT acronym transcribed "aynt"
    under a caller lexicon hard-failed as a changed negation. Tokens named in
    a pair are CANONICALIZED into the critical form, never exempted — an
    exemption excused omission too: with a ("knot", "not") pair, dropping the
    grammatical "not" itself passed at 0.93. Substitution is invisible; a drop
    or insertion still counts. Without the pair, the change still hard-fails."""
    ref = "The AINT algorithm tracks storms across the continent today."
    hyp = "The aynt algorithm tracks storms across the continent today."
    assert coverage(ref, hyp, sound_alikes=(("AINT", "aynt"),))[0] == 1.0
    assert coverage(ref, hyp)[0] == 0.0
    knot = "Tie this knot securely but do not open the sealed harbor door before sunrise."
    assert coverage(knot, knot.replace("do not open", "do open"),
                    sound_alikes=(("knot", "not"),))[0] == 0.0
    assert coverage(knot, knot, sound_alikes=(("knot", "not"),))[0] == 1.0


def test_sound_alike_chains_and_protected_pairs() -> None:
    """Alignment composes pairs — ("AINT","aynt") plus ("aynt","eint") verifies
    "eint" — so the critical canonicalization must group them into classes, or
    it hard-fails the chained spelling alignment already accepted. And a pair
    naming TWO protected tokens refuses to map: honoring ("cannot","cant")
    would erase a distinction the critical list exists to keep."""
    ref = "The AINT algorithm tracks storms across the continent today."
    chain = (("AINT", "aynt"), ("aynt", "eint"))
    assert coverage(ref, ref.replace("AINT", "eint"), sound_alikes=chain)[0] == 1.0
    assert coverage(ref, ref.replace("AINT ", ""), sound_alikes=chain)[0] == 0.0
    both = "The visitor cannot open these doors after presenting the matching front key."
    assert coverage(both, both.replace("cannot", "cant"),
                    sound_alikes=(("cannot", "cant"),))[0] == 0.0


def test_bare_contraction_spellings_stay_meaning_critical() -> None:
    """The regressions a bare-token expansion table bought, each measured against
    the pre-change behavior it broke:

    - "won't" vs a transcript's bare "wouldnt" normalized to will not /
      would not, agreed on the critical "not", and a changed auxiliary passed
      at 0.923 where it had hard-failed as a changed token.
    - deleting "wont" (habit) or "cant" (jargon) from the audio passed at
      0.91-0.92 once the words left the critical list.

    Bare spellings expand nowhere and hard-fail as critical tokens instead."""
    ref = "The visitor won't open these doors after presenting the matching front key."
    assert coverage(ref, ref.replace("won't", "wouldnt"))[0] == 0.0
    assert coverage("He was wont to visit the old harbor every winter before sunrise.",
                    "He was to visit the old harbor every winter before sunrise.")[0] == 0.0
    assert coverage("The engineer discussed cant in the old harbor before sunrise today.",
                    "The engineer discussed in the old harbor before sunrise today.")[0] == 0.0


def test_contraction_tables_are_english_language_data() -> None:
    """The tables are English orthography, gated on lang exactly as fold() gates
    Czech phonology. Applied language-blind, the uppercase acronym ISNT in a
    Czech sentence expanded to "is not" and certified audio that never spelled
    the letters — and the same script/transcript pair must still verify when
    both sides carry the acronym."""
    ref = "Oční lékař použil pravidlo ISNT při vyšetření okraje zrakového nervu."
    assert coverage(ref, ref.replace("ISNT", "is not"), "cs")[0] < 0.90
    assert coverage(ref, ref, "cs")[0] == 1.0


def test_ain_t_fold_does_not_weld_sentences() -> None:
    """The "ain t" -> "aint" fold first ran AFTER punctuation stripping, where
    "the Ain. T cells" had also become "ain t" — the fold welded two sentences
    together, per-sentence word counts stopped matching the full-chunk tokens,
    and an IDENTICAL ref/hyp pair scored 0.857. The fold now runs while the
    terminator still stands. Ain is a real river; ref == hyp must be 1.0."""
    ref = "Our route follows the Ain. T cells protect the body during infection."
    assert coverage(ref, ref)[0] == 1.0


def test_rare_bare_lookalikes_are_not_critical_tokens() -> None:
    """The critical list briefly held every bare n't spelling, and the rare ones
    vetoed real vocabulary: "Shant" (an Armenian given name) hard-failed even
    through the caller's own sound_alikes pair, and river "Darent" against the
    ASR's "Darenth" went from a rescued pass to 0.0. The list holds only the
    common bare forms. A rare bare form against its expanded transcript still
    fails, hard, on the asymmetric "not" the expansion introduces — fail-closed
    for a script that misspelled its own contraction, and pinned as such."""
    ref = "Shant will inspect these unusual harbor doors before sunrise today."
    hyp = "Shahnt will inspect these unusual harbor doors before sunrise today."
    assert coverage(ref, hyp, sound_alikes=(("Shant", "Shahnt"),))[0] == 1.0
    assert coverage("The river Darent flows quietly past the village mill before sunrise.",
                    "The river Darenth flows quietly past the village mill before sunrise.")[0] >= 0.90
    assert coverage("They havent finished the harbor inspection before sunrise today.",
                    "They haven't finished the harbor inspection before sunrise today.")[0] == 0.0


def test_uppercase_acronym_is_never_a_contraction() -> None:
    """English too, not only Czech: lowercasing fed the acronym ISNT into the
    bare-token expansion table and "the ISNT rule" scored a perfect 1.0 against
    audio that said "is not" instead of spelling the letters. Bare spellings no
    longer expand in any language, so the acronym round-trips as itself and the
    wrong audio hard-fails."""
    ref = "The clinician applied the ISNT rule while examining the optic nerve rim today."
    assert coverage(ref, ref)[0] == 1.0
    assert coverage(ref, ref.replace("ISNT", "is not"))[0] == 0.0


def test_affirmative_contractions_are_not_collapsed_into_other_words() -> None:
    """Deleting apostrophes wholesale turned "we'll" into "well" — a changed
    word scoring a perfect 1.0 — and "he'll" into "hell". Only the n't negation
    class expands; every other apostrophe spaces out as punctuation always did."""
    assert coverage("We'll wait.", "Well, wait.")[0] < 0.90
    assert coverage("He'll return to the harbor before the tide turns tonight.",
                    "Hell return to the harbor before the tide turns tonight.")[0] < 0.90


def test_possessive_numeral_still_reaches_the_numeral_check() -> None:
    """Apostrophe deletion collapsed "two's" into the non-numberish "twos", so
    two's complement rendered as ten's complement passed at 0.909 — the
    isolated-numeral hard-fail saw [] == []. Spaced, "two" is isolated again."""
    assert coverage("The device stores signed integers using two's complement representation in memory.",
                    "The device stores signed integers using ten's complement representation in memory.")[0] == 0.0


def test_cannot_is_not_the_same_negation_as_will_not() -> None:
    """Expanding can't/cannot to "can not" left one "not" looking like any
    other: "cannot open" against "will not open" agreed on critical counts and
    passed at 0.923, turning inability into refusal. "can't" therefore expands
    to the single protected token "cannot", never to "can not"."""
    ref = "The visitor cannot open these doors after presenting the matching front key."
    assert coverage(ref, ref.replace("cannot", "will not"))[0] == 0.0


def test_yall_expanding_to_you_all_is_not_a_changed_critical_token() -> None:
    """"y'all" -> "yall" under apostrophe deletion, while the ASR's "you all"
    carries the protected token "all" — correct audio hard-failed at 0.0 with
    [meaning-critical token changed: all]. Spaced, both sides count one "all"."""
    assert coverage("Y'all can enter the vault after presenting the correct master key today.",
                    "You all can enter the vault after presenting the correct master key today.")[0] >= 0.90


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
    to š corrupted a correct word on a real render."""
    ref = "Špatné slovo shodí kontrolní součet patnáctkrát ze šestnácti; slovo mimo seznam selže okamžitě."
    hyp = "Špatné slovo schodí kontrolní součet patnáctkrát ze šestnácti. Slovo mimo seznam se lže okamžitě."
    assert coverage(ref, hyp, "cs")[0] >= 0.90


def test_sound_alikes_are_vocabulary_the_caller_supplies() -> None:
    """Phonology lives in fold(); a specific word's pronunciation belongs to
    the caller. A pronunciation-lexicon pair is a sound-alike by construction,
    and applied in fold space it covers inflections of the written form."""
    ref = "Pod kapotou je to samé hashování až dolů."
    hyp = "Pod kapotou je to samé hešování až dolů."
    assert coverage(ref, hyp, "cs")[0] < 0.90, "no hardcoded project vocabulary"
    assert coverage(ref, hyp, "cs", sound_alikes=(("hashování", "hešování"),))[0] >= 0.90


def test_pronunciation_pair_verifies_as_sound_alike() -> None:
    """The lexicon makes the engine say the spoken form; the ASR writes what
    it hears; the script holds the written form. The same pair that drives
    synthesis therefore closes the verification loop, with no per-word rules
    in the library."""
    ref = "Jde do hashe zvaného HMAC SHA pět set dvanáct a ven vypadne pět set dvanáct bitů."
    hyp = "Jde do haše zvaného HMAC šá 512 a ven vypadne 512 bytů."
    alikes = (("SHA", "ša"), ("hashe", "haše"))
    assert coverage(ref, hyp, "cs", sound_alikes=alikes)[0] >= 0.90
    assert CoverageVerifier(FakeASR(hyp), sound_alikes=alikes).verify(SILENCE, ref, "cs").ok


def test_numeral_welded_into_a_neighbour_is_not_a_changed_number() -> None:
    """"dva z" comes back as "dvaze": the value vanishes from the transcript's
    numeral list while its letters sit in plain sight inside the welded token.
    Failed 3 of 5 real renders of one chunk, on correct audio each time."""
    ref = "Libovolné dva z: dvě věci k ochraně místo jedné; hesla se zapomínají."
    hyp = "Libovolné dvaze: Dvě věci k ochraně místo jedné. Hesla se zapomínají."
    assert coverage(ref, hyp, "cs")[0] > 0.0, "must not hard-fail as a changed number"


def test_merge_rescue_does_not_excuse_a_genuinely_changed_number() -> None:
    """The rescue needs the numeral's letters inside a welded token; a value
    that actually changed, or appeared from nowhere, still fails outright."""
    assert coverage("It has four bytes here.", "It has nine bytes here.")[0] == 0.0
    assert coverage("Vezmi dva klíče domů.", "Vezmi klíče domů.", "cs")[0] == 0.0
    assert coverage("Vezmi klíče domů.", "Vezmi dva klíče domů.", "cs")[0] == 0.0


# ------------------------------------------------- word-level diagnostics

def test_diagnostics_report_a_dropped_sentence_as_a_run_of_d_codes() -> None:
    """The typed codes exist because the SHAPE is the signal: a contiguous run
    of d: is a drop or truncation, which no scalar score can distinguish from
    scattered spelling noise. Pinned to the same real chunk as the aggregate-
    similarity trap above."""
    detail = coverage_detail(CHUNK, ASR_DROPPED)
    assert detail.word_diagnostics == (
        "d:a", "d:d", "d:where", "d:a", "d:c", "d:should", "d:be",
    )


def test_hard_fail_diagnostics_name_the_dropped_negation() -> None:
    """The codes ride along on hard fails too: the bracketed reason says a
    critical token changed, the codes say which word and in which direction."""
    detail = coverage_detail(
        "Never share the master password with anyone.",
        "Share the master password with anyone.",
    )
    assert detail.score == 0.0
    assert "meaning-critical" in detail.worst_sentence
    assert detail.word_diagnostics == ("d:never",)


def test_inserted_negation_diagnostics_point_the_other_way() -> None:
    detail = coverage_detail("The key is safe.", "The key is not safe.")
    assert detail.score == 0.0
    assert detail.word_diagnostics == ("i:not",)


def test_boundary_rescued_word_is_not_reported_missing() -> None:
    """A rescue means the audio was right; a diagnostic that contradicts the
    score is a false alarm. The co-worker's split rescued at 1.0 above must
    produce neither d:coworkers nor i:co / i:worker / i:s."""
    detail = coverage_detail(
        "The coworkers reaction is not subtle here today.",
        "The co-worker's reaction is not subtle here today.",
    )
    assert detail.score == 1.0
    assert detail.word_diagnostics == ()


def test_short_sentence_rescue_suppresses_diagnostics() -> None:
    """Same rule for the other rescue: 'Ne znemožní.' returned merged as
    'Neznemožní.' is correct audio, so it must not surface as codes."""
    detail = coverage_detail(
        "Nová známka na každý dopis. Což třídění ztíží. Ne znemožní.",
        "Nová známka na každý dopis. Což třídění ztíží. Neznemožní.",
        "cs",
    )
    assert detail.score == 1.0
    assert detail.word_diagnostics == ()


def test_hallucinated_content_reads_as_a_mass_of_i_codes() -> None:
    detail = coverage_detail(
        "Alpha beta gamma.", "Alpha beta gamma. And then some invented extra sentence."
    )
    assert "[inserted content]" in detail.worst_sentence
    assert detail.word_diagnostics == (
        "i:and", "i:then", "i:some", "i:invented", "i:extra", "i:sentence",
    )


def test_substitution_reports_the_pair_in_reading_order() -> None:
    detail = coverage_detail("The key is safe here.", "The kay is safe here.")
    assert detail.score < 0.90
    assert detail.word_diagnostics == ("s:key/kay",)


def test_coverage_is_exactly_the_detail_pair() -> None:
    """The stable two-tuple view must never drift from the detail it fronts."""
    pairs = (
        (CHUNK, ASR_DROPPED, "en"),
        ("Never share the master password with anyone.",
         "Share the master password with anyone.", "en"),
        ("Nová známka na každý dopis. Což třídění ztíží. Ne znemožní.",
         "Nová známka na každý dopis. Což třídění ztíží. Neznemožní.", "cs"),
        ("", "anything", "en"),
    )
    for ref, hyp, lang in pairs:
        detail = coverage_detail(ref, hyp, lang)
        assert coverage(ref, hyp, lang) == (detail.score, detail.worst_sentence)
