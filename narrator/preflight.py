"""Would this script survive a PERFECT render? Ask before paying for one.

The verifier deliberately refuses what a text round-trip cannot see: an
all-numeral sentence ("Two fifty six.") scores 0.0 no matter what the audio
says, because scripts spell numerals out, every ASR writes digits back, and
number-blinding leaves nothing to compare. Right refusal, wrong moment — on a
real model it lands only after the chunk has burned every retry plus the
sentence-split fallback (each attempt an ASR pass too), and render() reports
it only after every OTHER chunk has been paid for as well. A script defect
priced like an audio defect.

The oracle here is the verifier itself, fed each chunk as its own transcript.
A perfect recogniser cannot do better than returning exactly the reference,
so a chunk that fails the identity round-trip fails as a property of the
SCRIPT, not of any audio — no backend, no voice, no model, milliseconds.
Running the real coverage code (not a hand-copied "shapes that fail" list)
means any future fail-closed rule in verify.py that is reference-intrinsic is
covered here automatically; a parallel list would rot the first time policy
moved.

Advisory by design: render() keeps its own refusal, because preflight speaks
for the DEFAULT policy (MIN_COVERAGE, the standard verifiers) and a caller's
custom verifier may legitimately disagree.

One direction only, and the asymmetry is the whole contract. A FAILING chunk
here is doomed — no transcript beats the reference verbatim. A PASSING chunk is
not thereby renderable, and the reasons are worth naming rather than implying.

The identity oracle models a PERFECT recogniser, which is exactly what does not
exist, so anything that fails because the ASR writes the words back DIFFERENTLY
is invisible: both sides of an identity round-trip are spelled the same. An
expressive control token (issue #17) is the measured case — it stays in the
expected reference and is simply absent from the hypothesis, because no
recogniser reads it aloud:

    preflight("<|emotion:surprise|> Jeden vous?")  -> every chunk can verify
    coverage(same, "Jeden vouz?")                  -> below the gate

Nor is verification the only way a chunk fails. `synth` refuses a take that hits
the frame cap, falls outside the duration bounds, or carries unscripted silence,
all before any transcript exists; a backend can raise; and a caller's own
verifier answers for itself. None of that is reference-intrinsic, so none of it
is visible here.

So this finds script defects cheaply. It never certifies a script.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from narrator.chunking import MAX_CHARS, plan_segments, split_sentences
from narrator.synth import SynthConfig, resolve_reference
from narrator.types import Gap, Segment
from narrator.verify import normalize, verdict_for_transcript

# The render's own pacing default, so the duration estimate and the frame-cap
# budget agree on how fast the narrator talks.
_WPS = SynthConfig().words_per_second


@dataclass(frozen=True)
class UnverifiableChunk:
    """A chunk no audio can ever get past the default verifier."""

    index: int
    """The chunk index render() will assign — Gaps consume none — so the
    preflight finding and the eventual RenderFailed line name the same chunk."""
    text: str
    reason: str
    """The identity verdict's worst_sentence, e.g. "[unverifiable, all-numeral]
    Two fifty six." — the same bracketed diagnostic a failed render prints."""


@dataclass(frozen=True)
class PreflightReport:
    chunks: int
    words: int
    speech_s: float
    """Estimated, at the default pacing — a planning figure, not a promise."""
    gap_s: float
    unverifiable: tuple[UnverifiableChunk, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.unverifiable

    def summary(self) -> str:
        verdict = (
            "every chunk can verify"
            if self.clean
            else f"{len(self.unverifiable)} chunk(s) can NEVER verify"
        )
        return (
            f"{self.chunks} chunks, {self.words} words, "
            f"~{(self.speech_s + self.gap_s) / 60:.1f} min "
            f"({self.gap_s:.1f} s of gaps) | {verdict}"
        )


def preflight(
    segments: list[Segment],
    lang: str = "en",
    max_chars: int = MAX_CHARS,
    sound_alikes: tuple[tuple[str, str], ...] = (),
    allow_sentence_split: bool = SynthConfig().allow_sentence_split,
    non_speech: tuple[str, ...] = (),
) -> PreflightReport:
    """Check a script against the verifier it will face, with no model loaded.

    `unverifiable` lists every chunk whose IDENTITY round-trip — reference
    compared against itself — already fails the default acceptance gate. No
    transcript can score higher than the reference verbatim, so each entry is
    a guaranteed refusal: rendering it is spending the whole retry ladder to
    learn what this function knows for free. Fix the script (spell the
    sentence so it carries a non-numeral content word) and re-run.

    Chunking runs on the original text, exactly as render() does — literally
    the same `plan_segments` walk, not a second one written to match. The
    pronunciation lexicon never shifts boundaries and never reaches the
    verifier, so there is no lexicon parameter here to misuse.

    `allow_sentence_split` must match the render it is predicting. The fallback
    rescues chunks that fail as a whole, so assuming it when the caller has
    turned it off reports clean on a chunk that render would refuse — preflight
    modelling a different ladder from the one being run.
    """
    findings: list[UnverifiableChunk] = []
    chunks = words = 0
    speech_s = gap_s = 0.0
    for segment in plan_segments(segments, max_chars):
        if isinstance(segment, Gap):
            gap_s += segment.seconds
            continue
        # Declared atoms are removed first, exactly as the render will do.
        # Without this, preflight round-trips the RAW text — the atom present
        # on both sides, cancelling out — and reports clean for a chunk the
        # render must refuse because its reference has no speech left.
        raw = segment.text
        piece = resolve_reference(raw, replace(SynthConfig(), non_speech=non_speech))
        if raw != piece and not normalize(piece, lang).split():
            # Every token was a declared atom, so there is no speech left to
            # compare and `coverage("", "")` answers 1.0 — a vacuous pass
            # that would certify whatever audio came back.
            #
            # It is flagged HERE and not in the verifier because only this
            # layer can tell the two empties apart: a punctuation-only
            # sentence the caller WROTE is a pause the script spells, which
            # `verify` blesses on purpose, while this one lost its content
            # to a removal. The verifier never sees the raw text.
            findings.append(UnverifiableChunk(
                chunks, raw, "[no speech left after removing declared non_speech]"))
            chunks += 1
            continue
        # The verifier's own function, not a re-spelling of it: doomed must
        # mean "fails the exact gate the audio will face", so preflight can
        # never flag something the render would in fact tolerate. Comparing a
        # score against `MIN_COVERAGE` here was that re-spelling, and it would
        # have gone stale the first time a fail-closed rule was added to the
        # verdict rather than to the score.
        verdict = verdict_for_transcript(piece, piece, lang, sound_alikes)
        rescued = allow_sentence_split and _split_rescues(piece, lang, sound_alikes)
        if not verdict.ok and not rescued:
            findings.append(UnverifiableChunk(chunks, piece, verdict.dropped_sentence))
        n = len(piece.split())
        words += n
        speech_s += n / _WPS
        chunks += 1
    return PreflightReport(chunks, words, speech_s, gap_s, tuple(findings))


def _split_rescues(piece: str, lang: str, sound_alikes: tuple[tuple[str, str], ...]) -> bool:
    """Would the sentence-split fallback save this chunk in a perfect render?

    Doomed must model the whole retry LADDER, not just its first rung. A chunk
    like "Look at the dial. Four." fails chunk-level identity — "Four." is an
    all-numeral sentence, unverifiable in context — but the fallback renders
    each sentence alone, where the isolated numeral IS comparable ("Four."
    against the ASR's "4" verifies by value). Judging the chunk alone declared
    it doomed while a real render recovered it cleanly (frontier review, this
    diff). Mirrors synth._sentence_split: available only from two sentences up,
    and only if EVERY sentence passes alone. Models the default policy —
    a caller who sets allow_sentence_split=False loses this rescue, which is why
    the caller passes that setting in rather than this assuming it.
    """
    sentences = split_sentences(piece)
    if len(sentences) < 2:
        return False
    return all(
        verdict_for_transcript(s, s, lang, sound_alikes).ok
        for s in sentences
    )
