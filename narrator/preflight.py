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

from dataclasses import dataclass

from narrator.chunking import MAX_CHARS, chunk, split_sentences
from narrator.synth import SynthConfig
from narrator.types import Gap, Segment, Text
from narrator.verify import MIN_COVERAGE, coverage_detail

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
) -> PreflightReport:
    """Check a script against the verifier it will face, with no model loaded.

    `unverifiable` lists every chunk whose IDENTITY round-trip — reference
    compared against itself — already fails the default acceptance gate. No
    transcript can score higher than the reference verbatim, so each entry is
    a guaranteed refusal: rendering it is spending the whole retry ladder to
    learn what this function knows for free. Fix the script (spell the
    sentence so it carries a non-numeral content word) and re-run.

    Chunking runs on the original text, exactly as render() does — the
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
    for segment in segments:
        if isinstance(segment, Gap):
            gap_s += segment.seconds
            continue
        if not isinstance(segment, Text):  # pragma: no cover - guarded by the type union
            raise TypeError(f"Not a segment: {segment!r}")
        for piece in chunk(segment.text, max_chars):
            detail = coverage_detail(piece, piece, lang, sound_alikes)
            # The verifier's real gate, not `< 1.0`: doomed must mean "fails
            # the exact threshold the audio will face", so preflight can never
            # flag something the render would in fact tolerate.
            rescued = allow_sentence_split and _split_rescues(piece, lang, sound_alikes)
            if detail.score < MIN_COVERAGE and not rescued:
                findings.append(UnverifiableChunk(chunks, piece, detail.worst_sentence))
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
        coverage_detail(s, s, lang, sound_alikes).score >= MIN_COVERAGE
        for s in sentences
    )
