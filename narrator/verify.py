"""Does the audio actually say the text?

This is the load-bearing component. Duration heuristics alone caught **zero of
eight** real content drops in the measurement that motivated this library — a
chunk that loses a third of its words still lands inside any duration bound
permissive enough not to fire constantly.

The approach is an ASR round-trip scored **per sentence**, not in aggregate.
Three traps, each of which cost real debugging:

1. **`difflib` needs `autojunk=False`.** By default SequenceMatcher discards
   elements appearing in more than 1% of a sequence of 200+ items. On a
   250-character chunk that junks the common letters, and it scored
   98.2%-identical text at **0.231** — a false failure that sent good audio into
   recovery and made a correct chunk look catastrophic.

2. **Aggregate similarity cannot separate the two cases.** Measured on one real
   227-character chunk: correct audio where the ASR wrote "20" for "twenty"
   scored 0.982, and audio with a whole sentence genuinely missing scored 0.942.
   Four points apart, and inverted once the autojunk bug is fixed. Per-sentence
   coverage on the same chunk: 0.947 versus **0.000**.

3. **It must be number-blind.** Scripts written for TTS spell numerals out
   ("SHA two fifty six") because digits are unspeakable; every ASR writes them
   straight back as digits. Source and verifier are in guaranteed conflict on
   exactly that token class, and a two-word sentence containing one ("Episode
   one.") cannot survive any threshold without this.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from narrator.types import Audio, Verdict

MIN_COVERAGE = 0.60
SHORT_SENTENCE_WORDS = 3

# Spelled-out numerals in the languages this is used on. Anything matching is
# dropped from BOTH sides before comparison — see trap 3.
_NUMBER_WORDS = set(
    """
    zero one two three four five six seven eight nine ten eleven twelve thirteen
    fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
    sixty seventy eighty ninety hundred thousand million billion
    nula jedna jeden jedno dva dvě tři čtyři pět šest sedm osm devět deset
    jedenáct dvanáct třináct čtrnáct patnáct šestnáct sedmnáct osmnáct devatenáct
    dvacet třicet čtyřicet padesát šedesát sedmdesát osmdesát devadesát
    sto stě sta set tisíc tisíce milion miliony milionů miliarda miliard miliardy
    """.split()
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class ASR(Protocol):
    """Speech recognition, behind a seam so verification is testable without a model."""

    def transcribe(self, audio: Audio, lang: str) -> str: ...


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def is_numberish(word: str) -> bool:
    return any(c.isdigit() for c in word) or word in _NUMBER_WORDS


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if not is_numberish(w)]


def _squashed(text: str) -> str:
    """Letters only, no spaces. Immune to word-boundary disagreement."""
    return re.sub(r"\s+", "", normalize(text))


def coverage(reference: str, hypothesis: str) -> tuple[float, str]:
    """Worst per-sentence coverage in [0,1], and the sentence that scored it.

    A dropped sentence scores ~0 regardless of how long the surrounding chunk is;
    an ASR spelling quirk costs a word or two inside an otherwise intact sentence.
    That separation is the entire reason this is per-sentence.
    """
    ref_words = content_words(reference)
    hyp_words = content_words(hypothesis)
    if not ref_words:
        return 1.0, ""

    covered = [False] * len(ref_words)
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    for i, _, size in matcher.get_matching_blocks():
        for k in range(i, i + size):
            covered[k] = True

    hyp_squashed = _squashed(hypothesis)
    worst, worst_sentence, pos = 1.0, "", 0
    for sentence in (s for s in _SENTENCE_END.split(reference.strip()) if s.strip()):
        n = len(content_words(sentence))
        if n == 0:
            continue
        score = sum(covered[pos:pos + n]) / n
        pos += n

        if n < SHORT_SENTENCE_WORDS:
            # Short sentences are fragile at word level: Czech "Ne znemožní."
            # comes back as one token, "Neznemožní", and neither source word
            # matches, so word coverage reads 0.0 on correct audio. Squashing
            # both sides makes boundary disagreement invisible.
            present = _squashed(sentence) in hyp_squashed
            score = 1.0 if (present or score > 0) else 0.0

        if score < worst:
            worst, worst_sentence = score, sentence.strip()

    return worst, worst_sentence


@dataclass
class CoverageVerifier:
    """The default verifier: transcribe, then score per-sentence coverage."""

    asr: ASR
    min_coverage: float = MIN_COVERAGE

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        transcript = self.asr.transcribe(audio, lang)
        score, dropped = coverage(text, transcript)
        return Verdict(
            ok=score >= self.min_coverage,
            coverage=score,
            dropped_sentence="" if score >= self.min_coverage else dropped,
            transcript=transcript,
        )


@dataclass
class NullVerifier:
    """Accepts everything. For callers who want speed over safety, explicitly.

    Named rather than implied: the predecessor's `--no-asr` flag silently made
    retries useless, because a skipped check returned a perfect score that no
    later attempt could beat.
    """

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        return Verdict(ok=True, coverage=1.0)
