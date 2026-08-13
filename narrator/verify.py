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

MIN_COVERAGE = 0.90
"""Near-complete, deliberately.

At 0.60 a sentence could lose almost 40% of its words and pass: "Never share the
master password to anyone." rendered as "Share the master password with anyone."
scored 0.857. Dropping a single negation inverts the meaning, and no threshold
that tolerates it is defensible for teaching material. Number-blinding plus the
short-sentence rule already absorb the ASR disagreements that made a loose
threshold seem necessary."""
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

# Closing quotes and brackets may sit between the terminator and the space.
# Missing them merged sentences, which silently restored the aggregate
# scoring that per-sentence coverage exists to replace.
_SENTENCE_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')


@runtime_checkable
class ASR(Protocol):
    """Speech recognition, behind a seam so verification is testable without a model."""

    def transcribe(self, audio: Audio, lang: str) -> str: ...


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def is_numberish(word: str) -> bool:
    """A token that carries only numeric value, so it can be ignored.

    Deliberately NOT "contains a digit": that deleted `utf8`, `iso9001`, `rfc822`
    and `base64` from both sides, which are content words in this domain — the
    verifier was blind to whether the audio said them at all.
    """
    return word.isdigit() or word in _NUMBER_WORDS


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if not is_numberish(w)]


def _bounded_count(hyp_words: list[str], needle_words: list[str]) -> int:
    """How many times `needle_words` appears in `hyp_words`, allowing the words to
    have merged or split in the transcript but not to straddle other words."""
    if not needle_words:
        return 0
    needle = "".join(needle_words)
    count = 0
    for start in range(len(hyp_words)):
        joined = ""
        for end in range(start, min(start + len(needle_words) + 2, len(hyp_words))):
            joined += hyp_words[end]
            if joined == needle:
                count += 1
                break
            if len(joined) > len(needle):
                break
    return count


def _squashed(text: str) -> str:
    """Letters only, no spaces. Immune to word-boundary disagreement."""
    return re.sub(r"\s+", "", normalize(text))


def coverage(reference: str, hypothesis: str) -> tuple[float, str]:
    """Worst per-sentence coverage in [0,1], and the sentence that scored it.

    A dropped sentence scores ~0 regardless of how long the surrounding chunk is;
    an ASR spelling quirk costs a word or two inside an otherwise intact sentence.
    That separation is the entire reason this is per-sentence.

    Known limit, stated because it is not obvious: a sentence whose content words
    are ALL numerals cannot be verified this way. Scripts spell numerals out
    because digits are unspeakable, every ASR writes them back as digits, and
    number-blinding — which is what makes the rest of this work — leaves such a
    sentence with nothing to compare. Those are counted, not silently skipped, and
    the duration bounds remain the only guard on them.
    """
    ref_words = content_words(reference)
    hyp_words = content_words(hypothesis)
    if not ref_words:
        return 1.0, ""

    covered = [False] * len(ref_words)
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    matched = 0
    for i, _, size in matcher.get_matching_blocks():
        matched += size
        for k in range(i, i + size):
            covered[k] = True

    # Recall alone is not enough, and this was the library's worst blind spot.
    # Marking only reference words made everything the engine ADDED invisible:
    # "The key is safe." rendered as "The key is not safe." scored a perfect 1.0,
    # as did correct audio followed by a hallucinated extra sentence. An inserted
    # negation is the most damaging corruption possible in teaching material.
    #
    # Measured on SQUASHED CHARACTERS, not words, for the same reason the
    # short-sentence rule is: a transcript that merges "ne znemožní" into
    # "neznemožní" has inserted nothing, but word-level precision reads the merge
    # as one unmatched token and one missing pair.
    ref_squashed = "".join(ref_words)
    hyp_squashed_words = "".join(hyp_words)
    char_matcher = difflib.SequenceMatcher(None, ref_squashed, hyp_squashed_words, autojunk=False)
    matched_chars = sum(size for _, _, size in char_matcher.get_matching_blocks())
    precision = matched_chars / len(hyp_squashed_words) if hyp_squashed_words else 1.0

    hyp_squashed = _squashed(hypothesis)
    seen: dict[str, int] = {}
    worst, worst_sentence, pos = 1.0, "", 0
    unverifiable: list[str] = []

    for sentence in (s for s in _SENTENCE_END.split(reference.strip()) if s.strip()):
        n = len(content_words(sentence))

        if n == 0:
            # Every content word was a numeral, so number-blinding left nothing to
            # compare. This is genuinely unverifiable by a text round-trip: the
            # script says "two fifty six" precisely because digits are unspeakable,
            # and the ASR says "256", so no squashed form matches either. Recorded
            # rather than skipped — silently passing it is how a dropped sentence
            # gets through, which this used to do.
            unverifiable.append(sentence.strip())
            continue

        score = sum(covered[pos:pos + n]) / n
        pos += n

        if n < SHORT_SENTENCE_WORDS:
            # Short sentences are fragile at word level: Czech "Ne znemožní."
            # returns as one token, "Neznemožní", so neither source word matches
            # and correct audio reads 0.0. Squashing both sides hides boundary
            # disagreement — but containment alone is not enough, because an
            # identical sentence elsewhere in the chunk satisfies it while this
            # one is genuinely absent. Count occurrences instead of asking
            # "does it appear at all".
            needle = _squashed(sentence)
            seen[needle] = seen.get(needle, 0) + 1
            # Boundaries matter: an unanchored substring search found "go now"
            # inside "under-go now-here". Require the match to sit at a word
            # boundary in the squashed hypothesis, reconstructed from the words.
            present = _bounded_count(hyp_words, content_words(sentence)) >= seen[needle]
            # `or score > 0` used to turn ANY partial coverage into a pass, so a
            # two-word sentence rendered as one word scored 1.0. Only genuine
            # containment rescues a short sentence now.
            score = 1.0 if present else score

        if score < worst:
            worst, worst_sentence = score, sentence.strip()

    if precision < worst:
        # Something was inserted rather than dropped. Report the whole chunk,
        # since an insertion does not belong to any one reference sentence.
        return precision, f"[inserted content] {reference.strip()[:70]}"
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
