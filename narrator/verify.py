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
# Words whose loss or insertion inverts meaning. A fuzzy threshold cannot protect
# these: dropping one word from a twelve-word sentence is ~8% of its coverage, so
# "The keeper can open these doors" rendered as "cannot open" scored 0.9167
# and passed a 0.90 gate. Meaning is not proportional to word count.
# NOTE: standalone Czech "ne" is deliberately absent. Czech negation is
# morphological — a ne- prefix (nelze, nesmí, neznemožní) — so "ne" as a separate
# word is rare, and listing it collided with the word-boundary case this verifier
# must tolerate: the ASR returns "Ne znemožní." as "Neznemožní.", which then read
# as a dropped negation. Prefix negation is caught by coverage instead, since the
# prefixed and unprefixed forms are different tokens.
_CRITICAL_TOKENS = frozenset(
    """
    not no never none nor cannot cant dont doesnt didnt isnt arent wasnt werent
    wont wouldnt shouldnt couldnt without nothing neither
    nikdy nic nikdo žádný žádná žádné nelze bez ani nesmí nesmíš nemůže
    always must all every only
    vždy musí všechny každý pouze jen
    """.split()
)


def critical_counts(words: list[str]) -> dict[str, int]:
    return {w: words.count(w) for w in set(words) & _CRITICAL_TOKENS}


# Values for the simple cardinals, so an ISOLATED numeral can be compared rather
# than merely ignored. Number-blinding is what makes the rest of this verifier
# work, but it left the wrong number scoring a perfect 1.0 — and this pipeline
# teaches "four harbor lights" and "a twenty-page logbook", where the number IS the
# content.
_NUMERAL_VALUES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9,
    "nula": 0, "jedna": 1, "jeden": 1, "jedno": 1, "dva": 2, "dvě": 2, "tři": 3,
    "čtyři": 4, "pět": 5, "šest": 6, "sedm": 7, "osm": 8, "devět": 9, "deset": 10,
    "jedenáct": 11, "dvanáct": 12, "třináct": 13, "čtrnáct": 14, "patnáct": 15,
    "šestnáct": 16, "sedmnáct": 17, "osmnáct": 18, "devatenáct": 19, "dvacet": 20,
    "třicet": 30, "čtyřicet": 40, "padesát": 50, "šedesát": 60, "sedmdesát": 70,
    "osmdesát": 80, "devadesát": 90, "sto": 100, "tisíc": 1000,
}


def has_compound_numeral(words: list[str]) -> bool:
    """Two numerals side by side, e.g. "two fifty six"."""
    return any(
        is_numberish(w) and is_numberish(words[i + 1])
        for i, w in enumerate(words[:-1])
    )


def isolated_numerals(words: list[str]) -> list[int]:
    """Values of numerals that stand ALONE, with no numeral either side.

    Compounds are skipped deliberately. "two fifty six" and "256" denote the same
    quantity but tokenize as [2, 50, 6] versus [256], so comparing them would
    manufacture exactly the false failures number-blinding exists to prevent.
    An isolated numeral has no such ambiguity: "four bytes" against "nine bytes"
    is unambiguously wrong.
    """
    values: list[int] = []
    for i, word in enumerate(words):
        if not is_numberish(word):
            continue
        prev_num = i > 0 and is_numberish(words[i - 1])
        next_num = i + 1 < len(words) and is_numberish(words[i + 1])
        if prev_num or next_num:
            continue          # part of a compound; ambiguous, so skip
        if word.isdigit():
            values.append(int(word))
        elif word in _NUMERAL_VALUES:
            values.append(_NUMERAL_VALUES[word])
    return values


_SENTENCE_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')


@runtime_checkable
class ASR(Protocol):
    """Speech recognition, behind a seam so verification is testable without a model."""

    def transcribe(self, audio: Audio, lang: str) -> str: ...


# Czech orthography encodes distinctions that its phonology does not, so an ASR
# and a script routinely disagree in spelling about identical sound. Measured
# across 201 real Czech chunks, EVERY rejection was of this kind: lisa/lise,
# tipovat/typovat, odpověz/odpověs, cokoli/cokoliv, hashe/haše.
# Czech failed at 18.4% against English at 2.4% for this reason alone.
#
# Folding is safe for the thing that matters: it can make two spellings of the
# same word match, but it cannot conjure a word that is absent. Drop detection
# is unaffected.
_FOLD = str.maketrans({
    # i/y carry no sound difference in modern Czech, and vowel LENGTH is what an
    # ASR most often gets wrong. Nothing here merges two different consonants:
    # a blanket voiced/voiceless collapse (z->s, d->t, b->p) was tried and made
    # things worse — too many distinct words collided, which scrambled the
    # alignment and broke 9 chunks that had been passing.
    "y": "i", "ý": "i", "í": "i",
    "á": "a", "é": "e", "ě": "e", "ú": "u", "ů": "u", "ó": "o",
})


# Voicing assimilation in clusters: an obstruent takes the voicing of what
# follows, so ztíží/stíží and spraví/zpraví are one pronunciation with two
# spellings — measured as hard 0.00 rejections from three independent
# recognisers on correct audio. Unlike the blanket voiced/voiceless collapse
# (which merged distinct words and was reverted), this fires only in the
# cluster positions where Czech phonology actually neutralises the contrast;
# s and z between vowels stay distinct.
_DOUBLED = re.compile(r"(.)\1+")
_Z_BEFORE_VOICELESS = re.compile(r"z(?=[ptťkfsšcč])")
_S_BEFORE_VOICED = re.compile(r"s(?=[bdďgzž])")


def fold(word: str, lang: str) -> str:
    """Collapse spelling differences that carry no difference in sound."""
    if not lang.startswith("cs"):
        return word
    w = word.translate(_FOLD)
    w = w.replace("sh", "š")                 # English loanwords: hashe / haše
    w = _DOUBLED.sub(r"\1", w)               # doubled letters
    w = _Z_BEFORE_VOICELESS.sub("s", w)
    w = _S_BEFORE_VOICED.sub("z", w)
    if w.endswith("z"):                      # final devoicing: odpověz / odpověs
        w = w[:-1] + "s"
    return w


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


def coverage(reference: str, hypothesis: str, lang: str = "en") -> tuple[float, str]:
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
    ref_words = [fold(w, lang) for w in content_words(reference)]
    hyp_words = [fold(w, lang) for w in content_words(hypothesis)]
    if not ref_words:
        return 1.0, ""

    covered = [False] * len(ref_words)
    hyp_claimed = [False] * len(hyp_words)
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    matched = 0
    for i, j, size in matcher.get_matching_blocks():
        matched += size
        for k in range(i, i + size):
            covered[k] = True
        for k in range(j, j + size):
            hyp_claimed[k] = True

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

    # Word-boundary rescue, applied to INDIVIDUAL words rather than only to short
    # sentences. Measured on real Whisper output: the script says "coworkers" and
    # the transcript says "co-worker's", which normalises to three tokens and
    # aligns with none of them — correct audio scoring 0.83 and failing. The same
    # mechanism as the Czech "Ne znemožní" -> "Neznemožní" merge, in reverse.
    #
    # Restricted to UNCLAIMED hypothesis text, so a word cannot be rescued by an
    # occurrence that another sentence already matched. That restriction is what
    # keeps a genuinely dropped sentence detectable.
    unclaimed = "".join(w for w, taken in zip(hyp_words, hyp_claimed) if not taken)
    for k, word in enumerate(ref_words):
        if not covered[k] and len(word) > 2 and word in unclaimed:
            covered[k] = True
            matched += 1

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
            needle = "".join(fold(w, lang) for w in content_words(sentence))
            seen[needle] = seen.get(needle, 0) + 1
            # Boundaries matter: an unanchored substring search found "go now"
            # inside "under-go now-here". Require the match to sit at a word
            # boundary in the squashed hypothesis, reconstructed from the words.
            # Search only from where this sentence should begin. Counting
            # globally let a later sentence rescue an earlier missing one:
            # "It burns." dropped still scored 1.0 because "it burns" occurred
            # inside the following sentence.
            # Search only hypothesis words that no other sentence matched.
            #
            # This is the distinction that makes the rescue safe. When the ASR
            # merges "ne znemožní" into "neznemožní", that merged token is
            # UNCLAIMED — no reference word aligned to it — so containment finds
            # it and correctly rescues the sentence. When a sentence is genuinely
            # dropped but its words appear elsewhere ("It burns." recurring inside
            # a later sentence, or one of three identical "Again."), every
            # candidate word is already claimed by another sentence's alignment,
            # so there is nothing left to rescue it with. Searching the raw
            # transcript could not tell those apart at any window size, because
            # global alignment had already matched the sentence to the wrong
            # occurrence.
            leftover = [w for w, claimed in zip(hyp_words, hyp_claimed) if not claimed]
            present = _bounded_count(leftover, [fold(w, lang) for w in content_words(sentence)]) >= 1
            # `or score > 0` used to turn ANY partial coverage into a pass, so a
            # two-word sentence rendered as one word scored 1.0. Only genuine
            # containment rescues a short sentence now.
            score = 1.0 if present else score

        if score < worst:
            worst, worst_sentence = score, sentence.strip()

    # An isolated numeral that changed value is a content error, not an ASR
    # spelling difference. Compounds are excluded above, so this cannot fire on
    # "two fifty six" vs "256".
    ref_tokens = normalize(reference).split()
    hyp_tokens = normalize(hypothesis).split()
    # Skip if EITHER side compounds. The check must be symmetric: "two fifty six"
    # is three adjacent numerals in the script and collapses to the single
    # isolated "256" in the transcript, so an asymmetric rule reads a correct
    # transcription as a changed number.
    compound = has_compound_numeral(ref_tokens) or has_compound_numeral(hyp_tokens)
    ref_nums = [] if compound else sorted(isolated_numerals(ref_tokens))
    hyp_nums = [] if compound else sorted(isolated_numerals(hyp_tokens))
    if ref_nums != hyp_nums:
        return 0.0, f"[numeral changed: {ref_nums} became {hyp_nums}]"

    # A meaning-inverting token that appears or disappears fails outright,
    # regardless of how good the surrounding coverage looks.
    ref_critical = critical_counts(normalize(reference).split())
    hyp_critical = critical_counts(normalize(hypothesis).split())
    if ref_critical != hyp_critical:
        changed = sorted(set(ref_critical) ^ set(hyp_critical)) or sorted(
            w for w in ref_critical if ref_critical[w] != hyp_critical.get(w)
        )
        return 0.0, f"[meaning-critical token changed: {', '.join(changed)}]"

    if precision < worst:
        # Something was inserted rather than dropped. Report the whole chunk,
        # since an insertion does not belong to any one reference sentence.
        return precision, f"[inserted content] {reference.strip()[:70]}"

    if unverifiable:
        # Fail closed. These sentences are all-numeral, so number-blinding left
        # nothing to compare and a text round-trip genuinely cannot check them.
        # The list was previously built and then ignored, which meant a dropped
        # all-numeral sentence scored a clean 1.0 — documenting a blind spot is
        # not the same as refusing to certify what you cannot see.
        return 0.0, f"[unverifiable, all-numeral] {unverifiable[0]}"

    return worst, worst_sentence


@dataclass
class CoverageVerifier:
    """The default verifier: transcribe, then score per-sentence coverage."""

    asr: ASR
    min_coverage: float = MIN_COVERAGE

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        transcript = self.asr.transcribe(audio, lang)
        score, dropped = coverage(text, transcript, lang)
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
