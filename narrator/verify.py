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

from narrator.chunking import split_sentences
from narrator.types import ASR, Audio, Verdict, Verifier

MIN_COVERAGE = 0.90
"""Near-complete, deliberately.

At 0.60 a sentence could lose almost 40% of its words and pass: "Never share the
master password with anyone." rendered as "Share the master password with anyone."
scored 0.857. Dropping a single negation inverts the meaning, and no threshold
that tolerates it is defensible for teaching material. Number-blinding plus the
short-sentence rule already absorb the ASR disagreements that made a loose
threshold seem necessary."""
SHORT_SENTENCE_WORDS = 3

# Spelled-out numerals in the languages this is used on. Anything matching is
# dropped from BOTH sides before comparison — see trap 3.
#
# Per-language, NOT one blind pool. "set" is the genitive plural of Czech sto
# ("pět set" = 500) and also core English vocabulary, so the pooled list
# deleted a real English content word from both sides — and once the
# all-numeral fail-closed reached whole references, "One set." rejected a
# PERFECT transcript as unverifiable (frontier review, this diff). English
# text therefore blinds only English numerals. Czech keeps the UNION: these
# scripts narrate English tech content, where trap 3's "SHA two fifty six"
# appears verbatim inside Czech prose, so English numerals must stay blind
# there. The union carries one KNOWN collision the other way, inherited
# unchanged from the pooled list: English "ten" is blinded inside Czech text,
# where "ten" is the demonstrative pronoun — "Vyber ten." against the ASR's
# "Vyber deset." passes today exactly as it did on main. Left in place
# deliberately: removing "ten" from the Czech blind set changes cs verdicts
# on real renders and needs its own measured pass, not a drive-by.
_NUMBER_WORDS_EN = set(
    """
    zero one two three four five six seven eight nine ten eleven twelve thirteen
    fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
    sixty seventy eighty ninety hundred thousand million billion
    """.split()
)
# Czech declines its numerals, and prose uses the oblique forms constantly:
# "dvou tisíc čtyřiceti osmi slov" is 2048 in the genitive. The ASR writes
# "2048", so a blind list holding only the citation forms left every inflected
# numeral unmatched — measured on a real render as four chunks failing at
# 0.52-0.89 on orthography, not audio. The oblique forms are numerals and
# nothing else in Czech, so blinding them collides with no content word.
_NUMBER_WORDS_CS = _NUMBER_WORDS_EN | set(
    """
    nula jedna jeden jedno dva dvě tři čtyři pět šest sedm osm devět deset
    jedenáct dvanáct třináct čtrnáct patnáct šestnáct sedmnáct osmnáct devatenáct
    dvacet třicet čtyřicet padesát šedesát sedmdesát osmdesát devadesát
    sto stě sta set tisíc tisíce milion miliony milionů miliarda miliard miliardy
    jedné jednoho jednomu jedním jednou dvou dvěma tří třech třem třemi
    čtyř čtyřech čtyřem čtyřmi pěti šesti sedmi osmi devíti deseti
    jedenácti dvanácti třinácti čtrnácti patnácti šestnácti sedmnácti osmnácti
    devatenácti dvaceti třiceti čtyřiceti padesáti šedesáti sedmdesáti
    osmdesáti devadesáti stu tisíci tisících milionu miliardě
    """.split()
)

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
# APOSTROPHIZED contractions never reach this count as themselves: `normalize`
# expands them to their two-word forms — except can't and ain't, which land on
# the single protected tokens cannot and aint — so a listed token carries every
# one of them. Listing
# them was expected to work and did not — normalize spaced the apostrophe into
# "don t", no token ever matched, and "can't open" rendered as "can open"
# scored 0.923 and PASSED while "don't" against an ASR's "do not" hard-failed
# correct audio. The COMMON bare spellings (dont, wont, cant...) are listed and
# reachable: they occur only when a side genuinely contains that spelling
# (a sloppy script, or bare "wont"/"cant" as real words), and there the
# fail-closed hard fail is the cheap direction — expanding them instead cost
# three false-accept classes, recorded at the expansion tables. The RARE bare
# forms (shant, darent, maynt, havent...) are deliberately NOT listed: Shant
# and Darent are real proper names (an Armenian given name; a Kentish river)
# that reached the veto through ordinary text and even through caller
# sound_alikes, while the misspellings they would protect are emitted by no
# recogniser. A rare bare form against its expanded transcript still fails,
# and fails HARD: the expansion puts a "not" on one side only, and the
# asymmetric critical count returns 0.0 — the fail-closed direction for a
# script that misspelled its own contraction.
# "cannot" IS listed, and "can't" expands to it rather than to "can not":
# collapsing both to a bare "not" made "cannot open" and "will not open" agree
# on their critical counts, and inability became refusal at 0.923 — auxiliary
# identity is meaning, and one "not" looks like any other. That protection stops
# at cannot: pairing every "not" with its auxiliary (so won't/wouldn't differ)
# was considered and rejected, because the auxiliary token is unstable on
# correct audio — "it's not" against "it is not" pairs "s not" vs "is not" and
# would hard-fail a routine contraction. "aint" is listed because ain't
# collapses to that single token — no expansion is faithful to its
# subject-dependent auxiliary, but a dropped ain't must still be a drop.
_CRITICAL_TOKENS = frozenset(
    """
    not no never none nor cannot aint without nothing neither
    arent cant couldnt didnt doesnt dont isnt shouldnt wasnt werent
    wont wouldnt
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
    # Oblique-case forms, so an inflected numeral is COMPARED, not merely
    # blinded — without these the ref side skipped "šestnácti" while the hyp
    # side counted "16", and a correct transcription hard-failed as a changed
    # number.
    "jedné": 1, "jednoho": 1, "jednomu": 1, "jedním": 1, "jednou": 1,
    "dvou": 2, "dvěma": 2, "tří": 3, "třech": 3, "třem": 3, "třemi": 3,
    "čtyř": 4, "čtyřech": 4, "čtyřem": 4, "čtyřmi": 4, "pěti": 5, "šesti": 6,
    "sedmi": 7, "osmi": 8, "devíti": 9, "deseti": 10, "jedenácti": 11,
    "dvanácti": 12, "třinácti": 13, "čtrnácti": 14, "patnácti": 15,
    "šestnácti": 16, "sedmnácti": 17, "osmnácti": 18, "devatenácti": 19,
    "dvaceti": 20, "třiceti": 30, "čtyřiceti": 40, "padesáti": 50,
    "šedesáti": 60, "sedmdesáti": 70, "osmdesáti": 80, "devadesáti": 90,
    "stu": 100, "tisíci": 1000,
    # DETERMINATE forms of the large units only. "milion" and its singular
    # obliques mean exactly 10**6; a bare PLURAL does not — "Byly jich
    # miliony." means "there were millions (of them)", and mapping
    # miliony=10**6 certified a transcript's literal "1000000" against it at
    # 1.0 (gate review of this change). The indeterminate forms — sta/set
    # (bare "hundreds" / the counted form after five and up), tisíce/tisících,
    # miliony/milionů/miliard/miliardy — therefore carry NO value on purpose:
    # they stay blinded, and the _valued guard in the by-value branch refuses
    # them as non-comparable, which the silence sweep in the tests enforces
    # word by word. In compounds ("pět set", "dva tisíce") they are
    # suppressed with the rest of the compound and never reach a value
    # lookup. "stě" IS determinate — it occurs as the singular locative
    # ("ve stě případech"), parallel to "stu" above.
    "stě": 100,
    "milion": 10**6, "milionu": 10**6,
    "miliarda": 10**9, "miliardě": 10**9,
}


def has_compound_numeral(words: list[str], lang: str = "en") -> bool:
    """Two numerals side by side, e.g. "two fifty six"."""
    return any(
        is_numberish(w, lang) and is_numberish(words[i + 1], lang)
        for i, w in enumerate(words[:-1])
    )


def isolated_numerals(words: list[str], lang: str = "en") -> list[int]:
    """Values of numerals that stand ALONE, with no numeral either side.

    Compounds are skipped deliberately. "two fifty six" and "256" denote the same
    quantity but tokenize as [2, 50, 6] versus [256], so comparing them would
    manufacture exactly the false failures number-blinding exists to prevent.
    An isolated numeral has no such ambiguity: "four bytes" against "nine bytes"
    is unambiguously wrong.
    """
    values: list[int] = []
    for i, word in enumerate(words):
        if not is_numberish(word, lang):
            continue
        prev_num = i > 0 and is_numberish(words[i - 1], lang)
        next_num = i + 1 < len(words) and is_numberish(words[i + 1], lang)
        if prev_num or next_num:
            continue          # part of a compound; ambiguous, so skip
        if _plain_digits(word):
            values.append(int(word))
        elif word in _NUMERAL_VALUES:
            values.append(_NUMERAL_VALUES[word])
    return values




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
_SH_AFTER_VOWEL = re.compile(r"(?<=[aeiou])sh")
_Z_BEFORE_VOICELESS = re.compile(r"z(?=[ptťkfsšcč])")
_S_BEFORE_VOICED = re.compile(r"s(?=[bdďgzž])")


def fold(word: str, lang: str) -> str:
    """Collapse spelling differences that carry no difference in sound."""
    if not lang.startswith("cs"):
        return word
    w = word.translate(_FOLD)
    # [sx] spelled two ways: the script writes native s+h (shodí), the ASR
    # writes sch (schodí). One spelling before the loanword rule below.
    w = w.replace("sch", "sh")
    # English loanwords: hashe / haše. Only after a vowel — word-initial sh is
    # the native s+h prefix (shodí, shoda, shora), pronounced [sx], not [š];
    # folding it corrupted a correct native word on a real render.
    w = _SH_AFTER_VOWEL.sub("š", w)
    w = w.replace("ee", "i")                 # English loanwords: seed / síd —
    # no native Czech word contains "ee", so this collides with nothing
    w = _DOUBLED.sub(r"\1", w)               # doubled letters
    w = _Z_BEFORE_VOICELESS.sub("s", w)
    w = _S_BEFORE_VOICED.sub("z", w)
    if w in ("ze", "ke", "ve", "se"):
        # Vocalized prepositions: the -e exists only for pronunciation before
        # certain clusters, so "ze šestnácti" and "z 16" are the same word.
        # Surfaced by number-blinding: the ASR writes the digit, drops the
        # vowel, and the preposition mismatched on spelling alone.
        w = w[0]
    # Final devoicing: Czech devoices every word-final obstruent, so odpověz /
    # odpověs and seed[síd] / sít are one pronunciation each. Positional like
    # the cluster rule — led (ice) and let (flight) genuinely are homophones,
    # and the verifier argues about sound.
    final = {"z": "s", "d": "t", "b": "p", "ž": "š", "v": "f", "h": "ch"}
    if w and w[-1] in final:
        w = w[:-1] + final[w[-1]]
    return w


# Script and ASR freely disagree on contracting a negation — the script writes
# "don't", Whisper writes "do not", or the reverse. Expanding both sides to the
# two-word form aligns the tokens and lets "not" carry the meaning-critical
# protection. The apostrophe is REQUIRED: bare "wont" (habit) and "cant"
# (jargon) are genuine English words, and an unconditional table read "he was
# wont to visit" as "he was will not to visit" — a fabricated negation the
# critical check then trusted. "ain't" collapses to the single token "aint"
# rather than expanding: no expansion is faithful to its subject-dependent
# auxiliary — mapping it to bare "not" accepted "I not ready" at 1.0 — but the
# single token can be protected, so a dropped ain't is still a drop. "can't"
# expands to the single token "cannot", not "can not" — see _CRITICAL_TOKENS.
# These tables are ENGLISH ORTHOGRAPHY and apply only to English, the same way
# fold() applies only to Czech: language-blind expansion turned the uppercase
# acronym ISNT in a Czech sentence into "is not" and certified audio that
# never spelled the letters.
# STRICT "n't" only: the apostrophe directly between the n and the t, no
# whitespace on either side, no guard on what follows. Every looser separator
# was tried and each rewrote real English, because a loosened contraction is
# indistinguishable BY STRING from quotation: whitespace-only welded
# "Don T Harris" and "can T-test" into negations; whitespace-with-apostrophe
# welded the quoted letter in "Can 'T' represent" and, spaced, "Can ' T '";
# apostrophe-then-whitespace welded the closing quote plus initial in
# "After 'can' T. S. Eliot". Trailing-apostrophe guards then hard-failed the
# QUOTED contraction — "the word 'Don't'" against its unquoted transcript.
# The strict form has none of these: a space on either side of the apostrophe
# breaks the match, and a following quote is fine because no real text
# attaches a quoted bare T to a word ("don'T'" exists nowhere).
#
# The cost, deliberate: tokenized-apart spellings ("don 't", "don' t",
# "don ' t", whitespace-only "ain t") do NOT expand and fail closed against
# their expanded counterparts. No recogniser emits them — the predecessor
# only "passed" them because it spaced every apostrophe and the two sides
# happened to collapse identically.
#
# Matched before punctuation is stripped — a sentence terminator must keep
# blocking any cross-boundary reading, as in "the Ain. T cells", where a
# post-strip fold once welded two sentences and an IDENTICAL ref/hyp pair
# scored 0.857 because per-sentence word counts stopped matching the
# full-chunk tokens.
_NT_GENERIC = re.compile(
    r"\b(am|are|could|dare|did|does|do|had|has|have|is|may|might|must|need|"
    r"ought|should|used|was|were|would)n't\b"
)
_NT_SPECIAL = (
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bain't\b"), "aint"),
)
# The APOSTROPHE-LESS spellings (dont, isnt, wouldnt...) deliberately do NOT
# expand. A bare-token expansion table was tried, as armor for a hypothetical
# punctuation-stripping ASR, and it opened three false-accept paths at once:
# the uppercase acronym ISNT lowercased into the table and "the ISNT rule"
# certified audio that said "is not"; "won't" against a transcript's bare
# "wouldnt" normalized to will not / would not, agreed on the critical "not",
# and a changed auxiliary passed at 0.923 where it had hard-failed; and
# keeping bare "wont"/"cant" out of the table forced them out of the critical
# list, so deleting either word from the audio passed at 0.91. Neither Whisper
# nor Parakeet ever emits the bare spelling — the armor protected against
# nothing and cost three real defect classes. Bare forms are critical tokens
# instead: reachable only when a side genuinely contains that spelling, where
# failing closed is the cheap direction.

# Typographic apostrophes an editor or ASR may emit — the right and left
# single quotation marks (U+2019, U+2018), the modifier letter apostrophe
# (U+02BC) and the fullwidth form (U+FF07). NFC does NOT unify them, and
# expansion regexes matching only the ASCII and U+2019 forms left the others
# unexpanded — the dropped-negation hole reopened through a side door, at the
# same 0.92 it was closed at.
_APOSTROPHES = re.compile(r"[’‘ʼ＇]")


def normalize(text: str, lang: str = "en") -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = _APOSTROPHES.sub("'", text)
    if lang.startswith("en"):
        text = _NT_GENERIC.sub(r"\1 not", text)
        for pattern, expansion in _NT_SPECIAL:
            text = pattern.sub(expansion, text)
    # Every remaining apostrophe is spaced along with the rest of the
    # punctuation, NOT deleted. Deletion was tried and it reached beyond the
    # negation class: "we'll" collapsed into "well" (a changed word scoring
    # 1.0), and "two's" into the non-numberish "twos", which blinded the
    # isolated-numeral hard-fail to two's/ten's complement.
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _plain_digits(token: str) -> bool:
    """ASCII digits only, so a value lookup can never raise.

    str.isdigit() accepts superscripts and other unicode digits that int()
    rejects — coverage("Four.", "²") CRASHED with ValueError mid-verdict
    (gate review), and a verifier crash is worse than any refusal: it takes
    the whole render down instead of failing one chunk closed. Such tokens
    carry no spoken value this round-trip can compare; every value path must
    gate on this, never on bare isdigit()."""
    return token.isascii() and token.isdigit()


def is_numberish(word: str, lang: str = "en") -> bool:
    """A token that carries only numeric value, so it can be ignored.

    Deliberately NOT "contains a digit": that deleted `utf8`, `iso9001`, `rfc822`
    and `base64` from both sides, which are content words in this domain — the
    verifier was blind to whether the audio said them at all. And per-language,
    not pooled — see the _NUMBER_WORDS split: Czech "set" is English content.
    """
    words = _NUMBER_WORDS_CS if lang.startswith("cs") else _NUMBER_WORDS_EN
    return word.isdigit() or word in words


def content_words(text: str, lang: str = "en") -> list[str]:
    return [w for w in normalize(text, lang).split() if not is_numberish(w, lang)]


def _bounded_matches(hyp_words: list[str], needle_words: list[str]) -> list[tuple[int, int]]:
    """Occurrences of `needle_words` in `hyp_words` as (start, end) index spans,
    allowing the words to have merged or split in the transcript but not to
    straddle other words. `_bounded_count` is the length of this list; the spans
    themselves exist so the diagnostics can tell which hypothesis tokens a
    short-sentence rescue consumed, rather than reporting them as insertions."""
    if not needle_words:
        return []
    needle = "".join(needle_words)
    matches: list[tuple[int, int]] = []
    for start in range(len(hyp_words)):
        joined = ""
        for end in range(start, min(start + len(needle_words) + 2, len(hyp_words))):
            joined += hyp_words[end]
            if joined == needle:
                matches.append((start, end + 1))
                break
            if len(joined) > len(needle):
                break
    return matches


def _bounded_count(hyp_words: list[str], needle_words: list[str]) -> int:
    """How many times `needle_words` appears in `hyp_words`, allowing the words to
    have merged or split in the transcript but not to straddle other words."""
    return len(_bounded_matches(hyp_words, needle_words))


@dataclass(frozen=True)
class CoverageDetail:
    """coverage() plus the word-level evidence behind it.

    `word_diagnostics` is an ordered tuple of typed codes — "d:word" (the
    reference word is missing from the transcript), "i:word" (the transcript
    added it), "s:ref/hyp" (one replaced the other) — in alignment order,
    because the shape is the signal: a trailing run of d: is truncation, a
    mass of i: is babble, scattered s: is mispronunciation. A scalar score
    throws exactly that away. Words are reported in normalized, unfolded
    spelling (lowercased, contractions expanded, punctuation stripped,
    numerals blinded out), so a Czech reader sees znemožní, not its fold.

    The codes reflect the FINAL covered state: a word the boundary rescue or
    the short-sentence rescue accepted is not reported, because a rescue
    means the audio was right and a diagnostic that contradicts the score is
    a false alarm. They ride along on every outcome, hard fails included, so
    a critical-token failure still names the dropped word; on the numeral
    paths they are naturally silent — numerals are blinded out of the
    alignment — and the bracketed reason in `worst_sentence` stays the
    diagnostic there.
    """

    score: float
    worst_sentence: str
    word_diagnostics: tuple[str, ...] = ()


def coverage_detail(
    reference: str,
    hypothesis: str,
    lang: str = "en",
    sound_alikes: tuple[tuple[str, str], ...] = (),
) -> CoverageDetail:
    """Worst per-sentence coverage in [0,1], and the sentence that scored it.

    A dropped sentence scores ~0 regardless of how long the surrounding chunk is;
    an ASR spelling quirk costs a word or two inside an otherwise intact sentence.
    That separation is the entire reason this is per-sentence.

    Known limit, stated because it is not obvious: a sentence whose content words
    are ALL numerals cannot be word-aligned this way. Scripts spell numerals out
    because digits are unspeakable, every ASR writes them back as digits, and
    number-blinding — which is what makes the rest of this work — leaves such a
    sentence with nothing to align. What happens then is decided here, never by
    the duration bounds (an earlier version said they were the only guard, and
    then never consulted its own unverifiable list — a dropped all-numeral
    sentence scored 1.0): when the whole reference is such a sentence and its
    numerals are isolated, the VALUES are compared instead — "Four." against
    the transcript's "4" passes, against "9" or nothing fails; everything else
    in the class — a compound like "two fifty six", or an all-numeral sentence
    inside a longer chunk — is refused as unverifiable rather than certified
    unseen.
    """
    # Caller-supplied equivalences — vocabulary, not phonology. A pronunciation
    # lexicon pair IS one by construction: the engine is told to say the spoken
    # form, the ASR writes what it hears, and the script holds the written form.
    # Applied in fold space so general rules and project vocabulary compose;
    # multi-word forms are skipped (the boundary rescue already covers merges).
    alike_pairs = []
    for written, spoken in sound_alikes:
        wf = [fold(w, lang) for w in content_words(written, lang)]
        sf = [fold(w, lang) for w in content_words(spoken, lang)]
        if len(wf) == 1 and len(sf) == 1 and wf[0] != sf[0]:
            alike_pairs.append((wf[0], sf[0]))

    def _fold(word: str) -> str:
        w = fold(word, lang)
        for a, b in alike_pairs:
            w = w.replace(a, b)
        return w

    # The display arrays are index-aligned with the folded ones by
    # construction, so the diagnostics can name real spellings for free.
    ref_display = content_words(reference, lang)
    hyp_display = content_words(hypothesis, lang)
    ref_words = [_fold(w) for w in ref_display]
    hyp_words = [_fold(w) for w in hyp_display]
    if not ref_words:
        # Empty CONTENT is not empty TEXT. A reference whose words were all
        # blinded numerals must hit the same unverifiable fail-closed as the
        # per-sentence loop below — this early return used to score it 1.0
        # against ANY transcript, including the empty one, so the sentence-split
        # fallback laundered "Two fifty six." rendered alone (or dropped
        # entirely) into a clean pass, at exactly the granularity the fallback
        # exists to make failures visible. Found by preflight's identity oracle:
        # a chunk it declared doomed rendered "clean" through this hole.
        ref_tokens = normalize(reference, lang).split()
        if ref_tokens:
            # ...but fail-closed only where there is genuinely nothing to
            # compare. ISOLATED numerals carry comparable value, so "Four."
            # against a transcript's "4" is verified, not unverifiable — the
            # first version of this fix refused that pair, which broke the
            # sentence-split rescue for every short answer sentence a teaching
            # script contains (frontier review, this diff). Comparable means:
            # the transcript has no non-numeral content the script never asked
            # for, and neither side compounds ("two fifty six" vs "256"
            # tokenize as [2,50,6] vs [256] — the ambiguity blinding exists
            # to absorb, still unverifiable here as everywhere).
            hyp_tokens = normalize(hypothesis, lang).split()

            # Comparable also requires every blinded token to CARRY a value.
            # isolated_numerals() silently omits a numberish word missing from
            # _NUMERAL_VALUES, so "Set." (a blind-list word without a value
            # entry at the time) against the empty transcript compared [] == []
            # and verified silence at 1.0 — the laundering this branch exists
            # to close, reopened by a vocabulary gap (counter-review). The
            # values table now covers every blind-list word; this guard turns
            # any future gap into a refusal instead of a vacuous pass.
            def _valued(tokens: list[str]) -> bool:
                return all(
                    _plain_digits(t) or t in _NUMERAL_VALUES
                    for t in tokens
                    if is_numberish(t, lang)
                )

            # Digits compare by VALUE, but only in canonical spelling: "04" is
            # what an ASR writes for digit-by-digit audio ("oh four"), which a
            # single value cannot confirm — int("04") folded it into a pass,
            # so refuse instead (gate review of this change). "-4" is
            # invisible by construction and therefore PASSES: normalize treats
            # the sign as punctuation like everywhere else in this verifier,
            # so it arrives as "4" — recorded, not hidden. Cross-language
            # equivalence follows the blind vocabulary: `not hyp_words` means
            # every transcript token is numberish IN LANG, which for cs
            # deliberately includes English (see _NUMBER_WORDS_CS) — "Sto."
            # against "hundred" passes there by design, while in en a foreign
            # numeral word is content and lands in hyp_words, refusing here.
            def _canonical(tokens: list[str]) -> bool:
                return all(str(int(t)) == t for t in tokens if _plain_digits(t))

            comparable = (
                not hyp_words
                and not has_compound_numeral(ref_tokens, lang)
                and not has_compound_numeral(hyp_tokens, lang)
                and _valued(ref_tokens)
                and _valued(hyp_tokens)
                and _canonical(ref_tokens)
                and _canonical(hyp_tokens)
            )
            if comparable:
                ref_nums = sorted(isolated_numerals(ref_tokens, lang))
                hyp_nums = sorted(isolated_numerals(hyp_tokens, lang))
                if ref_nums == hyp_nums:
                    return CoverageDetail(1.0, "")
                return CoverageDetail(
                    0.0, f"[numeral changed: {ref_nums} became {hyp_nums}]")
            return CoverageDetail(
                0.0, f"[unverifiable, all-numeral] {reference.strip()}")
        return CoverageDetail(1.0, "")

    covered = [False] * len(ref_words)
    hyp_claimed = [False] * len(hyp_words)
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    for i, j, size in matcher.get_matching_blocks():
        for k in range(i, i + size):
            covered[k] = True
        for k in range(j, j + size):
            hyp_claimed[k] = True

    # Diagnostic bookkeeping lives on COPIES, never on the arrays the score
    # reads. Marking `hyp_claimed` for a rescue would change `leftover` below
    # and could flip a short-sentence verdict — the diagnostics must describe
    # the decision, not participate in it.
    hyp_diag_claimed = list(hyp_claimed)

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
    unclaimed_spans: list[tuple[int, int, int]] = []   # (hyp index, start, end)
    offset = 0
    for j, (w, taken) in enumerate(zip(hyp_words, hyp_claimed, strict=True)):
        if not taken:
            unclaimed_spans.append((j, offset, offset + len(w)))
            offset += len(w)
    unclaimed = "".join(hyp_words[j] for j, _, _ in unclaimed_spans)
    consumed: dict[str, int] = {}
    for k, word in enumerate(ref_words):
        if not covered[k] and len(word) > 2 and word in unclaimed:
            covered[k] = True
            # The hypothesis tokens this rescue consumed are accounted for in
            # the diagnostics too: the script's "coworkers" rescued from the
            # transcript's "co worker s" must not then report i:co, i:worker,
            # i:s — a false alarm beside a passing score. Each repeat of the
            # same word consumes the NEXT occurrence: searching from the start
            # every time claimed one span twice, so the second of two rescued
            # co-worker's pairs still surfaced as insertions (found in review).
            at = unclaimed.find(word, consumed.get(word, 0))
            if at == -1:
                at = unclaimed.find(word)
            consumed[word] = at + len(word)
            for j, s, e in unclaimed_spans:
                if s < at + len(word) and e > at:
                    hyp_diag_claimed[j] = True

    worst, worst_sentence, pos = 1.0, "", 0
    unverifiable: list[str] = []
    short_rescued: list[tuple[int, int]] = []   # reference ranges, for diagnostics

    for sentence in split_sentences(reference):
        n = len(content_words(sentence, lang))

        if n == 0:
            # Every content word was a numeral, so number-blinding left nothing to
            # align. IN CONTEXT this is unverifiable even for isolated numerals:
            # the chunk-wide value multiset cannot place a value inside a specific
            # sentence, so "Four." dropped while a 4 survives elsewhere would
            # still balance — only the whole-reference path (early return above),
            # where the sentence stands alone, may compare by value. Compounds
            # are worse: the script says "two fifty six" precisely because digits
            # are unspeakable, the ASR says "256", and no squashed form matches
            # either. Recorded rather than skipped — silently passing it is how
            # a dropped sentence gets through, which this used to do.
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
            # Only a sentence that NEEDS rescuing consumes leftover text. At
            # score 1.0 the assignment below was a no-op, but the diagnostic
            # marking was not: "Go now." against "Go now. Go now." ate the
            # hallucinated duplicate, so the render rejected on precision with
            # word_diagnostics=() — a refusal that named nothing (found in
            # review). Skipping the whole block at 1.0 changes no verdict and
            # lets the duplicate surface as the i: codes it is.
            if score < 1.0:
                leftover_idx = [j for j, claimed in enumerate(hyp_claimed) if not claimed]
                leftover = [hyp_words[j] for j in leftover_idx]
                matches = _bounded_matches(leftover, [_fold(w) for w in content_words(sentence, lang)])
                # `or score > 0` used to turn ANY partial coverage into a pass, so a
                # two-word sentence rendered as one word scored 1.0. Only genuine
                # containment rescues a short sentence now.
                if matches:
                    # A rescued sentence is right, so its words must not surface in
                    # the diagnostics — mark its reference range and the hypothesis
                    # tokens its first match consumed, both on the diagnostic copies.
                    short_rescued.append((pos - n, pos))
                    for j in range(*matches[0]):
                        hyp_diag_claimed[leftover_idx[j]] = True
                    score = 1.0

        if score < worst:
            worst, worst_sentence = score, sentence.strip()

    # The typed codes, computed once and attached to every return below —
    # uniformly, so no path needs its own decision and no path can forget.
    # Walking get_opcodes() on the word matcher adds no alignment cost: the
    # equal opcodes are exactly the matching blocks already consumed above.
    # Every emission is gated on the final diagnostic state, so a rescued word
    # reports nothing. A replace block pairs positionally into s: codes for as
    # long as both sides last; the overhang degrades to d:/i:, which is honest
    # — alignment has no opinion on how a 1:3 replace pairs up.
    covered_diag = list(covered)
    for start, end in short_rescued:
        covered_diag[start:end] = [True] * (end - start)
    codes: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            codes.extend(f"d:{ref_display[k]}" for k in range(i1, i2) if not covered_diag[k])
        elif tag == "insert":
            codes.extend(f"i:{hyp_display[j]}" for j in range(j1, j2) if not hyp_diag_claimed[j])
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for o in range(paired):
                k, j = i1 + o, j1 + o
                if not covered_diag[k] and not hyp_diag_claimed[j]:
                    codes.append(f"s:{ref_display[k]}/{hyp_display[j]}")
                else:
                    if not covered_diag[k]:
                        codes.append(f"d:{ref_display[k]}")
                    if not hyp_diag_claimed[j]:
                        codes.append(f"i:{hyp_display[j]}")
            codes.extend(f"d:{ref_display[k]}" for k in range(i1 + paired, i2) if not covered_diag[k])
            codes.extend(f"i:{hyp_display[j]}" for j in range(j1 + paired, j2) if not hyp_diag_claimed[j])
    word_diagnostics = tuple(codes)

    # An isolated numeral that changed value is a content error, not an ASR
    # spelling difference. Compounds are excluded above, so this cannot fire on
    # "two fifty six" vs "256".
    ref_tokens = normalize(reference, lang).split()
    hyp_tokens = normalize(hypothesis, lang).split()
    # Skip if EITHER side compounds. The check must be symmetric: "two fifty six"
    # is three adjacent numerals in the script and collapses to the single
    # isolated "256" in the transcript, so an asymmetric rule reads a correct
    # transcription as a changed number.
    compound = has_compound_numeral(ref_tokens, lang) or has_compound_numeral(hyp_tokens, lang)
    ref_nums = [] if compound else sorted(isolated_numerals(ref_tokens, lang))
    hyp_nums = [] if compound else sorted(isolated_numerals(hyp_tokens, lang))
    if ref_nums != hyp_nums:
        # Merge rescue, one direction only. The ASR sometimes welds a numeral
        # to its neighbour — "dva z" comes back as "dvaze" — and the welded
        # token is no longer numberish, so the value vanishes from the hyp
        # side of a correct transcription. A value missing from hyp is
        # accounted for if one of its word forms survives inside a
        # non-numberish hyp token. Extra hyp values are never excused, and a
        # numeral that is genuinely gone has no containing token to hide in.
        missing = list(ref_nums)
        for v in hyp_nums:
            if v in missing:
                missing.remove(v)
        extra = list(hyp_nums)
        for v in ref_nums:
            if v in extra:
                extra.remove(v)
        welded = [t for t in hyp_tokens if not is_numberish(t, lang)]
        for v in list(missing):
            # Only the CURRENT language's spellings can hide inside a welded
            # token: the ASR that welded "dva z" into "dvaze" writes Czech, so
            # a Czech render searches Czech forms. Searching every language's
            # forms let the Czech "set" (= 100, briefly valued) rescue a
            # dropped English "hundred" through "set" ⊂ "settings" — correct-
            # looking audio missing its number, accepted at 1.0 (gate review).
            forms = [w for w, val in _NUMERAL_VALUES.items()
                     if val == v and is_numberish(w, lang)]
            if any(f in t for f in forms for t in welded):
                missing.remove(v)
        # Cross-language quoting, the one exception to "extra is never
        # excused". An English script QUOTES a foreign numeral — "The Czech
        # word is čtyři." — the ASR writes the digit, and per-language
        # detection sees a 4 on the hyp side it never saw on the ref side.
        # That is not a changed number: the transcript's value is the value
        # the script asked for, in another language's spelling. The pooled
        # list passed this; the split turned it into a fabricated
        # "[numeral changed: [] became [4]]" (gate review). Two gates keep
        # the excusal from becoming a hole:
        #
        # NON-ASCII only. The value table is keyed by spelling, not language,
        # so an ordinary English word can shadow a Czech form: through "set"
        # (briefly valued 100) the excusal accepted a transcript's "100"
        # where the audio should have said the word "set" — at exactly 0.90
        # (second gate review). No English dictionary exists here to tell
        # quoted-Czech from native-English, but the motivating case is
        # quoted Czech, whose diacritics no English text carries natively:
        # čtyři, pět, tisíc qualify; ASCII spellings two languages could
        # share ("sto", "dva") never excuse and soft-fail as missing words —
        # the fail-closed direction.
        #
        # UNCOVERED only, so the verdict is handed to coverage, which judges
        # honestly: the quoted word's absence dents its sentence — a short
        # sentence still rejects, as missing-word evidence, while a long one
        # absorbs it like any ASR spelling quirk, which is what main did.
        # The covered case stays a hard fail on purpose: a transcript
        # containing BOTH "čtyři" and an extra "4" heard a number the
        # script never asked for. Inert in cs, where the union leaves no
        # content word carrying a numeral value.
        #
        # Known limit, deferred deliberately: a quoted foreign COMPOUND —
        # "dvacet jedna" against the ASR's "21" — still hard-fails, where
        # the pooled list passed it. Composing a value across foreign word
        # sequences is exactly the [2,50,6]-vs-[256] ambiguity that compound
        # suppression exists to avoid, so it stays a refusal until real
        # renders motivate better.
        quoting = [
            t for k, t in enumerate(ref_display)
            if not covered[k] and not t.isascii() and t in _NUMERAL_VALUES
        ]
        for v in list(extra):
            for t in quoting:
                if _NUMERAL_VALUES[t] == v:
                    quoting.remove(t)
                    extra.remove(v)
                    break
        if missing or extra:
            return CoverageDetail(
                0.0, f"[numeral changed: {ref_nums} became {hyp_nums}]", word_diagnostics)

    # A meaning-inverting token that appears or disappears fails outright,
    # regardless of how good the surrounding coverage looks. Tokens the
    # caller's sound_alikes pair with a critical token are CANONICALIZED into
    # it first, never exempted: a pair is the caller declaring two spellings
    # one sound, and the hard fail must not veto that equivalence — the AINT
    # acronym transcribed "aynt" under a caller lexicon hard-failed as a
    # changed negation. But an early version exempted both endpoints from the
    # count, which excused OMISSION as well as substitution: with a
    # ("knot", "not") pair, dropping the grammatical "not" itself passed at
    # 0.93. Mapping the paired spellings into the critical form makes the
    # declared substitution invisible while a drop or insertion still counts.
    # Single-token pairs only, so a multi-word spoken form like "do not enter"
    # cannot touch the protection of "not". Pairs are grouped into equivalence
    # CLASSES first, because alignment composes them — ("AINT","aynt") plus
    # ("aynt","eint") verifies "eint", and mapping only direct partners
    # hard-failed the chained spelling alignment had already accepted. A class
    # holding TWO protected tokens (("cannot","cant")) refuses to map at all:
    # collapsing them would erase a distinction this list exists to keep, so
    # that pair fails closed.
    clusters: list[set[str]] = []
    for written, spoken in sound_alikes:
        wf = normalize(written, lang).split()
        sf = normalize(spoken, lang).split()
        if len(wf) != 1 or len(sf) != 1 or wf[0] == sf[0]:
            continue
        cluster = {wf[0], sf[0]}
        untouched = []
        for group in clusters:
            if group & cluster:
                cluster |= group
            else:
                untouched.append(group)
        clusters = [*untouched, cluster]
    critical_canon: dict[str, str] = {}
    for cluster in clusters:
        protected = sorted(cluster & _CRITICAL_TOKENS)
        if len(protected) == 1:
            critical_canon.update(
                {m: protected[0] for m in cluster if m != protected[0]})
    ref_critical = critical_counts(
        [critical_canon.get(w, w) for w in normalize(reference, lang).split()])
    hyp_critical = critical_counts(
        [critical_canon.get(w, w) for w in normalize(hypothesis, lang).split()])
    if ref_critical != hyp_critical:
        changed = sorted(set(ref_critical) ^ set(hyp_critical)) or sorted(
            w for w in ref_critical if ref_critical[w] != hyp_critical.get(w)
        )
        return CoverageDetail(
            0.0, f"[meaning-critical token changed: {', '.join(changed)}]", word_diagnostics)

    if precision < worst:
        # Something was inserted rather than dropped. Report the whole chunk,
        # since an insertion does not belong to any one reference sentence.
        return CoverageDetail(
            precision, f"[inserted content] {reference.strip()[:70]}", word_diagnostics)

    if unverifiable:
        # Fail closed. These sentences are all-numeral inside a longer chunk,
        # where even the isolated-numeral values cannot be PLACED (see the
        # sentence loop) — the whole-reference by-value path never applies
        # here. The list was previously built and then ignored, which meant a
        # dropped all-numeral sentence scored a clean 1.0 — documenting a blind
        # spot is not the same as refusing to certify what you cannot see. The
        # sentence-split fallback is the recovery: rendered alone, an isolated
        # numeral becomes the whole reference and verifies by value.
        return CoverageDetail(
            0.0, f"[unverifiable, all-numeral] {unverifiable[0]}", word_diagnostics)

    return CoverageDetail(worst, worst_sentence, word_diagnostics)


def coverage(
    reference: str,
    hypothesis: str,
    lang: str = "en",
    sound_alikes: tuple[tuple[str, str], ...] = (),
) -> tuple[float, str]:
    """The (score, worst_sentence) view of `coverage_detail`, kept stable.

    Existing callers — the tests and the bench tools — consume exactly this
    pair; the word-level evidence is additive and lives on CoverageDetail.
    """
    detail = coverage_detail(reference, hypothesis, lang, sound_alikes)
    return detail.score, detail.worst_sentence


def format_word_diagnostics(codes: tuple[str, ...], limit: int = 6) -> str:
    """One compact human line from the typed codes.

    Grouped for reading — "missing: never · inserted: not · substituted:
    key→kay" — while the raw ordered tuple stays on the dataclasses for
    anyone who needs the alignment shape. `limit` bounds each group with a
    +N more marker: a dropped paragraph should read as a headline, not a
    flood of every word it contained.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    groups: dict[str, list[str]] = {"d": [], "i": [], "s": []}
    for code in codes:
        kind, _, rest = code.partition(":")
        if kind in groups:
            groups[kind].append(rest.replace("/", "→", 1) if kind == "s" else rest)
    parts = []
    for kind, label in (("d", "missing"), ("i", "inserted"), ("s", "substituted")):
        words = groups[kind]
        if not words:
            continue
        shown = ", ".join(words[:limit])
        if len(words) > limit:
            shown += f" +{len(words) - limit} more"
        parts.append(f"{label}: {shown}")
    return " · ".join(parts)


@dataclass
class CoverageVerifier:
    """The default verifier: transcribe, then score per-sentence coverage."""

    asr: ASR
    min_coverage: float = MIN_COVERAGE
    sound_alikes: tuple[tuple[str, str], ...] = ()

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        transcript = self.asr.transcribe(audio, lang)
        detail = coverage_detail(text, transcript, lang, self.sound_alikes)
        ok = detail.score >= self.min_coverage
        # Diagnostics are gated exactly like dropped_sentence: a passing chunk
        # with tolerated ASR spelling quirks must not print alarming codes.
        return Verdict(
            ok=ok,
            coverage=detail.score,
            dropped_sentence="" if ok else detail.worst_sentence,
            transcript=transcript,
            word_diagnostics=() if ok else detail.word_diagnostics,
        )


@dataclass
class CascadeVerifier:
    """Accept when ANY verifier confirms the text; escalate only on rejection.

    A recogniser never sees the script, so a transcript that independently
    matches it is strong evidence the audio is right — no matter which model
    produced it. Requiring every recogniser to fail before rejecting therefore
    removes each model's idiosyncratic misreadings without weakening drop
    detection: defective audio doesn't transcribe into the correct script by
    accident.

    Measured on 82 real Czech chunks (bench/asr_headtohead.py): ~10% of
    single-model rejections were solo — the other recogniser read the same audio
    as correct — and every rejection costs up to three re-synthesis attempts
    plus a sentence-split fallback, each vastly more expensive than one extra
    ASR pass. Order verifiers fastest-first: later ones run only when earlier
    ones reject, so the escalation is nearly free in the common case.

    On total failure the verdict with the best coverage is returned, so the
    retry ladder ranks attempts the same way it would with one verifier. Two
    consequences of that, named because they are deliberate: a hard-fail 0.0
    (changed numeral, critical token) can be superseded in the REPORTED verdict
    by a sibling's higher soft score — accept/reject is unaffected, only the
    diagnostic and the ranking of already-failed attempts. The word_diagnostics
    ride inside the verdict, so they follow the same choice: the reported codes
    are the best-scoring sibling's, consistent with the sentence it names. And accepting on any
    single pass makes the false-accept rate the union of the members' — the
    price of removing their idiosyncratic false rejections. Requiring
    concurrence instead would re-buy the measured false-rejection class and
    double the ASR cost of every chunk; a recogniser that never saw the script
    transcribing it back is evidence enough.

    A verifier that RAISES (model download failed, backend broke mid-render) is
    skipped, not fatal — otherwise the fallback this class promises would be
    defeated by exactly the situations that need it. Only when every verifier
    errors is there nothing to report, and the last error propagates.
    """

    verifiers: list[Verifier]  # ordered fastest-first

    def __post_init__(self) -> None:
        if not self.verifiers:
            raise ValueError("CascadeVerifier needs at least one verifier")

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        best: Verdict | None = None
        error: Exception | None = None
        for verifier in self.verifiers:
            try:
                verdict = verifier.verify(audio, text, lang)
            except Exception as exc:
                error = exc
                continue
            if verdict.ok:
                return verdict
            if best is None or verdict.coverage > best.coverage:
                best = verdict
        if best is None:
            raise RuntimeError("every verifier in the cascade errored") from error
        return best


def default_verifier(
    source_rate: int, sound_alikes: tuple[tuple[str, str], ...] = ()
) -> Verifier:
    """The verification stack render entry points should use.

    One policy, owned here rather than assembled by every caller — the cascade
    ordering is derived from this library's own bench and callers were already
    diverging on it (one forgot `source_rate`, which silently corrupts every
    verdict). Parakeet-first when the `[parakeet]` extra is installed, plain
    Whisper otherwise.
    """
    from narrator.asr import WhisperASR

    whisper = CoverageVerifier(WhisperASR(source_rate=source_rate), sound_alikes=sound_alikes)
    import importlib.util
    if importlib.util.find_spec("parakeet_mlx") is None:
        return whisper
    from narrator.asr import ParakeetASR

    return CascadeVerifier([
        CoverageVerifier(ParakeetASR(source_rate=source_rate), sound_alikes=sound_alikes),
        whisper,
    ])


@dataclass
class NullVerifier:
    """Accepts everything. For callers who want speed over safety, explicitly.

    Named rather than implied: the predecessor's `--no-asr` flag silently made
    retries useless, because a skipped check returned a perfect score that no
    later attempt could beat.
    """

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        return Verdict(ok=True, coverage=1.0)
