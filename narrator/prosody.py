"""Terminal F0 contour measurement and question-intent helpers.

The measured story (bench/RESULTS.md §11): Higgs v3 ends 31/32 verified
wh/declarative takes going downward (§11.7 showed that is declination, not a
conclusive terminal event — but downward either way, so they must never be
rise-targeted), while yes/no question rises are stochastic — ~60% of verified
takes rise, the same text flips contour between takes, and reference-clip
engineering plateaus around 67%. The retry ladder in
synth.py already carries a budget of three attempts per chunk (returning at
the first verified take); this module supplies the contour signal that lets
it keep spending that budget in search of a rising verified take when the
CALLER says the text should rise.

Intent is the caller's, not punctuation's: wh-questions end in `?` too and
canonically FALL — a punctuation trigger would demand rises on the one
category that is already right. `yes_no_question` below is the offered
default policy; it is a heuristic, and callers with better knowledge of their
scripts should supply their own.

Everything here is total: `terminal_delta_st` never raises — degenerate or
unmeasurable audio yields None, which downstream means "no preference", never
a failure. Prosody must never be able to fail a verified chunk.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np

from narrator.audio import trim_silence
from narrator.types import Audio

# 60-450 Hz spans male and female speaking range without leaving octave errors
# much room. Window and hop are DURATIONS, not sample counts: the constants
# were measured at 24 kHz (Higgs) but the metric must classify identically at
# any backend rate — Supertonic declares 44.1 kHz — so frame lengths derive
# from these at call time. 85 ms is at least two periods of 60 Hz, pyin's
# practical floor; a 10 ms hop makes voiced-frame counts read as centiseconds.
FMIN_HZ = 60.0
FMAX_HZ = 450.0
FRAME_S = 0.085
HOP_S = 0.010
# Windows count VOICED frames, sidestepping unvoiced gaps. Below 60 accepted
# frames (600 ms of voicing) the contour is unmeasurable — the honest answer
# for short or creak-devoiced endings (a real, fully verified "Máš teď
# chvilku?" take was exactly this), not a forced class. Tail = last 300 ms of
# voicing; head = up to 500 ms preceding it (≥300 ms by construction at the
# floor, so the delta is defined for everything that clears it).
MIN_VOICED_FRAMES = 60
TAIL_FRAMES = 30
HEAD_FRAMES = 50
# Frames >8 st from the utterance median are halving/doubling errors or
# creak; genuine question rises stay within ~6 st of the median.
OCTAVE_GUARD_ST = 8.0
# Intonational contrasts become reliably perceptible around 1.5-2 st; the
# measured rises span 3-6 st and declination over 300 ms is ~0.1 st.
# Confirmed by an operator listening pass (2026-08-17): a delta-sorted
# ladder of takes from +1.4 to +5.2 st all read as acceptable questions,
# so the boundary stays where the literature put it. (Selection ships the
# FIRST verified take when nothing clears it, so a stricter threshold
# would cost attempts, never correctness.)
RISE_THRESHOLD_ST = 1.5


def voiced_f0(audio: Audio, sample_rate: int) -> np.ndarray:
    """Hz values of accepted voiced frames, time-ordered, one per 10 ms hop.

    Requires librosa (the `[higgs]` extra); callers that must not raise go
    through `terminal_delta_st` or gate on `rise_delta_checker()`.
    """
    import librosa

    f0, voiced, _prob = librosa.pyin(
        np.asarray(audio, dtype=np.float64),
        fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sample_rate,
        frame_length=round(FRAME_S * sample_rate),
        hop_length=round(HOP_S * sample_rate),
    )
    keep = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    values = f0[keep]
    if values.size == 0:
        return values
    deviation = np.abs(12.0 * np.log2(values / np.median(values)))
    return values[deviation <= OCTAVE_GUARD_ST]


def delta_from_f0(f0: np.ndarray) -> float | None:
    """Terminal delta in semitones over the voiced-frame windows, or None."""
    n = int(f0.size)
    if n < MIN_VOICED_FRAMES:
        return None
    tail = f0[-TAIL_FRAMES:]
    # Negative slice start clamps at the array head, so the head window is
    # "up to HEAD_FRAMES, at least MIN_VOICED_FRAMES - TAIL_FRAMES".
    head = f0[-(TAIL_FRAMES + HEAD_FRAMES):-TAIL_FRAMES]
    # Medians, never means: residual octave/creak outliers must not drag it.
    return 12.0 * math.log2(float(np.median(tail)) / float(np.median(head)))


def terminal_delta_st(audio: Audio, sample_rate: int) -> float | None:
    """Total, non-throwing terminal contour: semitone delta or None.

    None means unmeasurable (short voicing, degenerate audio, any analysis
    failure) — never an error. `trim_silence` returns all-silent input
    unchanged rather than empty, and pure silence simply yields zero voiced
    frames, so both land in the None path without an exception.
    """
    try:
        trimmed = trim_silence(np.asarray(audio, dtype=np.float32), sample_rate)
        return delta_from_f0(voiced_f0(trimmed, sample_rate))
    except Exception:
        return None


def rise_delta_checker() -> Callable[[Audio, int], float | None] | None:
    """The contour checker, or None when librosa is not installed.

    None disables the rise preference entirely — installing an extra may
    enable a feature the caller asked for, but the caller asks explicitly
    via SynthConfig.wants_rise, so environment never silently changes output.
    """
    try:
        import librosa  # noqa: F401
    except ImportError:
        return None
    return terminal_delta_st


# Wh-words per language — general linguistic data, the same standing as the
# letter-name and numeral tables (no project vocabulary). Inflected Czech
# forms included because questions open with any case of the pronoun.
_WH_WORDS = {
    "en": {
        "who", "whom", "whose", "what", "which", "when", "where", "why", "how",
    },
    "cs": {
        "kdo", "koho", "komu", "kom", "kým", "co", "čeho", "čemu", "čem", "čím",
        "který", "která", "které", "kteří", "kterou", "kterého", "kterému",
        "kterém", "kterým", "kterých", "kterými",
        "kdy", "kde", "kam", "odkud", "kudy", "proč", "nač",
        "jak", "jaký", "jaká", "jaké", "jakou", "jakého", "jakém", "jakým",
        "jakých", "jakými",
        "kolik", "kolika", "kolikrát", "čí",
    },
}

# `?` possibly followed by closing quotes/brackets — chunking's _SENTENCE_END
# closer class plus the Czech closing quotes (U+201C double, U+2018 single),
# which the English-centric class misses; a Czech quoted question must not
# lose its intent to its own quotation marks.
_FINAL_QUESTION = re.compile(r'\?["”“’‘\')\]»]*$')
_WORD = re.compile(r"\w+", re.UNICODE)


def yes_no_question(text: str, lang: str) -> bool:
    """Heuristic rising-intent policy: `?`-final and containing no wh-word.

    ANY wh-word anywhere in the sentence disqualifies, not just a fronted
    one, because the error directions are asymmetric: a yes/no question
    misread as wh merely keeps today's behavior (an embedded-wh question
    like "Víš, kdy přijde?" is a deliberately accepted false negative),
    while a wh-question misread as yes/no — "A kdy přijdeš?", "V čem je
    problém?", "And just why did it fail?" — would demand a rise where the
    measured contour goes downward (31/32 takes, bench/RESULTS.md §11/§11.7).
    Callers with real script knowledge should pass their own policy.
    """
    if not _FINAL_QUESTION.search(text.strip()):
        return False
    wh = _WH_WORDS.get(lang.split("-")[0])
    if wh is None:
        return False  # unknown language: no basis for demanding a contour
    words = [w.lower() for w in _WORD.findall(text)]
    return bool(words) and not any(w in wh for w in words)
