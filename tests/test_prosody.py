"""Rise-selection and prosody tests.

The selection contract under test (bench/RESULTS.md §11): yes/no rises are
stochastic (~60% of verified takes), so the ladder may prefer a rising
verified take — but prosody is a preference, never a gate. Every test here
pins one edge of that boundary: a rise can never rescue an unverified take,
a missing rise can never fail a verified chunk, statements pay nothing, and
attempt accounting stays truthful when selection breaks the "first success
returns immediately" identity.

Selection tests run on the fake backend with an injected checker keyed off
the fake's audio stamp — fake audio is a tone, and asking a real F0 tracker
to hear intonation in it would test nothing. The DSP itself is tested on
synthetic glides behind importorskip(librosa), so the fast suite stays
model-free and dependency-light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from narrator.backends.fake import Failure, FakeASR, FakeBackend
from narrator.prosody import terminal_delta_st, yes_no_question
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.types import Voice
from narrator.verify import CoverageVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
QUESTION = "Are you coming to the meeting tomorrow?"
STATEMENT = "You are coming to the meeting tomorrow."


def _stamp_index(audio) -> int:
    """Recover the fake backend's call index from its audio stamp."""
    return round(float(audio[0]) / 1e-4) - 1


def _checker(deltas: dict[int, float], calls: list[int] | None = None):
    """Injected rise check: per-call-index deltas, keyed off the stamp."""
    def check(audio, sample_rate):
        index = _stamp_index(audio)
        if calls is not None:
            calls.append(index)
        return deltas.get(index)
    return check


def run(text: str, deltas: dict[int, float], script=None, calls=None,
        wants_rise=lambda text, lang: text.rstrip().endswith("?")):
    backend = FakeBackend(script=script or {})
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=wants_rise, rise_check=_checker(deltas, calls))
    return synthesize_chunk(text, 0, backend, verifier, VOICE, cfg), backend


# ----------------------------------------------------------- selection

def test_later_rising_take_beats_earlier_flat_one() -> None:
    result, _ = run(QUESTION, deltas={0: 0.5, 1: 3.0})
    assert result.ok
    assert _stamp_index(result.audio) == 1
    assert result.attempts == 2
    # No failure was recovered — take 1 VERIFIED, the ladder just kept
    # looking. Reporting "retry" here would misreport a healthy chunk.
    assert result.recovered_by == ""


def test_all_flat_ships_first_verified_take() -> None:
    """Prosody is a preference, never a gate.

    Selecting the largest sub-threshold delta is unmeasured, so the
    conservative fallback is the FIRST verified take (here index 0, even
    though take 2 has the bigger delta)."""
    result, _ = run(QUESTION, deltas={0: 0.5, 1: 1.2, 2: 0.9})
    assert result.ok
    assert _stamp_index(result.audio) == 0
    assert result.attempts == 3, "cost of the search must be reported truthfully"
    assert result.recovered_by == ""


def test_statements_pay_nothing() -> None:
    calls: list[int] = []
    result, backend = run(STATEMENT, deltas={0: 5.0}, calls=calls)
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1, "no extra generations for a non-rising chunk"
    assert calls == [], "the checker must never run on a statement"


def test_checker_never_runs_on_failed_attempts() -> None:
    """F0 runs only on verified takes — and a rise can never rescue a failed
    one. The first attempt truncates (fails the duration floor before
    verification is even consulted); a high delta for it must be unreachable."""
    calls: list[int] = []
    result, _ = run(QUESTION, deltas={0: 9.0, 1: 3.0},
                    script={0: Failure.TRUNCATE}, calls=calls)
    assert result.ok
    assert _stamp_index(result.audio) == 1
    assert calls == [1], "checker consulted for the verified take only"
    assert result.recovered_by == "retry", "a real failure WAS recovered here"
    assert result.attempts == 2


def test_unmeasurable_contour_ships_immediately() -> None:
    """None from the checker means unmeasurable, not flat: a text too short
    to measure is too short on every take, so burning the remaining budget
    buys nothing. This distinction cost the probe a real verified take
    ("Máš teď chvilku?") before it was made explicit."""
    result, backend = run(QUESTION, deltas={})  # checker returns None
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1


def test_raising_checker_reads_as_no_preference() -> None:
    def exploding(audio, sample_rate):
        raise RuntimeError("pyin fell over")

    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=lambda t, lang: True, rise_check=exploding)
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, cfg)
    assert result.ok, "prosody must never be able to fail a verified chunk"
    assert result.attempts == 1


def test_raising_intent_policy_reads_as_no_intent() -> None:
    def exploding_intent(text, lang):
        raise ValueError("caller bug")

    calls: list[int] = []
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=exploding_intent, rise_check=_checker({}, calls))
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, cfg)
    assert result.ok
    assert result.attempts == 1
    assert calls == []


def test_default_config_is_byte_for_byte_todays_behavior() -> None:
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, SynthConfig())
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1


def test_wh_question_with_default_policy_is_not_reranked() -> None:
    """The finding that reversed the v1 design: wh-questions end in `?` and
    canonically FALL (97% correct, bench §11). The default policy must leave
    them alone."""
    calls: list[int] = []
    result, backend = run("Where did you leave the documentation?",
                          deltas={0: 0.0}, calls=calls,
                          wants_rise=yes_no_question)
    assert result.ok
    assert backend.calls == 1
    assert calls == [], "wh-question must not trigger rise selection"


# ------------------------------------------------------ intent heuristic

@pytest.mark.parametrize("text,lang,expected", [
    ("Are you coming to the meeting tomorrow?", "en", True),
    ("Where did you leave the documentation?", "en", False),
    ("Přijdeš zítra na tu schůzku?", "cs", True),
    ("Kdy přijdeš zítra na tu schůzku?", "cs", False),
    ("Kolik lidí přišlo na tu přednášku?", "cs", False),
    # Conjunction-fronted wh — the reason the window is two tokens.
    ("A kdy přijdeš?", "cs", False),
    # Trailing closer after the question mark (chunking's closer class).
    ("„Máš teď chvilku?“", "cs", True),
    ("You are coming to the meeting tomorrow.", "en", False),
    # Unknown language: no basis for demanding a contour.
    ("Kommst du morgen?", "de", False),
])
def test_yes_no_question_policy(text: str, lang: str, expected: bool) -> None:
    assert yes_no_question(text, lang) is expected


# ------------------------------------------------------------------ DSP

def _harmonic_glide(sample_rate: int, f_start: float, f_end: float,
                    duration_s: float = 2.0, glide_s: float = 0.3) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    freq = np.full(n, float(f_start))
    glide_start = duration_s - glide_s
    in_glide = t >= glide_start
    freq[in_glide] = f_start + (f_end - f_start) * (t[in_glide] - glide_start) / glide_s
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    signal = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)


def test_terminal_delta_classifies_glides() -> None:
    pytest.importorskip("librosa")
    sr = 24_000
    assert terminal_delta_st(_harmonic_glide(sr, 200, 300), sr) >= 1.5
    assert terminal_delta_st(_harmonic_glide(sr, 200, 130), sr) <= -1.5
    assert abs(terminal_delta_st(_harmonic_glide(sr, 200, 200), sr)) < 1.5


def test_terminal_delta_is_total() -> None:
    pytest.importorskip("librosa")
    sr = 24_000
    assert terminal_delta_st(np.zeros(sr, dtype=np.float32), sr) is None
    assert terminal_delta_st(np.zeros(0, dtype=np.float32), sr) is None
    assert terminal_delta_st(_harmonic_glide(sr, 200, 300, duration_s=0.3), sr) is None


def test_terminal_delta_sample_rate_parity() -> None:
    """The constants were measured at 24 kHz but the metric must classify
    identically at Supertonic's 44.1 kHz — frame/hop derive from durations,
    not sample counts, precisely so this holds."""
    pytest.importorskip("librosa")
    d24 = terminal_delta_st(_harmonic_glide(24_000, 200, 300), 24_000)
    d44 = terminal_delta_st(_harmonic_glide(44_100, 200, 300), 44_100)
    assert d24 is not None and d44 is not None
    assert d24 >= 1.5 and d44 >= 1.5
    assert abs(d24 - d44) < 0.75, "same signal, same measurement"
