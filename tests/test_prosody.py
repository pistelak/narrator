"""Rise-selection and prosody tests.

The selection contract under test (bench/RESULTS.md §11): yes/no rises are
stochastic (~60% of verified takes), so the ladder may prefer a rising
verified take — but prosody is a preference, never a gate. Every test here
pins one edge of that boundary: a rise can never rescue an unverified take,
a missing rise can never fail a verified chunk, statements pay nothing, and
attempt accounting stays truthful when selection breaks the "first success
returns immediately" identity.

Selection tests run on the fake backend with a stub checker keyed off the
fake's audio stamp, injected by monkeypatching prosody.rise_delta_checker —
fake audio is a tone, and asking a real F0 tracker to hear intonation in it
would test nothing. The DSP itself is tested on synthetic glides behind
importorskip(librosa), so the fast suite stays model-free.
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


def run(monkeypatch, text: str, deltas: dict[int, float], script=None, calls=None,
        wants_rise=lambda text, lang: text.rstrip().endswith("?"), threshold=None):
    import narrator.prosody

    checker = _checker(deltas, calls)
    monkeypatch.setattr(narrator.prosody, "rise_delta_checker", lambda: checker)
    backend = FakeBackend(script=script or {})
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = (SynthConfig(wants_rise=wants_rise) if threshold is None
           else SynthConfig(wants_rise=wants_rise, rise_threshold_st=threshold))
    return synthesize_chunk(text, 0, backend, verifier, VOICE, cfg), backend


# ----------------------------------------------------------- selection

def test_later_rising_take_beats_earlier_flat_one(monkeypatch) -> None:
    result, _ = run(monkeypatch, QUESTION, deltas={0: 0.5, 1: 3.0})
    assert result.ok
    assert _stamp_index(result.audio) == 1
    assert result.attempts == 2
    # No failure was recovered — take 1 VERIFIED, the ladder just kept
    # looking. Reporting "retry" here would misreport a healthy chunk.
    assert result.recovered_by == ""


def test_search_failures_after_verification_are_not_recoveries(monkeypatch) -> None:
    """verified-flat -> failed -> verified-rise must NOT report "retry".

    Success was secured on attempt 1; the failure happened during the
    optional rise search. The frontier merge review reproduced the misreport
    before this pin existed: provenance must come from the first verified
    take, not from whatever the shipped take saw."""
    result, backend = run(monkeypatch, QUESTION, deltas={0: 0.5, 2: 3.0},
                          script={1: Failure.TRUNCATE})
    assert result.ok
    assert _stamp_index(result.audio) == 2, "the rising take ships"
    assert result.attempts == 3
    assert backend.calls == 3
    assert result.recovered_by == "", "search failures are not recoveries"


def test_exhausted_search_ships_first_verified_without_recovery(monkeypatch) -> None:
    """verified-flat -> failed -> failed -> budget exhausted: the first
    verified take ships via the end-of-loop path with its own clean
    provenance — search failures after it are not recoveries there either."""
    result, backend = run(monkeypatch, QUESTION, deltas={0: 0.5},
                          script={1: Failure.TRUNCATE, 2: Failure.TRUNCATE})
    assert result.ok
    assert _stamp_index(result.audio) == 0
    assert result.attempts == 3
    assert backend.calls == 3
    assert result.recovered_by == ""


def test_threshold_is_config_not_constant(monkeypatch) -> None:
    """§11.5 names threshold softening as the next config-driven experiment;
    the knob must actually be live."""
    result, _ = run(monkeypatch, QUESTION, deltas={0: 2.0, 1: 2.6}, threshold=2.5)
    assert result.ok
    assert _stamp_index(result.audio) == 1, "2.0 st is below a 2.5 st threshold"
    assert result.attempts == 2


def test_all_flat_ships_first_verified_take(monkeypatch) -> None:
    """Prosody is a preference, never a gate.

    Selecting the largest sub-threshold delta is unmeasured, so the
    conservative fallback is the FIRST verified take (here index 0, even
    though take 2 has the bigger delta)."""
    result, _ = run(monkeypatch, QUESTION, deltas={0: 0.5, 1: 1.2, 2: 0.9})
    assert result.ok
    assert _stamp_index(result.audio) == 0
    assert result.attempts == 3, "cost of the search must be reported truthfully"
    assert result.recovered_by == ""


def test_statements_pay_nothing(monkeypatch) -> None:
    calls: list[int] = []
    result, backend = run(monkeypatch, STATEMENT, deltas={0: 5.0}, calls=calls)
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1, "no extra generations for a non-rising chunk"
    assert calls == [], "the checker must never run on a statement"


def test_checker_never_runs_on_failed_attempts(monkeypatch) -> None:
    """F0 runs only on verified takes — and a rise can never rescue a failed
    one. The first attempt truncates (fails the duration floor before
    verification is even consulted); a high delta for it must be unreachable."""
    calls: list[int] = []
    result, _ = run(monkeypatch, QUESTION, deltas={0: 9.0, 1: 3.0},
                    script={0: Failure.TRUNCATE}, calls=calls)
    assert result.ok
    assert _stamp_index(result.audio) == 1
    assert calls == [1], "checker consulted for the verified take only"
    assert result.recovered_by == "retry", "a real failure WAS recovered here"
    assert result.attempts == 2


def test_unmeasurable_contour_ships_immediately(monkeypatch) -> None:
    """None from the checker means unmeasurable, not flat — and shipping
    immediately is a COST policy, not a claim about the text: measurability
    is per-take stochastic (a real "Máš teď chvilku?" measured on two of
    three takes), but chasing a measurable contour has unknown payoff and
    None also covers a broken analysis, where retrying buys nothing."""
    result, backend = run(monkeypatch, QUESTION, deltas={})  # checker returns None
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1


def test_raising_checker_reads_as_no_preference(monkeypatch) -> None:
    import narrator.prosody

    def exploding(audio, sample_rate):
        raise RuntimeError("pyin fell over")

    monkeypatch.setattr(narrator.prosody, "rise_delta_checker", lambda: exploding)
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=lambda t, lang: True)
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, cfg)
    assert result.ok, "prosody must never be able to fail a verified chunk"
    assert result.attempts == 1


def test_raising_intent_policy_reads_as_no_intent(monkeypatch) -> None:
    import narrator.prosody

    def exploding_intent(text, lang):
        raise ValueError("caller bug")

    calls: list[int] = []
    monkeypatch.setattr(narrator.prosody, "rise_delta_checker",
                        lambda: _checker({}, calls))
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=exploding_intent)
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, cfg)
    assert result.ok
    assert result.attempts == 1
    assert calls == []


def test_broken_checker_resolution_reads_as_no_preference(monkeypatch) -> None:
    """A broken librosa install raises whatever it raises at import time —
    not ImportError — and resolution happens before the backend is called.
    Left unguarded, enabling the feature aborted renders that synthesized
    fine without it (Codex review finding, 2026-08-17)."""
    import narrator.prosody

    def exploding_resolver():
        raise AttributeError("broken librosa install")

    monkeypatch.setattr(narrator.prosody, "rise_delta_checker", exploding_resolver)
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = SynthConfig(wants_rise=lambda t, lang: True)
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, cfg)
    assert result.ok
    assert result.attempts == 1


def test_failed_split_reports_every_generation_paid_for() -> None:
    """Six real calls must not read as three: the failed sentence-split's
    generations were spent, and ChunkResult.attempts is the cost record."""
    two_sentences = "Not the keeper of anything. Not a stranger to anyone here."
    backend = FakeBackend(default=Failure.TRUNCATE)
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(two_sentences, 0, backend, verifier, VOICE, SynthConfig())
    assert not result.ok
    assert result.attempts == backend.calls


def test_default_config_is_byte_for_byte_todays_behavior() -> None:
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(QUESTION, 0, backend, verifier, VOICE, SynthConfig())
    assert result.ok
    assert result.attempts == 1
    assert backend.calls == 1


def test_wh_question_with_default_policy_is_not_reranked(monkeypatch) -> None:
    """The finding that reversed the v1 design: wh-questions end in `?` and
    measurably go DOWN (31/32 verified takes, bench §11/§11.7). The default
    policy must leave
    them alone."""
    calls: list[int] = []
    result, backend = run(monkeypatch, "Where did you leave the documentation?",
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
    # Non-fronted wh — the reason the scan covers every token, not a
    # prefix window: True here would demand a rise on a canonical fall.
    ("A kdy přijdeš?", "cs", False),
    ("V čem je problém?", "cs", False),
    ("Kterému člověku jsi to dal?", "cs", False),
    ("And just why did it fail?", "en", False),
    # Embedded-wh yes/no question — a deliberately accepted false negative
    # (safe direction: it merely keeps today's behavior).
    ("Víš, kdy přijde?", "cs", False),
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
