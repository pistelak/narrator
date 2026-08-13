"""Tests for the test double.

A fake that does not reproduce the real failure modes is worse than no fake: it
turns green into false confidence. So the double gets its own tests, and each one
pins a failure to the observed behaviour it imitates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from narrator.backends.fake import FakeASR, FakeBackend, Failure
from narrator.types import Voice
from narrator.verify import CoverageVerifier, coverage

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
TEXT = "Not the keeper. Not a stranger. Not any council with any mandate."


def build(script: dict[int, Failure] | None = None, **kw) -> tuple[FakeBackend, FakeASR]:
    backend = FakeBackend(script=script or {}, **kw)
    return backend, FakeASR(backend)


def synth(backend: FakeBackend, text: str = TEXT, max_frames: int = 10_000):
    return backend.synthesize(text, VOICE, max_frames=max_frames, temperature=0.4)


def test_clean_synthesis_round_trips_exactly() -> None:
    backend, asr = build()
    audio = synth(backend)
    assert asr.transcribe(audio, "en") == TEXT
    assert coverage(TEXT, asr.transcribe(audio, "en"))[0] == 1.0


def test_duration_is_proportional_to_what_was_said() -> None:
    backend, _ = build()
    short = synth(backend, "Three words here.")
    long = synth(backend, " ".join(["word"] * 60) + ".")
    assert len(long) > len(short) * 5


# ------------------------------------------------- each failure is detectable

def test_dropped_sentence_is_detected_by_the_real_verifier() -> None:
    backend, asr = build({0: Failure.DROP_SENTENCE})
    verdict = CoverageVerifier(asr).verify(synth(backend), TEXT, "en")
    assert not verdict.ok
    assert verdict.dropped_sentence, "must name what went wrong"


def test_repetition_loop_is_detected() -> None:
    backend, asr = build({0: Failure.REPEAT})
    verdict = CoverageVerifier(asr).verify(synth(backend), TEXT, "en")
    assert not verdict.ok


def test_truncation_is_detected() -> None:
    backend, asr = build({0: Failure.TRUNCATE})
    verdict = CoverageVerifier(asr).verify(synth(backend), TEXT, "en")
    assert not verdict.ok


def test_runaway_is_bounded_by_the_frame_cap() -> None:
    """A runaway must be bounded, or one bad chunk costs minutes of compute.

    Measured on the real engine: a babbling chunk ran 244 s against ~26 s
    expected and burned 197 s of render time, because the cap was 12x too loose.
    """
    backend, asr = build({0: Failure.RUNAWAY})
    audio = synth(backend, TEXT, max_frames=250)  # 10 s at 25 fps
    assert len(audio) / backend.sample_rate == pytest.approx(10.0, abs=0.01)
    assert not CoverageVerifier(asr).verify(audio, TEXT, "en").ok


def test_engine_exception_propagates() -> None:
    backend, _ = build({0: Failure.RAISE})
    with pytest.raises(RuntimeError):
        synth(backend)


# --------------------------------------------------- scripting across calls

def test_script_applies_per_call_so_retries_can_be_exercised() -> None:
    backend, asr = build({0: Failure.DROP_SENTENCE, 1: Failure.REPEAT})
    verifier = CoverageVerifier(asr)
    assert not verifier.verify(synth(backend), TEXT, "en").ok   # call 0
    assert not verifier.verify(synth(backend), TEXT, "en").ok   # call 1
    assert verifier.verify(synth(backend), TEXT, "en").ok       # call 2, unscripted
    assert backend.calls == 3


def test_backend_records_requests_and_caps() -> None:
    backend, _ = build()
    synth(backend, "One.", max_frames=100)
    synth(backend, "Two.", max_frames=200)
    assert backend.requests == ["One.", "Two."]
    assert backend.max_frames_seen == [100, 200]


# ------------------------------------------- the ASR's harmless disagreements

def test_imperfect_asr_disagreements_do_not_read_as_drops() -> None:
    """The verifier must tolerate what a real ASR actually does to correct audio."""
    backend = FakeBackend()
    asr = FakeASR(backend, perfect=False)
    text = "He copies the twenty byte code. Což třídění ztíží. Ne znemožní."
    audio = backend.synthesize(text, VOICE, max_frames=10_000, temperature=0.4)
    transcript = asr.transcribe(audio, "cs")
    assert transcript != text, "imperfect ASR should have altered something"
    assert coverage(text, transcript)[0] == 1.0
