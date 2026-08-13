"""Retry-ladder tests.

Both defects the predecessor shipped lived on paths that only execute when
something has already gone wrong, which is why a clean fourteen-minute render
sailed straight over one of them. Those paths get the most tests here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from narrator.backends.fake import FakeASR, FakeBackend, Failure
from narrator.synth import SynthConfig, duration_bounds, frame_cap, synthesize_chunk
from narrator.types import Voice
from narrator.verify import CoverageVerifier, NullVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
TEXT = "Not the keeper. Not a stranger. Not any council with any mandate."
CFG = SynthConfig()


def run(script=None, text=TEXT, cfg=CFG, perfect=True):
    backend = FakeBackend(script=script or {})
    verifier = CoverageVerifier(FakeASR(backend, perfect=perfect))
    return synthesize_chunk(text, 0, backend, verifier, VOICE, cfg), backend


# ------------------------------------------------------------- frame cap

def test_frame_cap_has_absolute_headroom_for_short_utterances() -> None:
    """The bug that broke the sentence-split fallback.

    "Jen firmu." is two words: 0.8 s expected. A pure 1.6x multiplier gives
    1.28 s, less than the real leading and trailing silence, so every short
    sentence was truncated by construction — and short sentences are exactly
    what the fallback produces.
    """
    fps = 25
    assert frame_cap(2, fps, CFG) / fps >= 4.0
    pure_multiplier = (2 / CFG.words_per_second) * CFG.frame_headroom
    assert pure_multiplier < 1.3, "the arithmetic that caused the bug"


def test_hitting_the_frame_cap_is_itself_a_failure() -> None:
    """The cap must detect a runaway, not merely bound its cost.

    With these constants the cap always lands below the duration ceiling, so a
    capped runaway passes the ceiling check. Before this was its own signal, a
    NullVerifier run accepted four minutes of babble as a valid chunk.
    """
    backend = FakeBackend(script={0: Failure.RUNAWAY})
    result = synthesize_chunk(TEXT, 0, backend, NullVerifier(), VOICE,
                              SynthConfig(max_attempts=1, allow_sentence_split=False))
    assert not result.ok


def test_frame_cap_still_scales_for_long_chunks() -> None:
    fps = 25
    assert frame_cap(40, fps, CFG) / fps == pytest.approx(16 * 1.6 + 2.0)


def test_frame_cap_is_passed_to_the_backend() -> None:
    _, backend = run()
    assert backend.max_frames_seen[0] == frame_cap(len(TEXT.split()), 25, CFG)


def test_duration_bounds_bracket_the_expected() -> None:
    floor, ceiling = duration_bounds(40, CFG)
    assert floor < 40 / CFG.words_per_second < ceiling


# ------------------------------------------------- the ranking bug (critical)

def test_a_passing_retry_is_never_discarded_for_a_failed_first_attempt() -> None:
    """The predecessor's critical bug, in one test.

    A duration-failed attempt received a fabricated perfect coverage score, which
    is the maximum, so no later attempt could outrank it. Attempt 2 could pass
    everything, break the loop, and the function still returned attempt 1's audio.
    """
    result, backend = run({0: Failure.RUNAWAY})   # attempt 1 fails duration
    assert result.ok
    assert result.attempts == 2
    assert result.recovered_by == "retry"
    assert backend.calls == 2


def test_duration_valid_failure_outranks_duration_invalid_failure() -> None:
    """When everything fails, report the least-bad — but never call it ok."""
    cfg = SynthConfig(max_attempts=2, allow_sentence_split=False)
    result, _ = run({0: Failure.RUNAWAY, 1: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok
    assert result.duration_s < 100, "kept the runaway instead of the plausible attempt"


def test_null_verifier_does_not_freeze_the_first_attempt() -> None:
    """--no-asr silently made retries useless in the predecessor."""
    backend = FakeBackend(script={0: Failure.RUNAWAY})
    result = synthesize_chunk(TEXT, 0, backend, NullVerifier(), VOICE, CFG)
    assert result.ok
    assert result.attempts == 2


# -------------------------------------------------------------- retrying

def test_succeeds_first_time_without_retrying() -> None:
    result, backend = run()
    assert result.ok and result.attempts == 1 and backend.calls == 1
    assert result.recovered_by == ""


def test_stops_retrying_once_it_passes() -> None:
    result, backend = run({0: Failure.DROP_SENTENCE})
    assert result.ok and backend.calls == 2


def test_exhausting_retries_falls_through_to_the_split() -> None:
    result, _ = run({0: Failure.DROP_SENTENCE, 1: Failure.DROP_SENTENCE, 2: Failure.DROP_SENTENCE})
    assert result.ok
    assert result.recovered_by == "sentence-split"


# ------------------------------------------------------------ exceptions

def test_one_raising_attempt_does_not_lose_the_chunk() -> None:
    """No guard here meant a transient error at chunk 80 discarded 15 minutes."""
    result, backend = run({0: Failure.RAISE})
    assert result.ok and backend.calls == 2


def test_all_attempts_raising_fails_loudly_with_no_audio() -> None:
    cfg = SynthConfig(max_attempts=2, allow_sentence_split=False)
    result, _ = run({0: Failure.RAISE, 1: Failure.RAISE}, cfg=cfg)
    assert not result.ok
    assert result.audio.size == 0, "silence here would read as a deliberate pause"


# -------------------------------------------------------- sentence split

def test_sentence_split_rescues_a_chunk_that_fails_every_attempt() -> None:
    always_drop = {i: Failure.DROP_SENTENCE for i in range(3)}
    result, backend = run(always_drop)
    assert result.ok
    assert result.recovered_by == "sentence-split"
    assert backend.calls > 3
    # Each sentence was rendered alone, where dropping one is not expressible.
    assert any(r == "Not a stranger." for r in backend.requests)


def test_sentence_split_declines_when_a_sentence_itself_fails() -> None:
    """Partial success must not be dressed up as success.

    TRUNCATE rather than DROP_SENTENCE, because you cannot drop a sentence from a
    one-sentence input — the split renders each sentence alone, so the injected
    failure has to be one that applies to a single sentence.
    """
    result, _ = run({i: Failure.TRUNCATE for i in range(40)})
    assert not result.ok


def test_single_sentence_chunk_has_no_split_to_fall_back_on() -> None:
    result, _ = run({i: Failure.TRUNCATE for i in range(10)}, text="One single sentence here.")
    assert not result.ok
    assert result.recovered_by == ""


def test_split_is_disabled_when_configured() -> None:
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, backend = run({0: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok and backend.calls == 1


# -------------------------------------------------------------- reporting

def test_failure_names_the_missing_sentence() -> None:
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, _ = run({0: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok
    assert result.dropped_sentence, "must name what went wrong"
    assert result.transcript


def test_tolerates_realistic_asr_disagreement() -> None:
    result, _ = run(text="He copies the twenty byte code. Ne znemožní.", perfect=False)
    assert result.ok and result.attempts == 1
