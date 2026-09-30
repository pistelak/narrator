"""Every refused generation is kept, with the evidence behind the refusal (issue #55).

The motivating data: 7 Czech episodes, ~540 chunks, 20 quarantines — all 20
transcription mismatches on correct audio — and ~40 chunks "recovered by retry"
whose rejected attempts nobody could inspect. Whether those were real defects or
recogniser noise can only be settled by listening, so these tests pin that the
record exists on every path, that the saved buffer is the one the failing check
measured, and that the files on disk can be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from narrator.backends.fake import Failure, FakeASR, FakeBackend
from narrator.render import RenderConfig, render
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.types import Text, Voice
from narrator.verify import CoverageVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
TEXT = "Not the keeper. Not a stranger. Not any council with any mandate."
NO_SPLIT = SynthConfig(allow_sentence_split=False)


def run(script=None, cfg=SynthConfig(), backend=None):
    backend = backend or FakeBackend(script=script or {})
    verifier = CoverageVerifier(FakeASR(backend))
    return synthesize_chunk(TEXT, 0, backend, verifier, VOICE, cfg), backend


class _Padded(FakeBackend):
    """Ends every take with 0.5 s of digital silence, which trimming removes —
    so the raw buffer and the trimmed one are distinguishable by length."""

    def synthesize(self, text, voice, *, max_frames, temperature):
        audio = super().synthesize(text, voice, max_frames=max_frames,
                                   temperature=temperature)
        return np.concatenate([audio, np.zeros(self.sample_rate // 2, dtype=np.float32)])


# ------------------------------------------------------------- the ladder

def test_a_recovered_chunk_keeps_what_was_refused() -> None:
    """The case the issue is about: the report said "retry" and nothing else."""
    result, _ = run({0: Failure.REPEAT})
    assert result.ok and result.recovered_by == "retry"
    [r] = result.rejected
    assert (r.attempt, r.sentence, r.reason) == (1, None, "verification")
    assert r.reference == TEXT
    assert r.transcript == "Not the keeper. Not the keeper. Not the keeper."
    assert r.coverage < 0.9
    assert r.dropped_sentence and any(c.startswith("d:") for c in r.word_diagnostics)
    assert r.audio is not None and r.audio.size


@pytest.mark.parametrize(("mode", "reason"), [
    (Failure.RUNAWAY, "cap"),
    (Failure.TRUNCATE, "duration"),
    (Failure.SILENT_HOLE, "silence"),
    (Failure.RAISE, "raised"),
])
def test_each_refusal_names_the_check_that_refused_it(mode, reason) -> None:
    result, _ = run({0: mode}, cfg=NO_SPLIT)
    assert result.ok
    [r] = result.rejected
    assert r.reason == reason
    # The cheap checks skip the recogniser by design; nothing is invented.
    assert r.transcript == ""
    assert (r.audio is None) == (reason == "raised")


def test_a_duration_refusal_keeps_the_raw_buffer_it_measured() -> None:
    """Duration reads the UNTRIMMED length. Saving the trimmed take would hand
    the listener audio without the padding that may be the whole defect."""
    result, backend = run({0: Failure.TRUNCATE}, cfg=NO_SPLIT,
                          backend=_Padded(script={0: Failure.TRUNCATE}))
    [r] = result.rejected
    assert r.reason == "duration"
    assert len(r.audio) == round(r.duration_s * backend.sample_rate)


def test_a_verification_refusal_keeps_the_trimmed_take_the_asr_heard() -> None:
    result, backend = run(cfg=NO_SPLIT,
                          backend=_Padded(script={0: Failure.REPEAT}))
    [r] = result.rejected
    assert r.reason == "verification"
    assert len(r.audio) < round(r.duration_s * backend.sample_rate) - backend.sample_rate // 4


def test_split_sentences_record_their_own_ladder_and_reference() -> None:
    """On the split path the assembly carries no transcript, so each sentence's
    refusal must say which sentence it was checked against."""
    # Calls 0-2: whole-chunk attempts; 3: sentence 0; 4: sentence 1, refused; 5: its retry.
    script = {0: Failure.REPEAT, 1: Failure.REPEAT,
              2: Failure.REPEAT, 4: Failure.TRUNCATE}
    result, _ = run(script)
    assert result.ok and result.recovered_by == "sentence-split"
    assert [(r.sentence, r.attempt) for r in result.rejected] == [
        (None, 1), (None, 2), (None, 3), (1, 1)]
    assert result.rejected[-1].reference == "Not a stranger."


def test_a_failed_split_still_reports_every_refusal() -> None:
    result, _ = run(script={}, backend=FakeBackend(default=Failure.TRUNCATE))
    assert not result.ok
    assert [(r.sentence, r.attempt) for r in result.rejected] == [
        (None, 1), (None, 2), (None, 3), (0, 1), (0, 2), (0, 3)]


def test_the_diagnostic_reverify_does_not_rewrite_the_record() -> None:
    """A capped attempt was refused without a transcript. The failure-only
    re-verify adds one to the REPORTED verdict; the rejection must keep what
    the ladder actually knew when it refused."""
    result, _ = run({0: Failure.RUNAWAY},
                    cfg=SynthConfig(max_attempts=1, allow_sentence_split=False))
    assert not result.ok and result.transcript
    [r] = result.rejected
    assert r.reason == "cap" and r.transcript == "" and r.coverage == 0.0


def test_a_verified_take_passed_over_for_a_rise_is_not_a_rejection(monkeypatch) -> None:
    import narrator.prosody

    def check(audio, sample_rate):
        return {0: 0.5, 2: 3.0}.get(round(float(audio[0]) / 1e-4) - 1)

    monkeypatch.setattr(narrator.prosody, "rise_delta_checker", lambda: check)
    backend = FakeBackend(script={1: Failure.TRUNCATE})
    cfg = SynthConfig(wants_rise=lambda text, lang: True)
    result = synthesize_chunk("Are you coming to the meeting tomorrow?", 0, backend,
                              CoverageVerifier(FakeASR(backend)), VOICE, cfg)
    assert result.ok
    # Take 1 verified flat and was passed over; only take 2 was refused.
    assert [(r.attempt, r.reason) for r in result.rejected] == [(2, "duration")]


# ------------------------------------------------------------- render

def _render(tmp_path: Path, script=None, backend=None, voice=VOICE, **cfg):
    backend = backend or FakeBackend(script=script or {})
    out = tmp_path / "episode.wav"
    report = render([Text(TEXT)], voice, backend, out,
                    CoverageVerifier(FakeASR(backend)),
                    RenderConfig(on_progress=None, **cfg))
    return report, out


def _rows(run_dir: Path) -> list[dict]:
    lines = (run_dir / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_render_writes_each_refused_take_and_its_line(tmp_path: Path) -> None:
    rejects = tmp_path / "rejects"
    report, _ = _render(tmp_path, {0: Failure.REPEAT, 1: Failure.RAISE},
                        rejects=rejects)
    [run_dir] = rejects.iterdir()
    rows = _rows(run_dir)
    assert [(row["attempt"], row["reason"]) for row in rows] == [
        (1, "verification"), (2, "raised")]
    first = rows[0]
    assert first["ok"] and first["recovered_by"] == "retry"
    assert first["final_transcript"] == TEXT
    assert first["transcript"] == "Not the keeper. Not the keeper. Not the keeper."
    assert first["wav"] == "c0000-c-a1.wav"
    audio, rate = sf.read(run_dir / first["wav"], dtype="float32")
    assert rate == 24000 and audio.size
    assert rows[1]["wav"] == "", "a raised attempt has no audio to keep"
    # The report keeps the record and drops the audio: a long render must not
    # hold every refused take until it ends.
    assert [r.audio for r in report.chunks[0].rejected] == [None, None]


def test_rendering_without_a_rejects_dir_still_drops_the_audio(tmp_path: Path) -> None:
    report, _ = _render(tmp_path, {0: Failure.REPEAT})
    [r] = report.chunks[0].rejected
    assert r.reason == "verification" and r.audio is None


def test_two_renders_into_one_dir_never_overwrite_each_other(tmp_path: Path) -> None:
    """Chunk indices restart at 0 every render, so fixed filenames would let the
    second render replace the first one's audio while both logs survived."""
    rejects = tmp_path / "rejects"
    _render(tmp_path, {0: Failure.REPEAT}, rejects=rejects)
    _render(tmp_path, {0: Failure.TRUNCATE}, rejects=rejects)
    runs = sorted(rejects.iterdir())
    assert len(runs) == 2
    assert {_rows(r)[0]["reason"] for r in runs} == {"verification", "duration"}


def test_a_failed_chunk_says_its_transcript_is_final_not_accepted(tmp_path: Path) -> None:
    rejects = tmp_path / "rejects"
    report, _ = _render(tmp_path, backend=FakeBackend(default=Failure.REPEAT),
                        rejects=rejects, quarantine=False)
    [run_dir] = rejects.iterdir()
    rows = _rows(run_dir)
    assert rows and all(not row["ok"] for row in rows)
    assert {row["final_transcript"] for row in rows} == {report.chunks[0].transcript}


def test_an_unusable_rejects_dir_fails_before_any_synthesis(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    backend = FakeBackend()
    with pytest.raises(OSError):
        _render(tmp_path, backend=backend, rejects=blocker)
    assert backend.calls == 0
    assert not (tmp_path / "episode.wav").exists()


def test_a_failed_wav_write_stops_the_render_and_keeps_earlier_evidence(
        tmp_path: Path, monkeypatch) -> None:
    """The evidence is what the caller asked for, unlike the take store's cache:
    a refused take whose write failed is gone for good, so this raises."""
    real_write = sf.write
    calls = []

    def flaky(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(sf, "write", flaky)
    rejects = tmp_path / "rejects"
    with pytest.raises(OSError, match="disk full"):
        _render(tmp_path, {0: Failure.REPEAT, 1: Failure.REPEAT},
                rejects=rejects)
    [run_dir] = rejects.iterdir()
    [row] = _rows(run_dir)
    assert (run_dir / row["wav"]).is_file()
    assert not (tmp_path / "episode.wav").exists()


def test_a_failed_log_write_stops_the_render_and_keeps_earlier_evidence(
        tmp_path: Path, monkeypatch) -> None:
    import importlib

    render_mod = importlib.import_module("narrator.render")

    class _FlakyJson:
        calls = 0

        @staticmethod
        def dumps(obj, **kwargs):
            _FlakyJson.calls += 1
            if _FlakyJson.calls == 2:
                raise OSError("log unwritable")
            return json.dumps(obj, **kwargs)

    monkeypatch.setattr(render_mod, "json", _FlakyJson)
    rejects = tmp_path / "rejects"
    with pytest.raises(OSError, match="log unwritable"):
        _render(tmp_path, {0: Failure.REPEAT, 1: Failure.REPEAT},
                rejects=rejects)
    [run_dir] = rejects.iterdir()
    [row] = _rows(run_dir)
    assert row["attempt"] == 1 and (run_dir / row["wav"]).is_file()
    assert not (tmp_path / "episode.wav").exists()


def test_a_reused_take_brings_no_rejections(tmp_path: Path) -> None:
    """This run made no attempts; the run that made the take holds its evidence."""
    takes = tmp_path / "takes"
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"reference-bytes")   # a real file, so the store can key on it
    voice = Voice(clip, "reference", "en")
    first, _ = _render(tmp_path, {0: Failure.REPEAT}, voice=voice, takes=takes)
    assert first.chunks[0].rejected
    # Same script, so the same backend identity: the second render reuses.
    again, _ = _render(tmp_path, {0: Failure.REPEAT}, voice=voice, takes=takes)
    assert again.chunks[0].reused
    assert again.chunks[0].rejected == ()
    assert "rejected" not in again.summary()


def test_the_summary_counts_refusals_by_check(tmp_path: Path) -> None:
    report, _ = _render(tmp_path, {0: Failure.TRUNCATE, 1: Failure.REPEAT})
    assert "| 2 rejected attempts (duration 1, verification 1)" in report.summary()
