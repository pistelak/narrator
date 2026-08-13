"""End-to-end pipeline tests, still with no model involved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from narrator.audio import MasterConfig, concatenate, declick, soft_limit, to_channels, trim_silence
from narrator.backends.fake import FakeASR, FakeBackend, Failure
from narrator.render import RenderConfig, RenderFailed, render
from narrator.synth import SynthConfig
from narrator.types import Gap, Text, Voice
from narrator.verify import CoverageVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
SEGMENTS = [
    Text("Not the keeper. Not a stranger. Not any council with any mandate."),
    Gap(3.0),
    Text("They were sent to a destination that does not exist."),
]


def build(script=None):
    backend = FakeBackend(script=script or {})
    return backend, CoverageVerifier(FakeASR(backend))


# ------------------------------------------------------------ happy path

def test_writes_a_file_and_reports_clean(tmp_path: Path) -> None:
    backend, verifier = build()
    out = tmp_path / "episode.wav"
    report = render(SEGMENTS, VOICE, backend, verifier, out)
    assert report.clean
    assert out.is_file()
    assert len(report.chunks) == 2


def test_gap_duration_is_honoured_exactly(tmp_path: Path) -> None:
    """A pause is content. A renderer that 'improves' the timing is editing."""
    backend, verifier = build()
    short = render([Text("One two three four.")], VOICE, backend, verifier, tmp_path / "a.wav")

    backend2, verifier2 = build()
    withgap = render(
        [Text("One two three four."), Gap(5.0)], VOICE, backend2, verifier2, tmp_path / "b.wav"
    )
    assert withgap.duration_s - short.duration_s == pytest.approx(5.0, abs=0.05)


def test_long_text_is_chunked(tmp_path: Path) -> None:
    backend, verifier = build()
    long = Text(" ".join(["This is a sentence of moderate length here."] * 30))
    report = render([long], VOICE, backend, verifier, tmp_path / "c.wav")
    assert len(report.chunks) > 3
    assert all(len(c.text) <= 250 for c in report.chunks)


# ------------------------------------------------------------ quarantine

def test_unrecoverable_chunk_raises_and_writes_nothing(tmp_path: Path) -> None:
    """The whole point. A plausible file nobody knows is wrong is the failure."""
    out = tmp_path / "episode.wav"
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    with pytest.raises(RenderFailed) as excinfo:
        render(SEGMENTS, VOICE, backend, verifier, out)
    assert not out.exists()
    assert excinfo.value.report.failures


def test_quarantine_can_be_waived_but_the_report_still_says_so(tmp_path: Path) -> None:
    out = tmp_path / "episode.wav"
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    report = render(
        SEGMENTS, VOICE, backend, verifier, out, RenderConfig(quarantine=False)
    )
    assert out.is_file()
    assert not report.clean
    assert report.failures


def test_failure_message_names_chunks_and_content(tmp_path: Path) -> None:
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    with pytest.raises(RenderFailed, match="failed verification"):
        render(SEGMENTS, VOICE, backend, verifier, tmp_path / "e.wav")


# ------------------------------------------------------------- reporting

def test_report_counts_recoveries(tmp_path: Path) -> None:
    backend, verifier = build({0: Failure.DROP_SENTENCE})
    report = render(SEGMENTS, VOICE, backend, verifier, tmp_path / "f.wav")
    assert report.clean
    assert any(c.recovered_by for c in report.chunks)
    assert "chunks" in report.summary()


def test_progress_callback_fires_per_chunk(tmp_path: Path) -> None:
    seen = []
    backend, verifier = build()
    render(SEGMENTS, VOICE, backend, verifier, tmp_path / "g.wav",
           RenderConfig(on_progress=lambda r, total: seen.append((r.index, total))))
    assert [i for i, _ in seen] == [0, 1]
    assert all(total == 2 for _, total in seen)


# ---------------------------------------------------------------- audio

def test_output_is_dual_mono_by_default(tmp_path: Path) -> None:
    """Sidesteps the BS.1770 mono offset without assuming player behaviour."""
    backend, verifier = build()
    out = tmp_path / "h.wav"
    render(SEGMENTS, VOICE, backend, verifier, out)
    data, _ = sf.read(str(out))
    assert data.ndim == 2 and data.shape[1] == 2
    assert np.allclose(data[:, 0], data[:, 1])


def test_mono_output_when_requested(tmp_path: Path) -> None:
    backend, verifier = build()
    out = tmp_path / "i.wav"
    render(SEGMENTS, VOICE, backend, verifier, out,
           RenderConfig(mastering=MasterConfig(channels=1)))
    data, _ = sf.read(str(out))
    assert data.ndim == 1


def test_trim_removes_silence_but_keeps_a_guard() -> None:
    sr = 24000
    speech = np.sin(np.linspace(0, 200, sr)).astype(np.float32)
    padded = np.concatenate([np.zeros(sr, np.float32), speech, np.zeros(sr, np.float32)])
    trimmed = trim_silence(padded, sr)
    assert trimmed.size < padded.size
    assert trimmed.size > speech.size * 0.9


def test_trim_leaves_all_silence_alone() -> None:
    silence = np.zeros(24000, dtype=np.float32)
    assert trim_silence(silence, 24000).size == silence.size


def test_declick_fades_both_edges() -> None:
    audio = np.ones(24000, dtype=np.float32)
    faded = declick(audio, 24000)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    assert faded[12000] == 1.0


def test_soft_limit_guarantees_the_ceiling() -> None:
    loud = np.linspace(-3, 3, 1000).astype(np.float32)
    limited = soft_limit(loud, 0.9)
    assert np.max(np.abs(limited)) < 0.9


def test_soft_limit_leaves_quiet_material_untouched() -> None:
    quiet = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    assert np.allclose(soft_limit(quiet, 0.9), quiet, atol=1e-6)


def test_concatenate_of_nothing_is_empty() -> None:
    assert concatenate([]).size == 0


def test_to_channels_duplicates_exactly() -> None:
    mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    stereo = to_channels(mono, 2)
    assert stereo.shape == (3, 2)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])


# ------------------------------------------------------------------- CLI

def test_cli_parses_blank_lines_as_paragraph_gaps() -> None:
    from narrator.cli import parse_text
    segments = parse_text("First para.\n\nSecond para.\n\n\nThird.", 0.5)
    assert [type(s).__name__ for s in segments] == ["Text", "Gap", "Text", "Gap", "Text"]
    assert all(s.seconds == 0.5 for s in segments if isinstance(s, Gap))


def test_cli_collapses_internal_whitespace() -> None:
    from narrator.cli import parse_text
    segments = parse_text("Line one\nline two\n   spaced", 0.35)
    assert segments == [Text("Line one line two spaced")]


def test_cli_ignores_empty_input() -> None:
    from narrator.cli import parse_text
    assert parse_text("\n\n   \n\n", 0.35) == []


def test_backend_with_unknown_sample_rate_is_refused(tmp_path: Path) -> None:
    """A leading Gap against sample_rate == 0 allocated zero samples and vanished
    from a render that still reported itself clean."""
    backend, verifier = build()
    backend.sample_rate = 0
    with pytest.raises(ValueError, match="sample_rate is 0"):
        render([Gap(3.0), Text("Alpha beta gamma.")], VOICE, backend, verifier, tmp_path / "x.wav")


def test_leading_gap_is_rendered_at_full_length(tmp_path: Path) -> None:
    backend, verifier = build()
    a = render([Text("Alpha beta gamma.")], VOICE, backend, verifier, tmp_path / "a.wav")
    b = render(*[[Gap(3.0), Text("Alpha beta gamma.")], VOICE, *build(), tmp_path / "b.wav"])
    assert b.duration_s - a.duration_s == pytest.approx(3.0, abs=0.05)
