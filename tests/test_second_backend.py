"""Does the Backend protocol actually admit a second engine?

The library claims that chunking, verification, retries, stitching and mastering
are engine-independent. With one backend that is an assertion. These tests make it
a check, using Supertonic — an engine that differs from Higgs in exactly the ways
that matter: it has no generation bound, no temperature, and a voice bank instead
of a reference clip.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from narrator.backends.supertonic import SupertonicBackend
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.types import Backend, Voice
from narrator.verify import NullVerifier

TEXT = "Not the keeper. Not a stranger. Not any council with any mandate."


class FakeTTS:
    """Stands in for the supertonic package."""

    def __init__(self, rate: int = 44100) -> None:
        self.calls: list[dict] = []
        self.styles_by_name: list[str] = []
        self.styles_by_path: list[str] = []
        self.rate = rate

    def get_voice_style(self, name: str):
        if name not in {"M1", "M2", "F1"}:
            raise KeyError(name)
        self.styles_by_name.append(name)
        return f"style::{name}"

    def get_voice_style_from_path(self, path: str):
        self.styles_by_path.append(path)
        return f"style::{path}"

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        seconds = len(kwargs["text"].split()) / 2.5
        return np.zeros(int(seconds * self.rate), dtype=np.float32), None

    def save_audio(self, wav, path):
        import soundfile as sf
        sf.write(path, wav, self.rate)


@pytest.fixture
def backend() -> SupertonicBackend:
    b = SupertonicBackend()
    b._tts = FakeTTS()
    return b


def test_satisfies_the_backend_protocol() -> None:
    assert isinstance(SupertonicBackend(), Backend)


def test_declares_that_it_cannot_honour_a_frame_cap() -> None:
    """The finding that drove the protocol change.

    Supertonic's synthesize takes no generation bound. Left undeclared, a backend
    that never reaches its cap and one that always does look identical to a check
    that only compares durations.
    """
    assert SupertonicBackend().honours_frame_cap is False


def test_cap_is_not_treated_as_a_runaway_signal(backend: SupertonicBackend) -> None:
    """Without the capability flag this chunk would be rejected on every attempt."""
    voice = Voice(preset="M1", lang="en")
    result = synthesize_chunk(
        TEXT, 0, backend, NullVerifier(), voice,
        SynthConfig(max_attempts=1, allow_sentence_split=False),
    )
    assert result.ok


def test_preset_voice_is_resolved_from_the_voice_bank(backend: SupertonicBackend) -> None:
    """The second protocol finding: some engines ship voices instead of cloning."""
    backend.synthesize(TEXT, Voice(preset="M2", lang="en"), max_frames=999, temperature=0.4)
    assert backend._tts.styles_by_name == ["M2"]
    assert backend._tts.styles_by_path == []


def test_reference_clip_still_works(backend: SupertonicBackend, tmp_path: Path) -> None:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    backend.synthesize(TEXT, Voice(ref, "t", "en"), max_frames=999, temperature=0.4)
    assert backend._tts.styles_by_path == [str(ref)]


def test_style_is_resolved_once_and_cached(backend: SupertonicBackend) -> None:
    for _ in range(4):
        backend.synthesize(TEXT, Voice(preset="M1", lang="en"), max_frames=999, temperature=0.4)
    assert backend._tts.styles_by_name == ["M1"]


def test_unknown_preset_names_the_voice_bank(backend: SupertonicBackend) -> None:
    with pytest.raises(ValueError, match="M1..M5"):
        backend.synthesize(TEXT, Voice(preset="nope", lang="en"), max_frames=999, temperature=0.4)


def test_language_reaches_the_engine(backend: SupertonicBackend) -> None:
    backend.synthesize(TEXT, Voice(preset="M1", lang="cs"), max_frames=999, temperature=0.4)
    assert backend._tts.calls[0]["lang"] == "cs"


def test_sample_rate_is_known_at_construction_then_verified() -> None:
    """A zero rate made a LEADING Gap allocate zero samples and vanish from a
    render that still reported itself clean, so the documented 44.1 kHz is known
    up front — and checked against the real output on first synthesis."""
    b = SupertonicBackend()
    assert b.sample_rate == 44100, "render() needs this before the first synthesis"
    b._tts = FakeTTS(rate=22050)
    b.synthesize(TEXT, Voice(preset="M1", lang="en"), max_frames=999, temperature=0.4)
    assert b.sample_rate == 22050, "a real mismatch must correct the assumption"


def test_documented_waveform_shape_is_not_collapsed(backend: SupertonicBackend) -> None:
    """Supertonic documents (1, num_samples); mean(axis=1) on that returns ONE
    sample — the whole utterance averaged to a point. The double returned 1-D,
    so no test caught it."""
    import numpy as np

    class TwoDimTTS(FakeTTS):
        def synthesize(self, **kwargs):
            mono, _ = super().synthesize(**kwargs)
            return mono.reshape(1, -1), None

    backend._tts = TwoDimTTS()
    audio = backend.synthesize(TEXT, Voice(preset="M1", lang="en"),
                               max_frames=999, temperature=0.4)
    assert audio.ndim == 1
    assert audio.size > 1000


def test_voice_requires_a_clip_or_a_preset() -> None:
    with pytest.raises(ValueError, match="audio_path or preset"):
        Voice()


def test_higgs_refuses_a_preset_only_voice() -> None:
    """A cloning engine cannot use a voice bank, and should say why."""
    from narrator.backends.higgs import HiggsBackend

    b = HiggsBackend()
    b._model = SimpleNamespace()
    with pytest.raises(ValueError, match="no voice bank"):
        b._codes_for(Voice(preset="M1", lang="en"))
