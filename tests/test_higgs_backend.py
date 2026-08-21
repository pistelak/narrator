"""Higgs backend tests.

The model itself is not exercised here — it is 8.7 GB and takes ~18 minutes to
render an episode. What IS testable without it is everything around the call:
protocol conformance, the reference-caching contract, and the error paths a
caller will actually hit (missing extra, missing reference file, wrong sample
rate). Those are where a wrapper goes wrong.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from narrator.backends.higgs import FPS, SAMPLE_RATE, HiggsBackend
from narrator.types import Backend, Voice


class FakeModel:
    """Stands in for mlx_audio's loaded model."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.encode_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.sample_rate = sample_rate

    def encode_reference_audio(self, path: str):
        self.encode_calls.append(path)
        # Numbered, so a test can tell *which* encode's codes were used. Codes
        # derived from the path alone cannot distinguish "re-encoded the edited
        # clip" from "re-encoded it and then generated from the stale ones".
        return f"codes::{path}::{len(self.encode_calls)}"

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        frames = kwargs["max_new_frames"]
        yield SimpleNamespace(
            audio=np.zeros(int(frames / FPS * SAMPLE_RATE), dtype=np.float32),
            sample_rate=self.sample_rate,
        )


@pytest.fixture
def voice(tmp_path: Path) -> Voice:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")  # only its existence matters here
    return Voice(ref, "reference transcript", "en")


@pytest.fixture
def backend() -> HiggsBackend:
    b = HiggsBackend()
    b._model = FakeModel()
    return b


def test_satisfies_the_backend_protocol() -> None:
    assert isinstance(HiggsBackend(), Backend)


def test_frames_per_second_is_the_measured_rate() -> None:
    assert HiggsBackend().frames_per_second() == 25


def test_synthesize_passes_the_frame_cap_through(backend: HiggsBackend, voice: Voice) -> None:
    """The cap is the only bound on a runaway; it must reach the engine intact."""
    backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
    assert backend._model.generate_calls[0]["max_new_frames"] == 250


def test_synthesize_passes_temperature_through(backend: HiggsBackend, voice: Voice) -> None:
    backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
    assert backend._model.generate_calls[0]["temperature"] == 0.4


def test_reference_is_encoded_once_and_reused(backend: HiggsBackend, voice: Voice) -> None:
    """One narrator, not a hundred.

    Re-encoding per chunk would reintroduce the encoder as a source of per-chunk
    variation, which is the drift a pinned reference exists to remove.
    """
    for _ in range(5):
        backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
    assert len(backend._model.encode_calls) == 1
    assert all(c["ref_audio_codes"] == f"codes::{voice.audio_path}::1"
               for c in backend._model.generate_calls)


def test_switching_voice_re_encodes(backend: HiggsBackend, voice: Voice, tmp_path: Path) -> None:
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF")
    backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
    backend.synthesize("Hello.", Voice(other, "other", "cs"), max_frames=250, temperature=0.4)
    assert len(backend._model.encode_calls) == 2


def test_alternating_voices_encode_once_each(
    backend: HiggsBackend, voice: Voice, tmp_path: Path
) -> None:
    """A dialogue alternates speakers every chunk; the cache must survive that.

    The one-slot cache this replaced re-encoded on every speaker change — about
    a hundred encodes across a render instead of two.
    """
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF-other")
    second = Voice(other, "other", "en")
    for _ in range(10):
        backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
        backend.synthesize("Hello.", second, max_frames=250, temperature=0.4)
    assert len(backend._model.encode_calls) == 2


def test_editing_a_reference_in_place_serves_the_new_speaker(
    backend: HiggsBackend, voice: Voice
) -> None:
    """A replaced clip must not keep serving the old speaker.

    The original size and mtime are both restored, because that is exactly the
    case a (path, size, mtime) key cannot catch: it would hand back the previous
    speaker's codes, and nothing downstream would notice — ASR verifies the
    words, not who spoke them. The assertion is on the codes that reached
    `generate`, not merely on the encode count, so re-encoding and then
    generating from the stale codes fails too.
    """
    import os
    before = voice.audio_path.stat()
    backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)

    voice.audio_path.write_bytes(b"X" * before.st_size)   # same size, new speaker
    os.utime(voice.audio_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = voice.audio_path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    backend.synthesize("Hello.", voice, max_frames=250, temperature=0.4)
    assert len(backend._model.encode_calls) == 2
    assert backend._model.generate_calls[-1]["ref_audio_codes"].endswith("::2")


def test_the_reference_cache_is_bounded(backend: HiggsBackend, tmp_path: Path) -> None:
    """A long-lived backend handed many references must not grow without bound.

    The re-use check at the end is the other half: a cache that evicted every
    entry the moment it landed would also stay under the cap, while re-encoding
    on every single call.
    """
    from narrator.backends.higgs import REF_CACHE_MAX
    for index in range(REF_CACHE_MAX + 4):
        clip = tmp_path / f"v{index}.wav"
        clip.write_bytes(f"RIFF{index}".encode())
        backend.synthesize("Hello.", Voice(clip, "t", "en"), max_frames=250, temperature=0.4)
    assert len(backend._ref_codes) == REF_CACHE_MAX

    encodes = len(backend._model.encode_calls)
    newest = Voice(tmp_path / f"v{REF_CACHE_MAX + 3}.wav", "t", "en")
    backend.synthesize("Hello.", newest, max_frames=250, temperature=0.4)
    assert len(backend._model.encode_calls) == encodes  # still cached


def test_missing_reference_fails_with_an_actionable_message(
    backend: HiggsBackend, tmp_path: Path
) -> None:
    missing = Voice(tmp_path / "nope.wav", "t", "en")
    with pytest.raises(FileNotFoundError, match="required, not optional"):
        backend.synthesize("Hello.", missing, max_frames=250, temperature=0.4)


def test_unexpected_sample_rate_is_refused(voice: Voice) -> None:
    """Silently resampling would be worse: it would change pitch and pace."""
    b = HiggsBackend()
    b._model = FakeModel(sample_rate=48_000)
    with pytest.raises(RuntimeError, match="48000 Hz"):
        b.synthesize("Hello.", voice, max_frames=250, temperature=0.4)


def test_missing_extra_names_the_torch_surprise() -> None:
    """The error must mention torch — it took a runtime failure to discover that
    an 'MLX' backend needs it, and nobody should discover it twice."""
    import narrator.backends.higgs as mod
    src = Path(mod.__file__).read_text()
    assert "narrator[higgs]" in src
    assert "torch is required" in src
