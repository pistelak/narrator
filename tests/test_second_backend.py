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
        return f"style::{name}::{len(self.styles_by_name)}"

    def get_voice_style_from_path(self, path: str):
        # Numbered by resolution, like the Higgs double: a style that merely
        # names its path cannot tell a FRESH resolution from a stale cached one,
        # so a backend that re-resolves and then synthesizes with the old style
        # would still pass.
        self.styles_by_path.append(path)
        return f"style::{path}::{len(self.styles_by_path)}"

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


def test_a_clip_path_cannot_collide_with_a_preset_name(
    backend: SupertonicBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip whose path reads like a preset must not be served the preset.

    The cache key used to be the bare string, so `Voice(audio_path=Path("M1"))`
    and `Voice(preset="M1")` landed on one entry: whichever arrived first
    supplied the style for both, and a dialogue that mixes cloned and bank
    voices would have synthesized one speaker in the other's voice — while
    verifying clean, because ASR checks the words, not who says them.
    """
    # The path is spelled exactly "M1" — the literal case in the docstring, and
    # the only one that collides, since an absolute path never equals a preset.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "M1").write_bytes(b"RIFF")
    backend.synthesize(TEXT, Voice(Path("M1"), "t", "en"), max_frames=999, temperature=0.4)
    backend.synthesize(TEXT, Voice(preset="M1", lang="en"), max_frames=999, temperature=0.4)
    assert backend._tts.styles_by_path == ["M1"]
    assert backend._tts.styles_by_name == ["M1"]   # resolved separately, not reused


def test_an_edited_reference_re_resolves_the_style(
    backend: SupertonicBackend, tmp_path: Path
) -> None:
    """Replace the file at a path and the next synthesis must use the new speaker.

    Path text is not content identity. Keyed on the path alone, the cache hits
    after an in-place edit and the engine synthesizes the PREVIOUS speaker while
    the render verifies clean — ASR checks the words, not who says them. The
    take store makes it reachable in a new way too: it misses correctly on the
    edited reference, and a long-lived backend then refills that miss with the
    old speaker's audio, now filed under the new reference's digest.

    Mirrors the Higgs test of the same name. Size and mtime are both restored, so
    a regression to `(path, size, mtime)` identity fails here rather than passing
    by luck; and the assertion is on the style that reached `synthesize`, not on
    the resolver count, so re-resolving and then synthesizing with the stale
    style fails too.
    """
    import os

    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF-speaker-A")
    voice = Voice(clip, "t", "en")
    before = clip.stat()
    backend.synthesize(TEXT, voice, max_frames=999, temperature=0.4)

    clip.write_bytes(b"RIFF-speaker-B")   # same path, same size, new speaker
    os.utime(clip, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = clip.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    backend.synthesize(TEXT, voice, max_frames=999, temperature=0.4)

    assert len(backend._tts.styles_by_path) == 2, "an edited clip must re-resolve"
    assert backend._tts.calls[-1]["voice_style"].endswith("::2"), "the FRESH style must ship"


def _clip(tmp_path: Path, index: int) -> Voice:
    clip = tmp_path / f"v{index}.wav"
    clip.write_bytes(f"RIFF{index}".encode())
    return Voice(clip, "t", "en")


def test_the_clip_cache_is_bounded_and_evicts_oldest_first(
    backend: SupertonicBackend, tmp_path: Path
) -> None:
    """A long-lived backend handed many references must not grow without bound.

    Both halves are asserted, because either alone is passed by a wrong policy:
    a cache that evicted every entry on arrival also stays under the cap while
    re-resolving every call, and "the newest is still cached" is satisfied by
    MRU, LRU or anything else that keeps the newest. So the boundary itself is
    pinned — oldest out, newest in — which is the documented FIFO contract.
    """
    from narrator.backends.supertonic import CLIP_CACHE_MAX

    for index in range(CLIP_CACHE_MAX + 4):
        backend.synthesize(TEXT, _clip(tmp_path, index), max_frames=999, temperature=0.4)
    assert len(backend._styles) == CLIP_CACHE_MAX

    resolved = len(backend._tts.styles_by_path)
    backend.synthesize(TEXT, _clip(tmp_path, CLIP_CACHE_MAX + 3), max_frames=999,
                       temperature=0.4)
    assert len(backend._tts.styles_by_path) == resolved, "the newest entry is still cached"

    backend.synthesize(TEXT, _clip(tmp_path, 0), max_frames=999, temperature=0.4)
    assert len(backend._tts.styles_by_path) == resolved + 1, "the oldest was evicted"


def test_presets_are_exempt_from_the_clip_bound(
    backend: SupertonicBackend, tmp_path: Path
) -> None:
    """Supertonic ships ten presets; a bound of eight must not evict the bank.

    Capping clips and presets together thrashed a case this engine explicitly
    supports — a dialogue cycling M1..M5/F1..F5 would evict its own speakers
    every turn, and clip styles would evict reusable preset styles. Presets
    cannot grow without bound anyway: an unknown one raises rather than caching.
    """
    from narrator.backends.supertonic import CLIP_CACHE_MAX

    backend.synthesize(TEXT, Voice(preset="M1", lang="en"), max_frames=999, temperature=0.4)
    for index in range(CLIP_CACHE_MAX + 4):
        backend.synthesize(TEXT, _clip(tmp_path, index), max_frames=999, temperature=0.4)

    resolved = len(backend._tts.styles_by_name)
    backend.synthesize(TEXT, Voice(preset="M1", lang="en"), max_frames=999, temperature=0.4)
    assert len(backend._tts.styles_by_name) == resolved, "a clip flood must not evict a preset"


def test_unknown_preset_names_the_voice_bank(backend: SupertonicBackend) -> None:
    with pytest.raises(ValueError, match=r"M1\.\.M5"):
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
