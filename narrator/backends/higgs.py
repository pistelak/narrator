"""Higgs Audio v3 via mlx-audio, on Apple Silicon.

Chosen over Supertonic 3, Piper, XTTS-v2 and Bark on a Czech and code-switched
test set. The deciding case was Czech sentences carrying English technical terms
— the pipeline's hardest input — where Supertonic produced "koudování base
64Hack" for "kódování base 64 check", mispronouncing a plain Czech word and
turning *check* into *hack*. Measured intelligibility was only ~1 CER point
apart, so naturalness decided it, on a full-length listening pass.

Two things about this backend that are not obvious:

- **torch is required** despite the MLX path. mlx-audio's Higgs codec loader
  reads safetensors with `framework="pt"`. Discovered at runtime, on a machine
  that had deliberately installed no torch.
- **The MLX port has no repetition-aware sampling.** `generation.py` imports only
  top-k and top-p; there is no `ras`, `repetition` or `penalty` anywhere in the
  module, while Boson's reference implementation runs RAS by default. That is the
  leading hypothesis for the babble mode this library defends against, though the
  causal claim is untested — absence is verified, causation is not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from narrator.types import Audio, Voice

MODEL = "bosonai/higgs-audio-v3-tts-4b"
SAMPLE_RATE = 24_000
FPS = 25
"""Acoustic frames per second — 40 ms per frame. From the SGLang-Omni cookbook."""

REF_CACHE_MAX = 8
"""How many encoded references one backend keeps. Comfortably above any
dialogue's speaker count, low enough that a long-lived backend handed a stream
of different references does not grow without bound."""


def _cache_key(path: Path) -> tuple[str, str]:
    """Resolved path plus a digest of the file, so an edited reference re-encodes.

    Content, not size and mtime: a same-size replacement that preserves the
    timestamp — or is written inside a filesystem's timestamp granularity, which
    `st_mtime_ns` reports in nanoseconds without resolving to them — keeps the
    old key and serves the previous speaker's codes. Verification would not
    catch it, because ASR checks the words, not who said them, so the wrong
    voice ships with a clean report. Measured, because it runs once per
    synthesis call: 0.33 ms for a 10 s reference, 2.2 ms for a 60 s one — 0.03 s
    and 0.22 s respectively across a hundred chunks, against a render measured
    in minutes.
    """
    digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    return (str(path.resolve()), digest)


@dataclass
class HiggsBackend:
    """Wraps mlx-audio. Loads the model once; encodes the voice reference once.

    The reference is the difference between one narrator and a hundred. Zero-shot
    models invent a voice per call, and across ~100 chunks that drifts audibly.
    Encoding it once also removes the encoder as a source of per-chunk variation.
    """

    model_id: str = MODEL
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    honours_frame_cap: bool = True
    """Autoregressive: max_new_frames is a real hard stop."""

    _model: Any = field(default=None, repr=False)
    _ref_codes: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)

    def frames_per_second(self) -> int:
        return self.fps

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_audio.tts import load
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "The Higgs backend needs the 'higgs' extra: pip install 'narrator[higgs]'. "
                "Note torch is required despite the MLX path — mlx-audio's codec loader "
                "reads safetensors with framework='pt'."
            ) from exc
        self._model = load(self.model_id)

    def _codes_for(self, voice: Voice) -> Any:
        """Encode each reference once and cache it for the life of the backend.

        Several references, not one slot: a dialogue render alternates between
        two pinned voices per chunk, and a one-slot cache would re-encode on
        every speaker change — ~100 encodes instead of 2. Transcripts are NOT
        part of the key on purpose: codes encode only the audio, and
        `synthesize` passes the transcript fresh on every call.

        The key digests the file's contents as well as its path, so editing a
        reference in place invalidates it. Path alone would not, and that would
        be a regression rather than a wash: the one-slot cache re-encoded on
        every speaker change, so A -> B -> edited A happened to pick the edit
        up, where a plain dict serves the old speaker while the render still
        verifies clean. Nothing in the API pins a backend to one render, so the
        size cap below bounds what a long-lived one accumulates.

        Validation comes before the lookup because the key reads the file."""
        if voice.audio_path is None:
            raise ValueError(
                "Higgs clones from a reference clip; it has no voice bank, so a preset-only "
                "Voice cannot be used. Supply audio_path and transcript."
            )
        if not voice.audio_path.is_file():
            raise FileNotFoundError(
                f"Voice reference not found: {voice.audio_path}. "
                "A pinned reference is required, not optional: without one the voice "
                "drifts audibly across a long render."
            )
        key = _cache_key(voice.audio_path)
        cached = self._ref_codes.get(key)
        if cached is not None:
            return cached
        codes = self._model.encode_reference_audio(str(voice.audio_path))
        self._ref_codes[key] = codes
        if len(self._ref_codes) > REF_CACHE_MAX:
            # Oldest out, not least-recently-used: a dialogue cycles a handful
            # of speakers and never reaches the cap, so the policy only ever
            # decides for a backend a service keeps alive across many uploaded
            # references — where the hazard is unbounded retention, not a miss.
            del self._ref_codes[next(iter(self._ref_codes))]
        return codes

    def synthesize(
        self, text: str, voice: Voice, *, max_frames: int, temperature: float
    ) -> Audio:
        self.load()
        result = next(
            self._model.generate(
                text=text,
                ref_audio_codes=self._codes_for(voice),
                ref_text=voice.transcript,
                temperature=temperature,
                max_new_frames=max_frames,
            )
        )
        audio = np.asarray(result.audio, dtype=np.float32)
        if getattr(result, "sample_rate", self.sample_rate) != self.sample_rate:
            raise RuntimeError(
                f"Backend returned {result.sample_rate} Hz, expected {self.sample_rate}"
            )
        return audio


# WhisperASR moved to narrator.asr; re-exported for existing imports.
from narrator.asr import WhisperASR as WhisperASR  # noqa: E402
