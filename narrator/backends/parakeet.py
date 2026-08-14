"""Parakeet TDT 0.6B v3 — the second opinion for verification.

An ASR here, not a TTS backend: this exists so a `CascadeVerifier` can consult a
recogniser that shares no architecture with Whisper. Whisper is an autoregressive
transformer with a strong internal language model; Parakeet is a TDT decoder that
transcribes what it hears with little smoothing. They fail differently, which is
the entire point — a chunk both misread the same way is evidence about the audio,
not about either model.

Measured on 82 real Czech chunks (bench/asr_headtohead.py): the two disagree on
which chunks to reject about 11% of the time, near-symmetrically, and Parakeet
runs ~2.4x faster than whisper-large-v3-turbo on Apple Silicon — which is why it
is the PRIMARY in the cascade and Whisper the escalation.

**v3 only.** v2 is English-only and returns nonsense on Czech ("Ahi Proceed event
three pudding infinite, dk."), which would fail every chunk. v3 covers 25
European languages and detects the language itself — `lang` is accepted for the
ASR protocol but not passed on.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any

from narrator.types import Audio


@dataclass
class ParakeetASR:
    repo: str = "mlx-community/parakeet-tdt-0.6b-v3"
    source_rate: int = 24_000
    """Sample rate of the audio handed to `transcribe`. Must match the BACKEND —
    same trap as WhisperASR.source_rate, see that docstring."""

    _model: Any = field(default=None, repr=False, compare=False)

    def transcribe(self, audio: Audio, lang: str) -> str:
        if audio.size == 0:
            return ""
        if self._model is None:
            try:
                from parakeet_mlx import from_pretrained
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "Parakeet verification needs: pip install 'narrator[parakeet]'"
                ) from exc
            self._model = from_pretrained(self.repo)

        import soundfile as sf

        # parakeet-mlx takes a file path and resamples internally.
        with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
            sf.write(fh.name, audio, self.source_rate)
            return self._model.transcribe(fh.name).text.strip()
