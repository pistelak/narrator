"""The recognisers behind the round-trip check.

Speech recognition is verification infrastructure, not a TTS engine, which is
why these do not live in `backends/`: importing a 4-billion-parameter TTS
module to verify a Supertonic render was the wrong dependency direction, and
its install hint pointed at the wrong extra.

Two models on purpose. Whisper is an autoregressive transformer with a strong
internal language model; Parakeet is a TDT decoder that transcribes what it
hears with little smoothing. They fail differently, which is what makes a
second opinion worth having — a chunk both misread the same way is evidence
about the audio, not about either model. `default_verifier` runs Parakeet
first (measured ~2.4x faster on Apple Silicon) and Whisper on escalation.

Both carry the same trap: `source_rate` must match the BACKEND that produced
the audio, not these defaults. Paired with a 44.1 kHz engine while assuming
24 kHz, the resample silently changes time and pitch and every verdict becomes
unreliable.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any

from narrator.takes import package_version
from narrator.types import Audio


@dataclass
class WhisperASR:
    """mlx-whisper, for the round-trip check.

    large-v3-turbo specifically: it is rated best-in-class for Czech in the
    sibling STT benchmark. Do NOT substitute Parakeet TDT 0.6B **v2** — it is
    English-only and returns nonsense on Czech ("Ahi Proceed event three pudding
    infinite, dk."), which would fail every chunk. v3 is multilingual and viable.
    """

    repo: str = "mlx-community/whisper-large-v3-turbo"
    source_rate: int = 24_000
    """See the module docstring: must match the backend's actual rate."""

    @property
    def identity(self) -> str:
        """For the take store. `source_rate` is IN it, deliberately.

        A verdict obtained at the wrong rate is unreliable in the way this
        module's docstring describes, and a stored take carries its verdict
        rather than being re-verified. Leaving the rate out would let a take
        checked by a misconfigured verifier be picked up by a correctly
        configured render later — laundering exactly the corruption the
        source_rate rule exists to prevent.

        Known limit: `repo` names a mutable model repository, so re-published
        weights under an unchanged package version are invisible here. Pinning
        would mean resolving a revision from the hub cache on every render, and
        released recogniser weights are not re-published in practice — but a
        take made before such a change would be reused without the new
        recogniser ever hearing it. Bump SEMANTICS in verify.py if that happens.
        """
        return (f"whisper/{type(self).__qualname__}/{self.repo}/{self.source_rate}/"
                f"{package_version('mlx-whisper')}")

    def transcribe(self, audio: Audio, lang: str) -> str:
        try:
            import mlx_whisper
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Whisper verification needs: pip install 'narrator[higgs]'") from exc

        if audio.size == 0:
            return ""
        from narrator.audio import resample_to_16k
        audio_16k = resample_to_16k(audio, self.source_rate)
        return mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=self.repo,
            language=lang,
            temperature=0.0,
            condition_on_previous_text=False,
        )["text"].strip()


@dataclass
class ParakeetASR:
    """parakeet-tdt-0.6b-v3, the cascade's fast first opinion.

    **v3 only.** v2 is English-only — see WhisperASR's docstring. v3 covers 25
    European languages and detects the language itself; `lang` is accepted for
    the ASR protocol but not passed on.
    """

    repo: str = "mlx-community/parakeet-tdt-0.6b-v3"
    source_rate: int = 24_000
    """See the module docstring: must match the backend's actual rate."""

    _model: Any = field(default=None, repr=False, compare=False)

    @property
    def identity(self) -> str:
        """See WhisperASR.identity for why the rate belongs in this string."""
        return (f"parakeet/{type(self).__qualname__}/{self.repo}/{self.source_rate}/"
                f"{package_version('parakeet-mlx')}")

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
