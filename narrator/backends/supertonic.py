"""Supertonic 3 — the engine this library's Higgs backend replaced.

Kept, and worth keeping, for three reasons beyond nostalgia:

1. It is the **second implementation of `Backend`**, which is the only way the
   claim "swapping engines does not mean re-earning the long-form machinery" gets
   tested rather than asserted.
2. It is tiny and fast — 99M parameters, 404 MB on disk, RTF around 0.02 against
   Higgs' 0.75. For a smoke test or a draft listen, forty times faster matters.
3. The existing back-catalogue was rendered with it, so re-rendering an old
   episode on the old engine stays possible.

Writing it found two things the protocol had wrong, both now fixed upstream:

- **No frame cap.** `TTS.synthesize` has no parameter that bounds generation, so
  `max_frames` cannot be honoured. `honours_frame_cap = False` says so, and the
  retry ladder stops treating "reached the cap" as evidence of a runaway — which
  it would otherwise have done on every single chunk, since a backend that never
  hits its cap and one that always does are indistinguishable to a check that
  only compares durations.
- **Preset voices.** Supertonic ships a voice bank (M1..M5, F1..F5) as well as
  accepting a reference clip. `Voice` only had the clip, so a whole class of
  engine could not be addressed at all.

The engine is not otherwise recommended. On the Czech-with-embedded-English case
that this library's consumer cares most about, it produced "koudování base 64Hack"
for "kódování base 64 check" — mispronouncing a plain Czech word and turning
*check* into *hack*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from narrator.types import Audio, Voice

MODEL = "supertonic-3"
FPS = 25
"""Nominal. Unused for capping — see `honours_frame_cap` — but the protocol needs
a number to turn a duration budget into a frame budget."""


@dataclass
class SupertonicBackend:
    """Wraps the supertonic ONNX runtime."""

    model_id: str = MODEL
    default_preset: str = "M1"
    total_steps: int = 10
    """Diffusion steps. 10 is smoother than the package default of 8."""

    speed: float = 0.9
    """Measured for learning material; 1.0 was reported as "quite fast"."""

    sample_rate: int = 44_100
    """Supertonic 3 is documented at 44.1 kHz.

    Known at construction rather than discovered on first synthesis: `render`
    allocates gap silence from this, so a zero here made a LEADING Gap allocate
    zero samples and vanish from a render still reporting itself clean. Verified
    against the written file on first synthesis and corrected if it differs."""

    fps: int = FPS
    honours_frame_cap: bool = False
    """`TTS.synthesize` accepts no generation bound. Saying so keeps the retry
    ladder from reading a meaningless signal as a runaway."""

    _rate_verified: bool = field(default=False, repr=False)
    _tts: Any = field(default=None, repr=False)
    _styles: dict[str, Any] = field(default_factory=dict, repr=False)

    def frames_per_second(self) -> int:
        return self.fps

    def load(self) -> None:
        if self._tts is not None:
            return
        try:
            from supertonic import TTS
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "The Supertonic backend needs the 'supertonic' extra: "
                "pip install 'narrator[supertonic]'"
            ) from exc
        self._tts = TTS(model=self.model_id, auto_download=True)

    def _style_for(self, voice: Voice) -> Any:
        """Resolve a Voice to a Supertonic style, clip or preset, cached."""
        key = str(voice.audio_path) if voice.audio_path else (voice.preset or self.default_preset)
        if key in self._styles:
            return self._styles[key]

        if voice.audio_path is not None:
            if not voice.audio_path.is_file():
                raise FileNotFoundError(f"Voice reference not found: {voice.audio_path}")
            style = self._tts.get_voice_style_from_path(str(voice.audio_path))
        else:
            try:
                style = self._tts.get_voice_style(key)
            except Exception as exc:
                raise ValueError(
                    f"Unknown Supertonic preset {key!r}. The voice bank is M1..M5 and F1..F5."
                ) from exc

        self._styles[key] = style
        return style

    def synthesize(
        self, text: str, voice: Voice, *, max_frames: int, temperature: float
    ) -> Audio:
        """`max_frames` and `temperature` are both ignored.

        Supertonic exposes neither. Silently accepting them is the honest shape
        here — the protocol documents them as advisory when `honours_frame_cap`
        is False, and pretending to sample at a temperature this engine does not
        have would be worse than ignoring it.
        """
        self.load()
        wav, _ = self._tts.synthesize(
            text=text,
            voice_style=self._style_for(voice),
            lang=voice.lang,
            total_steps=self.total_steps,
            speed=self.speed,
        )
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim == 2:
            # Supertonic documents shape (1, num_samples). `mean(axis=1)` on that
            # returns a SINGLE sample — the whole utterance averaged to a point.
            # The test double returned 1-D, so no test caught it.
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[1] in (1, 2):
                audio = audio.mean(axis=1)
            else:
                raise RuntimeError(f"Unexpected waveform shape {audio.shape}")
        if not self._rate_verified:
            actual = self._discover_rate(audio)
            if actual != self.sample_rate:
                self.sample_rate = actual
            self._rate_verified = True
        return audio

    def _discover_rate(self, audio: Audio) -> int:
        """The package returns samples without a rate; ask it to write a file once."""
        import tempfile
        from pathlib import Path

        import soundfile as sf

        tmp = Path(tempfile.gettempdir()) / "narrator_supertonic_probe.wav"
        self._tts.save_audio(audio, str(tmp))
        with sf.SoundFile(str(tmp)) as handle:
            return int(handle.samplerate)
