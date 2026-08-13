"""A backend that fakes synthesis, and an ASR that can hear what it faked.

Every defect the predecessor pipeline shipped lived in orchestration, not in the
model: a ranking bug that discarded a passing retry, a frame cap that truncated
short sentences by construction, an exception that aborted a whole render, a
similarity metric that failed in both directions. Each was found by burning a
twenty-minute render. Each is a millisecond here.

The pair works by conspiracy. `FakeBackend` stamps an id into the first sample of
the audio it returns and remembers what that audio "says" — including when it
says the wrong thing. `FakeASR` reads the stamp and reports it. So a test can
inject "this call drops a sentence" and the verifier will genuinely detect a
dropped sentence, through the real coverage code, without a model anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from narrator.types import Audio, Voice

# Closing quotes and brackets may sit between the terminator and the space.
# Missing them merged sentences, which silently restored the aggregate
# scoring that per-sentence coverage exists to replace.
_SENTENCE_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')


class Failure(str, Enum):
    """Failure modes observed in real engines, reproducible on demand."""

    NONE = "none"
    DROP_SENTENCE = "drop_sentence"   # silently omits a sentence — the headline failure
    REPEAT = "repeat"                 # loops one sentence; "Not the keeper" x4
    TRUNCATE = "truncate"             # stops partway, mid-thought
    RUNAWAY = "runaway"               # generates until the frame cap stops it
    RAISE = "raise"                   # engine throws


@dataclass
class FakeBackend:
    """Synthesises silence whose length is proportional to what it decided to say.

    `script` maps a 0-based call index to the failure it should exhibit, so a test
    can say "fail the first two attempts, then succeed" and exercise the retry
    ladder exactly.
    """

    sample_rate: int = 24000
    words_per_second: float = 2.5
    fps: int = 25
    honours_frame_cap: bool = True
    script: dict[int, Failure] = field(default_factory=dict)
    default: Failure = Failure.NONE

    calls: int = 0
    requests: list[str] = field(default_factory=list)
    max_frames_seen: list[int] = field(default_factory=list)
    _spoken: dict[int, str] = field(default_factory=dict)

    def frames_per_second(self) -> int:
        return self.fps

    def synthesize(self, text: str, voice: Voice, *, max_frames: int, temperature: float) -> Audio:
        index = self.calls
        self.calls += 1
        self.requests.append(text)
        self.max_frames_seen.append(max_frames)

        mode = self.script.get(index, self.default)
        if mode is Failure.RAISE:
            raise RuntimeError(f"fake engine failure on call {index}")

        spoken = self._apply(text, mode)
        duration = len(spoken.split()) / self.words_per_second

        # The frame cap is a hard stop in real engines, so honour it here — this
        # is what makes RUNAWAY bounded and detectable rather than unbounded.
        cap_seconds = max_frames / self.fps
        if mode is Failure.RUNAWAY or duration > cap_seconds:
            duration = cap_seconds

        samples = max(int(duration * self.sample_rate), 2)
        # A quiet tone, not silence. Silence made trim_silence collapse every
        # chunk to its guard band, so end-to-end tests exercised assembly on
        # 60 ms of nothing and pyloudnorm rejected the result as shorter than its
        # analysis block. A fake that does not survive the real pipeline tests
        # nothing downstream of itself.
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        audio = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        audio[0] = self._stamp(index)
        self._spoken[index] = spoken
        return audio

    def _apply(self, text: str, mode: Failure) -> str:
        sentences = [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]
        if mode is Failure.DROP_SENTENCE and len(sentences) > 1:
            return " ".join(sentences[:-1])
        if mode is Failure.REPEAT and sentences:
            return " ".join([sentences[0]] * max(len(sentences), 3))
        if mode is Failure.TRUNCATE:
            words = text.split()
            return " ".join(words[: max(len(words) // 3, 1)])
        if mode is Failure.RUNAWAY:
            return "babble " * 200
        return text

    # An id in the first sample survives trimming and concatenation well enough
    # for tests, and keeps the pair honest: the ASR cannot see the request, only
    # the audio, exactly as a real one cannot.
    @staticmethod
    def _stamp(index: int) -> np.float32:
        return np.float32((index + 1) * 1e-4)

    def heard(self, audio: Audio) -> str:
        if audio.size == 0:
            return ""
        index = int(round(float(audio[0]) / 1e-4)) - 1
        return self._spoken.get(index, "")


@dataclass
class FakeASR:
    """Transcribes by asking the backend what it actually said."""

    backend: FakeBackend
    perfect: bool = True

    def transcribe(self, audio: Audio, lang: str) -> str:
        spoken = self.backend.heard(audio)
        if self.perfect or not spoken:
            return spoken
        # Imitate the two harmless disagreements a real ASR produces, so tests
        # can confirm the verifier tolerates them rather than only tolerating
        # a transcript identical to the request.
        spoken = re.sub(r"\btwenty\b", "20", spoken)
        return re.sub(r"\bne (\w)", r"ne\1", spoken)
