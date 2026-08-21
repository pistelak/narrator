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
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np

from narrator.chunking import split_sentences
from narrator.types import Audio, Voice


class Failure(StrEnum):
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
    # New fields go AFTER the pre-existing ones, private included: this is a
    # dataclass, so inserting mid-list silently rebinds positional constructor
    # calls.
    voices_seen: list[Voice] = field(default_factory=list)
    amplitude: float = 0.1
    voice_amplitude: dict[Voice, float] = field(default_factory=dict)
    """Per-voice tone amplitude, so a test can stage the lopsided dialogue that
    `Voice.gain_db` exists for. Without it every voice speaks at one level and
    no test can tell a level correction from a no-op.

    Matched ignoring `gain_db`, because that is what it models: the level a
    reference clip happens to come out at, which the caller's declared correction
    does not change. It also has to be, now — `synth` zeroes `gain_db` before the
    engine call, so a lookup that saw it would miss every entry a test declared."""
    amplitude_script: dict[int, float] = field(default_factory=dict)
    """Call index -> amplitude, overriding `voice_amplitude`. Models one voice
    delivering one turn quietly — the whisper that three inference designs kept
    mistaking for a quiet reference clip."""

    @property
    def identity(self) -> str:
        """For the take store: what this fake was CONFIGURED to do, never what it
        has done so far.

        The injected failure script belongs here. A test that renders once, then
        re-renders with `{0: Failure.DROP_SENTENCE}` to prove the ladder recovers,
        would otherwise be served the take from the clean run and prove nothing.
        Call counters and remembered utterances are excluded for the opposite
        reason: they move on every synthesis, and a key that changes mid-render
        never hits.
        """
        script = ",".join(f"{i}:{m}" for i, m in sorted(self.script.items()))
        levels = ",".join(f"{i}:{a}" for i, a in sorted(self.amplitude_script.items()))
        voices = ",".join(f"{v}:{a}" for v, a in sorted(
            self.voice_amplitude.items(), key=lambda kv: str(kv[0])))
        return (f"fake/{self.sample_rate}/{self.fps}/{self.words_per_second}/"
                f"{self.honours_frame_cap}/{self.default}/[{script}]/"
                f"{self.amplitude}/[{levels}]/[{voices}]")

    def frames_per_second(self) -> int:
        return self.fps

    def synthesize(self, text: str, voice: Voice, *, max_frames: int, temperature: float) -> Audio:
        index = self.calls
        self.calls += 1
        self.requests.append(text)
        self.voices_seen.append(voice)
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
        level = self.amplitude_script.get(index, self._level_for(voice))
        audio = (level * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        audio[0] = self._stamp(index)
        self._spoken[index] = spoken
        return audio

    def _level_for(self, voice: Voice) -> float:
        """Declared amplitude for this voice's reference, `gain_db` ignored."""
        for candidate, level in self.voice_amplitude.items():
            if replace(candidate, gain_db=0.0) == replace(voice, gain_db=0.0):
                return level
        return self.amplitude

    def _apply(self, text: str, mode: Failure) -> str:
        sentences = split_sentences(text)
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
        index = round(float(audio[0]) / 1e-4) - 1
        return self._spoken.get(index, "")


@dataclass
class FakeASR:
    """Transcribes by asking the backend what it actually said."""

    backend: FakeBackend
    perfect: bool = True
    orthography: dict[str, str] = field(default_factory=dict)
    """Spoken form -> conventional spelling, to model a real ASR.

    An ASR hears sound and writes normal orthography. Handed audio synthesised
    from the respelling "Kalleh", it returns "Kalle" — which is precisely why a
    pronunciation lexicon must not reach the verifier. Without this the fake
    reports the respelling back and the asymmetry that caused three real render
    failures cannot be reproduced."""

    @property
    def identity(self) -> str:
        """Composed from the backend's, since this fake hears only what that fake
        said — the same recursion a real CoverageVerifier does over its ASR."""
        spellings = ",".join(f"{k}>{v}" for k, v in sorted(self.orthography.items()))
        return f"fake-asr/{self.backend.identity}/{self.perfect}/[{spellings}]"

    def transcribe(self, audio: Audio, lang: str) -> str:
        spoken = self.backend.heard(audio)
        for said, written in self.orthography.items():
            spoken = re.sub(rf"\b{re.escape(said)}\b", written, spoken)
        if self.perfect or not spoken:
            return spoken
        # Imitate the two harmless disagreements a real ASR produces, so tests
        # can confirm the verifier tolerates them rather than only tolerating
        # a transcript identical to the request.
        spoken = re.sub(r"\btwenty\b", "20", spoken)
        return re.sub(r"\bne (\w)", r"ne\1", spoken)
