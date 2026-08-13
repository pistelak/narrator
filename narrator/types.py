"""Core types.

Deliberately dataclasses rather than tuples. The predecessor of this library
passed a six-field tuple between its retry loop and three call sites; adding a
seventh field broke two of them silently, and one of the breaks only fired on a
code path that ran when synthesis failed — so it survived a clean fourteen-minute
render before killing a later one. Named fields make that class of bug a type
error instead of a runtime surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# Mono float32 PCM. Every audio value in this library is this and nothing else.
Audio = np.ndarray


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Text:
    """Something to be spoken."""
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Text segment is empty")


@dataclass(frozen=True)
class Gap:
    """Deliberate silence, in seconds.

    Callers own the meaning: a breath between paragraphs, a beat at a section
    boundary, or a retrieval pause the listener is meant to think through. This
    library only renders the duration — it never invents, lengthens or shortens
    one, because in a teaching script a pause is content.
    """
    seconds: float

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"Gap must be positive, got {self.seconds}")


Segment = Text | Gap
"""A script is an ordered sequence of these.

This is the entire interface between a caller and this library. Callers keep
their own markup — `[PAUSE 3]` lines, SSML, segment headers, a lexicon — and
resolve it into segments. Narrator never learns what a `[PAUSE]` is, and callers
never learn what a frame cap is.
"""


@dataclass(frozen=True)
class Voice:
    """A pinned narrator.

    Zero-shot models invent a voice per call. Across ~100 chunks that drifts
    audibly, so a reference is required rather than optional. Reuse the same one
    across every episode of a series: listeners adapt to a specific synthetic
    voice, and the adaptation is large and long-lived (42% -> 78% intelligibility
    over eight days of exposure in Schwab, Nusbaum & Pisoni 1985, still +32 points
    at six months).
    """
    audio_path: Path
    transcript: str
    lang: str = "en"


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """What happened to one chunk. The unit of the render report."""
    index: int
    text: str
    audio: Audio
    duration_s: float
    attempts: int
    ok: bool
    coverage: float = 1.0
    dropped_sentence: str = ""
    transcript: str = ""
    recovered_by: str = ""      # "", "retry", or "sentence-split"

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class RenderReport:
    """The outcome of a render. Truthful about what failed.

    A caller must be able to tell "this audio is what I asked for" from "this
    audio is plausible". The predecessor could not: it shipped an episode with
    seven silently dropped sentences, one of them a whole question that left a
    pause and an answer with nothing between them, and every chunk passed.
    """
    out_path: Path
    duration_s: float
    chunks: list[ChunkResult] = field(default_factory=list)
    loudness_lufs: float = 0.0
    peak_dbfs: float = 0.0
    render_s: float = 0.0

    @property
    def failures(self) -> list[ChunkResult]:
        return [c for c in self.chunks if not c.ok]

    @property
    def clean(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        rec = sum(1 for c in self.chunks if c.recovered_by)
        return (
            f"{self.out_path.name}: {self.duration_s / 60:.1f} min | "
            f"{len(self.chunks)} chunks, {len(self.failures)} failed, {rec} recovered | "
            f"{self.loudness_lufs:.1f} LUFS, peak {self.peak_dbfs:.1f} dBFS | "
            f"rendered in {self.render_s / 60:.1f} min"
        )


# --------------------------------------------------------------------------
# Extension points
# --------------------------------------------------------------------------

@runtime_checkable
class Backend(Protocol):
    """A text-to-speech engine.

    Narrator drives the engine; the engine does not drive narrator. Everything
    that makes long-form work — chunking, verification, retries, stitching — sits
    above this line and is engine-independent, so swapping engines does not mean
    re-earning those lessons.
    """

    sample_rate: int

    def synthesize(self, text: str, voice: Voice, *, max_frames: int, temperature: float) -> Audio:
        """Speak `text` once. May return wrong or truncated audio — that is the
        caller's problem to detect, and `max_frames` is a hard stop so a runaway
        generation is bounded rather than unbounded."""
        ...

    def frames_per_second(self) -> int:
        """Acoustic frames per second, for turning a duration budget into `max_frames`."""
        ...


@dataclass
class Verdict:
    ok: bool
    coverage: float
    dropped_sentence: str = ""
    transcript: str = ""


@runtime_checkable
class Verifier(Protocol):
    """Decides whether synthesized audio actually says the text.

    This is the load-bearing component. Duration heuristics alone caught zero of
    eight real content drops in the measurement that motivated this library.
    """

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        ...
