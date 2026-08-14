"""The pipeline: segments in, one mastered file out, and the truth about it.

    render(segments, voice, backend, out) -> RenderReport

`RenderReport.clean` is the field that matters. By default a render that could not
produce correct audio for some chunk **raises** rather than writing a file, because
the failure this library exists to prevent is a plausible file that nobody knows is
wrong. Pass `quarantine=False` to get the file plus a report that says so.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from narrator.audio import MasterConfig, concatenate, declick, master, trim_silence
from narrator.chunking import MAX_CHARS, chunk
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.types import (
    Audio,
    Backend,
    ChunkResult,
    Gap,
    RenderReport,
    Segment,
    Text,
    Verifier,
    Voice,
)
from narrator.verify import default_verifier


class RenderFailed(RuntimeError):
    """Some chunk could not be rendered correctly, and no file was written."""

    def __init__(self, report: RenderReport) -> None:
        failures = report.failures
        detail = "\n".join(
            f"  chunk {c.index}: coverage {c.coverage:.2f}"
            + (f", dropped {c.dropped_sentence!r}" if c.dropped_sentence else "")
            + f" :: {c.text[:60]}..."
            for c in failures[:10]
        )
        super().__init__(
            f"{len(failures)} of {len(report.chunks)} chunks failed verification:\n{detail}\n"
            "No file written. Pass quarantine=False to write anyway."
        )
        self.report = report


@dataclass(frozen=True)
class RenderConfig:
    max_chars: int = MAX_CHARS
    synth: SynthConfig = SynthConfig()
    """Carries `pronunciation`, applied at synthesis only — see SynthConfig."""
    mastering: MasterConfig = MasterConfig()
    quarantine: bool = True
    on_progress: Callable[[ChunkResult, int], None] | None = None


def render(
    segments: list[Segment],
    voice: Voice,
    backend: Backend,
    out: Path,
    verifier: Verifier | None = None,
    cfg: RenderConfig = RenderConfig(),
) -> RenderReport:
    """Render `segments` to `out`.

    Gaps are honoured exactly as given. This library never invents, lengthens or
    shortens one: in a teaching script a pause is content — the listener is meant
    to answer during it — and a renderer that "improves" the timing is editing.

    `verifier=None` means the library's own policy, `default_verifier`, built
    against the backend's actual sample rate — the assembly callers used to do
    by hand, and got wrong (one forgot `source_rate`, which silently corrupts
    every verdict). Pass `NullVerifier()` to opt out of verification, or a
    custom verifier to override.
    """
    if not getattr(backend, "sample_rate", 0):
        raise ValueError(
            f"{type(backend).__name__}.sample_rate is 0. A backend must know its rate "
            "before rendering: a leading Gap would otherwise allocate zero samples and "
            "vanish from an otherwise clean render. Call the backend's load()/prepare "
            "step, or set sample_rate explicitly."
        )
    if verifier is None:
        # The pronunciation lexicon doubles as the verifier's sound-alike list:
        # each pair names a written form and what the audio will actually say.
        verifier = default_verifier(backend.sample_rate,
                                    sound_alikes=cfg.synth.pronunciation)
    started = time.perf_counter()
    plan = _plan(segments, cfg.max_chars)
    total = sum(1 for s in plan if isinstance(s, Text))

    pieces: list[Audio] = []
    results: list[ChunkResult] = []
    index = 0

    for segment in plan:
        if isinstance(segment, Gap):
            pieces.append(np.zeros(int(segment.seconds * backend.sample_rate), dtype=np.float32))
            continue

        result = synthesize_chunk(segment.text, index, backend, verifier, voice, cfg.synth)
        results.append(result)
        index += 1
        if cfg.on_progress is not None:
            cfg.on_progress(result, total)
        if result.audio.size:
            pieces.append(declick(trim_silence(result.audio, backend.sample_rate), backend.sample_rate))

    raw = concatenate(pieces)
    audio, lufs, peak = master(raw, backend.sample_rate, cfg.mastering)

    frames = audio.shape[0] if audio.ndim else 0
    report = RenderReport(
        out_path=out,
        duration_s=frames / backend.sample_rate,
        chunks=results,
        loudness_lufs=lufs,
        peak_dbfs=peak,
        render_s=time.perf_counter() - started,
    )

    if cfg.quarantine and not report.clean:
        raise RenderFailed(report)

    _write(out, audio, backend.sample_rate)   # master() already laid out the channels
    return report


def _plan(segments: list[Segment], max_chars: int) -> list[Segment]:
    """Flatten segments: gaps pass through, texts become chunk-sized Texts."""
    plan: list[Segment] = []
    for segment in segments:
        if isinstance(segment, Gap):
            plan.append(segment)
        elif isinstance(segment, Text):
            plan.extend(Text(piece) for piece in chunk(segment.text, max_chars))
        else:  # pragma: no cover - guarded by the type union
            raise TypeError(f"Not a segment: {segment!r}")
    return plan


def _write(out: Path, audio: Audio, sample_rate: int) -> None:
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sample_rate, subtype="PCM_16")
