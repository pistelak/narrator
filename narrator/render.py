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

from narrator.audio import (
    MasterConfig,
    apply_gain,
    concatenate,
    declick,
    master,
    trim_silence,
)
from narrator.chunking import MAX_CHARS, chunk
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.takes import TakeStore, identity_of
from narrator.types import (
    Audio,
    Backend,
    ChunkResult,
    Gap,
    RenderReport,
    Segment,
    Text,
    Verdict,
    Verifier,
    Voice,
)
from narrator.verify import default_verifier, format_word_diagnostics


class RenderFailed(RuntimeError):
    """Some chunk could not be rendered correctly, and no file was written."""

    def __init__(self, report: RenderReport, takes: Path | None = None) -> None:
        failures = report.failures
        detail = "\n".join(
            f"  chunk {c.index}: coverage {c.coverage:.2f}"
            + (f", dropped {c.dropped_sentence!r}" if c.dropped_sentence else "")
            + f" :: {c.text[:60]}..."
            + (f"\n    {format_word_diagnostics(c.word_diagnostics)}"
               if c.word_diagnostics else "")
            for c in failures[:10]
        )
        # The good takes are already on disk when a store is in use, which is
        # what makes this refusal cheap to act on: fix the line, run again, pay
        # for that line. Without it the 87 chunks that passed are discarded along
        # with the one that did not.
        resume = (f"\nThe {len(report.chunks) - len(failures)} chunk(s) that passed are cached "
                  f"in {takes}; re-running after a fix re-synthesises only what changed."
                  if takes is not None else "")
        super().__init__(
            f"{len(failures)} of {len(report.chunks)} chunks failed verification:\n{detail}\n"
            "No file written. Pass quarantine=False to write anyway." + resume
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

    takes: Path | None = None
    """Directory of verified takes to reuse and extend. None disables it.

    A directory rather than an injected store, and opt-in rather than default:
    narrator owns the one caching policy the way it owns the one verification
    policy, and a store nobody asked for would quietly write ~90 MB of takes next
    to a caller's output. With it, an edited script re-synthesises only the chunks
    whose inputs changed, and a killed render resumes from what it had finished."""

    reroll: frozenset[int] = frozenset()
    """Chunk indices to generate fresh, ignoring any stored take.

    How a caller asks a sampled model for another take of audio that verifies but
    does not SOUND right — a content key would otherwise return the same take
    forever. It bypasses the lookup rather than deleting the entry first: two
    identical paragraphs share one key, so an earlier occurrence would refill a
    deleted entry before the requested index was ever reached."""


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
        verifier = _DeferredDefaultVerifier(backend, cfg.synth.pronunciation)
    started = time.perf_counter()
    store = TakeStore(cfg.takes) if cfg.takes is not None else None
    plan = _plan(segments, cfg.max_chars)
    total = sum(1 for s in plan if isinstance(s, Text))

    pieces: list[Audio | Gap] = []
    results: list[ChunkResult] = []
    index = 0

    for segment in plan:
        if isinstance(segment, Gap):
            # Kept as a placeholder, not allocated here: a gap must be sized at
            # the rate the file is written at, and that rate may not be settled
            # yet. Supertonic declares 44100 at construction and corrects it to
            # the true rate during the first synthesis, so a leading 3.0 s Gap
            # allocated in this loop landed at the stale rate and played as
            # 5.51 s — the same corruption class _DeferredDefaultVerifier below
            # exists to prevent. Materializing after the loop makes allocation
            # and _write read the same value by construction; a gaps-only
            # render never settles the rate, but then the file is written at
            # the declared rate too, so samples and header still agree.
            pieces.append(segment)
            continue

        chunk_voice = segment.voice or voice
        result = synthesize_chunk(segment.text, index, backend, verifier,
                                  chunk_voice, cfg.synth,
                                  store=store, reuse=index not in cfg.reroll)
        results.append(result)
        index += 1
        if cfg.on_progress is not None:
            cfg.on_progress(result, total)
        if result.audio.size:
            # The voice's declared gain lands here, before stitching: a level
            # offset between reference clips belongs to the speaker, not to a
            # chunk, so every chunk of that voice moves by the same amount and
            # the performance inside each one is left as synthesised.
            trimmed = declick(trim_silence(result.audio, backend.sample_rate), backend.sample_rate)
            pieces.append(apply_gain(trimmed, chunk_voice.gain_db))

    raw = concatenate([
        np.zeros(int(p.seconds * backend.sample_rate), dtype=np.float32)
        if isinstance(p, Gap) else p
        for p in pieces
    ])
    audio, lufs, peak = master(raw, backend.sample_rate, cfg.mastering)

    frames = audio.shape[0] if audio.ndim else 0
    report = RenderReport(
        out_path=out,
        duration_s=frames / backend.sample_rate,
        chunks=results,
        loudness_lufs=lufs,
        peak_dbfs=peak,
        render_s=time.perf_counter() - started,
        takes_unwritten=store.write_failures if store is not None else 0,
    )

    if cfg.quarantine and not report.clean:
        raise RenderFailed(report, cfg.takes)

    _write(out, audio, backend.sample_rate)   # master() already laid out the channels
    return report


@dataclass
class _DeferredDefaultVerifier:
    """default_verifier, constructed on first use rather than up front.

    A backend may only learn its true sample rate during its first synthesis —
    Supertonic corrects 44100 to 22050 there — and ASRs freeze the rate they
    are constructed with. Building eagerly would bake in the stale rate, the
    exact silent corruption the source_rate rule exists to prevent. The first
    verification necessarily runs after the first synthesis, so building here
    always sees the corrected rate.
    """

    backend: Backend
    sound_alikes: tuple[tuple[str, str], ...]
    _verifier: Verifier | None = None

    @property
    def identity(self) -> str | None:
        """Absent until the inner verifier exists — so chunk 0 is never cached.

        The identity has to name the rate its ASRs were built with, and that rate
        is only settled by the first synthesis. Resolving it early to win the
        lookup would bake in the stale value, which is the corruption this class
        exists to prevent; claiming an identity we have not resolved would be
        worse still. So the first chunk of a render on the default policy pays a
        generation nobody gets to reuse, and every chunk after it hits. One in
        eighty-eight, against a rule that must not be softened.
        """
        return identity_of(self._verifier)

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        if self._verifier is None:
            self._verifier = default_verifier(self.backend.sample_rate,
                                              sound_alikes=self.sound_alikes)
        return self._verifier.verify(audio, text, lang)


def _plan(segments: list[Segment], max_chars: int) -> list[Segment]:
    """Flatten segments: gaps pass through, texts become chunk-sized Texts."""
    plan: list[Segment] = []
    for segment in segments:
        if isinstance(segment, Gap):
            plan.append(segment)
        elif isinstance(segment, Text):
            plan.extend(Text(piece, voice=segment.voice)
                        for piece in chunk(segment.text, max_chars))
        else:  # pragma: no cover - guarded by the type union
            raise TypeError(f"Not a segment: {segment!r}")
    return plan


def _write(out: Path, audio: Audio, sample_rate: int) -> None:
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sample_rate, subtype="PCM_16")
