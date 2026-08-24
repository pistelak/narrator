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
    _silent_runs,
    apply_gain,
    concatenate,
    declick,
    master,
    trim_silence,
)
from narrator.chunking import MAX_CHARS, plan_segments
from narrator.synth import SynthConfig, resolve_reference, synthesize_chunk
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
from narrator.verify import default_verifier, format_word_diagnostics, normalize


class RenderFailed(RuntimeError):
    """Some chunk could not be rendered correctly, and no file was written."""

    def __init__(self, report: RenderReport, takes: Path | None = None,
                 cached: int = 0) -> None:
        failures = report.failures
        detail = "\n".join(
            f"  chunk {c.index}: coverage {c.coverage:.2f}"
            + (f", dropped {c.dropped_sentence!r}" if c.dropped_sentence else "")
            + f" :: {c.text[:60]}..."
            + (f"\n    {format_word_diagnostics(c.word_diagnostics)}"
               if c.word_diagnostics else "")
            for c in failures[:10]
        )
        # What makes this refusal cheap to act on: fix the line, run again, pay
        # for that line, where before the 87 chunks that passed were discarded
        # along with the one that did not.
        #
        # `cached` counts THIS render's chunks that the store read or filed —
        # not the chunks that passed, and not the files in the directory. Not
        # every passing chunk is stored (an unidentifiable verifier, a
        # rise-wanting chunk and a failed write each produce none), and a
        # directory holding another script's takes proves nothing about this one.
        # Promising reuse that will not happen sends someone into a 25-minute
        # render expecting a 20-second one.
        resume = ""
        if takes is not None and cached:
            resume = (f"\n{cached} of this render's chunks are cached in {takes}; "
                      "re-running after a fix re-synthesises only what changed.")
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
    deleted entry before the requested index was ever reached.

    That sharing is worth stating plainly, because it is what content-addressing
    means: chunks with identical inputs are one entry, so rerolling one of them
    replaces the take that any LATER identical chunk will then reuse. The
    alternative is the chunk index in the key, which was rejected for a bigger
    reason — it would re-render an entire episode because a paragraph was
    inserted at the top.

    An index past the end of the render is refused rather than ignored: it is
    almost always a stale number from a previous script, and silently reusing
    everything looks exactly like a reroll that produced the same take again.

    One corner where a reroll does not propagate: chunk 0 under the default
    verifier has no key at all (see `_DeferredDefaultVerifier.identity`), so
    rerolling it cannot replace a stored entry, and a later identical chunk goes
    on serving the take a previous run filed."""


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
    plan = plan_segments(segments, cfg.max_chars)
    total = sum(1 for s in plan if isinstance(s, Text))
    if cfg.reroll and (max(cfg.reroll) >= total or min(cfg.reroll) < 0):
        raise ValueError(
            f"reroll={sorted(cfg.reroll)} names chunks outside this render, which has "
            f"{total} (0..{total - 1}). Nothing would be re-generated and the render "
            "would look like a reroll that changed nothing."
        )

    if cfg.synth.non_speech:
        # A chunk whose declared atoms leave no speech cannot be verified: the
        # reference is empty, `coverage("", anything)` answers 1.0, and the take
        # is certified and STORED. Preflight reports this too, but preflight is
        # advisory and nothing obliges a caller to run it.
        #
        # Refused rather than failed, because it is an input error of the same
        # kind `Text.__post_init__` already refuses — "something to be spoken"
        # with nothing to speak. It only becomes visible one layer later, after
        # removal. A verification verdict would be the wrong shape: it would
        # override `NullVerifier`, which opts out of checking the AUDIO, not out
        # of the library refusing an incoherent request.
        for at, planned in enumerate(s for s in plan if isinstance(s, Text)):
            # The question is whether any WORD survives removal, not whether any
            # character does. An earlier version asked `.strip()`, which lets
            # punctuation stand in for speech: `Text("<|pause|> ...")` resolved to
            # "...", walked past this guard, and was certified at 1.0 against a
            # transcript of totally different words — the exact vacuous pass this
            # guard exists to refuse, one character class further out. `normalize`
            # is what the verifier itself reduces the reference to, so asking it
            # here is asking the same question the gate will ask.
            #
            # It also restores preflight's one contract. Preflight already asked
            # the normalize question, so it reported this chunk doomed while
            # render shipped it clean — a FAILING preflight chunk that was not in
            # fact doomed. Two predicates for one question; there is now one, and
            # it is the stricter one, because the looser one was certifying audio
            # nobody had checked.
            #
            # Scoped to chunks that REMOVAL emptied, which is preflight's
            # condition too. Caller-written punctuation is a different question:
            # `Text("...")` certifies vacuously whether or not any atom is
            # declared elsewhere in the render, so refusing it only when some
            # unrelated atom happens to be configured would make the same script
            # legal or illegal depending on a setting it never used. That hole is
            # real and named here rather than half-closed (a review caught the
            # half-closed version); closing it is a verifier-policy change with
            # its own measurement, not a side effect of this guard.
            resolved = resolve_reference(planned.text, cfg.synth)
            if resolved != planned.text and not normalize(
                    resolved, (planned.voice or voice).lang).split():
                raise ValueError(
                    f"chunk {at} has no speech left after declared non_speech is "
                    f"removed, so there is nothing to verify: {planned.text[:60]!r}"
                )

    pieces: list[Audio | Gap] = []
    results: list[ChunkResult] = []
    # Where each result begins, as an index into `pieces`. One number per result,
    # recorded BEFORE its audio is appended, so a chunk with no audio needs no
    # special case: it begins at the piece that follows it, which is exactly the
    # position it would have occupied. A caller reading the report needs that
    # position to line the report up with the file, and the previous shape —
    # a parallel `owners` list plus an `owners_for_empty` side table, reconciled
    # in two passes — spent a re-sum of every earlier piece on each empty result
    # to say the same thing.
    starts_at: list[int] = []
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
        starts_at.append(len(pieces))
        index += 1
        if result.audio.size:
            # The voice's declared gain lands here, before stitching: a level
            # offset between reference clips belongs to the speaker, not to a
            # chunk, so every chunk of that voice moves by the same amount and
            # the performance inside each one is left as synthesised.
            trimmed = declick(trim_silence(result.audio, backend.sample_rate), backend.sample_rate)
            # Measured HERE, on the buffer that actually ships. synth cannot
            # compute it: the sentence-split path is trimmed per sentence and
            # then trimmed AGAIN at the assembly's outer edges, against a
            # threshold relative to the whole assembly, so only this trim's
            # output is the true length.
            result.shipped_s = len(trimmed) / backend.sample_rate
            pieces.append(apply_gain(trimmed, chunk_voice.gain_db))
        else:
            # No audio, so no piece appended — and nothing else to record, since
            # `starts_at` already holds the position it would have occupied.
            result.shipped_s = 0.0
        # Fired between chunks, which is where a real kill lands and what makes
        # progress progress. `start_s` is necessarily still None here: it needs
        # the settled rate and the whole piece list, neither of which exists
        # mid-loop. Documented on the field rather than worked around.
        if cfg.on_progress is not None:
            cfg.on_progress(result, total)

    # Offsets are assigned AFTER the loop, at the settled rate — the same reason
    # gap samples are allocated here rather than where the Gap was seen. A
    # backend that only learns its true rate during the first synthesis would
    # otherwise place every earlier chunk with a stale one.
    at = 0
    offsets: list[int] = []
    for piece in pieces:
        offsets.append(at)
        at += (int(piece.seconds * backend.sample_rate)
               if isinstance(piece, Gap) else len(piece))
    # One past the end, so a result whose audio was empty and which nothing
    # follows still resolves — it begins where the file ends.
    offsets.append(at)
    for result, piece_at in zip(results, starts_at, strict=True):
        result.start_s = offsets[piece_at] / backend.sample_rate

    raw = concatenate([
        np.zeros(int(p.seconds * backend.sample_rate), dtype=np.float32)
        if isinstance(p, Gap) else p
        for p in pieces
    ])
    audio, lufs, peak = master(raw, backend.sample_rate, cfg.mastering)

    frames = audio.shape[0] if audio.ndim else 0
    unscripted = _unscripted_silence(pieces, audio, backend.sample_rate, cfg.synth)
    report = RenderReport(
        out_path=out,
        duration_s=frames / backend.sample_rate,
        chunks=results,
        loudness_lufs=lufs,
        peak_dbfs=peak,
        render_s=time.perf_counter() - started,
        unscripted_silence_s=unscripted,
        takes_unwritten=store.write_failures if store is not None else 0,
    )

    if cfg.quarantine and not report.clean:
        raise RenderFailed(report, cfg.takes, store.usable if store is not None else 0)

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


def _unscripted_silence(pieces: list[Audio | Gap], audio: Audio, sample_rate: int,
                        cfg: SynthConfig) -> float:
    """Longest silent run in the written audio that no `Gap` accounts for.

    Measured on the stitched result, because that is where the defect appears: a
    run can form across a join from material no single chunk contains enough of
    to fail. Declared gaps are subtracted rather than searched around — narrator
    allocated them itself, at known offsets, so their spans are exactly the
    silence the caller asked for.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if mono.size == 0:
        return 0.0

    declared: list[tuple[int, int]] = []
    at = 0
    for piece in pieces:
        if isinstance(piece, Gap):
            n = int(piece.seconds * sample_rate)
            declared.append((at, at + n))
            at += n
        else:
            at += len(piece)

    worst = 0.0
    for start, end in _silent_runs(mono, sample_rate, cfg.silence_drop_db):
        overlap = sum(min(end, b) - max(start, a) for a, b in declared
                      if min(end, b) > max(start, a))
        worst = max(worst, (end - start - overlap) / sample_rate)
    return worst


def _write(out: Path, audio: Audio, sample_rate: int) -> None:
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sample_rate, subtype="PCM_16")
