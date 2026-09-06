"""Core types.

Deliberately dataclasses rather than tuples. The predecessor of this library
passed a six-field tuple between its retry loop and three call sites; adding a
seventh field broke two of them silently, and one of the breaks only fired on a
code path that ran when synthesis failed — so it survived a clean fourteen-minute
render before killing a later one. Named fields make that class of bug a type
error instead of a runtime surprise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# Mono float32 PCM. Every audio value in this library is this and nothing else.
Audio = np.ndarray

MAX_GAIN_DB = 60.0
"""Bound on `Voice.gain_db`. Two reference clips are worth a few dB of each
other; sixty is already absurd, and the values that do real damage are far
past it."""


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Text:
    """Something to be spoken.

    `voice` optionally pins this segment to a specific narrator, overriding the
    render's default voice. This is how a caller renders dialogue: it resolves
    its own speaker markup into per-segment voices, narrator only ever sees a
    different pinned reference. Chunking never crosses a segment, so a voice
    can never bleed into another speaker's turn by construction. None means
    "the render's default voice" — the single-narrator case is unchanged.
    """
    text: str
    voice: Voice | None = None

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
    """A pinned narrator: either a reference clip or a named preset.

    Zero-shot models invent a voice per call. Across ~100 chunks that drifts
    audibly, so pinning is required rather than optional. Reuse the same voice
    across every episode of a series: listeners adapt to a specific synthetic
    voice, and the adaptation is large and long-lived (42% -> 78% intelligibility
    over eight days of exposure in Schwab, Nusbaum & Pisoni 1985, still +32 points
    at six months).

    Two shapes, because engines differ and the second one proved it: cloning
    engines want a clip plus its transcript, while engines that ship a voice bank
    want a name. A backend uses whichever it supports and says so plainly when
    handed the other.

    A reference is behavioural conditioning, not merely a timbre sample. Beyond
    identity and drift, it carries the speaker's register — which words they
    reach for — so a clip whose register differs from the script's can make the
    engine follow the voice rather than the page, which a word-for-word verifier
    then correctly rejects. Cast in the script's register, or write in the
    reference's; the measured case and why per-word equivalences do not fix it
    are in README, "Casting", and #9.
    """
    audio_path: Path | None = None
    transcript: str = ""
    lang: str = "en"
    preset: str | None = None
    gain_db: float = 0.0
    """Level correction for this voice, applied to every chunk it speaks.

    Two voices can arrive at different baseline levels, and mastering cannot
    repair it: loudness normalisation moves both speakers by the same amount,
    so the file gets no closer to balanced. This is where a caller states the
    offset.

    Deliberately cause-agnostic. Whether the difference came from recording gain
    in a reference clip, from how loudly someone performed, or from the engine's
    own behaviour on a given voice, it lands in the file the same way and this
    corrects it the same way. Narrator does not need to know which — and does
    not claim to: that a cloning engine carries its reference's level into its
    output is plausible and unmeasured here, so no rule rests on it.

    Declared, never inferred, and that boundary was expensive to find. Three
    designs that derived it from the rendered audio were built and measured, and
    each confused a quiet *delivery* with a quiet *reference*: pulling chunks
    toward the batch median boosted a deliberate whisper by 6 dB purely because
    a second speaker existed; a per-voice median then let one whispered aside
    cut a twenty-turn narrator by 6 dB; and guarding that with "a voice needs
    two chunks to count" only moved the failure to a cliff, where splitting the
    same aside in two flipped the outcome.

    Measuring the *reference clips* instead was tried too, and abandoned for the
    same reason one level deeper: separating a voice from the room it was
    recorded in needs voice-activity detection, and a threshold that is not one
    reads two seconds of room tone as 3 dB of level difference. So narrator does
    not estimate this. Get the number the way audio people already do — a level
    meter, or a calibration render compared against a reference — and state it
    here, and narrator will apply exactly that.

    Positive values are allowed but rarely wanted: prefer turning the loud
    voice down, so nothing new is pushed into the limiter.
    """

    def __post_init__(self) -> None:
        if self.audio_path is None and not self.preset:
            raise ValueError("A Voice needs either audio_path or preset — see the class docstring")
        if not math.isfinite(self.gain_db) or abs(self.gain_db) > MAX_GAIN_DB:
            # Anything unreasonable here is silent: it multiplies the audio
            # after verification has already passed, so the render still reports
            # clean. NaN or inf comes free from arithmetic on two unmeasurable
            # clips (-inf minus -inf); a wild finite value underflows to a file
            # of silence (-1e308) or overflows outright (+1e308). No real
            # correction between two reference clips is anywhere near this
            # bound, so refusing is never the wrong call.
            raise ValueError(
                f"Voice.gain_db must be finite and within ±{MAX_GAIN_DB:.0f} dB, "
                f"got {self.gain_db}"
            )


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
    """Length of the RAW synthesis, NOT the length of `audio` or of the file.

    The duration bounds were checked against this, so it keeps its diagnostic
    value — but synth trims the take before verifying it, and `audio` is that
    trimmed take, so a chunk reported here at 27.3 s can occupy 14.5 s of the
    file. Use `shipped_s` for the in-file length and `start_s` to locate it.

    One exception: a chunk recovered by sentence-split reports the ASSEMBLY's
    length, gaps included. No bound was ever checked against that aggregate —
    each sentence was bounded on its own raw take — so for those chunks this
    equals `len(audio) / sample_rate` and carries no raw-length evidence."""

    attempts: int
    ok: bool
    coverage: float = 1.0
    dropped_sentence: str = ""
    transcript: str = ""
    recovered_by: str = ""      # "", "retry", or "sentence-split"
    word_diagnostics: tuple[str, ...] = ()
    """The failed verdict's typed word codes — see verify.CoverageDetail."""
    reused: bool = False
    """This audio came from the take store, not from the engine on this run.

    Appended, like every field before it, because this is a dataclass and
    inserting mid-list silently rebinds positional constructor calls.

    A reused take carries the verdict it was stored with rather than being
    re-verified, which is only honest if the report says so. It is not a claim
    that the run measured anything: `attempts` is 0 on a reused chunk because
    this run spent no generations, while `recovered_by` is preserved, because it
    describes how the audio itself was made and that is still true."""

    silence_s: float = 0.0
    """Longest silence in this chunk's audio, in seconds, edges included.

    Appended, for the reason `reused` records above. Always reported, not only
    on failure: `SynthConfig.max_silence_s` deliberately refuses just the severe
    class that was measured, so this is the number that tells a caller whether
    the shorter band is real in their material — and the evidence for tightening
    the threshold later. A reused take reports what was stored with it, not 0.0.

    Edges included, because they are what `trim_silence` leaves behind rather
    than what it removes (issue #42). The one exception is a chunk recovered by
    sentence-split, which reports the interior measure of the assembly: its edges
    are measured against the whole assembly's speech level, where a deliberately
    quiet closing sentence reads as a multi-second run."""

    shipped_s: float | None = None
    """Seconds this chunk actually occupies in the written file, or None.

    Appended, for the reason `reused` records above — inserting it mid-list
    silently rebound every positional constructor call by one, which a review
    caught here.

    `duration_s` is the RAW synthesis length, measured before synth trims, so
    a chunk reported at 27.3 s can ship at 14.5 s and accumulating `duration_s`
    to locate a chunk drifts by every silence trimmed before it. That cost a real
    investigation in issue #21. This equals `len(audio) / sample_rate` once
    measured: `render` ships `audio` as verified, with only an edge fade and
    the voice's declared gain applied.

    None means "not measured" — a result that never went through `render`, or one
    restored from the take store before render overwrites it. Not 0.0, because
    0.0 is legitimate: a chunk whose every attempt raised occupies nothing."""

    start_s: float | None = None
    """Seconds from the start of the written file to this chunk, or None.

    `shipped_s` alone cannot locate a chunk: `Gap` segments never appear in
    `RenderReport.chunks`, so summing chunk lengths skips the silence between
    them. This is the number that answers "where in the file is chunk N".

    Still None inside an `on_progress` callback, necessarily: that fires between
    chunks — which is where a real kill lands — while this needs the settled rate
    and the complete piece list. Every result in the returned report has it."""


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
    unscripted_silence_s: float = 0.0
    """Longest silence in the written file that no `Gap` asked for.

    Reported, not refused, and deliberately so. Renders emit stretches of silence
    the script never declared (issue #21), and the per-chunk gate cannot see one
    that forms only after stitching — it measures one chunk's trimmed audio, so
    silence that only exists once two chunks meet belongs to this number alone.

    Refusing on this number was implemented and cut. Removing edge material by
    level alone deletes real audio: a whispered clause sits in the same band as
    an unwanted tail, and at the time verification ran BEFORE trimming, so the
    recogniser credited words that removal then took out of the shipped file —
    silent content loss, which is the failure this library exists to prevent.
    Measured: 1.5-2.0 s of a quiet opening removed after being verified. That
    ordering is gone (synth now trims once, before the ASR, and nothing after
    verification removes a span — declick, gain and mastering change sample
    values, never length; see `_best_attempt`), but the lesson is not: any
    level-based removal added AFTER verification reopens the same hole.

    So this measures and says so. Declared `Gap` spans are excluded, because
    those are exactly the silence the caller asked for. Once there is data on
    what a real corpus produces, a threshold can be argued from it rather than
    guessed."""

    takes_unwritten: int = 0
    """Verified takes the store could not file (a full disk, an unwritable path).

    Reported rather than raised, and reported rather than swallowed. The store is
    an optimisation, so a failure to write one must not fail a render that
    synthesised correctly — but a store that silently caches nothing looks
    identical to one that is working until the next render bills for it."""

    @property
    def failures(self) -> list[ChunkResult]:
        return [c for c in self.chunks if not c.ok]

    @property
    def clean(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        rec = sum(1 for c in self.chunks if c.recovered_by)
        # Reused chunks are named in the summary because a reused take carries a
        # verdict this run did not measure. The count is what tells a reader how
        # much of "0 failed" was checked just now and how much was checked before.
        reused = sum(1 for c in self.chunks if c.reused)
        cached = f", {reused} reused" if reused else ""
        unwritten = f" | {self.takes_unwritten} take(s) not cached" if self.takes_unwritten else ""
        return (
            f"{self.out_path.name}: {self.duration_s / 60:.1f} min | "
            f"{len(self.chunks)} chunks, {len(self.failures)} failed, {rec} recovered"
            f"{cached} | "
            f"{self.loudness_lufs:.1f} LUFS, peak {self.peak_dbfs:.1f} dBFS | "
            f"rendered in {self.render_s / 60:.1f} min{unwritten}"
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

    honours_frame_cap: bool
    """Whether `max_frames` actually bounds generation.

    Not every engine has the concept. Autoregressive models can run away and need
    a hard stop; a diffusion or flow model given a fixed step count cannot, and
    has no parameter to accept one. A backend that returns False is telling the
    caller that reaching the cap is not a detectable event for it, so that signal
    must not be used as evidence of a runaway.

    This exists because a second backend broke the assumption that it did not.
    """

    def synthesize(self, text: str, voice: Voice, *, max_frames: int, temperature: float) -> Audio:
        """Speak `text` once. May return wrong or truncated audio — that is the
        caller's problem to detect. When `honours_frame_cap` is True, `max_frames`
        is a hard stop so a runaway is bounded rather than unbounded; otherwise it
        is advisory and may be ignored."""
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
    word_diagnostics: tuple[str, ...] = ()
    """Typed word codes behind a rejection — see verify.CoverageDetail.
    Empty on a pass, like `dropped_sentence`. Trailing with a default on
    purpose: Verdict is constructed positionally on the cheap-check path."""


@runtime_checkable
class Verifier(Protocol):
    """Decides whether synthesized audio actually says the text.

    This is the load-bearing component. Duration heuristics alone caught zero of
    eight real content drops in the measurement that motivated this library.

    Exported for typing, not for policy: there is one verification policy
    (`default_verifier`) and good reason for it to stay singular — an
    alternative policy is far more likely to be a weaker one. The genuinely
    replaceable dependency is `ASR`: swapping the speech recogniser is a real
    need, swapping what "correct" means is not.
    """

    def verify(self, audio: Audio, text: str, lang: str) -> Verdict:
        ...


@runtime_checkable
class ASR(Protocol):
    """Speech recognition, behind a seam so verification is testable without a model."""

    def transcribe(self, audio: Audio, lang: str) -> str: ...


class Identified(Protocol):
    """Declares a stable string identity, so takes made with this object can be cached.

    Separate from `Backend`, `Verifier` and `ASR` on purpose, and NOT
    @runtime_checkable. Those three are: a protocol member is structurally
    required, so declaring `identity` on them would make every implementation
    without one — the third-party ones this is meant to leave working, and this
    repo's own `isinstance(HiggsBackend(), Backend)` assertions — stop satisfying
    the protocol. Read it with `takes.identity_of`, which returns None for
    anything that does not declare one, and None disables the store.

    An identity must cover everything about the object that reaches the audio or
    the verdict: its class, its model, the version of the package that does the
    work, and any setting that changes generation. Two objects sharing an
    identity is a promise that their output is interchangeable.

    **A subclass that adds configuration must extend this**, and the bundled
    implementations cannot do it for you. They name their own class, so a
    subclass is never confused with its parent, and `CoverageVerifier` walks its
    instance state — but a backend cannot, because its instance also holds a
    loaded model and a reference cache that change within a single render, and
    an identity built from those would never match itself twice. So a backend
    subclass that adds a pitch or a sampler setting is on its own: override
    `identity`, or return None from it and take the re-synthesis.
    """

    identity: str
