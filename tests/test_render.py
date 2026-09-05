"""End-to-end pipeline tests, still with no model involved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from narrator.audio import (
    MasterConfig,
    concatenate,
    declick,
    soft_limit,
    to_channels,
    trim_silence,
)
from narrator.backends.fake import Failure, FakeASR, FakeBackend
from narrator.render import RenderConfig, RenderFailed, render
from narrator.synth import SynthConfig
from narrator.types import ChunkResult, Gap, Text, Voice
from narrator.verify import CoverageVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
SEGMENTS = [
    Text("Not the keeper. Not a stranger. Not any council with any mandate."),
    Gap(3.0),
    Text("They were sent to a destination that does not exist."),
]


def build(script=None):
    backend = FakeBackend(script=script or {})
    return backend, CoverageVerifier(FakeASR(backend))


# ------------------------------------------------------------ happy path

def test_writes_a_file_and_reports_clean(tmp_path: Path) -> None:
    backend, verifier = build()
    out = tmp_path / "episode.wav"
    report = render(SEGMENTS, VOICE, backend, out, verifier)
    assert report.clean
    assert out.is_file()
    assert len(report.chunks) == 2


def test_gap_duration_is_honoured_exactly(tmp_path: Path) -> None:
    """A pause is content. A renderer that 'improves' the timing is editing.

    Non-numeral words on purpose: the original "One two three four." is
    all-numeral, which the verifier refuses to certify — it only ever passed
    here through the empty-reference hole test_verify pins closed.
    """
    backend, verifier = build()
    short = render([Text("Alpha beta gamma delta.")], VOICE, backend, tmp_path / "a.wav", verifier)

    backend2, verifier2 = build()
    withgap = render(
        [Text("Alpha beta gamma delta."), Gap(5.0)], VOICE, backend2, tmp_path / "b.wav", verifier2
    )
    assert withgap.duration_s - short.duration_s == pytest.approx(5.0, abs=0.05)


def test_long_text_is_chunked(tmp_path: Path) -> None:
    backend, verifier = build()
    long = Text(" ".join(["This is a sentence of moderate length here."] * 30))
    report = render([long], VOICE, backend, tmp_path / "c.wav", verifier)
    assert len(report.chunks) > 3
    assert all(len(c.text) <= 250 for c in report.chunks)


# ------------------------------------------------------------ quarantine

def test_unrecoverable_chunk_raises_and_writes_nothing(tmp_path: Path) -> None:
    """The whole point. A plausible file nobody knows is wrong is the failure."""
    out = tmp_path / "episode.wav"
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    with pytest.raises(RenderFailed) as excinfo:
        render(SEGMENTS, VOICE, backend, out, verifier)
    assert not out.exists()
    assert excinfo.value.report.failures


def test_quarantine_can_be_waived_but_the_report_still_says_so(tmp_path: Path) -> None:
    out = tmp_path / "episode.wav"
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    report = render(
        SEGMENTS, VOICE, backend, out, verifier, RenderConfig(quarantine=False)
    )
    assert out.is_file()
    assert not report.clean
    assert report.failures


def test_failure_message_names_chunks_and_content(tmp_path: Path) -> None:
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    with pytest.raises(RenderFailed, match="failed verification"):
        render(SEGMENTS, VOICE, backend, tmp_path / "e.wav", verifier)


# ------------------------------------------------------------- reporting

def test_report_counts_recoveries(tmp_path: Path) -> None:
    backend, verifier = build({0: Failure.DROP_SENTENCE})
    report = render(SEGMENTS, VOICE, backend, tmp_path / "f.wav", verifier)
    assert report.clean
    assert any(c.recovered_by for c in report.chunks)
    assert "chunks" in report.summary()


def test_progress_callback_fires_per_chunk(tmp_path: Path) -> None:
    seen = []
    backend, verifier = build()
    render(SEGMENTS, VOICE, backend, tmp_path / "g.wav", verifier,
           RenderConfig(on_progress=lambda r, total: seen.append((r.index, total))))
    assert [i for i, _ in seen] == [0, 1]
    assert all(total == 2 for _, total in seen)


# ---------------------------------------------------------------- audio

def test_output_is_dual_mono_by_default(tmp_path: Path) -> None:
    """Sidesteps the BS.1770 mono offset without assuming player behaviour."""
    backend, verifier = build()
    out = tmp_path / "h.wav"
    render(SEGMENTS, VOICE, backend, out, verifier)
    data, _ = sf.read(str(out))
    assert data.ndim == 2 and data.shape[1] == 2
    assert np.allclose(data[:, 0], data[:, 1])


def test_mono_output_when_requested(tmp_path: Path) -> None:
    backend, verifier = build()
    out = tmp_path / "i.wav"
    render(SEGMENTS, VOICE, backend, out, verifier,
           RenderConfig(mastering=MasterConfig(channels=1)))
    data, _ = sf.read(str(out))
    assert data.ndim == 1


def test_trim_removes_silence_but_keeps_a_guard() -> None:
    sr = 24000
    speech = np.sin(np.linspace(0, 200, sr)).astype(np.float32)
    padded = np.concatenate([np.zeros(sr, np.float32), speech, np.zeros(sr, np.float32)])
    trimmed = trim_silence(padded, sr)
    assert trimmed.size < padded.size
    assert trimmed.size > speech.size * 0.9


def test_trim_leaves_all_silence_alone() -> None:
    silence = np.zeros(24000, dtype=np.float32)
    assert trim_silence(silence, 24000).size == silence.size


def test_trim_deletes_a_quiet_spoken_edge() -> None:
    """The mechanism behind "verify what ships": trimming is peak-relative.

    One second of tone at amplitude 0.001 followed by one at 0.5 is 54 dB of
    range, and the floor sits 42 dB under the peak, so the first second is
    "silence" to `trim_silence` even though it is signal. Synthetic on purpose
    (a sine, not a recording): this pins the deletion path, not its incidence
    in real speech. It is why synth now trims BEFORE the ASR hears the take and
    `render` never trims — a quiet edge is either heard and shipped, or trimmed
    and refused, never credited by the ASR and deleted afterwards.
    """
    sr = 24000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 220 * t)
    audio = np.concatenate([0.001 * tone, 0.5 * tone]).astype(np.float32)
    trimmed = trim_silence(audio, sr)
    assert 1.0 < trimmed.size / sr < 1.1


class _SilentTail(FakeBackend):
    """Ends every take with 0.5 s of digital silence — see the twin in test_synth."""

    def synthesize(self, text, voice, *, max_frames, temperature):
        audio = super().synthesize(text, voice, max_frames=max_frames,
                                   temperature=temperature)
        return np.concatenate([audio, np.zeros(self.sample_rate // 2, dtype=np.float32)])


def test_render_ships_the_verified_buffer_without_trimming_it(tmp_path: Path) -> None:
    """`shipped_s` is the length of `audio` exactly: render no longer trims.

    It used to trim again here, against the chunk's own peak — for a
    sentence-split assembly a LOUDER reference than any one sentence saw — on
    the far side of verification. Whatever that removed was gone from the file
    and present in the verdict.
    """
    backend = _SilentTail()
    verifier = CoverageVerifier(FakeASR(backend))
    report = render(SEGMENTS, VOICE, backend, tmp_path / "s.wav", verifier)
    for chunk in report.chunks:
        assert chunk.shipped_s == chunk.audio.size / backend.sample_rate
        assert chunk.shipped_s < chunk.duration_s, "the tail was trimmed, in synth"


def test_a_quiet_final_sentence_survives_render_after_sentence_split(tmp_path: Path) -> None:
    """The louder-reference case: `render`'s former trim measured the ASSEMBLY.

    Three sentences rendered alone, the last at -46 dB relative to the others.
    Each verified against its own peak; the old outer trim then judged the
    whole assembly against the loudest sentence's peak, where -46 dB is under
    the -42 dB floor, and cut the quiet sentence off the end of a chunk the
    report called clean. Nothing on the far side of verification trims now, so
    the assembly ships at exactly its verified length.
    """
    backend = FakeBackend(script={i: Failure.DROP_SENTENCE for i in range(3)},
                          amplitude_script={5: 0.1 * 10 ** (-46 / 20)})
    verifier = CoverageVerifier(FakeASR(backend))
    report = render(SEGMENTS[:1], VOICE, backend, tmp_path / "q.wav", verifier)
    chunk = report.chunks[0]
    assert chunk.recovered_by == "sentence-split"
    assert chunk.shipped_s == chunk.audio.size / backend.sample_rate
    assert chunk.shipped_s == pytest.approx(chunk.duration_s)


def test_declick_fades_both_edges() -> None:
    audio = np.ones(24000, dtype=np.float32)
    faded = declick(audio, 24000)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    assert faded[12000] == 1.0


def test_soft_limit_guarantees_the_ceiling() -> None:
    loud = np.linspace(-3, 3, 1000).astype(np.float32)
    limited = soft_limit(loud, 0.9)
    assert np.max(np.abs(limited)) < 0.9


def test_soft_limit_leaves_quiet_material_untouched() -> None:
    quiet = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    assert np.allclose(soft_limit(quiet, 0.9), quiet, atol=1e-6)


def test_concatenate_of_nothing_is_empty() -> None:
    assert concatenate([]).size == 0


def test_to_channels_duplicates_exactly() -> None:
    mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    stereo = to_channels(mono, 2)
    assert stereo.shape == (3, 2)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])


# ------------------------------------------------------------------- CLI

def test_cli_parses_blank_lines_as_paragraph_gaps() -> None:
    from narrator.cli import parse_text
    segments = parse_text("First para.\n\nSecond para.\n\n\nThird.", 0.5)
    assert [type(s).__name__ for s in segments] == ["Text", "Gap", "Text", "Gap", "Text"]
    assert all(s.seconds == 0.5 for s in segments if isinstance(s, Gap))


def test_cli_collapses_internal_whitespace() -> None:
    from narrator.cli import parse_text
    segments = parse_text("Line one\nline two\n   spaced", 0.35)
    assert segments == [Text("Line one line two spaced")]


def test_cli_ignores_empty_input() -> None:
    from narrator.cli import parse_text
    assert parse_text("\n\n   \n\n", 0.35) == []


def test_backend_with_unknown_sample_rate_is_refused(tmp_path: Path) -> None:
    """A leading Gap against sample_rate == 0 allocated zero samples and vanished
    from a render that still reported itself clean."""
    backend, verifier = build()
    backend.sample_rate = 0
    with pytest.raises(ValueError, match="sample_rate is 0"):
        render([Gap(3.0), Text("Alpha beta gamma.")], VOICE, backend, tmp_path / "x.wav", verifier)


def test_leading_gap_is_rendered_at_full_length(tmp_path: Path) -> None:
    backend, verifier = build()
    a = render([Text("Alpha beta gamma.")], VOICE, backend, tmp_path / "a.wav", verifier)
    backend2, verifier2 = build()
    b = render([Gap(3.0), Text("Alpha beta gamma.")], VOICE, backend2, tmp_path / "b.wav", verifier2)
    assert b.duration_s - a.duration_s == pytest.approx(3.0, abs=0.05)


def test_default_verifier_is_built_after_the_backend_settles_its_rate(monkeypatch, tmp_path: Path) -> None:
    """Supertonic corrects its sample rate during the first synthesis; an ASR
    constructed before that freezes the stale rate — the silent-corruption case
    the source_rate rule exists to prevent. The default verifier must therefore
    be built on first use, which is always after the first synthesis."""
    import importlib

    render_mod = importlib.import_module("narrator.render")

    backend = FakeBackend()
    backend.sample_rate = 44100          # wrong until the engine first runs

    real_synthesize = backend.synthesize

    def settling_synthesize(*a, **kw):
        backend.sample_rate = 24000      # the engine's true rate
        return real_synthesize(*a, **kw)

    backend.synthesize = settling_synthesize

    seen = []

    def fake_default_verifier(rate, sound_alikes=()):
        seen.append(rate)
        return CoverageVerifier(FakeASR(backend))

    monkeypatch.setattr(render_mod, "default_verifier", fake_default_verifier)
    render(SEGMENTS, VOICE, backend, tmp_path / "v.wav")
    assert seen == [24000], "verifier was built with the pre-settlement rate"


def test_leading_gap_survives_a_backend_that_settles_its_rate(tmp_path: Path) -> None:
    """The gap-side twin of the deferred-verifier test above.

    Gaps used to be allocated inside the segment loop, at whatever rate the
    backend declared at that moment. Against a Supertonic-style backend that
    corrects 44100 to its true rate during the first synthesis, a leading
    3.0 s Gap was allocated at the stale rate and written at the corrected
    one: measured 5.51 s (3.0 x 44100 / 24000). A gap must be allocated at
    the same rate the file is written at, which is only settled after the
    segment loop.
    """
    def settling(backend: FakeBackend) -> None:
        real = backend.synthesize

        def synthesize(*a, **kw):
            backend.sample_rate = 24000   # the engine's true rate, discovered here
            return real(*a, **kw)

        backend.sample_rate = 44100       # wrong until the engine first runs
        backend.synthesize = synthesize

    backend, verifier = build()
    settling(backend)
    a = render([Text("Alpha beta gamma.")], VOICE, backend, tmp_path / "a.wav", verifier)
    backend2, verifier2 = build()
    settling(backend2)
    b = render([Gap(3.0), Text("Alpha beta gamma.")], VOICE, backend2, tmp_path / "b.wav", verifier2)
    assert b.duration_s - a.duration_s == pytest.approx(3.0, abs=0.05)


def test_render_failed_message_names_the_missing_words(tmp_path: Path) -> None:
    """The refusal is the product, so the refusal must explain itself: not just
    a score and a sentence, but which words the audio lost."""
    backend, verifier = build({i: Failure.TRUNCATE for i in range(50)})
    with pytest.raises(RenderFailed, match="missing:"):
        render(SEGMENTS, VOICE, backend, tmp_path / "d.wav", verifier)


def test_progress_line_names_the_missing_words(capsys: pytest.CaptureFixture[str]) -> None:
    from narrator.cli import _progress
    from narrator.types import ChunkResult
    result = ChunkResult(
        index=0, text="The keeper counts the lantern posts.",
        audio=np.zeros(0, dtype=np.float32), duration_s=1.0, attempts=3, ok=False,
        coverage=0.5, dropped_sentence="The keeper counts the lantern posts.",
        word_diagnostics=("d:lantern", "d:posts"),
    )
    _progress(result, 3)
    assert "missing: lantern, posts" in capsys.readouterr().out


# ------------------------------------------------------------ per-segment voices

Q_VOICE = Voice(Path("questioner.wav"), "q reference", "en")


def test_segment_voice_overrides_the_default(tmp_path: Path) -> None:
    """A dialogue turn pinned to a voice is synthesized with that voice."""
    backend, verifier = build()
    segments = [
        Text("Why would anyone burn money on purpose?", voice=Q_VOICE),
        Gap(3.0),
        Text("Nobody burns it on purpose. The typo does."),
    ]
    report = render(segments, VOICE, backend, tmp_path / "d.wav", verifier)
    assert report.clean
    by_request = dict(zip(backend.requests, backend.voices_seen, strict=True))
    assert by_request["Why would anyone burn money on purpose?"] is Q_VOICE
    assert by_request["Nobody burns it on purpose. The typo does."] is VOICE


def test_voice_never_bleeds_across_chunks_of_a_long_turn(tmp_path: Path) -> None:
    """A turn long enough to chunk keeps its speaker on every chunk.

    This is the dialogue failure that must stay inexpressible: chunking happens
    inside a segment, so no chunk can inherit the other speaker's voice."""
    backend, verifier = build()
    long_turn = Text(" ".join(["The explainer keeps talking at length here."] * 30))
    q_turn = Text(" ".join(["And the questioner asks a very long question now?"] * 30),
                  voice=Q_VOICE)
    report = render([long_turn, q_turn], VOICE, backend, tmp_path / "e.wav", verifier)
    assert report.clean
    assert len(report.chunks) > 4  # both turns actually chunked
    for request, voice in zip(backend.requests, backend.voices_seen, strict=True):
        expected = Q_VOICE if "questioner" in request else VOICE
        assert voice is expected, f"voice bleed on chunk: {request[:50]!r}"


def test_sentence_split_fallback_keeps_the_segment_voice(tmp_path: Path) -> None:
    """The rescue path must not fall back to the default narrator."""
    backend = FakeBackend(default=Failure.DROP_SENTENCE)
    # Every whole-chunk attempt drops a sentence; per-sentence calls say one
    # sentence each, which DROP_SENTENCE leaves intact, so the fallback passes.
    verifier = CoverageVerifier(FakeASR(backend))
    segments = [Text("First thought here. Second thought follows.", voice=Q_VOICE)]
    report = render(segments, VOICE, backend, tmp_path / "f.wav", verifier)
    assert report.clean
    assert report.chunks[0].recovered_by == "sentence-split"
    assert all(v is Q_VOICE for v in backend.voices_seen)


# ------------------------------------------------------------- per-voice level
#
# The correction is *declared* on the Voice, never inferred from the rendered
# audio. Three inference designs were built and measured before this one, and
# each mistook a quiet delivery for a quiet reference — see Voice.gain_db. So
# these tests ask two things: is a declared offset applied exactly, and does a
# performance survive it untouched.

RATE = 24000


def tone(amplitude: float, seconds: float = 1.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def rms(piece: np.ndarray) -> float:
    return float(np.sqrt((piece.astype(np.float64) ** 2).mean()))


def db(ratio: float) -> float:
    return 20 * np.log10(ratio)


def speech_levels(path: Path) -> list[float]:
    """RMS of each speech region in a written file, split on the Gap's silence.

    Frame envelope, not per-sample: a tone crosses zero 440 times a second, so a
    sample-wise threshold finds hundreds of "regions" per turn instead of one.
    """
    audio, _ = sf.read(str(path), dtype="float32")
    mono = audio[:, 0] if audio.ndim == 2 else audio
    frame = RATE // 50
    count = mono.size // frame
    frames = mono[: count * frame].reshape(count, frame).astype(np.float64)
    loud = np.sqrt((frames ** 2).mean(axis=1)) > 1e-3

    regions: list[tuple[int, int]] = []
    start: int | None = None
    for position, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = position
        elif not is_loud and start is not None:
            regions.append((start, position))
            start = None
    if start is not None:
        regions.append((start, count))
    return [float(np.sqrt((frames[s:e] ** 2).mean())) for s, e in regions if e - s >= 5]


def frame_levels(path: Path, region: int) -> np.ndarray:
    """Per-frame RMS inside one speech region, edges dropped.

    The declick fades at each end are real signal shaping, not a level change,
    so the first and last few frames are excluded before comparing.
    """
    audio, _ = sf.read(str(path), dtype="float32")
    mono = audio[:, 0] if audio.ndim == 2 else audio
    frame = RATE // 50
    count = mono.size // frame
    frames = mono[: count * frame].reshape(count, frame).astype(np.float64)
    power = (frames ** 2).mean(axis=1)
    loud = power > 1e-6

    spans, start = [], None
    for position, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = position
        elif not is_loud and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, count))
    spans = [s for s in spans if s[1] - s[0] >= 5]
    begin, end = spans[region]
    return np.sqrt(power[begin + 3:end - 3])


def test_apply_gain_is_identity_at_zero() -> None:
    """The default must not touch a single sample — every existing render is 0 dB."""
    from narrator.audio import apply_gain
    audio = tone(0.2)
    assert apply_gain(audio, 0.0) is audio


def test_apply_gain_applies_exactly_what_was_asked() -> None:
    from narrator.audio import apply_gain
    audio = tone(0.2)
    assert db(rms(apply_gain(audio, -6.0)) / rms(audio)) == pytest.approx(-6.0, abs=0.01)
    assert db(rms(apply_gain(audio, 3.0)) / rms(audio)) == pytest.approx(3.0, abs=0.01)


def test_declared_gain_evens_out_a_lopsided_dialogue(tmp_path: Path) -> None:
    """End to end: two speakers 12 dB apart ship level when the caller says so."""
    quiet = Voice(Path("nonexistent.wav"), "reference", "en")
    loud = Voice(Path("questioner.wav"), "q reference", "en", gain_db=-12.0)
    backend = FakeBackend(voice_amplitude={quiet: 0.05, loud: 0.20})
    verifier = CoverageVerifier(FakeASR(backend))
    segments = [
        Text("Nobody burns it on purpose. The typo does."),
        Gap(1.0),
        Text("Why would anyone burn money on purpose?", voice=loud),
    ]
    out = tmp_path / "declared.wav"
    render(segments, quiet, backend, out, verifier)
    first, second = speech_levels(out)
    assert db(second / first) == pytest.approx(0.0, abs=0.3)


def test_without_a_declared_gain_the_imbalance_ships(tmp_path: Path) -> None:
    """The control: narrator never invents the correction, so it stays 12 dB apart."""
    loud = Voice(Path("questioner.wav"), "q reference", "en")
    backend = FakeBackend(voice_amplitude={VOICE: 0.05, loud: 0.20})
    verifier = CoverageVerifier(FakeASR(backend))
    segments = [
        Text("Nobody burns it on purpose. The typo does."),
        Gap(1.0),
        Text("Why would anyone burn money on purpose?", voice=loud),
    ]
    out = tmp_path / "untouched.wav"
    render(segments, VOICE, backend, out, verifier)
    first, second = speech_levels(out)
    assert db(second / first) == pytest.approx(12.0, abs=0.3)


def test_a_declared_gain_never_touches_a_performance(tmp_path: Path) -> None:
    """A whisper stays exactly as far below its speaker's ordinary turns.

    This is what every inference design got wrong, in three different ways: a
    quiet turn is content, and only the *speaker's* offset is the renderer's to
    correct. A constant per-voice gain cannot flatten a delivery — the ratio
    between two of one voice's turns is identical with and without it.
    """
    ordinary = Text("Nobody burns it on purpose. The typo does.")
    whispered = Text("Almost nobody, anyway.")
    other = Voice(Path("questioner.wav"), "q reference", "en")

    def levels(gain_db: float, name: str) -> list[float]:
        voice = Voice(Path("nonexistent.wav"), "reference", "en", gain_db=gain_db)
        # One voice, two deliveries: the second turn is spoken 12 dB down. A
        # second speaker keeps the file's overall loudness — and so master's
        # makeup gain — fixed, leaving this voice's own level free to move.
        backend = FakeBackend(voice_amplitude={voice: 0.10, other: 0.10},
                              amplitude_script={1: 0.025})
        verifier = CoverageVerifier(FakeASR(backend))
        out = tmp_path / name
        render([ordinary, Gap(1.0), whispered, Gap(1.0),
                Text("A third voice holds the level steady.", voice=other)],
               voice, backend, out, verifier)
        return speech_levels(out)

    plain, shifted = levels(0.0, "plain.wav"), levels(-6.0, "shifted.wav")
    contrast = db(plain[1] / plain[0])
    assert contrast == pytest.approx(-12.0, abs=0.3)                 # the whisper is real
    assert db(shifted[1] / shifted[0]) == pytest.approx(contrast, abs=0.05)
    # ...and the gain genuinely moved this voice against the other speaker, so
    # an implementation that dropped it would fail here. Measured as a ratio:
    # master renormalises the finished file, which hides absolute levels.
    assert db(shifted[0] / shifted[2]) - db(plain[0] / plain[2]) == pytest.approx(-6.0, abs=0.2)


def test_chunking_a_long_turn_keeps_the_voices_gain(tmp_path: Path) -> None:
    """Every chunk of a turn moves together — the gain rides on the Voice.

    A turn long enough to chunk stitches into one continuous region, so a chunk
    left ungained shows up as a step in level partway through it rather than as
    a separate region. Both checks are ratios: `master` renormalises the file,
    so absolute levels say nothing.
    """
    other = Voice(Path("nonexistent.wav"), "reference", "en")

    def render_at(gain_db: float, name: str) -> tuple[np.ndarray, float]:
        speaker = Voice(Path("questioner.wav"), "q reference", "en", gain_db=gain_db)
        backend = FakeBackend(voice_amplitude={speaker: 0.10, other: 0.10})
        verifier = CoverageVerifier(FakeASR(backend))
        long_turn = Text(" ".join(["The questioner keeps asking at length here."] * 30),
                         voice=speaker)
        out = tmp_path / name
        report = render([long_turn, Gap(1.0),
                         Text("A steady closing line that anchors the loudness.",
                              voice=other)],
                        other, backend, out, verifier)
        assert len(report.chunks) > 4
        turn, anchor = speech_levels(out)
        return frame_levels(out, region=0), db(turn / anchor)

    (plain_frames, plain_ratio) = render_at(0.0, "plain-long.wav")
    (cut_frames, cut_ratio) = render_at(-6.0, "cut-long.wav")
    # No step anywhere inside the turn: every chunk of it carries the same gain.
    # Percentiles, because the 8 ms declick fade at each join dips one frame.
    # The window is safe in both directions: the dips are well under 5% of the
    # frames, and gain is applied per chunk, so the smallest omission possible
    # here is one chunk of six — 16.7% of them.
    for frames in (plain_frames, cut_frames):
        low, high = np.percentile(frames, [5, 95])
        assert high / low < 1.05
    assert cut_ratio - plain_ratio == pytest.approx(-6.0, abs=0.2)


def test_gain_db_must_be_finite_and_sane() -> None:
    """A wild gain multiplies the audio after verification and still reports clean."""
    for bad in (float("nan"), float("inf"), float("-inf"), -1e308, 1e308, 61.0):
        with pytest.raises(ValueError, match="finite"):
            Voice(Path("a.wav"), "t", "en", gain_db=bad)
    Voice(Path("a.wav"), "t", "en", gain_db=-60.0)   # the bound itself is allowed


# ------------------------------------------------- unscripted silence (#18)

def test_a_render_full_of_holes_is_quarantined(tmp_path: Path) -> None:
    """The headline: dead air must not ship, however clean the transcript is."""
    out = tmp_path / "episode.wav"
    backend = FakeBackend(default=Failure.SILENT_HOLE)
    verifier = CoverageVerifier(FakeASR(backend))
    with pytest.raises(RenderFailed):
        render(SEGMENTS, VOICE, backend, out, verifier)
    assert not out.exists()


def test_a_spelled_pause_under_the_gate_renders_clean(tmp_path: Path) -> None:
    """A spelled pause is fine until it reaches the length of a defect.

    `coverage` calls a punctuation-only sentence "a pause the script spells
    rather than speech", and the gate must not contradict that for ordinary
    pauses. It does NOT extend indefinitely: a rendered pause longer than
    `max_silence_s` is refused, because at that length nothing distinguishes it
    from the defect, and `Gap` is how this library takes a pause as an
    instruction. Documented in the README.

    The pause is injected, because the fake renders one uninterrupted tone — a
    version of this test using plain text asserted nothing at all.
    """
    backend = FakeBackend(default=Failure.SILENT_HOLE, hole_s=2.0)
    verifier = CoverageVerifier(FakeASR(backend))
    segments = [Text("Alpha beta gamma delta epsilon. ...")]
    report = render(segments, VOICE, backend, tmp_path / "p.wav", verifier)
    assert report.clean
    assert report.chunks[0].silence_s > 1.5, "and the pause is still reported"


def test_silence_is_reported_on_every_chunk(tmp_path: Path) -> None:
    """Reported, not only refused: the short band is evidence for later tuning."""
    backend, verifier = build()
    report = render(SEGMENTS, VOICE, backend, tmp_path / "s.wav", verifier)
    assert report.clean
    assert all(c.silence_s == 0.0 for c in report.chunks)


def test_a_chunk_that_is_only_atoms_is_refused(tmp_path: Path) -> None:
    """The vacuous pass: an empty reference scores 1.0 against ANY transcript.

    Left alone, such a chunk is certified, written, AND stored for reuse.
    Preflight reports it too, but preflight is advisory and nothing obliges a
    caller to run it, so the render refuses on its own.

    An input error rather than a verification verdict — `Text.__post_init__`
    already refuses "something to be spoken" with nothing to speak; this is the
    same refusal one layer later, where removal makes the emptiness visible.
    """
    atom = "<|sfx:laughter|>"
    backend = FakeBackend(consumes=(atom,))
    verifier = CoverageVerifier(FakeASR(backend))
    cfg = RenderConfig(synth=SynthConfig(non_speech=(atom,)))

    with pytest.raises(ValueError, match="no speech left after declared non_speech"):
        render([Text(atom)], VOICE, backend, tmp_path / "a.wav", verifier, cfg)
    assert backend.calls == 0, "refused before paying for a generation"

    # Punctuation is not speech, and the guard used to think it was. Asking
    # `.strip()` let `Text("<|sfx:laughter|> ...")` through with the reference
    # "...", which normalizes to nothing and therefore scores 1.0 against ANY
    # transcript — the same vacuous certification, one character class out.
    # Preflight already reported this chunk doomed, so the looser guard here was
    # also what broke preflight's one contract.
    with pytest.raises(ValueError, match="no speech left after declared non_speech"):
        render([Text(f"{atom} ...")], VOICE, backend, tmp_path / "c.wav", verifier, cfg)

    # ...but only where REMOVAL is what emptied it. Caller-written punctuation
    # certifies vacuously too, and that is a real hole — it is simply not this
    # guard's, since it exists whether or not any atom is configured. Refusing it
    # here would make the same script legal or illegal depending on an unrelated
    # setting, which a review caught in the first version of this fix.
    render([Text("...")], VOICE, backend, tmp_path / "d.wav", verifier, cfg)

    # A tag BESIDE real speech is the supported case and still renders.
    report = render([Text(f"{atom} Alpha beta gamma delta.")], VOICE, backend,
                    tmp_path / "b.wav", verifier, cfg)
    assert report.clean


def test_the_reported_chunk_spans_add_up_to_the_file(tmp_path: Path) -> None:
    """The invariant that was broken: the parts must sum to the whole.

    `duration_s` is the raw synthesis length and the take is trimmed before it
    ships (in synth now; in `render` when this was found), so accumulating it
    drifts by every silence removed before a given chunk.
    `RenderReport.duration_s` is measured on the finished audio, so the
    total was right while the parts did not reach it — the confusing direction,
    and it cost a real investigation in issue #21.

    Asserted in samples with `round()` on each chunk, because neither side is
    exact on its own: gap frames are floored per gap (`int(seconds * rate)`) and
    a length that went float -> seconds -> float does not always return to the
    same integer (`Gap(13 / 22050)` is the counterexample). Rounding each term
    recovers the sample count that was actually written.
    """
    backend, verifier = build()
    report = render(SEGMENTS, VOICE, backend, tmp_path / "a.wav", verifier)
    rate = backend.sample_rate

    chunk_samples = sum(round(c.shipped_s * rate) for c in report.chunks)
    gap_samples = sum(int(s.seconds * rate) for s in SEGMENTS if isinstance(s, Gap))
    assert round(report.duration_s * rate) == chunk_samples + gap_samples


def test_a_chunk_can_be_located_in_the_file(tmp_path: Path) -> None:
    """`shipped_s` alone is not enough, which is why `start_s` exists.

    `Gap` segments never appear in `RenderReport.chunks`, so summing chunk
    lengths skips the silence between them and lands short — a variant of the
    same drift, reachable the moment a script uses a Gap.
    """
    backend, verifier = build()
    report = render(SEGMENTS, VOICE, backend, tmp_path / "b.wav", verifier)

    first, second = report.chunks
    assert first.start_s == 0.0
    # The second chunk starts after the first PLUS the declared gap between them.
    assert second.start_s == pytest.approx(first.shipped_s + 3.0, abs=1 / backend.sample_rate)
    assert second.start_s + second.shipped_s == pytest.approx(report.duration_s, abs=0.01)


def test_a_chunk_with_no_audio_ships_nothing_rather_than_nothing_measured(
    tmp_path: Path,
) -> None:
    """0.0 and None mean different things and both are reachable.

    A chunk whose every attempt raised has no audio and occupies no file — that
    is 0.0. None is "never went through render", which is why the default is not
    0.0: a take restored from the store must not claim a span it never had.
    """
    # One chunk raises every attempt, one succeeds — a render where EVERY chunk
    # is empty crashes in `master`, which predates this change (filed separately).
    backend = FakeBackend(script={i: Failure.RAISE for i in range(3)})  # chunk 1 only
    verifier = CoverageVerifier(FakeASR(backend))
    report = render([Text("Alpha beta gamma delta."), Text("Not the keeper.")],
                    VOICE, backend, tmp_path / "c.wav", verifier,
                    RenderConfig(quarantine=False))

    empty = next(c for c in report.chunks if c.audio.size == 0)
    assert empty.shipped_s == 0.0
    # It occupies nothing, but it still HAS a position — the point it would have
    # been written at. Leaving that None would say "never went through render",
    # which is a different claim.
    assert empty.start_s is not None
    assert ChunkResult(index=0, text="t", audio=np.zeros(0), duration_s=1.0,
                       attempts=1, ok=True).shipped_s is None


def test_a_clean_render_reports_no_unscripted_silence(tmp_path: Path) -> None:
    """A declared Gap is not unscripted — it is exactly what was asked for."""
    backend, verifier = build()
    report = render(SEGMENTS, VOICE, backend, tmp_path / "a.wav", verifier)
    assert report.unscripted_silence_s == 0.0


def test_silence_no_gap_asked_for_is_reported(tmp_path: Path) -> None:
    """The defect made visible, which is what this reports.

    The per-chunk gate now sees a 3 s tail (issue #42) but does not refuse it —
    it is under `max_silence_s`, which refuses only the severe class. Summed
    across chunks it is still worth reporting, and the report is what does that.
    """
    # 3 s between the two silence floors (see `FakeBackend.tail_s`): seen by the
    # gate now, reported, and passing only because it is under `max_silence_s`.
    backend = FakeBackend(tail_s=3.0)
    verifier = CoverageVerifier(FakeASR(backend))
    report = render(SEGMENTS, VOICE, backend, tmp_path / "b.wav", verifier,
                    RenderConfig(quarantine=False))

    assert report.unscripted_silence_s > 1.0
    # The declared 3 s Gap in SEGMENTS is excluded, so this is not just the gap.
    for chunk in report.chunks:
        assert chunk.ok, "3 s is under the gate, which refuses only the severe class"
        assert chunk.silence_s > 1.0, "but the chunk now measures its own tail"


def test_a_chunk_with_no_audio_at_the_very_end_still_has_a_position(
    tmp_path: Path,
) -> None:
    """The one position the piece list cannot point at directly.

    Every result records where it begins as an index into the stitched piece
    list. A chunk whose every attempt raised contributes no piece, so it begins
    at whatever piece follows it — and when nothing follows, that index is one
    past the end. It resolves to the end of the file, which is exactly the point
    the chunk would have occupied.

    Pinned because it is the only index the offset table has to invent. The
    previous shape reached it through a separate side table and a re-sum of
    every earlier piece; this is the same answer from one pass.
    """
    backend = FakeBackend(script={1: Failure.RAISE, 2: Failure.RAISE, 3: Failure.RAISE})
    cfg = RenderConfig(quarantine=False,
                       synth=SynthConfig(max_attempts=3, allow_sentence_split=False))
    report = render(
        [Text("Alpha beta gamma delta."), Gap(2.0), Text("Epsilon zeta eta theta.")],
        VOICE, backend, tmp_path / "a.wav", CoverageVerifier(FakeASR(backend)), cfg,
    )

    shipped, dropped = report.chunks
    assert shipped.ok and dropped.ok is False
    assert dropped.audio.size == 0 and dropped.shipped_s == 0.0
    assert dropped.start_s == report.duration_s, (
        "a chunk that occupies nothing begins where the file ends"
    )
    assert shipped.start_s == 0.0
