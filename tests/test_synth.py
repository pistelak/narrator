"""Retry-ladder tests.

Both defects the predecessor shipped lived on paths that only execute when
something has already gone wrong, which is why a clean fourteen-minute render
sailed straight over one of them. Those paths get the most tests here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from narrator.backends.fake import Failure, FakeASR, FakeBackend
from narrator.synth import SynthConfig, duration_bounds, frame_cap, synthesize_chunk
from narrator.types import Voice
from narrator.verify import CoverageVerifier, NullVerifier

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")
TEXT = "Not the keeper. Not a stranger. Not any council with any mandate."
CFG = SynthConfig()


def run(script=None, text=TEXT, cfg=CFG, perfect=True):
    backend = FakeBackend(script=script or {})
    verifier = CoverageVerifier(FakeASR(backend, perfect=perfect))
    return synthesize_chunk(text, 0, backend, verifier, VOICE, cfg), backend


# ------------------------------------------------------------- frame cap

def test_frame_cap_has_absolute_headroom_for_short_utterances() -> None:
    """The bug that broke the sentence-split fallback.

    "Jen firmu." is two words: 0.8 s expected. A pure 1.6x multiplier gives
    1.28 s, less than the real leading and trailing silence, so every short
    sentence was truncated by construction — and short sentences are exactly
    what the fallback produces.
    """
    fps = 25
    assert frame_cap(2, fps, CFG) / fps >= 4.0
    pure_multiplier = (2 / CFG.words_per_second) * CFG.frame_headroom
    assert pure_multiplier < 1.3, "the arithmetic that caused the bug"


def test_hitting_the_frame_cap_is_itself_a_failure() -> None:
    """The cap must detect a runaway, not merely bound its cost.

    With these constants the cap always lands below the duration ceiling, so a
    capped runaway passes the ceiling check. Before this was its own signal, a
    NullVerifier run accepted four minutes of babble as a valid chunk.
    """
    backend = FakeBackend(script={0: Failure.RUNAWAY})
    result = synthesize_chunk(TEXT, 0, backend, NullVerifier(), VOICE,
                              SynthConfig(max_attempts=1, allow_sentence_split=False))
    assert not result.ok


def test_frame_cap_still_scales_for_long_chunks() -> None:
    fps = 25
    assert frame_cap(40, fps, CFG) / fps == pytest.approx(16 * 1.6 + 2.0)


def test_frame_cap_is_passed_to_the_backend() -> None:
    _, backend = run()
    assert backend.max_frames_seen[0] == frame_cap(len(TEXT.split()), 25, CFG)


def test_duration_bounds_bracket_the_expected() -> None:
    floor, ceiling = duration_bounds(40, CFG)
    assert floor < 40 / CFG.words_per_second < ceiling


# ------------------------------------------------- the ranking bug (critical)

def test_a_passing_retry_is_never_discarded_for_a_failed_first_attempt() -> None:
    """The predecessor's critical bug, in one test.

    A duration-failed attempt received a fabricated perfect coverage score, which
    is the maximum, so no later attempt could outrank it. Attempt 2 could pass
    everything, break the loop, and the function still returned attempt 1's audio.
    """
    result, backend = run({0: Failure.RUNAWAY})   # attempt 1 fails duration
    assert result.ok
    assert result.attempts == 2
    assert result.recovered_by == "retry"
    assert backend.calls == 2


def test_duration_valid_failure_outranks_duration_invalid_failure() -> None:
    """When everything fails, report the least-bad — but never call it ok."""
    cfg = SynthConfig(max_attempts=2, allow_sentence_split=False)
    result, _ = run({0: Failure.RUNAWAY, 1: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok
    assert result.duration_s < 100, "kept the runaway instead of the plausible attempt"


def test_null_verifier_does_not_freeze_the_first_attempt() -> None:
    """--no-asr silently made retries useless in the predecessor."""
    backend = FakeBackend(script={0: Failure.RUNAWAY})
    result = synthesize_chunk(TEXT, 0, backend, NullVerifier(), VOICE, CFG)
    assert result.ok
    assert result.attempts == 2


# -------------------------------------------------------------- retrying

def test_succeeds_first_time_without_retrying() -> None:
    result, backend = run()
    assert result.ok and result.attempts == 1 and backend.calls == 1
    assert result.recovered_by == ""


def test_stops_retrying_once_it_passes() -> None:
    result, backend = run({0: Failure.DROP_SENTENCE})
    assert result.ok and backend.calls == 2


def test_exhausting_retries_falls_through_to_the_split() -> None:
    result, _ = run({0: Failure.DROP_SENTENCE, 1: Failure.DROP_SENTENCE, 2: Failure.DROP_SENTENCE})
    assert result.ok
    assert result.recovered_by == "sentence-split"


# ------------------------------------------------------------ exceptions

def test_one_raising_attempt_does_not_lose_the_chunk() -> None:
    """No guard here meant a transient error at chunk 80 discarded 15 minutes."""
    result, backend = run({0: Failure.RAISE})
    assert result.ok and backend.calls == 2


def test_all_attempts_raising_fails_loudly_with_no_audio() -> None:
    cfg = SynthConfig(max_attempts=2, allow_sentence_split=False)
    result, _ = run({0: Failure.RAISE, 1: Failure.RAISE}, cfg=cfg)
    assert not result.ok
    assert result.audio.size == 0, "silence here would read as a deliberate pause"


# -------------------------------------------------------- sentence split

def test_sentence_split_rescues_a_chunk_that_fails_every_attempt() -> None:
    always_drop = {i: Failure.DROP_SENTENCE for i in range(3)}
    result, backend = run(always_drop)
    assert result.ok
    assert result.recovered_by == "sentence-split"
    assert backend.calls > 3
    # Each sentence was rendered alone, where dropping one is not expressible.
    assert any(r == "Not a stranger." for r in backend.requests)


def test_sentence_split_gap_uses_the_settled_rate() -> None:
    """The sentence-split twin of render's leading-gap case.

    When every whole-chunk attempt raises, the backend has never produced audio
    by the time the sentence-split fallback starts, so a Supertonic-style
    backend still declares its construction-time rate. The 0.12 s inter-sentence
    gap allocated at 44100 and written at the settled 24000 measured 0.22 s
    each — the joins audibly dragged. The gap must be allocated only after the
    rate has settled, i.e. after a sentence has actually been synthesized.
    """
    backend = FakeBackend(script={i: Failure.RAISE for i in range(3)})
    real = backend.synthesize

    def synthesize(*a, **kw):
        declared = backend.sample_rate
        backend.sample_rate = 24000        # the engine speaks at its true rate
        try:
            return real(*a, **kw)
        except Exception:
            backend.sample_rate = declared  # a failed call discovers nothing
            raise

    backend.sample_rate = 44100             # wrong until audio first comes back
    backend.synthesize = synthesize

    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(TEXT, 0, backend, verifier, VOICE, CFG)
    assert result.ok
    assert result.recovered_by == "sentence-split"
    expected = len(TEXT.split()) / backend.words_per_second + 2 * CFG.sentence_gap_s
    assert result.duration_s == pytest.approx(expected, abs=0.01)


def test_sentence_split_declines_when_a_sentence_itself_fails() -> None:
    """Partial success must not be dressed up as success.

    TRUNCATE rather than DROP_SENTENCE, because you cannot drop a sentence from a
    one-sentence input — the split renders each sentence alone, so the injected
    failure has to be one that applies to a single sentence.
    """
    result, _ = run({i: Failure.TRUNCATE for i in range(40)})
    assert not result.ok


def test_single_sentence_chunk_has_no_split_to_fall_back_on() -> None:
    result, _ = run({i: Failure.TRUNCATE for i in range(10)}, text="One single sentence here.")
    assert not result.ok
    assert result.recovered_by == ""


def test_split_is_disabled_when_configured() -> None:
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, backend = run({0: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok and backend.calls == 1


# -------------------------------------------------------------- reporting

def test_failure_names_the_missing_sentence() -> None:
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, _ = run({0: Failure.DROP_SENTENCE}, cfg=cfg)
    assert not result.ok
    assert result.dropped_sentence, "must name what went wrong"
    assert result.transcript


def test_tolerates_realistic_asr_disagreement() -> None:
    result, _ = run(text="He copies the twenty byte code. Ne znemožní.", perfect=False)
    assert result.ok and result.attempts == 1


# --------------------------- pronunciation lexicon (found by a real render)

def test_lexicon_is_applied_to_synthesis_but_not_to_verification() -> None:
    """A respelling and a round-trip verifier conflict unless they are separated.

    Substituting upstream made the verifier compare audio against "Kalleh" while
    the ASR reported "Kalle" — correct audio scoring 0.88 and failing. Three of
    five failures in a real 106-chunk render were this.
    """
    backend = FakeBackend()
    # The ASR hears the respelled audio and writes the conventional spelling.
    asr = FakeASR(backend, orthography={"Kalleh": "Kalle"})
    cfg = SynthConfig(pronunciation=(("Kalle", "Kalleh"),), max_attempts=1,
                      allow_sentence_split=False)
    result = synthesize_chunk("This is by Kalle Lindkvist here.", 0, backend,
                              CoverageVerifier(asr), VOICE, cfg)
    assert backend.requests == ["This is by Kalleh Lindkvist here."], "engine gets the respelling"
    assert result.text == "This is by Kalle Lindkvist here.", "report keeps the original"
    assert result.ok, "verification must use the original, not the respelling"


def test_longer_lexicon_keys_win() -> None:
    from narrator.synth import apply_pronunciation
    pairs = (("cafe", "café"), ("cafe counter", "café kaunter"))
    assert apply_pronunciation("at the cafe counter", pairs) == "at the café kaunter"


# ----------------------------------------------------------- acronym spelling

def test_acronyms_are_spelled_as_letter_names_when_enabled() -> None:
    """Reading an all-caps token letter by letter is how acronyms are read —
    a language rule, not project vocabulary, so no word lists anywhere."""
    from narrator.synth import spell_acronyms
    assert spell_acronyms("Jde do XY kódu.", "cs") == "Jde do iks ypsilon kódu."
    assert spell_acronyms("The QR code.", "en") == "The cue are code."
    assert spell_acronyms("Ahoj světe.", "cs") == "Ahoj světe."


def test_lexicon_overrides_acronym_spelling() -> None:
    """The lexicon runs first, so a project can give one acronym a word-like
    reading while everything else gets the general treatment."""
    backend = FakeBackend()
    cfg = SynthConfig(pronunciation=(("XY", "iksík"),), spell_acronyms=True,
                      max_attempts=1, allow_sentence_split=False)
    synthesize_chunk("The XY and the QR here.", 0, backend, NullVerifier(), VOICE, cfg)
    assert backend.requests == ["The iksík and the cue are here."]


def test_failed_chunk_result_carries_word_diagnostics() -> None:
    """The retry ladder must hand the verifier's word-level evidence to the
    report, or the render error can only say a score and a sentence."""
    cfg = SynthConfig(allow_sentence_split=False)
    result, _ = run({i: Failure.DROP_SENTENCE for i in range(3)}, cfg=cfg)
    assert not result.ok
    assert "d:council" in result.word_diagnostics


def test_cheap_check_failure_gains_codes_from_the_diagnostic_reverify() -> None:
    """A duration-failed attempt skips verification, so it has no transcript
    and no codes. The failure-path re-verify exists to add that evidence —
    it must carry the codes across AND must not rescue the attempt, the same
    rule already pinned for the transcript."""
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, _ = run({0: Failure.TRUNCATE}, cfg=cfg)
    assert not result.ok, "the re-verify must never rescue a cheap-check failure"
    assert result.transcript, "the re-verify ran and named what the audio said"
    assert "d:stranger" in result.word_diagnostics


def test_a_passing_diagnostic_reverify_cannot_rescue_a_capped_attempt() -> None:
    """The sharp end of the non-rescue rule: here the re-verification PASSES.

    A half-speed engine says the whole text but hits the frame cap doing it,
    so the cheap check fails while the ASR round-trip scores a perfect 1.0.
    That verdict must inform the report (coverage 1.0, full transcript: it
    stopped early, it did not say the wrong thing) and must still not flip
    the failure — a flaky ASR succeeding on this second call once turned a
    failed chunk into a passing one without generating any new audio."""
    backend = FakeBackend(words_per_second=1.0)   # cap fires before the text ends
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result = synthesize_chunk(TEXT, 0, backend, CoverageVerifier(FakeASR(backend)), VOICE, cfg)
    assert not result.ok, "a passing re-verification rescued a capped attempt"
    assert result.coverage == 1.0
    assert result.transcript == TEXT


# ---------------------------------------------------------- unscripted silence

def test_a_multi_second_hole_is_refused_though_every_word_is_present() -> None:
    """The defect issue #18 exists for: silence between words contains no words.

    An episode shipped 18.5 s of dead air reporting `failed=0`, `min coverage
    1.0`. The ASR round-trip cannot see it — the transcript is perfect — and the
    duration ceiling is far too loose, since a 20-word chunk permits 16.5 s
    against ~8 s of real speech.
    """
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    result, _ = run({0: Failure.SILENT_HOLE}, cfg=cfg)
    assert not result.ok, "a hole must not ship"
    assert result.silence_s > cfg.max_silence_s, "and the report must name it"
    # The sharp end of the issue: the round-trip is PERFECT on this audio. Every
    # word was spoken, so the transcript matches and coverage is 1.00 — and the
    # chunk is still a defect. Nothing that scores words could have caught it.
    assert result.coverage == 1.0
    assert result.transcript, "the diagnostic transcript still runs on failure"


def test_a_hole_is_recovered_by_the_retry_ladder() -> None:
    """Why the check lives per-chunk rather than post-stitch.

    The defect is stochastic — issue #18 measured ~1% of chunks — so a
    regenerated attempt almost certainly clears it. A whole-file check could
    only refuse the finished render, after paying for every chunk.
    """
    result, _ = run({0: Failure.SILENT_HOLE})
    assert result.ok
    assert result.recovered_by == "retry"
    assert result.silence_s <= SynthConfig().max_silence_s


def test_the_silence_check_is_scale_invariant() -> None:
    """A correct but QUIET render must not read as silence.

    `FakeBackend(amplitude=0.001)` is entirely correct speech peaking at
    -60 dBFS, which any absolute dBFS floor would classify as dead air — and the
    same shape is legal for a natively quiet backend, a quiet reference, or the
    deliberate whisper this library protects. The check runs before `apply_gain`
    and before mastering normalises loudness, so an absolute number would not
    even refer to the level that ships.
    """
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    for amplitude in (0.1, 0.001):
        backend = FakeBackend(amplitude=amplitude)
        verifier = CoverageVerifier(FakeASR(backend))
        result = synthesize_chunk(TEXT, 0, backend, verifier, VOICE, cfg)
        assert result.ok, f"correct speech at amplitude={amplitude} must pass"
        assert result.silence_s == 0.0


def test_a_short_hole_is_reported_but_not_refused() -> None:
    """The 1.0-1.6 s band is telemetry, not a refusal.

    Only the severe class was measured (7.32, 8.08, 10.44, 10.98 s). A pause the
    script spells — `verify` already treats a punctuation-only "..." as one —
    plausibly lives in the short band, so it is reported and left to the caller
    rather than refused on evidence nobody has.
    """
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    backend = FakeBackend(script={0: Failure.SILENT_HOLE}, hole_s=2.0)
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(TEXT, 0, backend, verifier, VOICE, cfg)
    assert result.ok, "a two-second hole is not the measured defect"
    assert 1.5 < result.silence_s < 2.5, "but the caller is told about it"


class _TrailingResidue(FakeBackend):
    """Ends every chunk with material between the two silence thresholds.

    `trim_silence` keeps everything above the chunk's PEAK frame minus TRIM_DB
    (-42); the gate calls silent everything below the chunk's p95 frame minus
    SILENCE_DROP_DB (35). On speech those references sit ~0.1 dB apart, so a
    ~7 dB band survives trimming and reads as silence — which is exactly what
    issue #42 shipped, 9.03 s of it, at `silence_s=0.00`.

    4.5 s rather than the observed 9.03 s so the OTHER cheap checks pass on
    their own: 4.8 s of speech plus this tail is 9.3 s, inside the 2.67-10.5 s
    duration bounds and under the 9.68 s frame cap for this chunk. A test that
    passes because the ceiling fired pins nothing.
    """

    tail_s = 4.5

    def synthesize(self, text, voice, *, max_frames, temperature):
        import numpy as np

        audio = super().synthesize(text, voice, max_frames=max_frames,
                                   temperature=temperature)
        frame = int(0.02 * self.sample_rate)
        count = len(audio) // frame
        rms = np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(1) + 1e-20)
        level = float(np.percentile(rms, 95)) * 10 ** (-38 / 20)
        band = np.zeros(int(self.tail_s * self.sample_rate), dtype=np.float32)
        band[::2], band[1::2] = level, -level
        return np.concatenate([audio, band])


def test_trailing_dead_air_that_trimming_keeps_is_refused() -> None:
    """Issue #42: the gate delegated edges to `trim_silence`, which did not want them.

    A take 20.28 s long carrying 11.25 s of speech reported `silence_s=0.00`,
    `ok=True` at coverage 1.0, was written to the take store as verified, and was
    then served to two later renders — so no retry could ever reach it. The
    episode-level report saw the full 9.12 s, because it measures with the edges
    included; the gate did not, because it did not.
    """
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    backend = _TrailingResidue()
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(TEXT, 0, backend, verifier, VOICE, cfg)

    assert not result.ok, "9 s of dead air must not ship, trailing or interior"
    assert result.silence_s > cfg.max_silence_s, "and the report must name it"
    # The silence gate is what refused it. Every other cheap check passes on this
    # audio, and the round-trip is perfect — the words are all there.
    floor, ceiling = duration_bounds(len(TEXT.split()), cfg)
    assert floor <= result.duration_s <= ceiling
    assert result.duration_s < frame_cap(len(TEXT.split()), backend.fps, cfg) / backend.fps
    assert result.coverage == 1.0


def test_a_sentence_gap_at_the_gate_threshold_is_refused_up_front() -> None:
    """The rescue path inserts real zeros, so it must stay under the gate.

    Making the interaction unrepresentable is cheaper than teaching the check to
    recognise and skip its own joins.
    """
    with pytest.raises(ValueError, match="sentence_gap_s"):
        SynthConfig(sentence_gap_s=5.0)


def test_speech_that_merely_varies_is_not_a_hole() -> None:
    """The false positive that would matter most: quiet delivery is not dead air.

    A gate that refuses correct renders is worse than the defect it catches, so
    the margin is pinned. A passage 25 dB under the rest of its own chunk — far
    quieter than ordinary variation — must not register.
    """
    import numpy as np

    from narrator.audio import longest_silent_run

    sr = 24000

    def tone(seconds: float, amplitude: float) -> np.ndarray:
        t = np.arange(int(seconds * sr), dtype=np.float32) / sr
        return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    # 30 dB under, which is where a review found real low-level delivery living.
    # The default drop is 35 dB precisely so this shape has room.
    quiet = np.concatenate([tone(2, 0.1), tone(4, 0.1 * 10 ** (-30 / 20)), tone(2, 0.1)])
    assert longest_silent_run(quiet, sr) < 1.0


def test_an_extreme_hole_is_still_refused_by_another_gate() -> None:
    """Past ~96% dead air the percentile lands inside the hole and this fails open.

    Left that way on purpose: audio that silent cannot have said its words, so
    the frame cap and duration ceiling reject it first. Pinned so the reasoning
    is checkable rather than asserted.
    """
    cfg = SynthConfig(max_attempts=1, allow_sentence_split=False)
    backend = FakeBackend(script={0: Failure.SILENT_HOLE}, hole_s=60.0)
    verifier = CoverageVerifier(FakeASR(backend))
    result = synthesize_chunk(TEXT, 0, backend, verifier, VOICE, cfg)
    assert not result.ok


def test_a_sentence_split_assembly_is_checked_as_a_whole() -> None:
    """The parts passing does not make the assembly clean.

    Two sentences can each clear the gate and still meet across a join, because
    `trim_silence` is peak-relative and low-level residue survives it while
    counting as silence here. A review reproduced an assembly measuring 9.12 s of
    interior silence that returned ok=True, recovered_by="sentence-split", and
    was then stored for reuse.
    """
    import numpy as np

    from narrator.audio import longest_silent_run

    backend = FakeBackend()
    sr = backend.sample_rate
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    speech = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    assembly = np.concatenate([speech, np.zeros(int(6.0 * sr), dtype=np.float32), speech])

    cfg = SynthConfig()
    assert longest_silent_run(assembly, sr, cfg.silence_drop_db) > cfg.max_silence_s, (
        "the assembly is over the gate, so _synthesize must not report it clean"
    )


def test_a_non_finite_drop_is_refused() -> None:
    """A NaN threshold compares False against everything and disables the gate."""
    with pytest.raises(ValueError, match="silence_drop_db"):
        SynthConfig(silence_drop_db=float("nan"))


# ------------------------------------------------ declared non-speech atoms

ATOM = "<|emotion:surprise|>"
ATOM_CFG = SynthConfig(non_speech=(ATOM,))


def _tagged(script: str, cfg: SynthConfig = ATOM_CFG, script_modes=None):
    backend = FakeBackend(consumes=(ATOM,), script=script_modes or {})
    verifier = CoverageVerifier(FakeASR(backend))
    return synthesize_chunk(script, 0, backend, verifier, VOICE, cfg), backend


def test_a_tagged_chunk_verifies_against_its_speech() -> None:
    """The defect: every tagged chunk failed by construction.

    73 chunks, 11 tagged, exactly those 11 failed with the audio fine. The tag
    is one token the model acts on and never says, so no transcriber can return
    it — but it was part of the text the round-trip compared against.
    """
    result, backend = _tagged(f"{ATOM} Not the keeper. Not a stranger.")
    assert result.ok
    assert backend.requests[0].startswith(ATOM), "the engine still receives the atom"
    assert result.text.startswith(ATOM), "the report still shows the caller's script"


def test_the_guard_still_guards_a_tagged_chunk() -> None:
    """Declaring an atom must not become a way to launder a bad take."""
    cfg = replace(ATOM_CFG, max_attempts=1, allow_sentence_split=False)
    result, _ = _tagged(f"{ATOM} Not the keeper. Not a stranger.", cfg,
                        {0: Failure.DROP_SENTENCE})
    assert not result.ok


def test_word_count_comes_from_the_speech_not_the_markup() -> None:
    """Atoms inflate the duration FLOOR, which rejects correct short audio.

    Two real words carrying three atoms read as five, and the floor then demands
    ~1.11 s of audio for ~0.8 s of correct speech.
    """
    spoken = "Not the keeper."
    result, backend = _tagged(f"{ATOM} {spoken}")
    assert result.ok
    assert backend.max_frames_seen[0] == frame_cap(len(spoken.split()), 25, ATOM_CFG)


def test_atoms_are_opaque_to_the_pronunciation_lexicon() -> None:
    """Measured: a lexicon pair rewrote the token NAME.

    ("joy", "radost") turned `<|emotion:joy|>` into `<|emotion:radost|>` — a
    control token that does not exist — silently, because the tag never reaches
    verification either way. Declared-not-speech means not compared, not
    counted, and not respelled.
    """
    from narrator.synth import resolve_spoken

    cfg = SynthConfig(non_speech=("<|emotion:joy|>",), pronunciation=(("joy", "radost"),))
    assert resolve_spoken("<|emotion:joy|> joy", "en", cfg) == "<|emotion:joy|> radost"


def test_resolve_reference_never_fuses_two_words() -> None:
    from narrator.synth import resolve_reference

    assert resolve_reference(f"a{ATOM}b", ATOM_CFG) == "a b"
    # Untouched when no atom occurs, so declaring one cannot invalidate a take
    # for a chunk that does not contain it.
    assert resolve_reference("plain text", ATOM_CFG) == "plain text"


def test_an_atom_only_sentence_is_folded_into_its_neighbour() -> None:
    """A trailing atom must not cost a chunk its rescue path.

    Rendered alone, an atom-only sentence can never verify, so the split aborts
    and every tagged chunk loses the recovery that exists because the failure it
    rescues is stochastic.
    """
    from narrator.synth import _coalesce_atom_only

    assert _coalesce_atom_only(["Je to tak?", ATOM], ATOM_CFG) == [f"Je to tak? {ATOM}"]
    # A leading run folds FORWARD; there is no predecessor to fold into.
    assert _coalesce_atom_only([ATOM, "Je to tak?"], ATOM_CFG) == [f"{ATOM} Je to tak?"]


def test_an_atom_may_not_carry_whitespace_or_a_terminator() -> None:
    """Both rules are load-bearing, not tidiness.

    Whitespace would let removal stop commuting with sentence splitting; a
    terminator deletes a boundary from the reference, merging two scored
    sentences into one and eroding the granularity that stops a dropped
    sentence hiding in an aggregate.
    """
    with pytest.raises(ValueError, match="whitespace"):
        SynthConfig(non_speech=("<|a b|>",))
    with pytest.raises(ValueError, match="terminator"):
        SynthConfig(non_speech=("<|a.|>",))
