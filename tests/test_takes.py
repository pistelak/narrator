"""The take store: what may be reused, and everything that must not be.

Two costs motivated this and both are pinned below: a one-word edit that
re-synthesised all 88 chunks of an episode, and a run killed at chunk 60 of 88
that left nothing to resume from. The rest of the file is the other half of the
job — every input that must invalidate a take, because serving one that was made
for different inputs puts wrong audio under a report that still says clean.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from narrator.backends.fake import Failure, FakeASR, FakeBackend
from narrator.render import RenderConfig, RenderFailed, render
from narrator.synth import SynthConfig, synthesize_chunk
from narrator.takes import TakeStore, identity_of
from narrator.types import Gap, Text, Voice
from narrator.verify import CoverageVerifier, NullVerifier

SEGMENTS = [
    Text("Not the keeper. Not a stranger. Not any council with any mandate."),
    Gap(3.0),
    Text("They were sent to a destination that does not exist."),
]


def voice_at(tmp_path: Path, content: bytes = b"reference-bytes") -> Voice:
    """A voice whose reference clip really exists, so it can really be digested."""
    path = tmp_path / "voice.wav"
    path.write_bytes(content)
    return Voice(path, "reference", "en")


def build(script=None):
    backend = FakeBackend(script=script or {})
    return backend, CoverageVerifier(FakeASR(backend))


def render_with(tmp_path, store_dir, segments=SEGMENTS, voice=None, out="episode.wav",
                script=None, cfg_kwargs=None, synth=None):
    backend, verifier = build(script)
    cfg = RenderConfig(takes=store_dir, on_progress=None, **(cfg_kwargs or {}))
    if synth is not None:
        cfg = RenderConfig(takes=store_dir, synth=synth, **(cfg_kwargs or {}))
    report = render(segments, voice or voice_at(tmp_path), backend,
                    tmp_path / out, verifier, cfg)
    return backend, report


# ------------------------------------------------------------ the two costs

def test_a_second_identical_render_synthesizes_nothing(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    first_backend, _ = render_with(tmp_path, takes, out="a.wav")
    assert first_backend.calls == 2

    second_backend, second = render_with(tmp_path, takes, out="b.wav")
    assert second_backend.calls == 0
    assert all(c.reused for c in second.chunks)
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_one_edited_word_resynthesizes_exactly_one_chunk(tmp_path: Path) -> None:
    """The issue, pinned. Six characters changed 88 chunks; here it changes one."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")

    edited = [SEGMENTS[0], SEGMENTS[1],
              Text("They were sent to a destination that does not appear.")]
    backend, report = render_with(tmp_path, takes, segments=edited, out="b.wav")

    assert backend.calls == 1
    assert backend.requests == ["They were sent to a destination that does not appear."]
    assert [c.reused for c in report.chunks] == [True, False]


def test_a_killed_render_leaves_its_completed_takes(tmp_path: Path) -> None:
    """A run stopped at chunk 60 of 88 left no artifact at all. Now it resumes.

    Interrupted from the progress callback, which is where a real kill lands:
    between chunks, with the backend configured exactly as it will be on the
    re-run.
    """
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)

    def kill_after_the_first(result, total):
        if result.index == 0:
            raise KeyboardInterrupt

    first = FakeBackend()
    with pytest.raises(KeyboardInterrupt):
        render(SEGMENTS, voice, first, tmp_path / "a.wav", CoverageVerifier(FakeASR(first)),
               RenderConfig(takes=takes, on_progress=kill_after_the_first))
    assert first.calls == 1
    assert len(list(takes.glob("*.json"))) == 1

    backend, report = render_with(tmp_path, takes, voice=voice, out="b.wav")
    assert backend.calls == 1          # only the chunk the killed run never reached
    assert report.clean
    assert [c.reused for c in report.chunks] == [True, False]


def test_a_refused_render_keeps_the_chunks_that_passed(tmp_path: Path) -> None:
    """Quarantine used to discard 87 good chunks along with the one bad one.

    The whole edit loop the refusal is meant to drive — read the failure, fix that
    line, run again — now costs one chunk instead of eighty-eight.
    """
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    # Truncates everything except the first generation, so chunk 0 lands and
    # chunk 1 exhausts the ladder.
    script = dict.fromkeys(range(1, 20), Failure.TRUNCATE)

    with pytest.raises(RenderFailed) as exc:
        render_with(tmp_path, takes, voice=voice, out="a.wav", script=script)
    assert str(takes) in str(exc.value)
    assert len(list(takes.glob("*.json"))) == 1

    fixed = [SEGMENTS[0], SEGMENTS[1], Text("They were sent somewhere else entirely.")]
    backend, report = render_with(tmp_path, takes, segments=fixed, voice=voice,
                                  out="b.wav", script=script)
    assert backend.calls == 1
    assert report.clean
    assert [c.reused for c in report.chunks] == [True, False]


# ------------------------------------------------------------ what must NOT invalidate

def test_a_level_change_reuses_every_take_and_still_moves_the_audio(tmp_path: Path) -> None:
    """`gain_db` is applied after synthesis, so tuning it must cost no generations.

    Two speakers, because one cannot show it: loudness normalisation moves a
    single-voice render by whatever the gain moved it, and the file comes out the
    same. A RELATIVE change between two voices is the thing mastering cannot
    absorb — and is what the field exists for.
    """
    takes = tmp_path / "takes"
    narrator = voice_at(tmp_path)
    other = tmp_path / "other.wav"
    other.write_bytes(b"another-speaker")

    def dialogue(gain: float) -> list:
        return [SEGMENTS[0], Gap(1.0),
                Text(SEGMENTS[2].text, voice=Voice(other, "q reference", "en", gain_db=gain))]

    render_with(tmp_path, takes, segments=dialogue(0.0), voice=narrator, out="a.wav")
    backend, _ = render_with(tmp_path, takes, segments=dialogue(-12.0), voice=narrator,
                             out="b.wav")

    assert backend.calls == 0
    assert (tmp_path / "a.wav").read_bytes() != (tmp_path / "b.wav").read_bytes()


def test_the_engine_never_sees_a_declared_gain(tmp_path: Path) -> None:
    """The structural half of the rule above: a backend cannot act on `gain_db`."""
    backend, verifier = build()
    voice = Voice(tmp_path / "v.wav", "reference", "en", gain_db=-6.0)
    synthesize_chunk("Not the keeper.", 0, backend, verifier, voice)
    assert [v.gain_db for v in backend.voices_seen] == [0.0]


def test_a_lexicon_entry_only_invalidates_the_chunks_that_use_it(tmp_path: Path) -> None:
    """Keying on the SPOKEN form is what keeps a pronunciation edit local.

    True for a caller that supplies its own verifier. The default policy does not
    get this — see the next test, which pins why.
    """
    takes = tmp_path / "takes"
    segments = [Text("Kalle wrote the manual."), Text("Nobody read it.")]
    render_with(tmp_path, takes, segments=segments, out="a.wav")

    backend, _ = render_with(tmp_path, takes, segments=segments, out="b.wav",
                             synth=SynthConfig(pronunciation=(("Kalle", "Kalleh"),)))
    assert backend.requests == ["Kalleh wrote the manual."]


def test_a_lexicon_entry_invalidates_the_episode_under_the_default_policy(
    tmp_path: Path,
) -> None:
    """The limitation, pinned rather than glossed.

    `render`'s default verifier takes the pronunciation lexicon as its
    sound-alike list, so the lexicon is part of what "correct" means and lands in
    the verifier's identity — every chunk's, not just the chunks containing the
    word. It cannot be projected per chunk either: a pair applies to the
    TRANSCRIPT as well as the text, and the transcript does not exist when the key
    is computed. So adding one entry re-renders the episode.
    """
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    segments = [Text("Kalle wrote the manual."), Text("Nobody read it.")]
    lexicon = (("Kalle", "Kalleh"),)

    first = FakeBackend()
    render(segments, voice, first, tmp_path / "a.wav",
           CoverageVerifier(FakeASR(first)), RenderConfig(takes=takes))
    assert first.calls == 2

    # What render builds for itself when verifier=None: the lexicon, doubling as
    # the sound-alike list.
    second = FakeBackend()
    with_lexicon = CoverageVerifier(FakeASR(second), sound_alikes=lexicon)
    render(segments, voice, second, tmp_path / "b.wav", with_lexicon,
           RenderConfig(takes=takes, synth=SynthConfig(pronunciation=lexicon)))
    assert second.calls == 2


# ------------------------------------------------------------ what MUST invalidate

def test_an_edited_reference_clip_invalidates_its_takes(tmp_path: Path) -> None:
    """Same path, same size, same mtime — a digest is the only thing that notices.

    ASR checks the words, not who said them, so a take served for the previous
    speaker would ship under a clean report.
    """
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path, b"speaker-one-aa")
    render_with(tmp_path, takes, voice=voice, out="a.wav")

    stat = voice.audio_path.stat()
    voice.audio_path.write_bytes(b"speaker-two-bb")            # identical length
    import os
    os.utime(voice.audio_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert voice.audio_path.stat().st_size == stat.st_size

    backend, _ = render_with(tmp_path, takes, voice=voice, out="b.wav")
    assert backend.calls == 2


def test_a_different_verifier_policy_invalidates(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    render_with(tmp_path, takes, voice=voice, out="a.wav")

    backend = FakeBackend()
    strict = CoverageVerifier(FakeASR(backend), min_coverage=0.99)
    render(SEGMENTS, voice, backend, tmp_path / "b.wav", strict, RenderConfig(takes=takes))
    assert backend.calls == 2


def test_an_unverified_take_is_never_reused_by_a_verified_render(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    unchecked = FakeBackend()
    render(SEGMENTS, voice, unchecked, tmp_path / "a.wav", NullVerifier(),
           RenderConfig(takes=takes))
    assert unchecked.calls == 2

    backend, verifier = build()
    render(SEGMENTS, voice, backend, tmp_path / "b.wav", verifier, RenderConfig(takes=takes))
    assert backend.calls == 2


def test_an_unidentifiable_object_disables_the_store(tmp_path: Path) -> None:
    """No identity, no cache — narrator will not guess what a stranger is."""

    class Anonymous:
        """A caller's own verifier, with nothing to say about what it is."""

        def verify(self, audio, text, lang):
            from narrator.types import Verdict
            return Verdict(ok=True, coverage=1.0)

    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    assert identity_of(Anonymous()) is None

    first = FakeBackend()
    render(SEGMENTS, voice, first, tmp_path / "a.wav", Anonymous(), RenderConfig(takes=takes))
    assert not list(takes.glob("*.json"))

    second = FakeBackend()
    render(SEGMENTS, voice, second, tmp_path / "b.wav", Anonymous(), RenderConfig(takes=takes))
    assert second.calls == 2


def test_a_failure_is_never_stored(tmp_path: Path) -> None:
    """A stored failure would make a transient failure permanent."""
    takes = tmp_path / "takes"
    with pytest.raises(RenderFailed):
        render_with(tmp_path, takes, segments=[SEGMENTS[0]],
                    script=dict.fromkeys(range(20), Failure.TRUNCATE))
    assert not list(takes.glob("*.json"))


def test_a_rise_wanting_chunk_is_not_stored(tmp_path: Path) -> None:
    """Rise selection picks WHICH verified take ships, and cannot be keyed on."""
    takes = tmp_path / "takes"
    segments = [Text("Máš teď chvilku?"), Text("Tady je odpověď.")]
    synth = SynthConfig(wants_rise=lambda text, lang: text.endswith("?"))
    render_with(tmp_path, takes, segments=segments, out="a.wav", synth=synth)
    assert len(list(takes.glob("*.json"))) == 1

    backend, report = render_with(tmp_path, takes, segments=segments, out="b.wav", synth=synth)
    assert backend.requests == ["Máš teď chvilku?"]
    assert [c.reused for c in report.chunks] == [False, True]


# ------------------------------------------------------------ integrity

def test_a_corrupt_take_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    wav = next(takes.glob("*.wav"))
    audio, rate = sf.read(str(wav), dtype="float32")
    sf.write(str(wav), audio[: len(audio) // 2], rate, subtype="FLOAT")

    backend, report = render_with(tmp_path, takes, out="b.wav")
    assert backend.calls == 1
    assert report.clean


def test_a_take_with_no_sidecar_is_invisible(tmp_path: Path) -> None:
    """The sidecar is the commit marker: audio alone is an unfinished write."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    for sidecar in takes.glob("*.json"):
        sidecar.unlink()

    backend, _ = render_with(tmp_path, takes, out="b.wav")
    assert backend.calls == 2


def test_a_rate_mismatch_is_a_miss(tmp_path: Path) -> None:
    """A take is only reusable at the rate the file will be written at."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    for sidecar in takes.glob("*.json"):
        meta = json.loads(sidecar.read_text())
        meta["sample_rate"] = 44_100
        sidecar.write_text(json.dumps(meta))

    backend, _ = render_with(tmp_path, takes, out="b.wav")
    assert backend.calls == 2


def test_a_store_that_cannot_be_written_never_fails_the_render(tmp_path: Path) -> None:
    """The store is an optimisation; a full disk must not lose a good render."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    backend, report = render_with(tmp_path, blocked, out="a.wav")
    assert report.clean
    assert backend.calls == 2
    # Said out loud, though: a store that caches nothing looks exactly like one
    # that works, until the next render bills for it.
    assert report.takes_unwritten == 2
    assert "2 take(s) not cached" in report.summary()


# ------------------------------------------------------------ reporting and reroll

def test_a_reused_chunk_reports_what_this_run_actually_did(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    _, report = render_with(tmp_path, takes, out="b.wav")

    assert all(c.attempts == 0 for c in report.chunks)
    assert "2 reused" in report.summary()


def test_a_recovery_survives_reuse_but_is_not_re_paid(tmp_path: Path) -> None:
    """`recovered_by` describes the audio; `attempts` describes the run."""
    takes = tmp_path / "takes"
    _, first = render_with(tmp_path, takes, segments=[SEGMENTS[0]], out="a.wav",
                           script={0: Failure.DROP_SENTENCE})
    assert first.chunks[0].recovered_by == "retry"
    assert first.chunks[0].attempts == 2

    _, second = render_with(tmp_path, takes, segments=[SEGMENTS[0]], out="b.wav",
                            script={0: Failure.DROP_SENTENCE})
    assert second.chunks[0].recovered_by == "retry"
    assert second.chunks[0].attempts == 0
    assert second.chunks[0].reused


def test_reroll_regenerates_one_chunk_even_when_another_shares_its_key(tmp_path: Path) -> None:
    """Two identical paragraphs share one key, which is why reroll bypasses the
    lookup instead of deleting the entry: the first would refill it."""
    takes = tmp_path / "takes"
    twice = [Text("Not the keeper."), Gap(1.0), Text("Not the keeper.")]
    render_with(tmp_path, takes, segments=twice, out="a.wav")

    backend, report = render_with(tmp_path, takes, segments=twice, out="b.wav",
                                  cfg_kwargs={"reroll": frozenset({1})})
    assert backend.calls == 1
    assert [c.reused for c in report.chunks] == [True, False]


def test_the_store_survives_a_take_written_by_another_run(tmp_path: Path) -> None:
    """Overwriting an entry mid-read is a digest mismatch, which is a miss."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    store = TakeStore(takes)
    key = next(takes.glob("*.json")).stem
    sf.write(str(takes / f"{key}.wav"), np.zeros(1000, dtype=np.float32), 24000, subtype="FLOAT")
    assert store.get(key, 24000, 0, "whatever") is None


# ------------------------------------------------------------ frontier-review regressions

def test_the_intent_policy_is_not_asked_about_sentences_nobody_renders(
    tmp_path: Path,
) -> None:
    """A caller's policy need not be pure.

    "Rise on the first question of a paragraph" is a reasonable policy and is
    stateful. Resolving every sentence up front, so the store could decide
    whether to file the chunk, consumed answers for sentences the ladder never
    reaches — the split fallback only runs when the whole chunk fails. So the
    chunk is asked once, the sentences are asked only if the fallback runs, and
    the store reads what actually happened.
    """
    asked: list[str] = []

    def policy(text: str, lang: str) -> bool:
        asked.append(text)
        return True

    takes = tmp_path / "takes"
    chunk = Text("Máš teď chvilku? Tohle je důležité.")
    render_with(tmp_path, takes, segments=[chunk], out="a.wav",
                synth=SynthConfig(wants_rise=policy))

    assert asked == [chunk.text]
    assert not list(takes.glob("*.json"))


def test_a_sentence_that_wants_a_rise_keeps_a_recovered_chunk_out_of_the_store(
    tmp_path: Path,
) -> None:
    """The fallback ran, so the sentences were asked — and one wanted a rise.

    The decision to file is taken after synthesis for exactly this case: the
    chunk itself wanted nothing, so a key existed, and rise selection still
    reached the audio that shipped.
    """
    asked: list[str] = []

    def policy(text: str, lang: str) -> bool:
        asked.append(text)
        return text.endswith("?")

    takes = tmp_path / "takes"
    chunk = Text("Tohle je důležité. Máš teď chvilku?")
    # The whole-chunk attempts all fail, so the ladder falls through to the
    # split, which renders each sentence alone.
    _, report = render_with(tmp_path, takes, segments=[chunk], out="a.wav",
                            synth=SynthConfig(wants_rise=policy),
                            script=dict.fromkeys(range(3), Failure.DROP_SENTENCE))

    assert report.chunks[0].recovered_by == "sentence-split"
    assert asked == [chunk.text, "Tohle je důležité.", "Máš teď chvilku?"]
    assert not list(takes.glob("*.json"))


def test_two_voices_differing_only_inside_a_field_are_not_one_voice(
    tmp_path: Path,
) -> None:
    """A transcript is arbitrary caller text, so it cannot be a delimiter's problem."""
    from narrator.takes import _voice_identity

    clip = tmp_path / "v.wav"
    clip.write_bytes(b"reference")
    assert (_voice_identity(Voice(clip, "a|b", "en"))
            != _voice_identity(Voice(clip, "a", "b|en")))


def test_two_sound_alike_policies_differing_inside_a_pair_are_not_one_policy() -> None:
    backend = FakeBackend()
    lumped = CoverageVerifier(FakeASR(backend), sound_alikes=(("a", "b,c>d"),))
    split = CoverageVerifier(FakeASR(backend), sound_alikes=(("a", "b"), ("c", "d")))
    assert identity_of(lumped) != identity_of(split)


def test_a_frame_cap_that_is_not_enforced_is_part_of_the_backend(tmp_path: Path) -> None:
    """`honours_frame_cap` decides whether a cap-length take is a runaway.

    A take stored while the check was off must not be reused by a backend that
    would now reject it.
    """
    capped = FakeBackend()
    uncapped = FakeBackend(honours_frame_cap=False)
    assert identity_of(capped) != identity_of(uncapped)


def test_a_sidecar_missing_a_field_is_a_miss(tmp_path: Path) -> None:
    """Corruption that parses is still corruption."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    for sidecar in takes.glob("*.json"):
        meta = json.loads(sidecar.read_text())
        del meta["coverage"]
        sidecar.write_text(json.dumps(meta))

    backend, report = render_with(tmp_path, takes, out="b.wav")
    assert backend.calls == 2
    assert report.clean


def test_the_refusal_only_promises_takes_that_exist(tmp_path: Path) -> None:
    """Every chunk here is rise-wanting, so nothing was stored — say nothing."""
    takes = tmp_path / "takes"
    synth = SynthConfig(wants_rise=lambda text, lang: True)
    with pytest.raises(RenderFailed) as exc:
        render_with(tmp_path, takes, segments=[SEGMENTS[0]], out="a.wav", synth=synth,
                    script=dict.fromkeys(range(20), Failure.TRUNCATE))
    assert "cached in" not in str(exc.value)


def test_the_refusal_counts_this_render_not_the_directory(tmp_path: Path) -> None:
    """Another script's takes in the same directory prove nothing about this one."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, segments=[Text("An unrelated paragraph entirely.")],
                out="other.wav")
    assert len(list(takes.glob("*.json"))) == 1

    # This render caches nothing of its own: every chunk fails.
    with pytest.raises(RenderFailed) as exc:
        render_with(tmp_path, takes, segments=[SEGMENTS[0]], out="a.wav",
                    script=dict.fromkeys(range(20), Failure.TRUNCATE))
    assert "cached in" not in str(exc.value)


def test_the_refusal_counts_reused_chunks_as_already_paid(tmp_path: Path) -> None:
    """A render that reuses and then still fails is cheap to re-run, and says so.

    The failing chunk comes first on purpose: `FakeBackend`'s failures are keyed
    by CALL index, and a reused chunk makes no call, so a failure scripted after
    one would shift onto a different generation the second time round.
    """
    takes = tmp_path / "takes"
    voice = voice_at(tmp_path)
    segments = [Text("They were sent to a destination that does not exist."),
                Text("Not the keeper.")]
    # The first chunk's three attempts, and nothing else. It is a single
    # sentence, so the split fallback declines and it spends exactly three.
    script = dict.fromkeys(range(3), Failure.TRUNCATE)

    with pytest.raises(RenderFailed):
        render_with(tmp_path, takes, segments=segments, voice=voice, out="a.wav",
                    script=script)
    assert len(list(takes.glob("*.json"))) == 1

    backend, _ = build(script)
    with pytest.raises(RenderFailed) as exc:
        render(segments, voice, backend, tmp_path / "b.wav",
               CoverageVerifier(FakeASR(backend)), RenderConfig(takes=takes))
    assert backend.calls == 3          # the failing chunk, again; the other reused
    assert "1 of this render's chunks are cached" in str(exc.value)


def test_the_fake_cannot_claim_a_configuration_it_does_not_honour() -> None:
    """Identity and behaviour read one normalised mapping, so order cannot split them."""
    quiet = Voice(Path("a.wav"), "r", "en")
    same_but_corrected = Voice(Path("a.wav"), "r", "en", gain_db=-6.0)

    one = FakeBackend(voice_amplitude={quiet: 0.05, same_but_corrected: 0.20})
    other = FakeBackend(voice_amplitude={same_but_corrected: 0.20, quiet: 0.05})
    assert (identity_of(one) == identity_of(other)) == (
        one._level_for(quiet) == other._level_for(quiet))


def test_reroll_refuses_a_chunk_number_that_cannot_exist(tmp_path: Path) -> None:
    """A silently ignored reroll looks exactly like one that changed nothing.

    Refused before anything is read or loaded, so a typo costs no time.
    """
    from narrator.cli import main

    argv = [str(tmp_path / "absent.txt"), str(tmp_path / "o.wav"),
            "--voice", "v.wav", "--voice-text", "x", "--reroll"]
    assert main([*argv, "0"]) == 2
    assert main([*argv, "two"]) == 2


def test_a_raising_sentence_query_does_not_cost_the_chunk_its_rise(tmp_path: Path) -> None:
    """A policy that raises reads as "no preference", never as a failed render.

    And it costs only the answer it was asked for: the chunk's own preference
    stands, while a policy that could not be asked leaves nothing cacheable.
    """
    from narrator.synth import resolve_rise_intent

    voice = Voice(tmp_path / "v.wav", "r", "cs")
    cfg = SynthConfig(wants_rise=lambda text, lang: text.endswith("?"))
    intent = resolve_rise_intent("Máš teď chvilku?", voice, cfg)
    assert intent.chunk is True
    assert not intent.cacheable

    def exploding(text: str, lang: str) -> bool:
        raise RuntimeError("unanswerable")

    chunk_intent = resolve_rise_intent("Tohle je důležité.", voice,
                                       SynthConfig(wants_rise=lambda t, lang: False))
    assert chunk_intent.cacheable
    # A sentence query that raises during the fallback: no preference for that
    # sentence, and the chunk stops being cacheable.
    assert chunk_intent.wants("Tohle je důležité.", voice,
                              SynthConfig(wants_rise=exploding)) is False
    assert not chunk_intent.cacheable


def test_reroll_past_the_end_of_the_render_is_refused(tmp_path: Path) -> None:
    backend, verifier = build()
    with pytest.raises(ValueError, match="outside this render"):
        render(SEGMENTS, voice_at(tmp_path), backend, tmp_path / "a.wav", verifier,
               RenderConfig(reroll=frozenset({999})))


def test_a_single_sentence_chunk_asks_the_intent_policy_once(tmp_path: Path) -> None:
    """One question about the chunk, and no speculation about its sentences."""
    from narrator.synth import resolve_rise_intent

    asked: list[str] = []

    def policy(text: str, lang: str) -> bool:
        asked.append(text)
        return True

    resolve_rise_intent("Máš teď chvilku?", Voice(tmp_path / "v.wav", "r", "cs"),
                        SynthConfig(wants_rise=policy))
    assert asked == ["Máš teď chvilku?"]


def test_a_sidecar_field_of_the_wrong_type_is_a_miss(tmp_path: Path) -> None:
    """It parses and the audio is intact, so only a coercion catches it."""
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    for sidecar in takes.glob("*.json"):
        meta = json.loads(sidecar.read_text())
        meta["duration_s"] = "bad"
        sidecar.write_text(json.dumps(meta))

    backend, report = render_with(tmp_path, takes, out="b.wav")
    assert backend.calls == 2
    assert report.clean


def test_a_take_copied_under_another_key_is_not_served(tmp_path: Path) -> None:
    """The entry names the key it was filed under, so the filename is not the claim.

    A directory merged from another machine, or a file copied by hand, would
    otherwise hand one chunk's audio to a chunk it was never made for — with a
    verdict, so the report would call it clean.
    """
    takes = tmp_path / "takes"
    render_with(tmp_path, takes, out="a.wav")
    sidecar = sorted(takes.glob("*.json"))[0]
    stolen = "0" * len(sidecar.stem)
    (takes / f"{stolen}.json").write_text(sidecar.read_text())
    (takes / f"{stolen}.wav").write_bytes((takes / f"{sidecar.stem}.wav").read_bytes())

    store = TakeStore(takes)
    assert store.get(stolen, 24000, 0, "whatever") is None
    assert store.get(sidecar.stem, 24000, 0, "whatever") is not None


def test_an_override_that_changes_the_rules_changes_the_identity() -> None:
    """A subclass inherits every field the identity is built from, so the class
    name is part of it — otherwise a stricter verifier reuses what a laxer one
    accepted."""
    backend = FakeBackend()

    class Stricter(CoverageVerifier):
        def verify(self, audio, text, lang):
            return replace(super().verify(audio, text, lang), ok=False)

    assert identity_of(Stricter(FakeASR(backend))) != identity_of(CoverageVerifier(
        FakeASR(backend)))


def test_an_identity_that_raises_disables_the_store_and_not_the_render(
    tmp_path: Path,
) -> None:
    """`getattr`'s default only covers AttributeError."""

    class Broken:
        def verify(self, audio, text, lang):
            from narrator.types import Verdict
            return Verdict(ok=True, coverage=1.0)

        @property
        def identity(self) -> str:
            raise RuntimeError("no idea who I am")

    assert identity_of(Broken()) is None
    takes = tmp_path / "takes"
    backend = FakeBackend()
    report = render(SEGMENTS, voice_at(tmp_path), backend, tmp_path / "a.wav", Broken(),
                    RenderConfig(takes=takes))
    assert report.clean
    assert not list(takes.glob("*.json"))


def test_a_dependency_with_no_metadata_disables_the_store(monkeypatch) -> None:
    """One spelling of "unknown" would make two different unknowns look alike.

    A vendored or editable checkout carrying no distribution metadata is exactly
    the case `identity_of` refuses to guess at everywhere else.
    """
    import narrator.asr
    from narrator.asr import WhisperASR

    monkeypatch.setattr(narrator.asr, "package_version", lambda name: "0.4.3")
    assert identity_of(WhisperASR()) is not None

    monkeypatch.setattr(narrator.asr, "package_version", lambda name: None)
    assert identity_of(WhisperASR()) is None


def test_two_subclasses_sharing_a_name_are_not_one_verifier() -> None:
    """`__qualname__` alone collides across modules; the rules do not."""
    from narrator.takes import _class_id

    def define(module: str) -> type:
        class Stricter(CoverageVerifier):
            pass

        Stricter.__module__ = module
        return Stricter

    here, elsewhere = define("one.place"), define("other.place")
    assert here.__qualname__ == elsewhere.__qualname__

    backend = FakeBackend()
    assert _class_id(here(FakeASR(backend))) != _class_id(elsewhere(FakeASR(backend)))
