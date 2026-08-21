"""The take store: what may be reused, and everything that must not be.

Two costs motivated this and both are pinned below: a one-word edit that
re-synthesised all 88 chunks of an episode, and a run killed at chunk 60 of 88
that left nothing to resume from. The rest of the file is the other half of the
job — every input that must invalidate a take, because serving one that was made
for different inputs puts wrong audio under a report that still says clean.
"""

from __future__ import annotations

import json
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
