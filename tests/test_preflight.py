"""Preflight predicts the render's refusal — before any model runs.

The oracle claim is the whole feature: a chunk is "doomed" iff even a PERFECT
recogniser (one that returns the reference verbatim) cannot get it past the
default gate. Both directions are pinned here against the real pipeline via
the fake backend, so preflight and render can never silently disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from narrator.backends.fake import FakeASR, FakeBackend
from narrator.preflight import preflight
from narrator.render import RenderFailed, render
from narrator.types import Gap, Text, Voice
from narrator.verify import MIN_COVERAGE, CoverageVerifier, coverage

VOICE = Voice(Path("nonexistent.wav"), "reference", "en")

# The exact shape from verify's measurement: scripts spell numerals out because
# digits are unspeakable, every ASR writes them back as digits, and
# number-blinding leaves an all-numeral sentence with nothing to compare.
DOOMED = "The port is easy to remember. Two fifty six."


# ------------------------------------------------------------------ doom

def test_all_numeral_sentence_is_flagged_and_named() -> None:
    report = preflight([Text(DOOMED)])
    assert not report.clean
    [finding] = report.unverifiable
    assert finding.index == 0
    assert "all-numeral" in finding.reason
    assert "Two fifty six" in finding.reason


def test_preflight_tracks_the_numeral_policy_it_is_predicting() -> None:
    """Preflight follows verify.py without being edited.

    "Dvou tisíc čtyřiceti osmi." is 2048 in the genitive — every word an
    inflected numeral, so blinding leaves nothing to align and the chunk is
    doomed. This verdict has moved twice while preflight itself stayed
    untouched: composing Czech compounds made it rescuable, and abandoning
    composition made it doomed again.

    That is the whole reason preflight runs the real coverage code rather than a
    hand-copied list of shapes that fail — a parallel list would have been wrong
    on both flips.
    """
    report = preflight(
        [Text("Heslo má přesně osm znaků. Dvou tisíc čtyřiceti osmi.")], lang="cs"
    )
    assert not report.clean
    assert "Dvou tisíc" in report.unverifiable[0].reason

    # Unrescuable in English, where the compound stays ambiguous by design.
    english = preflight([Text("The dial is set. Two fifty six.")], lang="en")
    assert not english.clean


def test_finding_index_matches_the_render_chunk_index() -> None:
    # Gaps consume no chunk index in render()'s numbering; a finding that
    # numbered them would name the wrong chunk in the eventual failure report.
    report = preflight([Text("Alpha beta gamma delta."), Gap(3.0), Text(DOOMED)])
    assert [f.index for f in report.unverifiable] == [1]
    assert report.chunks == 2


# ---------------------------------------------------------------- oracle

def test_doomed_script_is_refused_even_by_a_perfect_render(tmp_path: Path) -> None:
    """What preflight dooms, a flawless engine cannot save: the fake backend
    says exactly the text and the fake ASR hears it exactly, and the render
    still refuses — the failure is the script's, which is the entire point of
    detecting it before synthesis."""
    segments = [Text(DOOMED)]
    assert not preflight(segments).clean
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    with pytest.raises(RenderFailed, match="all-numeral"):
        render(segments, VOICE, backend, tmp_path / "doomed.wav", verifier)


def test_doom_models_the_whole_ladder_not_its_first_rung(tmp_path: Path) -> None:
    """"Look at the dial. Four." fails chunk-level identity — "Four." is
    all-numeral in context — but the sentence-split fallback renders it alone,
    where the isolated numeral verifies by value against the ASR's "4". An
    earlier preflight judged the chunk alone and declared doomed what a real
    render recovered cleanly (frontier review). Doomed means: no rung of the
    default ladder can ever pass."""
    segments = [Text("Look at the dial. Four.")]
    assert preflight(segments).clean
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    report = render(segments, VOICE, backend, tmp_path / "dial.wav", verifier)
    assert report.clean
    assert report.chunks[0].recovered_by == "sentence-split"


def test_clean_preflight_means_a_perfect_render_passes(tmp_path: Path) -> None:
    segments = [Text("Not the keeper. Not a stranger."), Gap(1.0),
                Text("They were sent to a destination that does not exist.")]
    assert preflight(segments).clean
    backend = FakeBackend()
    verifier = CoverageVerifier(FakeASR(backend))
    report = render(segments, VOICE, backend, tmp_path / "clean.wav", verifier)
    assert report.clean


# --------------------------------------------------------------- figures

def test_counts_and_estimates() -> None:
    report = preflight([Text("Alpha beta gamma delta epsilon."), Gap(2.5)])
    assert report.clean
    assert report.chunks == 1
    assert report.words == 5
    assert report.gap_s == 2.5
    assert report.speech_s == pytest.approx(5 / 2.5)
    assert "1 chunks" in report.summary()
    assert "every chunk can verify" in report.summary()


def test_summary_says_never_when_doomed() -> None:
    assert "NEVER" in preflight([Text(DOOMED)]).summary()


# ------------------------------------------------------------------- CLI

def test_cli_preflight_needs_no_voice_and_exits_nonzero_on_doom(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from narrator.cli import main

    script = tmp_path / "script.txt"
    script.write_text(DOOMED, encoding="utf-8")
    code = main([str(script), str(tmp_path / "out.wav"), "--preflight"])
    assert code == 1
    captured = capsys.readouterr()
    assert "NEVER" in captured.out
    assert "all-numeral" in captured.err


def test_cli_preflight_exits_zero_on_a_clean_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from narrator.cli import main

    script = tmp_path / "script.txt"
    script.write_text("Not the keeper.\n\nNot a stranger.", encoding="utf-8")
    code = main([str(script), str(tmp_path / "out.wav"), "--preflight"])
    assert code == 0
    assert "every chunk can verify" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--no-verify", "--write-anyway", "--reroll", "--takes"])
def test_cli_preflight_refuses_render_only_flags(tmp_path: Path, flag: str) -> None:
    """One exit code per outcome, not one per flag combination.

    An earlier version let --no-verify turn a doomed script into exit 0, on the
    reasoning that a render checking nothing cannot be doomed. Coherent in
    isolation, and unusable in aggregate: the SAME script then exited 0 under
    --no-verify, 1 under --write-anyway, and silently accepted a --reroll naming
    chunks that do not exist. A CI job reading "can NEVER verify" as success is
    the failure this library exists to prevent, one level up.
    """
    from narrator.cli import main

    script = tmp_path / "script.txt"
    script.write_text(DOOMED, encoding="utf-8")
    argv = [str(script), str(tmp_path / "out.wav"), "--preflight", flag]
    if flag in ("--reroll", "--takes"):
        argv.append("1" if flag == "--reroll" else str(tmp_path))
    with pytest.raises(SystemExit):
        main(argv)


def test_cli_still_demands_a_voice_for_a_real_render(tmp_path: Path) -> None:
    from narrator.cli import main

    script = tmp_path / "script.txt"
    script.write_text("Not the keeper.", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(script), str(tmp_path / "out.wav")])


def test_a_missing_script_does_not_mask_a_missing_voice(tmp_path: Path) -> None:
    """The required-option error must survive an unreadable input file.

    Moving the voice check out of argparse (so --preflight needs no voice) put
    it after the script read, where a missing file raised a filesystem
    traceback instead of naming the option the user forgot.
    """
    from narrator.cli import main

    with pytest.raises(SystemExit):
        main([str(tmp_path / "absent.txt"), str(tmp_path / "out.wav")])


def test_preflight_honours_a_disabled_sentence_split() -> None:
    """Preflight must model the ladder being RUN, not the default one.

    "Four." is unverifiable inside its chunk and comparable alone, so the split
    fallback rescues the chunk — but only if the caller left the fallback on.
    Assuming it reports clean on a chunk that render would refuse.
    """
    segments = [Text("Look at the dial. Four.")]
    assert preflight(segments).clean
    assert not preflight(segments, allow_sentence_split=False).clean


def test_a_clean_preflight_is_not_a_promise_that_the_render_will_pass() -> None:
    """One direction only: doomed is certain, clean is not.

    The identity oracle models a PERFECT recogniser, which is exactly what does
    not exist. Anything failing because the ASR writes the words back
    differently is invisible here, since both sides of an identity round-trip
    are spelled the same. A Czech fused numeral is the measured case (issue #8):
    it passes here and hard-failed 10/10 real attempts with correct audio.

    Pinned so nobody later mistakes a clean preflight for a guarantee and
    weakens the render's own refusal on the strength of it.
    """
    # An expressive control token (issue #17). The tag is in the reference and
    # absent from any real transcript, because no recogniser reads it aloud —
    # but an identity round-trip has it on BOTH sides, so it cancels out here.
    script = "<|emotion:surprise|> Jeden vous?"
    assert preflight([Text(script)], lang="cs").clean
    assert coverage(script, "Jeden vouz?", "cs")[0] < MIN_COVERAGE


def test_preflight_sees_declared_atoms_too() -> None:
    """Otherwise it reports clean for a chunk the render must refuse.

    Preflight round-trips the reference against itself. Given raw text, a
    declared atom sits on BOTH sides and cancels out, so a chunk that is nothing
    but atoms scores 1.0 here while its render has no speech left to verify.
    """
    atom = "<|sfx:laughter|>"
    segments = [Text(f"Alpha beta gamma delta. {atom}")]

    assert preflight(segments).clean, "unknown markup is just content"
    # Declared, the atom is removed first — exactly as the render will do.
    report = preflight(segments, non_speech=(atom,))
    assert report.clean, "a tag beside real speech is fine"

    only_atoms = preflight([Text(atom)], non_speech=(atom,))
    assert not only_atoms.clean, "a chunk with no speech left cannot be verified"
