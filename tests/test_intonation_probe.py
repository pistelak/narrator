"""The probe's resume guard, which protects a MEASUREMENT rather than a render.

`bench/` results justify design decisions — §11 is why `prosody.py` exists — so
a tag that silently mixes two speakers produces a wrong conclusion, and a wrong
conclusion is not re-checked the way a wrong audio file is.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# A test-only module name, restored after loading. `dataclasses` resolves the
# defining module through sys.modules, so the entry has to exist while the module
# executes — but claiming the generic name "intonation_probe" permanently made
# any later import in the same pytest process receive this bench script, which
# is an order-dependent suite waiting to happen.
_NAME = "_bench_intonation_probe_under_test"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, Path(__file__).resolve().parent.parent / "bench" / "intonation_probe.py"
)
probe = importlib.util.module_from_spec(_SPEC)
_PREVIOUS = sys.modules.get(_NAME)
sys.modules[_NAME] = probe
try:
    _SPEC.loader.exec_module(probe)
finally:
    if _PREVIOUS is None:
        sys.modules.pop(_NAME, None)
    else:
        sys.modules[_NAME] = _PREVIOUS


def _header(voice: Path) -> dict:
    return {
        "tag": "t",
        "voice_path": str(voice),
        "voice_digest": probe.content_digest(voice),
        "engine": probe._execution_identity(),
        "voice_transcript": "reference",
        "takes": 3,
        "params": {"a": 1},
    }


def _write(results: Path, header: dict) -> None:
    results.write_text(json.dumps({"header": header, "records": [{"case": "c"}]}),
                       encoding="utf-8")


def test_an_unchanged_reference_resumes(tmp_path: Path) -> None:
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF-speaker-A")
    results = tmp_path / "results.json"
    _write(results, _header(voice))

    assert probe._load_existing(results, _header(voice)) == [{"case": "c"}]


def test_a_reference_replaced_in_place_refuses_to_resume(tmp_path: Path) -> None:
    """The motivating case: same path, different speaker.

    Path text is not content identity. Guarded on the path alone, the probe
    resumed and appended a second speaker's rows to the first speaker's tag —
    exactly the mixed provenance this guard exists to refuse, and invisible
    afterwards because both runs look like one tag.
    """
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF-speaker-A")
    results = tmp_path / "results.json"
    _write(results, _header(voice))

    voice.write_bytes(b"RIFF-speaker-B")          # same path, same size, new speaker
    with pytest.raises(SystemExit, match="voice_digest"):
        probe._load_existing(results, _header(voice))


def test_a_results_file_with_no_digest_is_refused(tmp_path: Path) -> None:
    """Provenance that cannot be established must not be extended.

    A results.json written before this guard existed carries no digest, so it
    compares None against a real one and is refused rather than trusted — the
    conservative direction for a file whose whole purpose is a controlled
    comparison.
    """
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF-speaker-A")
    results = tmp_path / "results.json"
    legacy = _header(voice)
    del legacy["voice_digest"]
    _write(results, legacy)

    with pytest.raises(SystemExit, match="voice_digest"):
        probe._load_existing(results, _header(voice))


def test_a_semantics_change_refuses_to_resume(tmp_path: Path) -> None:
    """Rows measured under different verification policies must not be mixed.

    The guard covered the voice and the parameters but not the BUILD, and these
    constants move: `verify.SEMANTICS` went 1 -> 4 in one afternoon. A partial
    run from this morning would otherwise resume tonight and pool verdicts from
    four different definitions of "correct" into one clean report — the same
    mixed provenance the digest was added to prevent, on the other axis.
    """
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF-speaker-A")
    results = tmp_path / "results.json"
    stale = _header(voice)
    stale["engine"] = json.dumps({"verify": 1, "synth": 1})
    _write(results, stale)

    with pytest.raises(SystemExit, match="engine"):
        probe._load_existing(results, _header(voice))


def test_the_run_renders_from_its_own_snapshot(tmp_path: Path) -> None:
    """The digest must name the bytes actually synthesized.

    It was taken when the header was written and the clip read minutes later,
    per case, so replacing the file mid-run appended a second speaker's rows
    under the first speaker's digest. A run-owned copy cannot be edited out from
    under the rows it produced.
    """
    source = tmp_path / "voice.wav"
    source.write_bytes(b"RIFF-speaker-A")
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    snapshot = probe._snapshot_reference(source, out_dir)
    assert snapshot != source
    digest = probe.content_digest(snapshot)

    source.write_bytes(b"RIFF-speaker-B")          # edited mid-run
    assert probe.content_digest(snapshot) == digest, "the run's own bytes are untouched"
