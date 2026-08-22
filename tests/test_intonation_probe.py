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

_SPEC = importlib.util.spec_from_file_location(
    "intonation_probe", Path(__file__).resolve().parent.parent / "bench" / "intonation_probe.py"
)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["intonation_probe"] = probe          # dataclasses resolve via sys.modules
_SPEC.loader.exec_module(probe)


def _header(voice: Path) -> dict:
    return {
        "tag": "t",
        "voice_path": str(voice),
        "voice_digest": probe.content_digest(voice),
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
