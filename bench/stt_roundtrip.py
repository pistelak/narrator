#!/usr/bin/env python3
"""STT round-trip check for synthesized WAVs.

For every WAV in outputs/, runs mlx-whisper large-v3-turbo on the file and
writes outputs/<stem>.stt.txt. Language is forced to match the source input
prefix (cs for cs_* and mixed_*, en for en_*). This is the objective
intelligibility signal — the transcript should match the input text closely.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
OUTPUTS_DIR = ROOT / "outputs"

LANGUAGE_BY_PREFIX = {
    "cs_": "cs",
    "mixed_": "cs",
    "en_": "en",
}


def language_for(stem):
    for prefix, lang in LANGUAGE_BY_PREFIX.items():
        if stem.startswith(prefix):
            return lang
    return "cs"


def main():
    try:
        import librosa
        import mlx_whisper
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Activate the virtualenv and run: pip install mlx-whisper librosa"
        )

    wavs = sorted(p for p in OUTPUTS_DIR.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        sys.exit(f"No WAVs found in {OUTPUTS_DIR}. Run bench_xtts.py first.")

    repo = "mlx-community/whisper-large-v3-turbo"
    print(f"Using {repo}")

    for wav in wavs:
        engine_stem = wav.stem
        source_stem = engine_stem.split("__", 1)[0]
        lang = language_for(source_stem)
        print(f"  transcribing {wav.name} (lang={lang}) ...", flush=True)
        audio, _ = librosa.load(str(wav), sr=16000, mono=True)
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=lang)
        text = result["text"].strip()

        out_path = OUTPUTS_DIR / f"{engine_stem}.stt.txt"
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"    -> {out_path.name}: {text[:80]}{'...' if len(text) > 80 else ''}")


if __name__ == "__main__":
    main()
