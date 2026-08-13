#!/usr/bin/env python3
"""Coqui XTTS-v2 text-to-speech benchmark.

Reads every .txt in inputs/, synthesizes one WAV per input via XTTS-v2 with a
built-in studio speaker (no reference WAV needed), writes the WAV into
outputs/, and appends timing metrics to outputs/results.csv.

The timed region is the synthesis call only. Model load and a one-shot warm-up
call are excluded.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"

CSV_FIELDS = [
    "input_file",
    "engine",
    "model",
    "voice",
    "text_chars",
    "synthesis_time_seconds",
    "audio_duration_seconds",
    "realtime_factor",
    "output_path",
    "sample_rate",
]

# mixed_* is mostly Czech, so route it through the Czech tokenizer.
LANGUAGE_BY_PREFIX = {
    "cs_": "cs",
    "mixed_": "cs",
    "en_": "en",
}


def language_for(input_path):
    name = input_path.name
    for prefix, lang in LANGUAGE_BY_PREFIX.items():
        if name.startswith(prefix):
            return lang
    return "cs"


def find_inputs(inputs_dir):
    return sorted(p for p in inputs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def main():
    parser = argparse.ArgumentParser(description="Czech / mixed CZ-EN TTS benchmark (XTTS-v2).")
    parser.add_argument(
        "--speaker",
        default="Claribel Dervla",
        help="XTTS-v2 built-in studio speaker name. Default: 'Claribel Dervla'.",
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui model id. Default: XTTS-v2.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded warm-up synthesis (default: warm up first).",
    )
    args = parser.parse_args()

    try:
        from TTS.api import TTS
        import soundfile as sf
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Activate the virtualenv and run: pip install 'coqui-tts[codec]'"
        )

    inputs = find_inputs(INPUTS_DIR)
    if not inputs:
        sys.exit(f"No .txt inputs found in {INPUTS_DIR}.")

    print(f"Loading model '{args.model}' (voice='{args.speaker}') ...")
    tts = TTS(args.model, progress_bar=False)

    if args.speaker not in tts.synthesizer.tts_model.speaker_manager.name_to_id:
        sys.exit(
            f"Unknown speaker '{args.speaker}'. Examples: "
            f"{list(tts.synthesizer.tts_model.speaker_manager.name_to_id)[:5]}"
        )

    if not args.no_warmup:
        print("Warming up ...", flush=True)
        tts.tts(text="Ahoj.", speaker=args.speaker, language="cs")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    csv_exists = RESULTS_CSV.exists()
    rows = []

    for input_path in inputs:
        text = input_path.read_text(encoding="utf-8").strip()
        lang = language_for(input_path)
        output_path = OUTPUTS_DIR / f"{input_path.stem}__xtts.wav"
        print(f"  synthesizing {input_path.name} (lang={lang}, {len(text)} chars) ...", flush=True)

        start = time.perf_counter()
        tts.tts_to_file(
            text=text,
            speaker=args.speaker,
            language=lang,
            file_path=str(output_path),
        )
        elapsed = time.perf_counter() - start

        data, sample_rate = sf.read(str(output_path))
        duration = len(data) / sample_rate
        rtf = elapsed / duration if duration else ""

        rows.append({
            "input_file": input_path.name,
            "engine": "coqui-xtts",
            "model": "xtts_v2",
            "voice": args.speaker,
            "text_chars": len(text),
            "synthesis_time_seconds": round(elapsed, 2),
            "audio_duration_seconds": round(duration, 2),
            "realtime_factor": round(rtf, 3) if rtf != "" else "",
            "output_path": str(output_path.relative_to(ROOT)),
            "sample_rate": sample_rate,
        })
        rtf_str = f"{rtf:.3f}" if rtf != "" else "n/a"
        print(f"    done in {elapsed:.2f}s (audio {duration:.2f}s, RTF {rtf_str})")

    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nWAVs saved to {OUTPUTS_DIR}")
    print(f"Metrics appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
