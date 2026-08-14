#!/usr/bin/env python3
"""Piper text-to-speech benchmark.

Reads every .txt in inputs/, synthesizes one WAV per input via a Piper ONNX
voice, writes the WAV into outputs/, and appends timing metrics to
outputs/results.csv.

The Piper voice is single-language. English fragments inside Czech text are
read with Czech phonemes — that mismatch is the entire point of this row in
the comparison.

The timed region is the synthesize_wav call only. Voice load and a one-shot
warm-up call are excluded.
"""

import argparse
import csv
import io
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
OUTPUTS_DIR = ROOT / "outputs"
VOICES_DIR = ROOT / ".voices" / "piper"
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


def find_inputs(inputs_dir):
    return sorted(p for p in inputs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def ensure_voice(voice_name):
    onnx = VOICES_DIR / f"{voice_name}.onnx"
    if onnx.exists():
        return onnx
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Piper voice '{voice_name}' to {VOICES_DIR} ...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "piper.download_voices",
        "--download-dir", str(VOICES_DIR), voice_name,
    ])
    return onnx


def synthesize(voice, text, output_path):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    output_path.write_bytes(buf.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Czech TTS benchmark (Piper).")
    parser.add_argument(
        "--voice",
        default="cs_CZ-jirka-medium",
        help="Piper voice name (e.g. 'cs_CZ-jirka-medium'). Default: cs_CZ-jirka-medium.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded warm-up synthesis.",
    )
    args = parser.parse_args()

    try:
        from piper import PiperVoice
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Activate the virtualenv and run: pip install piper-tts"
        )

    inputs = find_inputs(INPUTS_DIR)
    if not inputs:
        sys.exit(f"No .txt inputs found in {INPUTS_DIR}.")

    onnx = ensure_voice(args.voice)
    print(f"Loading Piper voice '{args.voice}' ...")
    voice = PiperVoice.load(str(onnx))

    if not args.no_warmup:
        print("Warming up ...", flush=True)
        warm_buf = io.BytesIO()
        with wave.open(warm_buf, "wb") as wf:
            voice.synthesize_wav("Ahoj.", wf)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    csv_exists = RESULTS_CSV.exists()
    rows = []

    for input_path in inputs:
        text = input_path.read_text(encoding="utf-8").strip()
        output_path = OUTPUTS_DIR / f"{input_path.stem}__piper.wav"
        print(f"  synthesizing {input_path.name} ({len(text)} chars) ...", flush=True)

        start = time.perf_counter()
        synthesize(voice, text, output_path)
        elapsed = time.perf_counter() - start

        import soundfile as sf
        data, sr = sf.read(str(output_path))
        duration = len(data) / sr
        rtf = elapsed / duration if duration else ""

        rows.append({
            "input_file": input_path.name,
            "engine": "piper",
            "model": args.voice,
            "voice": args.voice,
            "text_chars": len(text),
            "synthesis_time_seconds": round(elapsed, 2),
            "audio_duration_seconds": round(duration, 2),
            "realtime_factor": round(rtf, 3) if rtf != "" else "",
            "output_path": str(output_path.relative_to(ROOT)),
            "sample_rate": sr,
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
