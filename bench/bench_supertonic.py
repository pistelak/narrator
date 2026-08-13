#!/usr/bin/env python3
"""Supertonic 3 text-to-speech benchmark.

Reads every .txt in inputs/, synthesizes one WAV per input via Supertonic 3
(on-device ONNX, 31 languages incl. Czech), writes the WAV into outputs/, and
appends timing metrics to outputs/results.csv.

Unlike Piper, Supertonic is multilingual: language is set per call (cs for
Czech and mixed inputs, en for English) the same way XTTS-v2 is routed. Voice
is one of the 10 built-in styles (M1..M5, F1..F5); the default M1 is male, to
match Piper's male `jirka` voice for a like-for-like comparison.

The timed region is the synthesize call only. Model load and a one-shot warm-up
call are excluded.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Czech / mixed CZ-EN TTS benchmark (Supertonic 3).")
    parser.add_argument(
        "--voice",
        default="M1",
        help="Supertonic built-in voice style: M1..M5 (male) or F1..F5 (female). Default: M1.",
    )
    parser.add_argument(
        "--model",
        default="supertonic-3",
        help="Supertonic model id. Default: supertonic-3.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=8,
        help="Diffusion steps: 5 (low quality) to 12 (high). Default: 8.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.05,
        help="Speaking rate, 0.7 to 2.0. Default: 1.05.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded warm-up synthesis (default: warm up first).",
    )
    args = parser.parse_args()

    try:
        from supertonic import TTS
        import soundfile as sf
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Activate the virtualenv and run: pip install supertonic"
        )

    inputs = find_inputs(INPUTS_DIR)
    if not inputs:
        sys.exit(f"No .txt inputs found in {INPUTS_DIR}.")

    print(f"Loading model '{args.model}' (voice='{args.voice}') ...")
    tts = TTS(model=args.model, auto_download=True)
    try:
        style = tts.get_voice_style(args.voice)
    except Exception as exc:
        sys.exit(f"Unknown voice '{args.voice}'. Use M1..M5 or F1..F5. ({exc})")

    if not args.no_warmup:
        print("Warming up ...", flush=True)
        tts.synthesize(text="Ahoj.", voice_style=style, lang="cs",
                       total_steps=args.total_steps, speed=args.speed)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    csv_exists = RESULTS_CSV.exists()
    rows = []

    for input_path in inputs:
        text = input_path.read_text(encoding="utf-8").strip()
        lang = language_for(input_path)
        output_path = OUTPUTS_DIR / f"{input_path.stem}__supertonic.wav"
        print(f"  synthesizing {input_path.name} (lang={lang}, {len(text)} chars) ...", flush=True)

        start = time.perf_counter()
        wav, _ = tts.synthesize(
            text=text,
            voice_style=style,
            lang=lang,
            total_steps=args.total_steps,
            speed=args.speed,
        )
        elapsed = time.perf_counter() - start

        tts.save_audio(wav, str(output_path))
        data, sample_rate = sf.read(str(output_path))
        duration = len(data) / sample_rate
        rtf = elapsed / duration if duration else ""

        rows.append({
            "input_file": input_path.name,
            "engine": "supertonic",
            "model": args.model,
            "voice": args.voice,
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
