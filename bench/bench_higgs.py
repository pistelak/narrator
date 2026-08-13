#!/usr/bin/env python3
"""Higgs Audio v3 TTS benchmark (MLX, Apple Silicon).

Reads .txt files from inputs/, synthesizes one WAV per input via
bosonai/higgs-audio-v3-tts-4b through mlx-audio, writes WAVs into outputs/,
and appends timing metrics to outputs/results.csv in the same schema the
other bench_*.py scripts use.

One structural difference worth knowing, and the reason this engine is being
tested at all: Higgs takes NO language parameter. Piper is Czech-only,
XTTS-v2 and Supertonic take an explicit `lang` that selects one phonemizer
for the whole utterance -- which is exactly why the mixed_* inputs degrade on
them, and why a caller needs a pronunciation lexicon. Higgs infers language
from the text itself, so mixed Czech/English should need no routing at all.
The `voice` column records the preset speaker instead.

Runs in the isolated .venv-higgs (mlx-audio); the main .venv is left alone so
synthesize.py keeps working.

    .venv-higgs/bin/python bench_higgs.py
    .venv-higgs/bin/python bench_higgs.py --only mixed_ --max-chars 2000
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

DEFAULT_MODEL = "bosonai/higgs-audio-v3-tts-4b"
# 8k training context is shared between the text prompt and the audio tokens,
# so a long input silently truncates. Skip loudly instead -- see --max-chars.
DEFAULT_MAX_CHARS = 1200


def find_inputs(inputs_dir, only):
    paths = sorted(p for p in inputs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if only:
        paths = [p for p in paths if any(p.name.startswith(prefix) for prefix in only)]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Czech / mixed CZ-EN TTS benchmark (Higgs Audio v3).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default=None,
                        help="Preset speaker name. Default: the model's own default.")
    parser.add_argument("--ref-audio", default=None,
                        help="Reference clip for zero-shot voice cloning.")
    parser.add_argument("--ref-text", default=None,
                        help="Transcript of --ref-audio (required when cloning).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only inputs whose filename starts with one of these prefixes.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help=f"Skip inputs longer than this. Default: {DEFAULT_MAX_CHARS}.")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_audio.audio_io import write as audio_write
        from mlx_audio.tts import load
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}\nRun with .venv-higgs/bin/python (pip install mlx-audio).")

    inputs = find_inputs(INPUTS_DIR, args.only)
    if not inputs:
        sys.exit(f"No matching .txt inputs in {INPUTS_DIR}.")

    kept, skipped = [], []
    for p in inputs:
        text = p.read_text(encoding="utf-8").strip()
        (kept if len(text) <= args.max_chars else skipped).append((p, text))
    for p, text in skipped:
        print(f"  SKIP {p.name}: {len(text)} chars > --max-chars {args.max_chars}")
    if not kept:
        sys.exit("Every input was skipped; raise --max-chars.")

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    model = load(args.model)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    gen_kwargs = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    if args.voice:
        gen_kwargs["voice"] = args.voice
    if args.ref_audio:
        if not args.ref_text:
            sys.exit("--ref-audio requires --ref-text.")
        gen_kwargs["ref_audio"] = args.ref_audio
        gen_kwargs["ref_text"] = args.ref_text

    if not args.no_warmup:
        print("Warming up ...", flush=True)
        next(model.generate(text="Ahoj.", **gen_kwargs))

    OUTPUTS_DIR.mkdir(exist_ok=True)
    csv_exists = RESULTS_CSV.exists()
    rows = []

    for input_path, text in kept:
        output_path = OUTPUTS_DIR / f"{input_path.stem}__higgs.wav"
        print(f"  synthesizing {input_path.name} ({len(text)} chars) ...", flush=True)

        start = time.perf_counter()
        result = next(model.generate(text=text, **gen_kwargs))
        elapsed = time.perf_counter() - start

        audio_write(str(output_path), result.audio, result.sample_rate)
        duration = len(result.audio) / result.sample_rate
        rtf = elapsed / duration if duration else ""

        rows.append({
            "input_file": input_path.name,
            "engine": "higgs",
            "model": args.model,
            "voice": args.voice or ("clone" if args.ref_audio else "default"),
            "text_chars": len(text),
            "synthesis_time_seconds": round(elapsed, 2),
            "audio_duration_seconds": round(duration, 2),
            "realtime_factor": round(rtf, 3) if rtf != "" else "",
            "output_path": str(output_path.relative_to(ROOT)),
            "sample_rate": result.sample_rate,
        })
        rtf_str = f"{rtf:.3f}" if rtf != "" else "n/a"
        print(f"    done in {elapsed:.2f}s (audio {duration:.2f}s, RTF {rtf_str})")

    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(rows)

    peak_gb = mx.get_peak_memory() / 1e9
    print(f"\nWAVs saved to {OUTPUTS_DIR}")
    print(f"Metrics appended to {RESULTS_CSV}")
    print(f"Peak MLX memory: {peak_gb:.2f} GB")
    if skipped:
        print(f"Skipped {len(skipped)} input(s) over --max-chars: "
              + ", ".join(p.name for p, _ in skipped))


if __name__ == "__main__":
    main()
