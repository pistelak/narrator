#!/usr/bin/env python3
"""Is the verifier catching TTS defects, or ASR mistakes?

The verifier assumes a transcript that disagrees with the script means the audio
is wrong. The competing explanation is that the recogniser is wrong, and a single
transcript cannot distinguish them. That matters: one adjudication in an earlier
run scored a chunk as a real drop because Whisper returned "thank you" — which is
Whisper's most notorious hallucination on unclear or quiet audio, not necessarily
evidence that anything was dropped.

Two independent recognisers settle it:

  both disagree with the script, and agree with EACH OTHER   -> the audio is wrong
  the recognisers disagree with each other                   -> the ASR is unreliable
                                                                here, and a rejection
                                                                is not evidence

Run with the narrator venv:
    .venv-higgs/bin/python bench/asr_crosscheck.py <report.json> [--lang cs] [--limit 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from narrator.backends.higgs import HiggsBackend, WhisperASR
from narrator.backends.parakeet import ParakeetASR
from narrator.types import Voice
from narrator.verify import coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--voice", type=Path, default=None)
    args = ap.parse_args()

    voice_path = args.voice or (
        Path.home() / "Developer/voices" /
        (f"voice-reference.{args.lang}.wav" if args.lang != "en" else "voice-reference.wav")
    )
    voice = Voice(voice_path, voice_path.with_suffix(".txt").read_text(encoding="utf-8").strip(),
                  args.lang)

    rows = json.load(args.report.open(encoding="utf-8"))["results"]
    failures = [r for r in rows if not r["ok"] and r["transcript"]][: args.limit]
    print(f"{len(failures)} rejected chunks from {args.report.name}\n")

    backend = HiggsBackend()
    whisper = WhisperASR(source_rate=backend.sample_rate)
    parakeet = ParakeetASR(source_rate=backend.sample_rate)

    tally = {"audio_wrong": 0, "asr_unreliable": 0, "both_fine": 0}

    for r in failures:
        text = r["text"]
        cap = int((len(text.split()) / 2.5 * 1.6 + 2.0) * backend.frames_per_second())
        audio = backend.synthesize(text, voice, max_frames=cap, temperature=0.4)

        w = whisper.transcribe(audio, args.lang)
        p = parakeet.transcribe(audio, args.lang)
        w_score = coverage(text, w, args.lang)[0]
        p_score = coverage(text, p, args.lang)[0]
        agree = coverage(w, p, args.lang)[0]

        if w_score >= 0.9 and p_score >= 0.9:
            verdict, key = "both recognisers say the audio is FINE", "both_fine"
        elif w_score < 0.9 and p_score < 0.9 and agree >= 0.8:
            verdict, key = "both reject AND agree -> AUDIO IS WRONG", "audio_wrong"
        else:
            verdict, key = "recognisers DISAGREE -> ASR unreliable here", "asr_unreliable"
        tally[key] += 1

        print(f"[{r['i']:3d}] whisper {w_score:.2f} | parakeet {p_score:.2f} | "
              f"mutual {agree:.2f}  {verdict}")
        if key != "both_fine":
            print(f"      script  : {text[:95]}")
            print(f"      whisper : {w[:95]}")
            print(f"      parakeet: {p[:95]}")

    print(f"\n{'=' * 60}")
    n = max(sum(tally.values()), 1)
    for key, label in (("audio_wrong", "genuine TTS defects"),
                       ("asr_unreliable", "ASR disagreement — rejection not evidence"),
                       ("both_fine", "re-synthesis came out clean (stochastic)")):
        print(f"  {label:44s} {tally[key]:3d}  ({tally[key]/n*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
