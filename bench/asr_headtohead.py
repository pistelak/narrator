#!/usr/bin/env python3
"""Which recogniser should be the primary verification oracle?

asr_crosscheck.py adjudicated *rejected* chunks only — a sample selected
precisely where ASR is struggling, so it can't rank the models overall. This
runs both recognisers over EVERY chunk of an episode, on the same freshly
synthesized audio, and buckets each chunk by who accepted what:

  both accept                    -> uninformative for ranking (the common case)
  only whisper rejects           -> parakeet rescued it: whisper false-rejection
                                    candidate (cheap to review, cascade fixes it)
  only parakeet rejects          -> the reverse
  both reject, transcripts agree -> genuine TTS defect (the cascade keeps these)
  both reject, transcripts differ-> ASR unreliable here, rejection is not evidence

The solo-reject buckets are what decide primary order: a cascade only reviews
rejections, so the primary's false ACCEPTS are the one unreviewed path. Neither
bucket measures false accepts directly (no ground truth without ears), so the
report prints every disagreement for manual reading.

Run with the narrator venv:
    .venv-higgs/bin/python bench/asr_headtohead.py <report.json> [more.json ...] \
        [--lang cs] [--out bench/outputs/asr_headtohead.cs.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from narrator.asr import ParakeetASR, WhisperASR
from narrator.audio import resample_to_16k
from narrator.backends.higgs import HiggsBackend
from narrator.synth import SynthConfig, _best_attempt
from narrator.types import Voice
from narrator.verify import MIN_COVERAGE, NullVerifier, coverage

_HERE = Path(__file__).parent

THRESHOLD = MIN_COVERAGE
MUTUAL = 0.80      # same bar asr_crosscheck.py uses for "the transcripts agree"


class CanaryASR:
    """Third opinion: canary-1b-v2 via onnx-asr (CPU/CoreML). Best published Czech
    WER of the locally-runnable candidates (7.86 FLEURS vs parakeet v3's 11.01);
    here to test whether a stronger oracle shrinks the rejection rate."""

    def __init__(self, source_rate: int) -> None:
        import onnx_asr
        self.model = onnx_asr.load_model("nemo-canary-1b-v2")
        self.source_rate = source_rate

    def transcribe(self, audio, lang: str) -> str:
        clip = resample_to_16k(audio, self.source_rate)
        return self.model.recognize(clip, language=lang).strip()


def classify(w_score: float, p_score: float, mutual: float) -> str:
    w_ok, p_ok = w_score >= THRESHOLD, p_score >= THRESHOLD
    if w_ok and p_ok:
        return "both_accept"
    if w_ok and not p_ok:
        return "parakeet_only_reject"
    if p_ok and not w_ok:
        return "whisper_only_reject"
    return "both_reject_agree" if mutual >= MUTUAL else "both_reject_disagree"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--out", type=Path, default=_HERE / "outputs" / "asr_headtohead.json")
    ap.add_argument("--voice", type=Path, default=None)
    args = ap.parse_args()

    voice_path = args.voice or (
        Path.home() / "Developer/voices" /
        (f"voice-reference.{args.lang}.wav" if args.lang != "en" else "voice-reference.wav")
    )
    voice = Voice(voice_path, voice_path.with_suffix(".txt").read_text(encoding="utf-8").strip(),
                  args.lang)

    backend = HiggsBackend()
    whisper = WhisperASR(source_rate=backend.sample_rate)
    parakeet = ParakeetASR(source_rate=backend.sample_rate)
    canary = CanaryASR(source_rate=backend.sample_rate)
    scfg = SynthConfig()

    def checkpoint() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    records: list[dict] = []
    for report_path in args.reports:
        rows = json.load(report_path.open(encoding="utf-8"))["results"]
        episode = report_path.parent.name
        print(f"\n=== {episode} ({len(rows)} chunks) ===", flush=True)

        for r in rows:
            text = r["text"]
            # Reuse synth.py's cheap-check gate directly: audio that hit the frame
            # cap or has an implausible duration never reaches ASR in the real
            # pipeline, so letting it through here would count harness truncations
            # as TTS defects. (The first run did exactly that: 24 of 42 "defects"
            # were cap truncations.) With a NullVerifier, `attempt.ok` is exactly
            # "cheap checks passed".
            t0 = time.perf_counter()
            attempt = _best_attempt(text, backend, NullVerifier(), voice, scfg)
            synth_s = time.perf_counter() - t0

            rec = {"episode": episode, "i": r["i"], "text": text,
                   "orig_ok": r["ok"], "orig_coverage": r["coverage"]}
            if attempt is None or not attempt.ok:
                rec |= {"whisper": "", "parakeet": "", "canary": "",
                        "w_score": 0.0, "p_score": 0.0, "c_score": 0.0,
                        "mutual": 0.0, "bucket": "cheap_fail"}
                records.append(rec)
                checkpoint()
                print(f"[{r['i']:3d}] cheap checks never passed "
                      f"({synth_s:4.1f}s synth)  <- cheap_fail", flush=True)
                continue

            audio = attempt.audio
            w = whisper.transcribe(audio, args.lang)
            p = parakeet.transcribe(audio, args.lang)
            c = canary.transcribe(audio, args.lang)
            w_score = coverage(text, w, args.lang)[0]
            p_score = coverage(text, p, args.lang)[0]
            c_score = coverage(text, c, args.lang)[0]
            mutual = coverage(w, p, args.lang)[0]
            bucket = classify(w_score, p_score, mutual)

            rec |= {"whisper": w, "parakeet": p, "canary": c,
                    "w_score": round(w_score, 3), "p_score": round(p_score, 3),
                    "c_score": round(c_score, 3),
                    "mutual": round(mutual, 3), "bucket": bucket}
            records.append(rec)
            checkpoint()
            flag = "" if bucket == "both_accept" else f"  <- {bucket}"
            print(f"[{r['i']:3d}] w {w_score:.2f} | p {p_score:.2f} | c {c_score:.2f} | "
                  f"mutual {mutual:.2f} ({synth_s:4.1f}s synth){flag}", flush=True)

    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(rec["bucket"], []).append(rec)
    scored = [r for r in records if r["bucket"] != "cheap_fail"]
    n = max(len(scored), 1)

    print(f"\n{'=' * 64}")
    print(f"{len(records)} chunks total, {len(buckets.get('cheap_fail', []))} never "
          f"passed cheap checks; {n} reached ASR, threshold {THRESHOLD}\n")
    w_rej = sum(1 for r in scored if r["w_score"] < THRESHOLD)
    p_rej = sum(1 for r in scored if r["p_score"] < THRESHOLD)
    c_rej = sum(1 for r in scored if r["c_score"] < THRESHOLD)
    print(f"  whisper rejects  {w_rej:3d}  ({w_rej / n * 100:.0f}%)")
    print(f"  parakeet rejects {p_rej:3d}  ({p_rej / n * 100:.0f}%)")
    print(f"  canary rejects   {c_rej:3d}  ({c_rej / n * 100:.0f}%)")
    rescued = sum(1 for r in scored
                  if r["bucket"].startswith("both_reject") and r["c_score"] >= THRESHOLD)
    solo = sum(1 for r in scored
               if r["c_score"] < THRESHOLD
               and r["w_score"] >= THRESHOLD and r["p_score"] >= THRESHOLD)
    any_rej = sum(1 for r in scored
                  if r["w_score"] < THRESHOLD and r["p_score"] < THRESHOLD
                  and r["c_score"] < THRESHOLD)
    print(f"  canary accepts a chunk w+p both rejected     {rescued:3d}")
    print(f"  canary-only reject (w+p both accepted)       {solo:3d}")
    print(f"  rejected by all three oracles                {any_rej:3d}  "
          f"({any_rej / n * 100:.0f}%)")
    for key, label in (
        ("both_accept", "both accept"),
        ("whisper_only_reject", "whisper-only reject (parakeet rescues)"),
        ("parakeet_only_reject", "parakeet-only reject (whisper rescues)"),
        ("both_reject_agree", "both reject, agree -> genuine defect"),
        ("both_reject_disagree", "both reject, disagree -> ASR unreliable"),
    ):
        k = len(buckets.get(key, []))
        print(f"  {label:44s} {k:3d}  ({k / n * 100:.0f}%)")

    print(f"\ndisagreements for manual review ({args.out}):")
    for rec in records:
        if rec["bucket"] in ("both_accept", "cheap_fail"):
            continue
        print(f"\n[{rec['episode']} #{rec['i']}] {rec['bucket']}  "
              f"w {rec['w_score']:.2f} p {rec['p_score']:.2f} c {rec['c_score']:.2f}")
        print(f"  script  : {rec['text'][:100]}")
        print(f"  whisper : {rec['whisper'][:100]}")
        print(f"  parakeet: {rec['parakeet'][:100]}")
        print(f"  canary  : {rec['canary'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
