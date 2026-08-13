#!/usr/bin/env python3
"""Measure the verifier against REAL audio, both directions.

The gap this closes: every number we have describes false *rejection* — chunks
flagged that were fine. Nothing measures false *acceptance*, because no render so
far contained a confirmed corruption for the verifier to catch. A gate that has
never been shown to reject a real defect is an untested gate, however many clean
renders it has waved through.

So: synthesize real script text, keep only the chunks the verifier passes, then
corrupt that audio surgically and re-verify. A corruption that survives is a
false accept — the exact failure this library exists to prevent, measured rather
than argued.

Corruptions are applied to the AUDIO, not the text, so the ASR genuinely has to
hear the difference:

  drop        excise the samples spanning one sentence
  duplicate   repeat one sentence's samples in place
  truncate    cut the last third
  swap        reverse the order of two sentences

Run with the narrator venv:
    .venv-higgs/bin/python bench/verifier_acceptance.py <script.md> [--lang cs] [--limit 40]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from narrator.backends.higgs import HiggsBackend, WhisperASR
from narrator.chunking import chunk
from narrator.verify import CoverageVerifier
from narrator.types import Voice

SENTENCE_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')


@dataclass
class Case:
    chunk_index: int
    corruption: str
    text: str
    ok: bool
    coverage: float
    transcript: str
    note: str = ""


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def spoken_lines(path: Path) -> str:
    return " ".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not re.fullmatch(r"\[PAUSE[^\]]*\]", line.strip())
    )


def corrupt(audio: np.ndarray, kind: str, n_sentences: int, rng: random.Random) -> np.ndarray:
    """Cut the waveform at proportional sentence boundaries.

    Proportional rather than acoustic: a real forced alignment would be better,
    but the point is to remove or repeat roughly one sentence of speech, and the
    ASR only has to notice that the words changed.
    """
    if n_sentences < 2:
        return audio
    bounds = [round(i * len(audio) / n_sentences) for i in range(n_sentences + 1)]
    pick = rng.randrange(n_sentences)
    start, end = bounds[pick], bounds[pick + 1]

    if kind == "drop":
        return np.concatenate([audio[:start], audio[end:]])
    if kind == "duplicate":
        return np.concatenate([audio[:end], audio[start:end], audio[end:]])
    if kind == "truncate":
        return audio[: int(len(audio) * 2 / 3)]
    if kind == "swap" and n_sentences >= 3 and pick + 2 <= n_sentences:
        a, b, c = bounds[pick], bounds[pick + 1], bounds[pick + 2]
        return np.concatenate([audio[:a], audio[b:c], audio[a:b], audio[c:]])
    return audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--voice", type=Path,
                    default=Path.home() / "Developer/voices/voice-reference.wav")
    ap.add_argument("--limit", type=int, default=40, help="chunks to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench/acceptance-results.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    transcript = args.voice.with_suffix(".txt").read_text(encoding="utf-8").strip()
    voice = Voice(args.voice, transcript, args.lang)

    backend = HiggsBackend()
    verifier = CoverageVerifier(WhisperASR(source_rate=backend.sample_rate))

    chunks = chunk(spoken_lines(args.script), 250)
    chunks = [c for c in chunks if len(sentences(c)) >= 2][: args.limit]
    print(f"{len(chunks)} multi-sentence chunks from {args.script.name} (lang={args.lang})\n")

    cases: list[Case] = []
    started = time.perf_counter()

    for i, text in enumerate(chunks):
        cap = int((len(text.split()) / 2.5 * 1.6 + 2.0) * backend.frames_per_second())
        audio = backend.synthesize(text, voice, max_frames=cap, temperature=0.4)

        clean = verifier.verify(audio, text, args.lang)
        cases.append(Case(i, "none", text, clean.ok, clean.coverage, clean.transcript))
        if not clean.ok:
            print(f"  [{i}] clean chunk REJECTED ({clean.coverage:.2f}) — adjudicate: {text[:60]}...")
            continue   # only corrupt audio the verifier already trusts

        for kind in ("drop", "duplicate", "truncate", "swap"):
            bad = corrupt(audio, kind, len(sentences(text)), rng)
            if bad.size == audio.size and kind != "swap":
                continue
            verdict = verifier.verify(bad, text, args.lang)
            cases.append(Case(i, kind, text, verdict.ok, verdict.coverage, verdict.transcript))
            if verdict.ok:
                print(f"  [{i}] FALSE ACCEPT ({kind}, {verdict.coverage:.2f}): {text[:55]}...")

        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{len(chunks)} ({(time.perf_counter()-started)/60:.1f} min)", flush=True)

    clean_cases = [c for c in cases if c.corruption == "none"]
    dirty = [c for c in cases if c.corruption != "none"]
    false_reject = [c for c in clean_cases if not c.ok]
    false_accept = [c for c in dirty if c.ok]

    print(f"\n{'=' * 62}")
    print(f"clean chunks      : {len(clean_cases):3d}   false rejections: {len(false_reject):3d} "
          f"({len(false_reject)/max(len(clean_cases),1)*100:.1f}%)")
    print(f"corrupted chunks  : {len(dirty):3d}   FALSE ACCEPTS   : {len(false_accept):3d} "
          f"({len(false_accept)/max(len(dirty),1)*100:.1f}%)")
    for kind in ("drop", "duplicate", "truncate", "swap"):
        group = [c for c in dirty if c.corruption == kind]
        if group:
            caught = sum(1 for c in group if not c.ok)
            print(f"   {kind:10s} caught {caught:3d}/{len(group):<3d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(c) for c in cases], ensure_ascii=False, indent=1))
    print(f"\nfull record -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
