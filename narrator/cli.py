"""`narrate` — render a text file to audio.

Deliberately minimal. This library's interesting surface is the Python API, where
a caller resolves its own markup into segments. The CLI covers the plain case:
a text file whose blank lines are paragraph breaks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from narrator.audio import MasterConfig
from narrator.render import RenderConfig, RenderFailed, render
from narrator.types import ChunkResult, Gap, Segment, Text, Voice


def parse_text(raw: str, paragraph_gap: float) -> list[Segment]:
    """Blank line -> paragraph gap. Everything else is spoken."""
    segments: list[Segment] = []
    for block in (b.strip() for b in raw.split("\n\n")):
        if not block:
            continue
        if segments:
            segments.append(Gap(paragraph_gap))
        segments.append(Text(" ".join(block.split())))
    return segments


def _progress(result: ChunkResult, total: int) -> None:
    mark = "ok " if result.ok else "FAIL"
    note = f" ({result.recovered_by}, coverage {result.coverage:.2f})" if result.recovered_by else ""
    if not result.ok:
        note = f" coverage {result.coverage:.2f}"
        if result.dropped_sentence:
            note += f", dropped: {result.dropped_sentence[:50]}..."
    print(f"  [{result.index + 1}/{total}] {mark} {result.duration_s:5.1f}s{note}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="narrate",
        description="Render a text file to a mastered audio file, verifying every chunk.",
    )
    parser.add_argument("text", type=Path, help="UTF-8 text file; blank lines are paragraph breaks")
    parser.add_argument("out", type=Path, help="output .wav")
    parser.add_argument("--voice", type=Path, required=True, help="reference clip (wav)")
    parser.add_argument("--voice-text", required=True, help="transcript of the reference clip")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--paragraph-gap", type=float, default=0.35)
    parser.add_argument("--max-chars", type=int, default=250)
    parser.add_argument("--mono", action="store_true",
                        help="mono output; use to match an existing mono back-catalogue")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the ASR round-trip. Faster, and silent content loss "
                             "becomes undetectable — the failure this tool exists to prevent")
    parser.add_argument("--write-anyway", action="store_true",
                        help="write the file even if some chunk failed verification")
    args = parser.parse_args(argv)

    from narrator.backends.higgs import HiggsBackend, WhisperASR
    from narrator.verify import CoverageVerifier, NullVerifier

    backend = HiggsBackend()
    verifier = NullVerifier() if args.no_verify else CoverageVerifier(WhisperASR())
    voice = Voice(args.voice, args.voice_text, args.lang)
    segments = parse_text(args.text.read_text(encoding="utf-8"), args.paragraph_gap)
    if not segments:
        print(f"{args.text}: nothing to say", file=sys.stderr)
        return 2

    cfg = RenderConfig(
        max_chars=args.max_chars,
        mastering=MasterConfig(channels=1 if args.mono else 2),
        quarantine=not args.write_anyway,
        on_progress=_progress,
    )

    try:
        report = render(segments, voice, backend, verifier, args.out, cfg)
    except RenderFailed as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\n{report.summary()}")
    if not report.clean:
        print(f"WARNING: {len(report.failures)} chunk(s) failed and were written anyway.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
