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
from narrator.chunking import MAX_CHARS
from narrator.render import RenderConfig, RenderFailed, render
from narrator.types import ChunkResult, Gap, Segment, Text, Voice
from narrator.verify import format_word_diagnostics


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
    if result.reused:
        # Said plainly: this chunk was not generated or checked on this run, and
        # a line that looked identical to a fresh one would hide that.
        note = f" (reused{', ' + result.recovered_by if result.recovered_by else ''})"
    if not result.ok:
        note = f" coverage {result.coverage:.2f}"
        if result.dropped_sentence:
            note += f", dropped: {result.dropped_sentence[:50]}..."
        if result.word_diagnostics:
            note += f", {format_word_diagnostics(result.word_diagnostics)}"
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
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--mono", action="store_true",
                        help="mono output; use to match an existing mono back-catalogue")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the ASR round-trip. Faster, and silent content loss "
                             "becomes undetectable — the failure this tool exists to prevent")
    parser.add_argument("--write-anyway", action="store_true",
                        help="write the file even if some chunk failed verification")
    parser.add_argument("--takes", type=Path,
                        help="directory of verified takes to reuse and extend. An edited "
                             "script then re-renders only the chunks that changed, and a "
                             "killed run resumes from what it finished")
    parser.add_argument("--reroll", default="",
                        help="comma-separated chunk numbers (as printed, 1-based) to "
                             "generate fresh, ignoring any stored take — for audio that "
                             "verifies but does not sound right")
    args = parser.parse_args(argv)

    from narrator.backends.higgs import HiggsBackend
    from narrator.verify import NullVerifier

    backend = HiggsBackend()
    verifier = NullVerifier() if args.no_verify else None  # None -> render's default
    voice = Voice(args.voice, args.voice_text, args.lang)
    segments = parse_text(args.text.read_text(encoding="utf-8"), args.paragraph_gap)
    if not segments:
        print(f"{args.text}: nothing to say", file=sys.stderr)
        return 2

    try:
        # 1-based on the way in, to match the progress lines the user is reading
        # them off; 0-based inside, where chunk indices live.
        reroll = frozenset(int(n) - 1 for n in args.reroll.split(",") if n.strip())
    except ValueError:
        print(f"--reroll takes chunk numbers, got {args.reroll!r}", file=sys.stderr)
        return 2

    cfg = RenderConfig(
        max_chars=args.max_chars,
        mastering=MasterConfig(channels=1 if args.mono else 2),
        quarantine=not args.write_anyway,
        on_progress=_progress,
        takes=args.takes,
        reroll=reroll,
    )

    try:
        report = render(segments, voice, backend, args.out, verifier, cfg)
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
