"""Split text into chunks small enough for a TTS engine to render reliably.

Budget rationale. Purpose-built audiobook tools converge on ~250 characters per
call: ebook2audiobook ships 125 for XTTS (deliberately half the model's stated
limit), tts-audiobook-tool 40 words, Coqui and Auralis 250, Chatterbox 300. The
measured cliff for Higgs v3 is 450-500, but it is *non-deterministic* — a 793-char
chunk truncated to half its content on one attempt and degenerated into four
minutes of babble on the next — so the budget leaves 2x margin rather than sitting
at the edge.

The right budget also depends on what the content can afford to lose. Audiobook
tools optimise for prose, where a dropped clause costs a detail. In a teaching
script where each sentence sets up the next, a dropped sentence makes the
following one unintelligible, so smaller is cheaper than it looks.
"""

from __future__ import annotations

import re

MAX_CHARS = 250
MIN_WORDS = 3

# Scored split points, Auralis-style. The distance decay is the important part:
# a comma *at* the target beats a full stop 25 characters away, where a plain
# priority ladder would always take the full stop and produce lopsided chunks.
_SPLIT_PRIORITY: tuple[tuple[str, float], ...] = (
    (". ", 1.0), ("! ", 1.0), ("? ", 1.0),
    (": ", 0.9), ("; ", 0.9),
    (", ", 0.8),
    (") ", 0.7), ("] ", 0.7),
    (" — ", 0.7), (" – ", 0.7), (" - ", 0.7),
    (" ", 0.5),
)
_WINDOW = 30

# Closing quotes and brackets may sit between the terminator and the space.
# Missing them merged sentences, which silently restored the aggregate
# scoring that per-sentence coverage exists to replace.
#
# This is THE sentence boundary for the whole library. synth's sentence-split
# fallback, verify's per-sentence scoring, and the fake backend's failure
# injection all import it: they must segment identically, or a "rescued"
# sentence and a "scored" sentence stop being the same unit.
_SENTENCE_END = re.compile(r'(?<=[.!?])["”’\')\]]*\s+')


def split_sentences(text: str) -> list[str]:
    """Sentence split.

    Deliberately regex rather than pysbd: pysbd scores far better on English
    (97.92% vs NLTK punkt's 56.25% on the Golden Rule Set) but ships no Czech,
    and this library is used on both. A caller with a single language and a
    strong segmenter should pass pre-split text.
    """
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _best_split(text: str, limit: int) -> int:
    """Index to cut at, scoring candidates by priority and nearness to `limit`."""
    best_at, best_score = -1, 0.0
    for token, priority in _SPLIT_PRIORITY:
        start = 0
        while True:
            at = text.find(token, start, limit + len(token))
            if at == -1:
                break
            cut = at + len(token)
            distance = abs(limit - cut)
            score = priority * (1 - distance / (_WINDOW * 2))
            if score > best_score:
                best_at, best_score = cut, score
            start = at + 1
    return best_at


def _split_oversized(sentence: str, max_chars: int) -> list[str]:
    """Break one over-budget sentence at the best internal punctuation."""
    parts: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        cut = _best_split(rest, max_chars)
        if cut <= 0:
            cut = rest.rfind(" ", 0, max_chars)
            if cut <= 0:
                break  # one unbreakable token; hand it over intact
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return [p for p in parts if p]


def chunk(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Pack `text` into chunks of at most `max_chars`, splitting on sentences.

    Guarantees, each earned from a reported failure:

    - No chunk begins with whitespace or is empty. Leading whitespace caused
      first-word crackling and hallucination in ebook2audiobook #1791.
    - No chunk is shorter than MIN_WORDS, except a whole input that is.
    - Concatenating the result with single spaces reproduces the input's words in
      order — the chunker never loses text. Verified by test, because a chunker
      that drops a sentence is indistinguishable from a TTS engine that does.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    buf = ""
    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            # Split the buffer *together with* the oversized sentence rather than
            # flushing it first. Flushing strands a short preceding sentence as its
            # own runt chunk that the backwards orphan merge cannot rescue, since
            # nothing precedes it — and short chunks are where engines are least
            # stable and duration checks have the least signal.
            combined = f"{buf} {sentence}".strip() if buf else sentence
            buf = ""
            chunks.extend(_split_oversized(combined, max_chars))
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) > max_chars:
            chunks.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        chunks.append(buf)

    return _merge_orphans([c.strip() for c in chunks if c.strip()], max_chars)


def _merge_orphans(chunks: list[str], max_chars: int) -> list[str]:
    """Fold runt chunks backwards where they fit.

    A two-word chunk is not just inefficient: short inputs are where engines are
    least stable, and where a duration-based check has the least signal.
    """
    out: list[str] = []
    for c in chunks:
        if (
            out
            and len(c.split()) < MIN_WORDS
            and len(out[-1]) + len(c) + 1 <= max_chars
        ):
            out[-1] = f"{out[-1]} {c}"
        else:
            out.append(c)
    return out
