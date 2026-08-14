# narrator

Long-form narration from text. Chunk, synthesize, **verify**, stitch, master.

Built for episodes and audiobooks — twenty to forty minutes of speech assembled
from ~100 independently generated chunks, where the hard problem is not making
audio but knowing whether the audio says what you asked for.

## Why this exists

Modern neural TTS is excellent per sentence and unreliable per paragraph. Above a
few hundred characters it non-deterministically truncates, repeats, or degenerates
— and it does so **silently**, producing a plausible waveform of plausible length
containing the wrong words.

The pipeline this was extracted from shipped a twenty-minute episode with seven
dropped sentences, including a question that left a pause and an answer with
nothing between them. Every chunk passed duration validation. That is the failure
class this library exists to make impossible.

## Design

    segments ──► chunk ──► synthesize ──► verify ──► retry ladder ──► stitch ──► master
                                            │
                                    ASR round-trip:
                                    does it say the words?

- **The caller owns markup.** Narrator's input is `Text` and `Gap` segments and
  nothing else. It never learns what a `[PAUSE]` marker or an SSML tag is.
- **The engine is behind a `Backend` protocol.** Chunking, verification and retry
  are engine-independent, so swapping engines does not mean re-earning them.
- **Verification is not optional.** Duration heuristics caught zero of eight real
  content drops in the measurements that motivated this library.

## Requirements

The core library (chunking, verification scoring, assembly, mastering) is pure
Python ≥3.12 and runs anywhere. The bundled engines and recognisers are
**Apple-Silicon only**: the `[higgs]` extra (Higgs Audio v3 + mlx-whisper) and
the `[parakeet]` extra (the cascade's fast first opinion) both build on MLX.
On other hardware, implement the small `Backend`/`ASR` protocols against your
engine of choice — that seam is the point of the library.

## Status

Early. Extracted from a working pipeline; being rebuilt with the tests that
pipeline never had.

## License

MIT.
