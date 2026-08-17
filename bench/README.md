# bench — engine evaluation harness

This produced the numbers in `../docs/engine-comparison.md` and picked the
engine `narrator` now ships. Kept so the next bake-off runs against the same
inputs and round-trip, and is therefore comparable rather than merely newer.

The winner is Higgs Audio v3; `bench_higgs.py` is the reference for how it is
driven. To evaluate a challenger, add a `bench_<engine>.py` alongside, run the
same `inputs/`, and compare with `stt_roundtrip.py`.

## Original notes

A minimal benchmark for evaluating **local** text-to-speech on Czech and
realistic mixed Czech/English text. Sibling project to `stt-lab/`; same
folder shape, same CSV style, same "no Docker, no paid APIs" rules.

Four engines were evaluated:

| Engine | Status |
|---|---|
| **Supertonic 3** (`M1`) | Full benchmark — **best measured intelligibility (6/6), pending listening pass** |
| **Piper** (`cs_CZ-jirka-medium`) | Full benchmark — fastest/smallest; current default |
| **Coqui XTTS-v2** | Full benchmark — kept as a code-switch candidate |
| **Bark** (`suno/bark`) | Smoke-tested and dropped — no Czech speaker preset, RTF ≈ 6 |

See `RESULTS.md` for the per-input comparison.

No web UI, no Docker, no paid APIs. Everything runs locally.

## Layout

```
bench/
├── bench_piper.py        # Piper ONNX benchmark — fast Czech-only
├── bench_supertonic.py   # Supertonic 3 ONNX benchmark — multilingual, 44.1 kHz
├── bench_xtts.py         # Coqui XTTS-v2 benchmark — multilingual + voice cloning
├── stt_roundtrip.py      # mlx-whisper round-trip — objective intelligibility
├── inputs/               # one .txt per test sentence/paragraph
├── outputs/              # generated *.wav, *.stt.txt, and results.csv
├── .voices/piper/        # downloaded Piper ONNX voice files
├── README.md
└── RESULTS.md
```

## Requirements

- macOS on Apple Silicon (M1 or newer).
- **Python 3.12** via `asdf` (or any 3.12 you have on PATH).
- No system `ffmpeg` required. Audio is read/written through `soundfile`,
  `librosa`, and `wave`.

## Setup (macOS)

1. From `bench/`, create a venv with Python 3.12:

   ```sh
   cd bench
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. Install the engines and the STT round-trip:

   ```sh
   pip install piper-tts 'coqui-tts[codec]' mlx-whisper supertonic
   pip install 'transformers<5,>=4.45'
   ```

   - `piper-tts` ships eSpeak-NG phonemization bundled, so no system `espeak-ng`.
   - `coqui-tts[codec]` pulls in `torchcodec` (required by PyTorch ≥ 2.9 for
     audio I/O in Coqui).
   - `transformers` must be pinned to `<5` — XTTS uses
     `transformers.pytorch_utils.isin_mps_friendly`, which transformers 5.x
     removed.

3. Piper downloads its Czech voice automatically on first run, into
   `.voices/piper/`.

Model weights are downloaded on first use:

- Piper `cs_CZ-jirka-medium` → `.voices/piper/` (~60 MB)
- Supertonic 3 (`Supertone/supertonic-3`) → `~/.cache/huggingface/` (~1.8 GB download; ~380 MB active ONNX)
- XTTS-v2 → `~/Library/Application Support/tts/` (~1.7 GB)
- mlx-whisper `large-v3-turbo` → `~/.cache/huggingface/` (~1.5 GB)

## Running

With the venv active:

```sh
# Piper — fastest / smallest.
python bench_piper.py

# Supertonic 3 — multilingual ONNX, best measured intelligibility.
python bench_supertonic.py

# Coqui XTTS-v2 — multilingual / studio-voice candidate.
python bench_xtts.py

# Objective intelligibility check on every WAV in outputs/.
python stt_roundtrip.py
```

Each `bench_*.py` script:

- reads every `.txt` in `inputs/`,
- synthesizes one WAV into `outputs/<stem>__<engine>.wav`,
- and appends one row per input to `outputs/results.csv`.

`stt_roundtrip.py` then transcribes each WAV with `mlx-whisper
large-v3-turbo`, language forced to match the source input (Czech for
`cs_*` and `mixed_*`, English for `en_*`), and writes
`outputs/<stem>__<engine>.stt.txt`.

`results.csv` columns:

| column | meaning |
|---|---|
| `input_file` | source .txt filename |
| `engine` | `piper` or `coqui-xtts` |
| `model` | engine-specific model id |
| `voice` | voice / speaker name |
| `text_chars` | character count of the input |
| `synthesis_time_seconds` | wall-clock time of the synthesis call (warm) |
| `audio_duration_seconds` | length of the produced WAV |
| `realtime_factor` | `synthesis_time / audio_duration` — **lower is faster** |
| `output_path` | relative path to the WAV |
| `sample_rate` | WAV sample rate (Piper: 22 050; XTTS-v2: 24 000) |

Rows are **appended**, so running both engines accumulates a side-by-side
comparison in one CSV.

## Switching voice

### Piper

Any voice from <https://huggingface.co/rhasspy/piper-voices>. Czech voices:

```sh
python bench_piper.py --voice cs_CZ-jirka-medium   # default
python bench_piper.py --voice cs_CZ-jirka-low      # smaller, lower quality
```

### Supertonic 3

Ten built-in voice styles, no reference WAV required: `M1`..`M5` (male),
`F1`..`F5` (female). `M1` is the default (matches Piper's male `jirka`):

```sh
python bench_supertonic.py --voice M1            # default, male
python bench_supertonic.py --voice F1            # female
python bench_supertonic.py --total-steps 12      # higher quality (5..12)
python bench_supertonic.py --speed 1.0           # speaking rate (0.7..2.0)
```

### XTTS-v2

XTTS-v2 ships 58 built-in studio speakers — no reference WAV required:

```sh
python bench_xtts.py --speaker "Claribel Dervla"   # default
python bench_xtts.py --speaker "Daisy Studious"
python bench_xtts.py --speaker "Andrew Chipper"
```

## Reference clips

Cloning-backend probes (`intonation_probe.py`, `verifier_acceptance.py`,
`asr_crosscheck.py`, `asr_headtohead.py` — every script with a `--voice`
flag) take an operator-supplied reference clip with its exact
transcript in a `.txt` sidecar of the same basename. Keep clips under
`bench/.voices/ref/` — the directory is gitignored because a person's voice
recording must never land in the repo.

What to record (the reference is an in-context prosody example, not just a
timbre sample — the model imitates what it hears):

- 15–30 s, mono, ≥ 24 kHz, quiet room, no reverb or music, consistent mic
  distance, no clipping, natural narration pace.
- Include at least **one genuine yes/no question spoken with a real rise** —
  a purely declarative reference gives the model no example of this
  speaker's interrogative contour.
- Include one code-switched sentence (Czech frame, English technical term) —
  the pipeline's hardest input should be demonstrated, not just hoped for.
- The sidecar `.txt` is the exact spoken text, **punctuation included** —
  the engine conditions on the reference transcript, so its `?` matters.

Suggested naming: `bench/.voices/ref/radek_v1.wav` + `radek_v1.txt`.

## How language is chosen

XTTS-v2 and Supertonic 3 require an explicit language code per call. Piper
voices are single-language and do not. All bench scripts and
`stt_roundtrip.py` route inputs by filename prefix:

| Prefix | Language |
|---|---|
| `cs_*` | `cs` |
| `mixed_*` | `cs` (mixed inputs are dominantly Czech) |
| `en_*` | `en` |

## Why Piper wins despite being Czech-only

For a Czech assistant, "Czech-only" doesn't mean "fails on mixed text"; it
means **English fragments are read with Czech phonemes**. In practice that
sounds like a Czech speaker pronouncing an English brand name with a Czech
accent — which is exactly what Czech speakers do in real life:

- `Claude Code` → `klaudecode` *(Piper)*  vs  `Clouded Sword` *(XTTS-v2)*
- `feature flag` → `Fiature Flak`  vs  `feature flag` *(XTTS)*

XTTS-v2's "feature flag" survives the round-trip verbatim — but it also
silently rewrote `Claude Code` to a completely different phrase and added
trailing hallucinations on four of six inputs. Piper's Czech-accented
English is *recognizable* even when it's not phonetically perfect. See
`RESULTS.md §5` for the full per-input diff.

## Why Bark is not included

Bark looked promising on paper (multilingual, code-switching) but **the
`suno/bark` model on HuggingFace does not ship a Czech speaker preset** —
only de/en/es/fr/hi/it/ja/ko/pl/pt/ru/tr/zh. A smoke test with
`v2/pl_speaker_0` on one Czech command produced unrecoverable output
(`kalendáři` → `kalendaryk`, `event` dropped, `večer` looped) at RTF 6.4
(~60 s synthesis per 10 s of audio). Documented in `RESULTS.md §4.3`; the
smoke WAV and round-trip are preserved as
`outputs/bark_smoke_cs_command_pl-speaker-0.{wav,stt.txt}`.

## How CPU/GPU is used

- **Piper** — ONNXRuntime on CPU. Very low overhead per op; CPU is the
  right path.
- **XTTS-v2** — PyTorch on CPU. MPS works (`tts.to("mps")`) but is
  *slower* than CPU on this M5 Pro (3.5 s vs 2.0 s warm); XTTS-v2's
  per-step autoregressive ops don't amortize MPS kernel overhead.
- **mlx-whisper** — MLX (Apple GPU), as in `stt-lab/`.

CUDA is N/A on Apple Silicon.

## How to judge the results

Open `outputs/results.csv` for speed and `RESULTS.md` for the per-input
analysis. The scripts do **not** judge naturalness — they only produce WAVs
and round-trip transcripts. Listen to each WAV in `outputs/` to confirm
the qualitative dimensions (diacritics, voice consistency, prosody,
intonation) the round-trip can't see.

## License notice

- **Piper** voices are MIT-licensed (per-voice — check each `.onnx.json`).
  Cleared for commercial use.
- **Supertonic 3** model is **OpenRAIL-M** (sample code MIT). Commercial use
  is permitted subject to OpenRAIL behavioural-use restrictions — read the
  license file before relying on it. Less restrictive than XTTS's CPML.
- **XTTS-v2** is released under the **Coqui Public Model License (CPML)
  1.0**. Free for research and personal use, **commercial deployment is
  not permitted**.
- **Bark** (`suno/bark`) is MIT-licensed but excluded from the benchmark.

## Engines not included in this round

- **MMS-TTS** (`facebook/mms-tts-ces`) — small Czech-only baseline.
  Closest comparable to Piper but with a different vocoder.
- **eSpeak-NG** — robotic floor.
- **ChatterboxTTS / OuteTTS / F5-TTS / WhisperSpeech / Kokoro** —
  Czech support is undocumented or weak.

Adding any of these would follow the same pattern: a new `bench_<engine>.py`
that reads `inputs/`, writes a WAV per input into `outputs/`, and appends
one row per input to the same `outputs/results.csv`.
