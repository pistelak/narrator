# Long-form rendering with Higgs Audio v3 — implementation plan

**Date:** 2026-08-11
**Problem:** Higgs v3 on MLX is reliable per-sentence and non-deterministically
catastrophic above ~450–500 characters. The failure is **silent** — plausible
waveform, plausible duration, wrong content. A 25-minute episode is ~57 chunks at
current sizing, ~94 at the recommended sizing.
**Companion:** `RESULTS-higgs.md` (the engine comparison and measurements)

Evidence tags: **[measured]** benchmark, study, or verified here ·
**[shipping]** a real tool's production constant · **[inference]** reasoning from
mechanism · **[folklore]** no evidence found.

---

## 0. What we measured

| | value |
|---|---|
| RTF | 0.763, stable across 125–450 char chunks |
| Peak MLX memory | 12.3 GB |
| 30 min of audio | ~23 min clean, ~30–35 min with retries |
| Failure rate at 450-char chunks | 2 of 6 in one sample, 1 of 6 on retry |

Failure is **stochastic, not deterministic** — a 330-char chunk that produced 214 s
of babble passed on 2 of 3 retries; a 125-char chunk that produced 103 s passed 3 of
3. This is what makes a validate-and-retry loop viable. **[measured]**

Failures are expensive: RTF stays ~0.76 *of produced audio*, and babble produces
8–10× the audio it should, so one failure costs roughly 8 normal chunks. A frame cap
(step 1) is what fixes the economics.

### Root cause

**The MLX port has no repetition-aware sampling.** `generation.py` imports only
`apply_top_k` and `apply_top_p` from `mlx_lm`; there is no `ras`, `repetition`, or
`penalty` anywhere in the module. Boson's reference implementation runs RAS by
default. Verified directly. **[measured]**

The truncation half is **premature EOS** — the same mechanism reported against XTTS,
where one user measured *"the same text produced 1, 157, and 194 audio codes across
runs"* (ebook2audiobook #2054, fixed by PR #2056 with a `min_new_tokens` floor).
**Reseeding alone re-rolls the same dice; a minimum-length floor attacks the
mechanism.**

---

## Phase A — make silent failure impossible (~1 day)

### 1. Hard frame cap on every generation

```python
expected_s     = words / 2.5                    # ~150 wpm; use /2.1 for Czech
max_new_frames = int(expected_s * 1.6 * 25)     # 25 fps at 24 kHz, 60% headroom
```

**This is the early abort** — decoding physically stops, no runtime heuristic needed.
25 fps / 40 ms per frame is from the SGLang-Omni Higgs cookbook **[measured]**; the
1.6 headroom multiplier is **[inference]**.

✅ **No patching required.** `max_new_frames` is already an accepted parameter and is
resolved by `_resolve_generation_limit` — verified at `model.py:505`, with call sites
at `model.py:563/613/749/766`. Just pass it.

The default `max_new_tokens=2048` ≈ 82 s is far too loose for a ~26 s chunk; our
length sweep used 8192, roughly 12× what a chunk needs — which is why a babble run
burned 197 s instead of a bounded ~40 s.

### 2. Pin the voice reference

```python
ref_codes = model.encode_reference_audio(ref_wav)   # once, at episode start
# pass the identical mx.array as ref_audio_codes to all ~94 calls
```

Removes encoder nondeterminism as a drift source and saves per-chunk encode cost.
Documented in mlx-audio's `higgs_audio_v3/README.md`. **[measured]** Zero downside —
do this unconditionally.

Independently supported by the listener-adaptation literature: listeners adapt to a
*specific* synthetic voice (42% → 78% intelligibility over 8 days, +32 points at six
months), so the same reference should persist across **all episodes**, not just
within one.

### 3. Free validation checks (before any ASR)

| Check | Rule | Tag |
|---|---|---|
| Frame cap hit | `frames_generated >= max_new_frames` → fail | **[inference]** |
| Duration ceiling | `dur > 1.5 + 0.75 * words` → fail; skip if any word has ≥3 digits | **[shipping]** |
| Duration floor | `dur < words / 4.5` → fail (premature EOS) | **[inference]** |
| Tail RMS | RMS of last 10 ms > −20 dBFS → possible truncation | **[shipping]** |

Ceiling and floor are **complementary**: tail-RMS catches "stopped mid-word while
loud"; the floor catches "generated almost nothing," which ends in clean silence and
otherwise reads as a valid short ending. Ceiling 0.75 s/word ≈ an 80 wpm floor
(deliberately permissive); floor divisor 4.5 ≈ 270 wpm — tune both against our own
fastest good chunk.

---

## Phase B — chunking (~1 day)

### 4. Text normalization before synthesis

Expand numbers, dates and abbreviations; collapse ellipses; strip punctuation-only
tokens; substitute em-dash and semicolon; de-capitalize all-caps spans. Halves the
noise in the step-8 ASR comparison as a side effect.

⚠️ Our scripts are **already** written TTS-safe (numbers as words, no parentheses),
so this stage is mostly a no-op for the pipeline's output — it matters if we ever feed raw text.

### 5. Split: sentence segmentation → scored packing → 250 chars

1. Sentence-segment the text.
2. Greedy-pack sentences to **40 words / ~250 chars**.
3. Oversized sentence → scored split, window ±30 chars:

```python
score = priority * (1 - distance / (window_size * 2))
# [.!?]=1.0  [\n\n]=1.0  [:;]=0.9  [,]=0.8  brackets=0.7  dashes=0.7  ws=0.5
```

4. Merge orphans under 20 words backward; minimum sub-chunk 3 words.

The distance decay is the good idea — a comma *at* the target beats a full stop 25
chars away. Every ladder-style splitter produces lopsided chunks. **[shipping** —
Auralis**]**

**Budget rationale.** Our measured cliff is 450–500, but it is *nondeterministic*, so
leave 2× margin. Cross-tool consensus for a single call:

| tool | budget |
|---|---|
| ebook2audiobook (XTTS) | **125** (`char_limits / 2`) |
| tts-audiobook-tool (Higgs) | 40 words ≈ 220–250 |
| Coqui XTTS core / Auralis | 250 |
| Chatterbox-TTS-Extended | 300 |
| Higgs official default | no chunking at all |

Median of purpose-built audiobook tools is **250**. Expect ~94 chunks for episode 1,
not 57. **[shipping]**

⚠️ **`pysbd` does not support Czech.** Verified on the installed package: 23
languages, no `cs`. It does have **`sk`** (Slovak), which shares Czech punctuation
and abbreviation conventions closely — the pragmatic substitute, but **unvalidated on
Czech; test before trusting**.

⚠️ **Benchmark figures corrected after independent review.** The paper reports **22**
languages (listing neither Czech nor Slovak — `sk` came in a later package release)
and reports English Golden Rule Set **accuracy: pysbd 97.92% vs NLTK 56.25%**. The
"93.0 F1 vs 72.33" pair in an earlier draft does not appear in the paper. The
direction holds — pysbd beats punkt decisively on English — but the paper provides
**no evidence for Slovak-on-Czech segmentation**. **[measured, English only** —
arXiv:2010.09657**]**

### 6. Boundary hygiene

- A chunk must **never start** with whitespace or a pause marker — attach trailing
  markers to the *preceding* chunk. Caused first-word crackling and hallucination,
  worst on short paragraphs. **[measured** — ebook2audiobook #1791**]** Directly
  relevant: our scripts carry `[PAUSE n]` markers on their own lines.
- Strip or balance unmatched quote characters. Maintainer: *"Quotes are a nightmare
  for A.I. TTS creating hallucinations or strange voice behavior."* **[shipping]**
- No chunk under 3 words.

---

## Phase C — generation-time robustness (~half day)

### 7. Sampling and decoder guards

| Control | Value | Tag |
|---|---|---|
| `temperature` | **0.4** (our measurement) / 0.8 (Boson default) | **[measured]** |
| `top_k` | 50 | **[shipping]** |
| `top_p` | 0.9 | **[shipping]** |
| `seed` | explicit per attempt | — |
| **Port RAS** | `ras_win_len=7`, `ras_win_max_num_repeat=2` | **[measured** — Boson default; VALL-E 2 reports it "circumvents the infinite loop issue"**]** |
| **eoc suppression** | mask `eoc_id` in codebook-0 logits until `min_frames = int(words / 4.5 * 25)` | **[inference]** |

**On temperature:** Boson's own example uses 0.8, but we measured acronym rendering
degrading monotonically with temperature on Czech technical text:

| temp | SHA rendered as |
|---|---|
| 0.4 | `SHA-256`, `SHA-256` |
| 0.6 | `SHA-E256`, `SHA-256` |
| 0.8 | `HESAA-256`, `Shad 256` |
| 1.0 | `ŠAA256`, `SHH256` |

Use **0.4** for this material. **[measured]**

**Both patches go in `mlx_audio/tts/models/higgs_audio_v3/generation.py::step`**,
alongside the existing stop logic — `elif int(codes_n[0].item()) == eoc_id:` and
`state.eoc_countdown = n - 2`. RAS is ~20 lines.

⚠️ **Two corrections after independent review.**

1. **There is a second sampler.** `batch_generate` goes through
   `Model._step_batch_sampler` (`model.py:455`) calling `sample_batch`
   (`model.py:464`) — patching `generation.py::step` alone leaves the batch path
   unprotected. Either patch both, or explicitly forbid `batch_generate` in our
   wrapper.
2. **"Root cause" is a hypothesis, not a measurement.** RAS's *absence* is verified;
   that it *causes* our babble is not — no controlled with/without comparison was
   run. Retag as **[inference]** until we A/B it. It remains the most plausible
   explanation and the cheapest thing to test.

---

## Phase D — ASR round-trip (~1–2 days)

Already partly built: `roundtrip_compare.py` in the session scratchpad does
transcribe → normalize → WER/CER → typed diff.

### 8. Transcribe and compare

1. **ASR: `mlx-whisper large-v3-turbo`.** ⚠️ Do **not** use Parakeet TDT 0.6B **v2** —
   it is English-only, and `stt-lab/RESULTS.md` already measured it producing
   `"Ahi Proceed event three pudding infinite, dk."` on Czech. Our own bench rates
   mlx-whisper large-v3-turbo best-in-class for Czech; `parakeet-tdt-0.6b-v3`
   (multilingual, already cached) is the viable alternative. **[measured, ours]**
   Whisper decode params: `temperature=0`, `condition_on_previous_text=False`,
   `compression_ratio_threshold=2.4`, `logprob_threshold=-1.0`,
   `no_speech_threshold=0.2`.
2. **Normalize both sides identically** — casefold, strip word-boundary apostrophes,
   dashes to spaces, number normalization, punctuation to spaces. Load-bearing:
   unnormalized number/date formatting alone contributes ~0.15 WER, enough to bury
   the signal. **[measured]** ⚠️ OpenAI's `EnglishTextNormalizer` is English-only;
   Czech needs our own (our current script does NFC + casefold + punctuation strip).
3. **Align with DP edit distance over words**, emitting typed codes `d:word` /
   `i:word` / `s:src/trans`. Typed codes distinguish truncation (trailing run of
   `d:`) from babble (mass of `i:`) from mispronunciation (`s:`); a WER scalar throws
   that away.
4. **Leniency** — homophone matching, uncommon-word wildcard, whitespace-only repair
   ("firefly"/"fire fly"). Without these, proper nouns dominate the count.
5. **Fail if `num_errors > ceil(words / 10)`** ≈ 10% WER. **[shipping** —
   tts-audiobook-tool's "Moderate" default**]**

⚠️ epub2tts's `fuzz.ratio >= 88` threshold is **[folklore]** — no test, no benchmark.
Use word alignment, not a fuzzy ratio.

**Known measurement trap, seen in our own runs:** Whisper normalizes spoken numbers
to digits ("dvě stě padesát šest" → "256"), costing 3–4 word edits for correct audio.
Read **CER, not WER**, or normalize numbers on both sides before comparing.

### 9. Trim instead of retry

If a chunk fails but some source-length window of the timestamped transcript scores
**zero** errors, trim the audio to that range, cutting at local amplitude minima.
**Refuse this if tail-RMS fired** — a real truncation must not be "repaired."

Recovers the common "correct audio + garbage on one end" case for free instead of a
~30 s re-render. **[shipping** — the only implementation in the entire OSS survey;
requested upstream in Coqui repeatedly and never built**]**

---

## Phase E — retry ladder

### 10. Escalating retries, keep the best

| Attempt | Change |
|---|---|
| 1 | baseline |
| 2 | new random seed **+ `min_frames` floor enforced** |
| 3 | new seed + split the chunk at its best internal boundary, render halves |
| — | keep the best attempt **that passes validation** |
| — | if **no** attempt passes: **quarantine the chunk and block the episode** |

⚠️ **Corrected after independent review.** An earlier draft said "keep the attempt
with the lowest error count; flag, don't block" — which directly contradicts this
document's own goal of making silent failure impossible. A best-of-N winner that
still fails validation is exactly the silently-corrupt audio the plan exists to
prevent, and the cited paper credits best-of-N only when a *verified acceptable*
candidate exists (its residual hard-failure rate at N=3 is 0.038, not 0). **Never
concatenate a chunk that failed every attempt.** Fail the episode loudly; a human
decides.

Best-of-N with ASR selection drives failures 0.269 → 0.038 at N=3 → 0.000 at N≥4
**[measured** — arXiv:2606.18323**]**.

**Not recommended:** raising temperature on retry — theoretically defensible but no
surveyed pipeline does it, and it trades repetition for mispronunciation, which our
own temperature sweep shows is the dominant cost on technical Czech.
**[inference, low confidence]**

**Fallback engine** breaks voice identity mid-episode. Prefer flagging for manual
review; at N=3 expect near-zero survivors.

**Expect a nonzero steady-state failure rate.** Coqui's maintainer on repetition
*within* the character limit: *"This is not a bug… not possible to avoid
completely."* Log a per-episode failure count so an unlucky run is distinguishable
from a pathological passage or a bad voice reference.

---

## Phase F — stitching

### 11. Assemble

1. **Trim** leading/trailing silence: RMS frames at **−42 dB relative to peak frame
   RMS**, 30 ms frame / 10 ms hop, 100 ms minimum silence run, keep a **20–50 ms
   guard** for plosives. Peak-relative self-calibrates per chunk. **[shipping** —
   tts-audiobook-tool and F5-TTS converge on −42 dB**]**
2. **5–8 ms fade ramps** each end to declick. `synthesize.py` already does 8 ms —
   don't stack another.
3. **Clamped per-chunk gain match**: measure integrated LUFS per chunk, apply
   `target − measured` **clamped to ±6 dB**. Cut short-term spread 13.77 → 3.67 LU,
   versus 0.59 for *full* matching which flattens dynamics. **[measured]** Any chunk
   exceeding the clamp is a re-render candidate, not a gain problem — log it.
   MagpieTTS specifically calls out XTTS for "severe loudness inconsistency due to
   independent per-chunk gain normalization."

   **[not implemented — 2026-08-18]** Never shipped, and do not reach for it to
   balance *speakers*: this measurement is about drift within one narrator. Built
   across two voices it read a deliberate whisper as an error and boosted it by
   the full clamp, because a chunk's level confounds the speaker with the
   performance. Level between voices is declared on `Voice.gain_db` instead — see
   AGENTS.md, "Load-bearing rules". Chunk drift within one narrator remains
   unaddressed and is still a fair thing to measure on its own terms.
4. **Insert deterministic silence by boundary type** — phrase 0.3 s, sentence 0.6 s,
   paragraph 0.9 s, section 2.0 s. **[shipping]** Note `synthesize.py`'s current
   `SENTENCE_GAP = 0.12` is well below this and is flagged elsewhere as the biggest
   lever on perceived flatness.
5. **Hard concatenate in float32.**

### ⚠️ Contradiction between sources: crossfade

One source recommends **50 ms crossfades** at joins; another recommends **no
crossfade**, warning about pydub's 100 ms default.

⚠️ Corrected: the 100 ms default belongs to **`AudioSegment.append()`**, not the `+`
operator — pydub documents `sound1 + sound2` as *no* crossfade. The hazard is real
but attaches to `.append()`.

**Resolution:** a crossfade is only safe on *untrimmed* boundaries. After step 1
trims silence to ~20–50 ms of guard, a 50 ms crossfade overlaps real speech. Keep
`np.concatenate` plus the existing 8 ms declick fades — which is what `synthesize.py`
already does. If pydub is ever introduced, `append(crossfade=0)` is mandatory.

### 12. Master and encode

- One normalization pass **on the final file only**. See `pipeline-research.md` §7 for
  the mono LUFS offset — the short version is `TARGET_LUFS` should be −19 for mono.
- **Never concatenate encoded MP3s** — LAME's 576-sample delay plus frame padding
  injects 10–50 ms per join. Keep float32/WAV throughout, encode once. **[measured]**
- If the ffmpeg chain is ever used instead of `pyloudnorm`: **pin `-ar 24000` after
  `loudnorm`**, which otherwise silently resamples to 192 kHz. **[measured]**

---

## Optional / evaluate by ear

### 13. Rolling audio context — probably decline

`prompt.py`'s `build_prompt(text, *, references: Iterable[ReferenceCodes])` accepts a
list, so `[fixed_voice_ref, prev_chunk_ref]` needs no code change. Policy if enabled:
history depth 2–3, reset at paragraph/section boundaries, **clear the chain on any
failure** (a bad chunk must never become context), batch size 1.

**Recommendation: don't.** The field report says it "tends to **flatten** output in
all cases" — the *xerox effect*. Prosodic variation is the best-evidenced lever
against listener fatigue over 25 minutes (see `pipeline-research.md` §9), so flattening
is precisely the wrong trade for this content.

### 14. Drift diagnostic

Cosine similarity of speaker embeddings vs the reference, plotted against chunk index.
**Look for a downward trend, not an absolute threshold** — two genuine recordings of
the same speaker score only ~0.68 median. Diagnostic, not a gate. **[measured]**

---

## Do-not-copy list

**epub2tts.** Its Whisper gate is disabled by default for most engines
(`config['minratio'] = 0` forced for openai/edge/kokoro/vits; runs only for XTTS), and
on failure **the bad audio is kept in the concat list anyway** — it detects failures
it then ignores. `sentance_chunk_length` and a second `retries = 2` are computed and
never read. Its Pedalboard chain contains `Compressor(threshold_db=12)` — positive
dBFS, can never engage — and `Gain(gain_db=0)`.

**The motivating anecdote for this whole plan**: the abogen maintainer, on silent
truncation in his own audiobook tool — *"when I tried, it gave me a 22-second output.
Since I don't know the Chinese alphabet or the language, I couldn't tell if it was
corrupt or not."* Issue still open ~8 months later, unconfirmed. An ASR round-trip
answers that in two seconds, and exactly one tool in the entire survey ships one.

---

## Open questions

- Whether ebook2audiobook PR #2056's duration heuristic is tuned or guessed.
- Kokoro's internal chunking below `split_pattern` (not relevant unless we add it).
- Whether `pysbd` with `language="sk"` actually segments Czech well — **untested**.
- The 1.6 frame-cap headroom multiplier is unvalidated; measure against our own
  longest legitimate chunk before trusting it.
