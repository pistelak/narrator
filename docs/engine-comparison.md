# Higgs Audio v3 TTS vs Supertonic 3 — Czech + embedded English

Addendum to `RESULTS.md`. Run 2026-08-11 on **M5 Pro / 64 GB** (the original
benchmark assumed "M1 or newer" and sized every candidate for that; this run
deliberately drops the footprint constraint).

- Engine: `bosonai/higgs-audio-v3-tts-4b` via `mlx-audio` 0.4.8, in `.venv-higgs`
  (isolated — the main `.venv` that the previous pipeline depends on is untouched).
- Baseline: Supertonic 3, voice `M1`, `total_steps 10`, `speed 0.9` — the exact
  settings `synthesize.py` renders episodes with.
- Intelligibility: `mlx-whisper large-v3-turbo` round-trip, the model `stt-lab`
  rates best-in-class for Czech. Script: `roundtrip_compare.py`.

## 1. Cost

| | Higgs v3 | Supertonic 3 |
|---|---:|---:|
| Params | 4B | 99M |
| Disk | 8.7 GB | 404 MB |
| Peak RAM | 12.3 GB | ~0.3 GB |
| Load | 2.6 s | ~1 s |
| RTF | **0.75** | ~0.02 |

Higgs is ~40× slower but still faster than realtime: a 25-minute episode costs
~19 minutes of compute. On this machine that is not a constraint.

Note: `mlx-audio` alone is not enough — the Higgs codec loader reads safetensors
with `framework="pt"`, so **torch is a hard dependency** despite the MLX path.

## 2. Intelligibility (WER / CER %, lower better)

| Input | Higgs | Supertonic |
|---|---:|---:|
| cs_command | 7.1 / 6.2 | 7.1 / 6.2 |
| cs_paragraph | **3.9 / 0.9** | 9.8 / 1.7 |
| en_only | **9.1** / 7.3 | 27.3 / 7.3 |
| mixed_brand | 25.0 / 12.1 | 25.0 / **8.6** |
| mixed_casual | **5.6 / 2.9** | 16.7 / 7.2 |
| mixed_dev | 17.6 / **3.8** | **11.8** / 5.0 |
| mixed_acronyms | **25.8 / ~20** † | 29.0 / 22.6 |
| mixed_encoding | **17.9 / 12.7** | 25.0 / 17.6 |
| mixed_names | 6.2 / 1.2 | 6.2 / 1.2 |

† at `temperature 0.4`; see §4.

**Read CER, not WER.** Much of the WER is Whisper normalization, not synthesis
error — it writes spoken "šest" as `6`, "dvě stě padesát šest" as `256`,
"padesát osm" as `58`. Those cost 1–4 word edits each and say nothing about audio.

Excluding the acronym row, mean CER is **5.9 % Higgs vs 6.9 % Supertonic** —
a real but modest edge.

## 3. Where they actually differ

**Higgs wins on embedded English.** `mixed_encoding`: Supertonic produced
`koudování base 64Hack` and `20-bytový haš` — it mispronounced a plain Czech
word (*kódování*) and turned *check* into *hack*. Higgs produced
`kódování base 64 check` and `hash` correctly. Same pattern on `mixed_casual`
(Supertonic 16.7 % vs Higgs 5.6 %).

**Declined foreign names are a tie, not a discriminator.** *"od Kalleho
Lindkvista"* — an English surname carrying a Czech genitive — came back
verbatim from **both** engines (6.2 % WER, identical). This was predicted to be
the deciding test and it wasn't; it is consistent with the note already in
the earlier listening notes, which record that Kalle/Lindkvist reads fine
raw in Czech (unlike in English, where Supertonic collapses *Kalle* → "Cal").

**Rare acronyms defeat both.** Each engine mangled them several distinct ways,
and never the same way twice. Neither is reliable; this still
needs a lexicon entry whichever engine is used.

## 4. Temperature matters more than expected

The README's default `temperature=1.0` is too high. One run at 1.0 **silently
dropped an entire clause** ("který ho zmáčkne zhruba na polovinu"). A sweep of
4 temperatures × 2 reps showed the clause surviving in all 8, so that was a
stochastic failure, not a systematic one — but acronym rendering degrades
monotonically with temperature:

| temp | SHA rendered as |
|---|---|
| 0.4 | `SHA-256`, `SHA-256` |
| 0.6 | `SHA-E256`, `SHA-256` |
| 0.8 | `HESAA-256`, `Shad 256` |
| 1.0 | `ŠAA256`, `SHH256` |

**Use `temperature 0.4`** for this material.

## 5. The blocker: long-form collapse

Duration should scale linearly with input length (~0.077 s/char for Czech).
Two reps per length, `temperature 0.4`:

| chars | audio | expected | ratio |
|---:|---:|---:|---:|
| 150 | 14.3 / 13.9 s | 11.6 s | 1.24 / 1.21 |
| 263 | 20.5 / 23.9 s | 20.3 s | 1.01 / 1.18 |
| 334 | 27.6 / 28.4 s | 25.7 s | 1.07 / 1.11 |
| 459 | 31.9 / 33.2 s | 35.3 s | 0.90 / 0.94 |
| **793** | **244.0 / 31.9 s** | 61.1 s | **4.00 / 0.52** |

At 793 characters one rep truncated to half the content and the other
degenerated into **four minutes of babble** (Whisper transcribes it as a
repeated `www.hradeckralove.org`, its signature output for non-speech). At
1546 chars the collapse was total.

**Reliable ceiling: ~450 characters.** Above roughly 500 the model is
non-deterministically unusable, and the failure is silent — plausible-length
audio containing the wrong thing.

This kills the main architectural argument for adopting Higgs. The hope was
that a long-context model would let `synthesize.py` drop per-sentence synthesis
and the manual `SENTENCE_GAP` stitching. It cannot: Higgs is *worse* at
long-form than Supertonic, which is stable because it never attempts more than
one sentence. The existing per-sentence architecture is correct and Higgs would
have to slot into it unchanged.

## 6. Verdict

Higgs is modestly more intelligible and clearly better at Czech sentences with
English terms in them — the exact failure mode the pronunciation lexicon exists to
patch. It costs 8.7 GB, 12.3 GB RAM, 40× the compute, a torch dependency, and
**demands hard chunking under 450 characters with per-chunk validation**,
because its failure mode is silent content loss rather than mispronunciation.

**ADOPTED — gate passed 2026-08-13.** The listener heard a full ~20-minute render
end to end and reported: *"honestly, the sound/audio is really good, I can imagine
listening to this."* That is the Step 0 gate in the project's strategy doc, cleared
on a real episode rather than a 13-second sample: no audible voice drift across ~100
independently-generated chunks, no fatigue verdict against it, joins acceptable.

**Adopted, pending the guardrails.** The open question was naturalness, which a
round-trip cannot measure. Listening pass done 2026-08-11 on
`mixed_encoding__higgs.wav`: clearly better than Supertonic by ear. Since the
measured intelligibility gap was only ~1 CER point, naturalness is what decides it,
and it decides for Higgs.

Adoption requires the chunking, validation and retry work in `LONGFORM-PLAN.md` —
without it the silent-failure mode makes long-form renders untrustworthy.
