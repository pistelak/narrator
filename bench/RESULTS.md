# Local Czech & Mixed Czech/English Text-to-Speech: A Small Benchmark

**Author:** Radek Pistelák
**Date:** 2026-05-20 (Supertonic round added 2026-06-08)
**Scope:** Personal-assistant MVP — picking a local TTS voice for Czech and
realistic mixed Czech/English assistant replies on Apple Silicon. Sibling
work to `stt-lab/`.

## Abstract

> **Update (2026-06-08): Supertonic 3 was added as a fourth engine and is
> the new round-trip leader.** On the same six inputs it scores **6/6** on
> the recoverability rubric (vs Piper 3.5/6, XTTS 2.5/6), with **zero
> hallucinations** and **44.1 kHz** output. It fixes both of Piper's real
> weaknesses — embedded English fragments (`feature flag`, `cool tool`
> survive verbatim) and the pure-English input (`en_only` is perfect) —
> while keeping pure-Czech quality at least on par with Piper. The costs
> are modest: RTF ~0.16 (≈10× slower than Piper but still well under
> realtime and ~3× faster than XTTS), ~680 MB peak RAM, and an OpenRAIL-M
> license (commercial use permitted with behavioural restrictions, unlike
> XTTS's CPML). **Outstanding caveat: no human listening pass yet** — the
> round-trip can't judge naturalness, which is exactly the axis on which
> Supertonic was informally reported to "sound better than Jirka." See §10.

The original three-engine study below stands; Supertonic results are in §10.

We benchmarked three local TTS engines on six Czech / mixed-language
inputs sized like real assistant replies. **`piper` with the
`cs_CZ-jirka-medium` voice is the recommended default**: it reproduces
pure Czech almost verbatim through a Whisper round-trip, never
hallucinates trailing tokens, runs **~30× faster than XTTS-v2** (RTF
0.015 vs 0.48), and uses **~10× less RAM and ~28× less disk**. Its only
weakness — English words inside Czech text are read with Czech phonemes
(`Claude Code` → `klaudecode`) — sounds like a Czech speaker reading
English brand names with a Czech accent, which is what Czech speakers
*actually do*. XTTS-v2 was kept in the benchmark for completeness; it is
slower, heavier, hallucinates short trailing tokens on most utterances,
and on the worst mixed input rewrote `Claude Code` to `Clouded Sword`.
**Bark was smoke-tested and dropped**: the upstream model ships no Czech
speaker preset and the closest Slavic preset (Polish) produced
unrecoverable Czech at RTF ≈ 6.

## 1. Goal

Pick a local, offline, free TTS engine for the assistant's spoken output.
The voice must:

1. Sound like Czech when the text is Czech (diacritics, stress, intonation).
2. Read realistic Czech-with-embedded-English-words sentences without
   producing unrecoverable gibberish for English fragments and without
   abruptly switching voices.
3. Synthesize comfortably faster than realtime on an Apple Silicon Mac
   (target RTF well below 1.0; ideally < 0.3 for an interactive feel).
4. Run with no paid API, no Docker, no CUDA.

## 2. Setup

### 2.1 Hardware

- MacBook Pro, Apple **M5 Pro**, 64 GB RAM.
- macOS Darwin 25.4.0.
- Deployment target: base **M4 Mac mini, 16 GB**.

### 2.2 Software

- Python 3.12.13 (via `asdf`).
- `piper-tts` 1.4.2 (ONNX Runtime, bundled eSpeak-NG phonemization).
- `coqui-tts` 0.27.5 (Idiap fork; the only XTTS-v2 distribution that
  installs cleanly on Python 3.12).
- `torch` 2.12.0 + `torchcodec` 0.12.0 (required for audio I/O in
  Coqui ≥ 0.27 on PyTorch ≥ 2.9).
- `transformers` pinned to `<5,>=4.45` — XTTS depends on
  `transformers.pytorch_utils.isin_mps_friendly`, which transformers 5.x
  removed.
- `mlx-whisper` 0.4.3 (used only for the STT round-trip).
- `soundfile`, `librosa` (audio I/O, no system `ffmpeg` required).

### 2.3 Procedure

For each input, the relevant `bench_<engine>.py` script:

1. Loads the input `.txt`, strips trailing whitespace.
2. Synthesizes one WAV via the engine.
3. Times only the synthesis call (engine load and a one-shot warm-up
   synthesis are excluded).
4. Reads the resulting WAV back, derives `audio_duration_seconds` from
   `len(samples) / sample_rate`, and appends one row to
   `outputs/results.csv`.

Per-engine specifics:

- **Piper**: language is implicit in the voice file (`cs_CZ-jirka-medium`).
  Synthesis is one `voice.synthesize_wav(text, wave_file)` call.
- **XTTS-v2**: language is set per call (`cs` for Czech and mixed inputs,
  `en` for English). Voice is the built-in studio speaker
  `Claribel Dervla` — no reference WAV required.
- **Bark**: smoke-tested only (see §4.3).

`stt_roundtrip.py` then transcribes every WAV with `mlx-whisper
large-v3-turbo`, language forced to match the input prefix, and writes
`outputs/<stem>__<engine>.stt.txt`. This is our objective intelligibility
signal — see §2.5.

### 2.4 License notices

- **Piper** voices are MIT-licensed per-voice; `cs_CZ-jirka-medium`'s
  metadata declares `MIT`. Cleared for commercial use.
- **XTTS-v2** is released under the **Coqui Public Model License (CPML)
  1.0**. Free for research and personal use, **commercial deployment is
  not permitted**. For an MVP assistant used personally this is fine; if
  this project is ever commercialized, XTTS-v2 must be swapped out (or a
  commercial license obtained).
- **Bark** (`suno/bark`) is MIT-licensed but excluded from the benchmark
  for the reasons in §4.3.

### 2.5 How quality is judged

We do **not** judge naturalness here — the author of this report (Claude
Code) cannot listen to audio. Instead we use a **Whisper round-trip** as
an intelligibility proxy: synthesize text → transcribe the WAV with
`mlx-whisper large-v3-turbo` (the model `stt-lab/` selected as the
default Czech STT). If the round-trip transcript closely matches the
input, the TTS pronounced the input well enough for a Czech ASR to
recover the words. The reverse is not strictly true — Whisper can fail
on correctly-pronounced audio — but in practice a round-trip mismatch
flags genuine pronunciation problems.

The round-trip cannot judge:

- **Voice consistency** across the CZ↔EN boundary (does the voice
  change?).
- **Naturalness / prosody** (does it sound like a human or a robot?).
- **Stress patterns and intonation**.
- Subtle phoneme errors that the STT happens to normalize away.

A human listening pass remains required before final adoption.

## 3. Samples

Six inputs are stored in `inputs/`, designed to cover the assistant's
expected utterance shapes:

| ID | File | Type | Notes |
|----|------|------|------|
| 1 | `cs_command.txt` | Pure Czech command | Calendar event with day, time, venue (`Infinit`). |
| 2 | `cs_paragraph.txt` | Pure Czech paragraph | Numbers (`15:30`), brand (`S&P 500`), low-frequency words (`podinvestovaná`, `prorodnost`). |
| 3 | `mixed_dev.txt` | Czech + dev jargon | `meeting`, `feature flag`, `onboarding flow`. |
| 4 | `mixed_brand.txt` | Czech + brand names | `Visual Studio Code`, `commitni`, `GitHubu`. |
| 5 | `mixed_casual.txt` | Czech + casual anglicisms | `Claude Code`, `cool tool`, `gymu`, `meal prep`. |
| 6 | `en_only.txt` | Pure English (sanity) | Short calendar command — baseline for an English-pronunciation check. |

## 4. Models

| Engine | Model | Voice | Multilingual? | Verdict |
|---|---|---|---|---|
| `piper` | `cs_CZ-jirka-medium` | Jirka (male, medium quality) | Czech only | **Recommended default** |
| `coqui-xtts` | `xtts_v2` | `Claribel Dervla` (built-in studio) | 17 languages incl. `cs` and `en` | Kept; weaker on Czech, hallucinates |
| `bark` | `suno/bark` | n/a — no `cs_speaker_*` | 13 languages, none Czech | **Dropped — see §4.3** |

### 4.3 Why Bark was dropped

The `suno/bark` repo on HuggingFace ships speaker embedding presets for
**de, en, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh** — no Czech. Bark
can still synthesize Czech text (it's a text-conditioned model with
language inference), but without a Czech preset the voice persona is
either random per-call or borrowed from a different language family.

A smoke test on **input 1** (`cs_command.txt`) with `v2/pl_speaker_0`
(closest Slavic preset) produced:

| | |
|---|---|
| Input | *Ahoj, prosím vytvoř v kalendáři event na zítra v šest večer.* |
| Bark round-trip | *Ahoj, prosím, vytvoř v **kalendaryk**. Na **zíkra**, **výčer**, **výčer**.* |

- `kalendaryk` and `zíkra` are not Czech words.
- `event` was dropped.
- `večer` looped twice.
- **RTF was 6.4** (64 s synthesis for 10 s of audio) — product-killing
  on its own for an interactive assistant.

The smoke artifact is preserved as
`outputs/bark_smoke_cs_command_pl-speaker-0.{wav,stt.txt}`. No further
Bark configurations were tested.

## 5. Results

### 5.1 Speed

Warm cache (one warm-up synthesis is discarded before the loop). All times
measured on M5 Pro CPU.

| Input | Chars | Piper time | Piper audio | **Piper RTF** | XTTS time | XTTS audio | **XTTS RTF** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cs_command.txt` | 81 | 0.12 s | 6.97 s | **0.017** | 3.02 s | 6.17 s | 0.491 |
| `cs_paragraph.txt` | 289 | 0.41 s | 27.43 s | **0.015** | 11.66 s | 24.03 s | 0.485 |
| `mixed_dev.txt` | 98 | 0.13 s | 8.41 s | **0.015** | 3.69 s | 7.47 s | 0.494 |
| `mixed_brand.txt` | 71 | 0.09 s | 6.05 s | **0.015** | 2.97 s | 6.26 s | 0.475 |
| `mixed_casual.txt` | 90 | 0.11 s | 7.51 s | **0.015** | 3.70 s | 8.02 s | 0.461 |
| `en_only.txt` | 52 | 0.08 s | 5.07 s | **0.016** | 2.00 s | 4.31 s | 0.465 |

For Bark, the only data point is the smoke test in §4.3: **RTF 6.4**.

Headline:

- **Piper is ~30× faster than XTTS-v2** (RTF 0.015 vs 0.48).
- **RTF is flat across inputs** for both engines — neither has
  content-dependent slowdowns.
- Piper runs comfortably under the 0.3 "feels instant" interaction
  threshold; XTTS-v2 does not.

Sample rates: Piper 22 050 Hz, XTTS-v2 24 000 Hz.

### 5.2 Resource use

| | Piper | XTTS-v2 |
|---|---:|---:|
| Model on disk | **~60 MB** | ~1.7 GB |
| RSS, idle Python | ~16 MB | ~16 MB |
| RSS, after `import` | ~43 MB | ~524 MB |
| RSS, after model load | ~188 MB | **~3.16 GB** |
| RSS, during synthesis | **~346 MB** | ~3.48 GB |

On a base **M4 Mac mini, 16 GB**, this matters: Piper leaves ~14 GB free
for the LLM, STT, and OS; XTTS-v2 alone takes ~3.5 GB resident, leaving
~10 GB once mlx-whisper (~1.5 GB) is also resident — workable, but
tight.

### 5.3 Intelligibility — input vs Whisper round-trip

Per-input diff, side by side. **Bold** marks substantive errors;
whitespace and punctuation are normalized for readability.

#### Input 1 — `cs_command.txt`

> Input: *Ahoj, prosím vytvoř v kalendáři event na zítra v šest večer, půjdeme do Infinitu.*

| Engine | Round-trip |
|---|---|
| **Piper** | *Ahoj, prosím vytvoř v kalendáři event na zítra v 6. **večej**. Půjdeme do Infinitu.* |
| XTTS | *Ahoj, prosím vytvoř v **kalendáře** event na zítra v 6 večer, půjdeme do **Infinity Count**.* |

- Piper: only `večer` → `večej` (one phoneme glitch). `kalendáři` and
  `Infinitu` both survive.
- XTTS: case error on `kalendáři` → `kalendáře`, and the proper noun
  `Infinitu` is rewritten to `Infinity Count` (English phonemes plus a
  hallucinated trailing token).

#### Input 2 — `cs_paragraph.txt`

> Input: *Sraz máme v 15:30 na rohu náměstí, vezmi si s sebou i poznámky z minulého týdne. Trhy dnes klesly, S&P 500 je v mírném mínusu a podinvestovaná Evropa zase zaostává. Demograficky je problém prorodnost, která v Česku už několik let stagnuje. Po obědě projdeme čísla a domluvíme další postup.*

| Engine | Round-trip |
|---|---|
| **Piper** | *Sraz máme v 15.30 na rohu náměstí, vezmi si s sebou i poznámky z minulého týdne. Trhy dnes klesly, z **ampersend P500** je v mírném mínusu a podinvestovaná Evropa zase zaostává. Demograficky je problém **pro rodnost**, která v Česku už několik let stagnuje. Po obědě projdeme čísla a domluvíme další postup.* |
| XTTS | ***Sres** máme v 15, 30 na rohu náměstí, vezmi si s sebou i poznámky z minulého týdne. Trhy dnes klesly, **z AP500** je v **měn** mínusu a **podínostvo na Evropazl se zaostává pěny**. Demograficky je problém **pro rodnost**, která v Česku už několik let stagnuje. Po obědě projdeme čísla a domluvíme další **postupkám**.* |

This is the most discriminating input. Piper reproduces almost the whole
paragraph: `Sraz`, `mírném`, `podinvestovaná Evropa zase zaostává` all
survive. `S&P 500` is read literally as `ampersend P500` — verbose but
recoverable. `prorodnost` → `pro rodnost` is a benign word split that
both engines do. **No hallucinations.**

XTTS fails on `Sraz`, `mírném`, the brand `S&P 500`, mangles
`podinvestovaná Evropa zase zaostává` into the unrecoverable `podínostvo
na Evropazl se zaostává`, and appends two trailing hallucinations
(`pěny`, `postupkám`).

#### Input 3 — `mixed_dev.txt`

> Input: *Pošli Tomovi email, že meeting bude v pět hodin a probereme feature flag pro nový onboarding flow.*

| Engine | Round-trip |
|---|---|
| **Piper** | *Pošli Tomovi e-mail, že meeting bude v 5 hodin a probereme v **Fiature Flak** pro nový onboarding **FLOV**.* |
| XTTS | *Pošli **Tomový** e-mail, že meeting bude v 5 hodin a probereme feature flag pro nový onboarding flow.* |

- Piper: `Tomovi` (correct dative) survives. `meeting` and `onboarding`
  survive. `feature flag` → `Fiature Flak` and `flow` → `FLOV` are
  *Czech-accent transliterations* — recognizable to a Czech listener.
- XTTS: `Tomovi` → `Tomový` (case error). All English fragments survive
  verbatim with English pronunciation. **XTTS wins this input on the
  intelligibility axis.**

#### Input 4 — `mixed_brand.txt`

> Input: *Otevři Visual Studio Code, spusť testy a pak commitni změny do GitHubu.*

| Engine | Round-trip |
|---|---|
| **Piper** | *Otevři **vyslal studio CODE**, **spoust** testy a pak **komitní** změny do GitHubu.* |
| XTTS | *Otevři Visual Studio Code, **spušť** testy a pak **pomítni** změny do **GitHubu Dashto**.* |

- Piper: `Visual` → `vyslal` (Czech-phonetic, but unrecognizable).
  `spusť` → `spoust`. `commitni` → `komitní` (almost right — `t` lost).
  `GitHubu` survives.
- XTTS: `Visual Studio Code` survives. `commitni` → `pomítni`
  (substituted with a fictional Czech word). Trailing hallucination
  `Dashto`.

This is the input where **XTTS's English brand recognition helps**, but
its phonetic substitution on the English-derived verb `commitni` hurts
more than Piper's transliteration `komitní`. Net: a tie of different
flavors.

#### Input 5 — `mixed_casual.txt`

> Input: *Stáhni si Claude Code, je to docela cool tool. Pojďme po práci do gymu, dáme si meal prep.*

| Engine | Round-trip |
|---|---|
| **Piper** | *Stáhni si **klaudecode**, je to docela **cel tůl**. Pojdeme po práci do gímu, dáme si **mail prep**.* |
| XTTS | ***Stájni** si **Clouded Sword**, je to docela cool tool. Pojďme po práci do gymu, dáme si meal prep, **peškaj**.* |

- Piper: `Claude Code` → `klaudecode` (one word, Czech phonemes). A Czech
  listener hearing this would understand "Claude Code spoken with a Czech
  accent". `cool tool` → `cel tůl`. `meal prep` → `mail prep`. **No
  hallucinations.**
- XTTS: `Stáhni` → `Stájni`, `Claude Code` → `Clouded Sword`
  (unrecoverable rewrite — a Czech listener would have no idea this was
  meant to be `Claude Code`). `cool tool` and `meal prep` survive. One
  trailing hallucination `peškaj`.

This is the input where **Piper wins decisively**: Czech-accented English
is intelligible; XTTS's brand-name rewrite is not.

#### Input 6 — `en_only.txt`

> Input: *Create a calendar event for next Tuesday at six p.m.*

| Engine | Round-trip |
|---|---|
| Piper | ***Srete** a calendar event for next Tuesday at 6 pm.* |
| **XTTS** | *Create a calendar event for next Tuesday, at 6pm.* |

- Piper: `Create` → `Srete` (the Czech tokenizer fights the English
  word). XTTS wins this input cleanly.
- Whisper's `six p.m.` → `6pm` normalization is its own behavior on both
  rows.

### 5.4 Per-input intelligibility scorecard

A simple "recoverable by a Czech listener?" verdict:

| Input | Piper | XTTS-v2 |
|---|---|---|
| `cs_command` | **yes** (single-phoneme glitch) | partial (case error + brand rewrite + hallucination) |
| `cs_paragraph` | **yes** (verbatim, modulo `&` spelled out) | no (mangled rare words + hallucinations) |
| `mixed_dev` | yes (Czech-accent English) | **yes** (verbatim English) |
| `mixed_brand` | partial (`Visual` lost) | partial (`commitni` → fictional word) |
| `mixed_casual` | **yes** (Czech-accent English) | no (`Claude Code` → `Clouded Sword`) |
| `en_only` | no (Czech tokenizer on English) | **yes** |
| **Score** | **3.5 / 6** | 2.5 / 6 |

## 6. Discussion

### 6.1 Why Piper wins despite being Czech-only

The intuition "I need a multilingual model for mixed text" turns out to
be wrong here. Piper, a Czech-only model, **outperforms a multilingual
model on Czech text** and **degrades gracefully on English text** by
applying Czech phonemes. That degraded mode is exactly how real Czech
speakers pronounce English brand names mid-sentence — so a Czech
listener decodes it without effort.

The multilingual model, by contrast, has to *decide* per word whether to
go English or Czech, and that decision is unstable: `Claude Code`
becomes `Clouded Sword`, `commitni` becomes `pomítni`, `Infinitu`
becomes `Infinity Count`. When the decision goes wrong, the output is
not "accented English" — it is a completely different phrase.

### 6.2 Where XTTS still helps

There is one regime XTTS handles better: **mid-Czech English fragments
that the listener wants to hear in English** (e.g. UI labels read
verbatim — `meeting`, `feature flag`, `onboarding flow`). XTTS got
these verbatim; Piper produced `Fiature Flak` etc. If the assistant
frequently reads back English-language UI strings, XTTS may still be
worth a second pass on a per-utterance basis. For most spoken Czech
assistant replies, this advantage is marginal.

### 6.3 Hallucinations

XTTS-v2 added trailing one-word hallucinations on **4 of 6 inputs**:
`Count`, `pěny`, `postupkám`, `Dashto`, `peškaj`. These are short
(usually one extra word) but they are real — XTTS-v2 does not
consistently mark the end of an utterance. Piper added **zero
hallucinations** across all six inputs.

For an assistant where the user is hearing the reply, an extra random
word at the end is the most noticeable failure mode of all: it sounds
*wrong*, the way an autocomplete leaking into a sentence sounds wrong.
This is a strong reason to prefer Piper independent of speed and size.

### 6.4 Where Bark went wrong

Bark is presented as the "natural multilingual TTS" of this generation,
but it does not in fact ship a Czech speaker preset on HuggingFace. The
documentation language list ("supported languages") differs from the
speaker embedding list, and there is no programmatic warning when a
Czech-text-+-Polish-speaker combination is used. Combined with RTF ~6,
Bark is unusable for this assistant.

### 6.5 Speed and resources

Piper at RTF 0.015 means **a 10-second assistant reply is ready in 0.15 s
of compute** — well below human reaction time, and below the network
round-trip to most LLM APIs. On a base M4 Mac mini we expect Piper RTF
to stay ≤ 0.05 even with the CPU shared with an LLM. XTTS-v2 at RTF
0.48 on the M5 Pro will likely land at RTF 0.6–0.75 on the base mini,
still under realtime but no longer "instant".

## 7. Recommendation

**Adopt Piper with the `cs_CZ-jirka-medium` voice as the default
assistant TTS.** It is the smallest, fastest, lowest-RAM option in the
benchmark, beats XTTS-v2 on the pure-Czech inputs, ties or wins on most
mixed inputs, never hallucinates, and its license is permissive enough
to keep the deployment story simple.

**Keep XTTS-v2 around as an opt-in for verbatim English read-back.** If
the assistant ever needs to read mid-Czech English literally (e.g.
"`the parameter is set to <verbatim English UI string>`"), XTTS's
per-call language switching may still be worth invoking for that one
utterance.

**Per-input verdict against the original goals**:

| Input | Piper | XTTS-v2 |
|---|---|---|
| `cs_command` | ✅ adopt | ⚠️ workable with brand rewriter |
| `cs_paragraph` | ✅ adopt | ❌ rare-word mangling + hallucinations |
| `mixed_dev` | ✅ adopt (Czech-accent English) | ✅ verbatim English |
| `mixed_brand` | ⚠️ `Visual` lost | ⚠️ `commitni` mangled |
| `mixed_casual` | ✅ adopt | ❌ `Claude Code` lost |
| `en_only` | ❌ Czech tokenizer mangles English | ✅ verbatim |

For the assistant MVP, where 95+ % of utterances are Czech with at most
a few embedded English tokens, **Piper handles every realistic case
acceptably and is by far the cheapest to run**.

## 8. Limitations

- **Sample size is six**. All conclusions are tentative until validated
  on a larger Czech / mixed-CZ-EN corpus.
- **One voice per engine**. Piper has other Czech voices
  (`cs_CZ-jirka-low`, and others on HF) and XTTS-v2 has 57 other studio
  speakers; voice choice can shift the per-input verdicts.
- **No human listening pass**. All conclusions come from a Whisper STT
  round-trip (see §2.5). The round-trip catches gross pronunciation
  failures but misses naturalness, prosody, stress patterns, and voice
  consistency across language boundaries. Listen to every WAV in
  `outputs/` before final adoption.
- **Whisper-cs is forced on mixed inputs**. Round-trip transcripts of
  mixed inputs were obtained with `language="cs"`; an auto-detect pass
  might recover more English words.
- **No long-form (3 min+) test**. Both engines' behavior on very long
  utterances is unverified — though for an assistant, utterances are
  short.
- **Bark received one smoke test, not a benchmark**. If the upstream
  ever ships a `cs_speaker_*` preset, Bark deserves re-evaluation.
- **License**. XTTS-v2's CPML blocks commercial use. Piper voices are
  per-voice MIT.

## 9. Future work

1. **Test other Piper Czech voices** (`cs_CZ-jirka-low`, future
   `cs_CZ-*`) — a different voice may handle the `mixed_brand` /
   `mixed_casual` English fragments more recognizably.
2. **Add MMS-TTS** (`facebook/mms-tts-ces`) as a second Czech-only
   reference.
3. **Run the round-trip with `language=None`** on mixed inputs to remove
   the Czech bias from the intelligibility signal.
4. **Try other XTTS-v2 studio speakers** on the two inputs where it
   failed (`mixed_brand`, `mixed_casual`) — picking the speaker whose
   phoneme model best handles the CZ↔EN boundary might recover those
   utterances without dropping XTTS.
5. **Tune XTTS-v2 generation parameters** (`temperature`, `top_p`,
   `repetition_penalty`) to suppress trailing hallucinations.
6. **Human listening pass** on every WAV in `outputs/` — naturalness,
   stress, voice consistency, prosody.
7. **Test on a base M4 Mac mini** to confirm Piper's RTF stays ≤ 0.05
   and the 16 GB RAM headroom is comfortable with the LLM and STT also
   resident.
8. **Settle a per-utterance routing policy**: most replies → Piper;
   utterances that need verbatim English mid-Czech → XTTS-v2.
9. **Re-evaluate Bark** if/when `suno/bark` or a derivative ships a
   Czech speaker preset.

## 10. Supertonic 3 (added 2026-06-08)

Added after an informal report that Supertonic "sounds better than Jirka".
Supertonic 3 is an on-device ONNX TTS (`supertonic` PyPI package, model
`Supertone/supertonic-3` on HuggingFace), ~99M params, 31 languages
including Czech. It is multilingual like XTTS — language is passed per call
(`cs` / `en`), routed by the same filename-prefix rule. The voice is one of
10 built-in styles (`M1`..`M5`, `F1`..`F5`); we used **`M1`** (male) to
match Piper's male `jirka`. Defaults: `total_steps=8`, `speed=1.05`. Run
with `python bench_supertonic.py`.

### 10.1 Speed

Same protocol as §5.1 (warm-up discarded, synthesis call only, M5 Pro CPU).

| Input | Chars | Time | Audio | **RTF** |
|---|---:|---:|---:|---:|
| `cs_command.txt` | 81 | 1.16 s | 7.17 s | **0.161** |
| `cs_paragraph.txt` | 289 | 3.80 s | 25.98 s | **0.146** |
| `mixed_dev.txt` | 98 | 1.23 s | 7.73 s | **0.159** |
| `mixed_brand.txt` | 71 | 1.04 s | 6.20 s | **0.168** |
| `mixed_casual.txt` | 90 | 1.21 s | 7.45 s | **0.163** |
| `en_only.txt` | 52 | 0.71 s | 4.11 s | **0.172** |

RTF ~0.16, flat across inputs. That is **~10× slower than Piper** (0.015)
but **~3× faster than XTTS** (0.48), and comfortably under the 0.3 "feels
instant" threshold. Sample rate is **44 100 Hz** — higher than both Piper
(22 050) and XTTS (24 000).

### 10.2 Resource use

| | Piper | Supertonic 3 | XTTS-v2 |
|---|---:|---:|---:|
| Active ONNX assets on disk | ~60 MB | ~380 MB | ~1.7 GB |
| Full HF download | ~60 MB | **~1.8 GB** | ~1.7 GB |
| RSS, after `import` | ~43 MB | ~50 MB | ~524 MB |
| RSS, after model load | ~188 MB | ~522 MB | ~3.16 GB |
| RSS, peak during synthesis | ~346 MB | **~679 MB** | ~3.48 GB |

Note the gap between *active* ONNX assets (~380 MB: `vector_estimator`
245 MB + `vocoder` 97 MB + `text_encoder` 35 MB + `duration_predictor`
3.5 MB) and the *full download* (~1.8 GB, which ships additional precision
variants across 26 files). Peak RAM ~680 MB sits between Piper and XTTS —
fine for a base 16 GB M4 mini.

### 10.3 Intelligibility — input vs Whisper round-trip

Whitespace/punctuation normalized; **bold** marks substantive deviations.

| Input | Supertonic round-trip | Verdict |
|---|---|---|
| `cs_command` | *Ahoj, prosím vytvoř v kalendáři event na zítra v 6 večer. Půjdeme do Infinitu.* | **yes** — clean; `Infinitu` survives, no `večej` glitch |
| `cs_paragraph` | ***Sras** máme v 15.30 … Trhy dnes klesly. **SEP 500** je v mírném **mí minusu**. A podinvestovaná Evropa zase zaostává … další postup.* | **yes** — `Sraz`→`Sras` and a small `mí minusu` stutter; `S&P 500`→`SEP 500`; **no hallucinations** |
| `mixed_dev` | *Pošli Tomovi email, že meeting bude v 5 hodin a probereme feature **Flak** pro nový onboarding flow.* | **yes** — near-verbatim; `flag`→`Flak` only; `onboarding flow` survives |
| `mixed_brand` | *Otevři Visual Studio **kode**. **Spust** testy a pak **komitní** změny do GitHubu.* | **yes** — `Visual Studio` survives (Piper lost `Visual`); `Code`→`kode` |
| `mixed_casual` | *Stáhni si **Cloud Code**, je to docela cool tool. Pojďme po práci do GIMu, dáme si **mil prep**.* | **yes** — `Claude`→`Cloud` is recoverable (vs XTTS's `Clouded Sword`); `cool tool` verbatim |
| `en_only` | *Create a calendar event for next Tuesday at 6pm.* | **yes** — perfect (Piper failed: `Srete`) |

### 10.4 Updated scorecard

| Input | Piper | XTTS-v2 | **Supertonic 3** |
|---|---|---|---|
| `cs_command` | yes | partial | **yes** |
| `cs_paragraph` | yes | no | **yes** |
| `mixed_dev` | yes | yes | **yes** |
| `mixed_brand` | partial | partial | **yes** |
| `mixed_casual` | yes | no | **yes** |
| `en_only` | no | yes | **yes** |
| **Score** | 3.5 / 6 | 2.5 / 6 | **6 / 6** |

### 10.5 Discussion

Supertonic does what XTTS promised but failed to deliver: it reads embedded
English fragments in something close to English (`cool tool`, `feature
flag`, full `Visual Studio`) **without** the trailing hallucinations that
hit XTTS on 4 of 6 inputs and **without** the catastrophic rewrites
(`Claude Code`→`Clouded Sword`, `Infinitu`→`Infinity Count`). Its Czech is
at least on par with Piper's — the only deviations are minor
(`Sraz`→`Sras`, a `mí minusu` stutter) and none are unrecoverable. It also
clears the two inputs Piper genuinely failed: `mixed_brand` (Piper lost
`Visual`) and `en_only` (Piper's Czech tokenizer mangled `Create`).

What this round-trip **cannot** confirm is the original claim that it
"sounds better than Jirka" — that is a naturalness judgement, and the
Whisper proxy is blind to prosody, stress, and timbre. The objective signal
strongly corroborates the claim (Supertonic is more intelligible across the
board), but final adoption still needs a human listening pass on the WAVs
in `outputs/*__supertonic.wav`.

Costs vs Piper: ~10× slower (still RTF ~0.16, well under realtime), ~2× the
peak RAM (~680 MB), and a larger first-run download (~1.8 GB). License is
**OpenRAIL-M** (sample code MIT) — commercial use permitted with
behavioural-use restrictions, materially better than XTTS's
commercial-prohibited CPML, though not as unconditionally clean as Piper's
per-voice MIT.

### 10.6 Recommendation update

**Supertonic 3 is now the front-runner on measured intelligibility** and
the only engine in this study that handles all six realistic inputs without
a recoverability failure. Pending a human listening pass (naturalness, voice
consistency, prosody), it is the candidate to beat Piper as the default. If
the listening pass confirms quality, the decision becomes: pay ~10× compute
and ~2× RAM over Piper in exchange for clean code-switching and perfect
English — almost certainly worth it for an assistant that mixes CZ/EN.
Piper remains the right pick where the ~0.015 RTF / 60 MB footprint is
paramount and pure-Czech utterances dominate.

## 11. Question intonation, Higgs v3 (added 2026-08-17)

Motivation: rendered narration sounded flat on questions. The pipeline was
ruled out first (chunking preserves `?`; chunk text reaches the backend
verbatim), leaving conditioning and sampling as levers. Before touching
either, this round measured what the shipped engine actually does with a
question — `intonation_probe.py`, 21 cases (Czech/English/code-switched
yes/no, wh-, and matched declarative controls), 3 takes each at the
production temperature 0.4, every take verified by the default cascade, then
classified by terminal F0 contour: median of the last 300 ms of voicing vs
the preceding ~500 ms, in semitones, ±1.5 st threshold (pyin, 60–450 Hz,
10 ms hop, octave guard, ≥600 ms voicing else "undef").

Reference clip: the operator's ad hoc 6.8 s Czech clip — **purely
declarative**, which is the hypothesis under test (a cloning model that has
never heard its speaker ask a question has no example of the interrogative
contour).

### 11.1 The numbers (tag `baseline`, 63 takes, 59 verified)

| category | expected | rise | flat | fall | undef | verified/total |
|---|---|---|---|---|---|---|
| cs_yesno  | rise | **10** | 2 | 2 | 1* | 15/15 |
| cs_wh     | fall | 0 | 1 | **10** | 0 | 11/12 |
| cs_decl   | fall | 0 | 0 | **6** | 0 | 6/9 |
| en_yesno  | rise | **2** | 4 | 0 | 0 | 6/6 |
| en_wh     | fall | 0 | 0 | **6** | 0 | 6/6 |
| en_decl   | fall | 0 | 0 | **3** | 0 | 3/3 |
| mix_yesno | rise | **4** | 2 | 0 | 0 | 6/6 |
| mix_wh    | fall | 0 | 0 | **3** | 0 | 3/3 |
| mix_decl  | fall | 0 | 0 | **3** | 0 | 3/3 |

\* one cs_yesno take ("Máš teď chvilku?") had <600 ms of voicing → undef —
too short for the metric, excluded from the 59% rise rate below.

### 11.2 Findings

- **Falls are essentially solved: 31/32 verified wh/declarative takes fall**
  (the one exception is a flat "Proč to trvá tak dlouho?"). The engine
  clearly reads terminal punctuation and sentence type.
- **The deficit is specific to the terminal rise, and it is stochastic, not
  absent: 16/26 verified yes/no takes with a defined contour rise (62%).** Some sentences rise on
  every take with textbook 4–7 st contours ("Máte to už nasazené v
  produkci?", "Poslal jsi mu ten dokument včera večer?", "Nasadil jsi ten
  pull request do produkce?"); others are flat on every take ("Are you
  coming to the meeting tomorrow?": +0.4/+0.05/+0.2 st). Worst,
  the *same text* flips between takes — "Přijdeš zítra na tu schůzku?"
  produced +5.35, −1.8, −1.7 st across three takes.
- Yes/no questions that fail to rise still mostly refuse to *fall* (flat,
  not declarative-shaped) — the model knows something is different, it just
  doesn't commit to the rise.
- The four unverified takes are all one phenomenon: the recogniser wrote
  "jsi" as colloquial "si" (identical pronunciation in connected Czech) —
  correct audio, orthographic sound-alike, the class `sound_alikes` exists
  for. Not an intonation problem.
- Borderline deltas cluster at the ±1.5 st threshold ("Funguje ti to i bez
  připojení k internetu?": +1.8/+1.15/+1.55; "Běží ti ten deployment na
  Kubernetes?": +1.5/+1.2/−0.45) — these WAVs are the listening-pass
  targets for tuning the threshold.

### 11.3 Decision this supports

The cheap lever is exactly the one hypothesised: **re-record the reference
with at least one genuine question in it** (spec in README, "Reference
clips") and re-run as `--tag newref`. If a question-bearing reference lifts
the yes/no rise rate materially above 62% — and above ~90% it makes questions
a solved case — the multi-reference / rolling-context work sketched (and
declined) in `docs/long-form.md` stays parked. Sampling changes
(temperature/top-p) remain unmeasured and out of scope until the reference
A/B lands.

Reproduce: `.venv-higgs/bin/python bench/intonation_probe.py --voice
bench/.voices/ref/adhoc.wav --tag baseline` (per-take data in
`bench/intonation_probe/baseline/`, gitignored working data;
`--selftest` checks the F0 classifier on synthetic glides).

### 11.4 Reference A/B and voice casting (same day)

The probe then screened the tutor project's eight zero-shot casting rolls
(same declarative text, temperature 0.7, no reference) and A/B-tested
question-bearing composite references (original clip + a spliced, verified,
F0-selected rising take of "Dává ti to smysl?" — a question deliberately
outside the test set):

| reference | yes/no rise (verified, defined) | failed verification | run time |
|---|---|---|---|
| adhoc 6.8 s (baseline) | 62% | 4/63 | 170 s |
| explainer-01 "Tom" (129 Hz) | 67% | 12/63 | 2155 s |
| explainer-03 (137 Hz) | 71%* | 14/63 | 208 s |
| explainer-07 (99 Hz) | 59% | 3/63 | 177 s |
| explainer-04 "Mira" (177 Hz) | 48% | 0/63 | 150 s |
| Mira + question splice | **59% (cs: 40%→67%)** | 2/63 | 161 s |
| explainer-07 + question splice | 59% (no change) | 3/63 | 175 s |

\* small denominator — the voice fails verification too often to trust it.

Findings:

- **Verification stability varies dramatically per zero-shot voice** and is
  invisible until measured: two of four screened voices (01, 03) fail the
  cascade on ~20% of takes and burn up to 13× the compute in retries. For
  casting, run the probe before falling in love with a voice. Cast picked:
  explainer-04 ("Mira") + explainer-07 (male), both effectively fully
  stable (their only rejections are the "jsi"→"si" orthographic
  sound-alike).
- **A question-bearing reference helps when the voice has headroom and does
  nothing when it doesn't**: Mira's Czech rise rate jumped 40%→67% from one
  spliced exemplar (itself only +2.45 st — the strongest of six takes);
  explainer-07 stayed exactly at his 67%-cs / 59%-overall ceiling.
- **Reference engineering alone plateaus around ~60-67%.** No configuration
  measured reaches reliable question intonation. The data now points at the
  synth ladder, not the reference: verified attempts at temperature 0.4
  rise on roughly 3 takes in 5, so best-of-N selection that *prefers a
  rising verified take for `?`-final chunks* would reach ~95% at N=3 —
  the same mechanism that already drove hard failures 0.269→0.038. That is
  a `narrator/` change (prosody-aware attempt ranking) and needs its own
  design + review round; parked as the measured next step.

## Appendix A — Reproducing

```sh
cd bench
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install piper-tts 'coqui-tts[codec]' mlx-whisper supertonic
pip install 'transformers<5,>=4.45'

# 1. Synthesize. Each script appends to outputs/results.csv.
python bench_piper.py
python bench_xtts.py
python bench_supertonic.py      # 44.1 kHz, multilingual ONNX, M1 voice

# 2. STT round-trip — objective intelligibility check.
python stt_roundtrip.py

# Inspect.
cat outputs/results.csv
ls outputs/*.stt.txt
```

All WAVs, round-trip transcripts, and `results.csv` from this run are
preserved in `bench/outputs/` (gitignored). The Bark smoke artifact is
`outputs/bark_smoke_cs_command_pl-speaker-0.{wav,stt.txt}`.

---

# ASR head-to-head (2026-08-14) — is the verifier's oracle the bottleneck?

`asr_headtohead.py` ran every chunk of two real Czech episodes (82 chunks) and
two English ones (40 chunks) through three recognisers on the same freshly
synthesized Higgs audio, gated by synth.py's own cheap checks. Aggregates
(threshold 0.90, per-attempt, no retries):

| | Czech (82) | English (40) |
|---|---|---|
| whisper-large-v3-turbo rejects | 44 (54%) | 14 (35%) |
| parakeet-tdt-0.6b-v3 rejects | 45 (55%) | 11 (28%) |
| canary-1b-v2 rejects | 44 (54%) | 12 (30%) |
| rejected by all three | 37 (45%) | 10 (25%) |
| whisper-only reject (parakeet rescues) | 4 | 3 |
| parakeet-only reject (whisper rescues) | 5 | 0 |

Findings that drove the library changes:

- **A better oracle does not reduce the rejection rate** — canary-1b-v2 has the
  best published Czech WER of the three and rejects exactly as much. The
  rejections decompose into genuine single-attempt truncations (the retry
  ladder's job), inaudible Czech orthography (fixed by the voicing-assimilation
  fold), and a small solo-reject band (rescued by the cascade).
- **Two-model agreement is not evidence of a TTS defect when the models share a
  weakness**: every Czech-mode recogniser mangles code-switched English words
  and foreign proper nouns the same way.
- Per-attempt rejections after the fold fix + cascade: 54% → 44% (cs). All
  three English whisper-only rejections were Whisper misreadings of correct
  audio, confirmed by both other models at 1.0.

The per-chunk transcripts backing these numbers are working data
(`bench/outputs/`, gitignored) — regenerate with `asr_headtohead.py
<report.json> --lang cs`.
