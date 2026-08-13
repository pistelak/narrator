#!/usr/bin/env python3
"""Lexicon A/B probe — does the pronunciation-lexicon respelling actually help?

For each problem term in a caller's pronunciation lexicon, synthesize the
same carrier sentence twice with Supertonic: once with the raw written token
(what the script literally contains), once with the spoken respelling
substituted (what the voice pipeline would feed the engine). Round-trip both
through mlx-whisper and check whether the term comes back recognizably.

Decision per term:
  raw ok,  say ok   -> respelling optional (engine already says it right)
  raw bad, say ok   -> respelling fixes a real mispronunciation: keep it
  raw ok,  say bad   -> raw was fine; respelling hurts: drop or fix it
  raw bad, say bad   -> neither works: the respelling needs more work

Self-contained: writes only into lexicon_probe/, never touches inputs/,
outputs/, or results.csv. Mirrors bench_supertonic.py and stt_roundtrip.py
engine config exactly (M1, lang=en, whisper large-v3-turbo).
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lexicon_probe"

VOICE = "M1"
MODEL = "supertonic-3"
TOTAL_STEPS = 8       # bench default; diffusion steps affect prosody, not phoneme identity
SPEED = 1.05          # bench default
WHISPER = "mlx-community/whisper-large-v3-turbo"

# Each probe: a carrier sentence with "{}" where the term goes, the raw written
# token, the respelling from the pronunciation lexicon, and an acceptance regex for
# the listener-correct recovery (same target for both variants).
PROBES = [
    ("sha",   "First the archive goes through one called {} two fifty-six.",
     "SHA", "shah", r"\bshaw?\b"),
    ("crc", "Then it runs through a second function called {} thirty-two.",
     "CRC", "see are see", r"\bcrc\b"),
    ("fcs",   "The result is called the frame check sequence, {} for short.",
     "FCS", "eff see ess", r"\bfcs\b|\bf\s*c\s*s\b|eff\s*see"),
    ("nsa",   "One was designed by government cryptographers at the {}.",
     "NSA", "en ess ay", r"\bnsa\b|\bn\s*s\s*a\b|en\s*ess"),
    ("qr",    "Labels live on parcels and in {} codes.",
     "QR", "cue are", r"\bqr\b|\bq\s*r\b|cue\s*are"),
    ("acme",  "Then an insurance company called {} gets hold of a copy.",
     "Acme", "ack mee", r"\bac+k?me\b|ack\s*mee"),
    # HMAC collides with the following SHA when raw -> the respelling separates them.
    ("hmac",  "The token goes into a hash called {} SHA five-twelve.",
     "HMAC", "aitch mack", r"\bhmac\b|h\s*mac|aitch\s*mack"),
    # Author first name: raw "Kalle" collapses to one syllable ("Cal"); the
    # respelling must keep two syllables AND not break the surname (no hyphens).
    ("author", "This is the field guide, by {}.",
     "Kalle Lindkvist", "Kalleh Lindkvist", r"kah?l(?:i|eh|e)\b"),
    ("amira", "A new colleague named {} joins the night shift.",
     "Amira", "ah-mee-rah", r"\bam[iy]?ra\b|ah\s*mee\s*ra|\bamera\b"),
]


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def recovered(transcript, pattern):
    return bool(re.search(pattern, normalize(transcript)))


def verdict(raw_ok, say_ok):
    if raw_ok and say_ok:
        return "optional ", "engine already says the raw token right"
    if say_ok:
        return "KEEP     ", "respelling fixes a real mispronunciation"
    if raw_ok:
        return "DROP     ", "raw was fine; respelling hurts"
    return "REWORK   ", "neither raw nor respelling survives"


def main():
    try:
        from supertonic import TTS
        import soundfile as sf
        import librosa
        import mlx_whisper
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}\nActivate .venv (see bench/README.md).")

    OUT.mkdir(exist_ok=True)
    print(f"Loading Supertonic '{MODEL}' (voice {VOICE}, steps {TOTAL_STEPS}) ...", flush=True)
    tts = TTS(model=MODEL, auto_download=True)
    style = tts.get_voice_style(VOICE)
    tts.synthesize(text="Warm up.", voice_style=style, lang="en",
                   total_steps=TOTAL_STEPS, speed=SPEED)

    def synth_and_transcribe(text, wav_path):
        wav, _ = tts.synthesize(text=text, voice_style=style, lang="en",
                                total_steps=TOTAL_STEPS, speed=SPEED)
        tts.save_audio(wav, str(wav_path))
        audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        return mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER, language="en")["text"].strip()

    rows = []
    t0 = time.perf_counter()
    for slug, carrier, raw_tok, say_tok, pattern in PROBES:
        raw_text = carrier.format(raw_tok)
        say_text = carrier.format(say_tok)
        print(f"\n[{slug}] raw={raw_tok!r}  say={say_tok!r}", flush=True)

        raw_tr = synth_and_transcribe(raw_text, OUT / f"{slug}__raw.wav")
        say_tr = synth_and_transcribe(say_text, OUT / f"{slug}__say.wav")
        raw_ok = recovered(raw_tr, pattern)
        say_ok = recovered(say_tr, pattern)
        tag, why = verdict(raw_ok, say_ok)

        print(f"   raw -> [{'ok ' if raw_ok else 'BAD'}] {raw_tr}")
        print(f"   say -> [{'ok ' if say_ok else 'BAD'}] {say_tr}")
        print(f"   => {tag.strip()}: {why}")
        rows.append((slug, raw_tok, say_tok, raw_ok, say_ok, tag.strip(), why, raw_tr, say_tr))

    elapsed = time.perf_counter() - t0

    report = OUT / "REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Lexicon A/B probe — pronunciation lexicon\n\n")
        f.write(f"Supertonic {MODEL} / voice {VOICE} / steps {TOTAL_STEPS} / speed {SPEED}; "
                f"round-trip {WHISPER}. {len(PROBES)} terms in {elapsed:.0f}s.\n\n")
        f.write("| term | respelling | raw RT | say RT | verdict |\n")
        f.write("|---|---|:--:|:--:|---|\n")
        for slug, raw_tok, say_tok, raw_ok, say_ok, tag, why, _, _ in rows:
            f.write(f"| `{raw_tok}` | `{say_tok}` | {'ok' if raw_ok else 'BAD'} | "
                    f"{'ok' if say_ok else 'BAD'} | {tag} — {why} |\n")
        f.write("\n## Transcripts\n\n")
        for slug, raw_tok, say_tok, raw_ok, say_ok, tag, why, raw_tr, say_tr in rows:
            f.write(f"**{raw_tok}**\n\n")
            f.write(f"- raw `{raw_tok}` → {raw_tr}\n")
            f.write(f"- say `{say_tok}` → {say_tr}\n\n")

    keep = sum(1 for r in rows if r[5] == "KEEP")
    opt = sum(1 for r in rows if r[5] == "optional")
    drop = sum(1 for r in rows if r[5] == "DROP")
    rew = sum(1 for r in rows if r[5] == "REWORK")
    print(f"\n=== {keep} keep, {opt} optional, {drop} drop, {rew} rework "
          f"(of {len(rows)}) in {elapsed:.0f}s ===")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
