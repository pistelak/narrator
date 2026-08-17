"""Measure terminal F0 contour on rendered questions — does Higgs v3 rise?

Motivation: rendered narration sounds flat on questions, and the pipeline is
not the culprit — chunking preserves `?` and chunk text reaches the backend
verbatim. The suspected lever is the reference clip: the operator's current
references are purely declarative, so the model has never "heard" this
speaker ask a question. This probe turns "questions sound flat" into a number
so a better reference (or any future prosody work) is measured, not vibed.

Per take: synthesize with the real retry ladder (`_best_attempt` — NOT
`synthesize_chunk`, whose sentence-split recovery concatenates audio at a
synthetic join and would change the prosody being measured), verify with the
default verifier, then extract F0 with pyin and compare the median of the
last 300 ms of voicing against the preceding ~500 ms, in semitones. Czech
yes/no questions canonically end in a terminal RISE; wh-questions and
declaratives in a FALL. The wh-FALL expectation is canonical, not exclusive —
colloquial Czech wh-questions can carry a politeness rise, so a wh-rise is
noteworthy, not automatically a failure.

A/B protocol: two runs with different --tag values (e.g. `baseline` with the
old reference, `newref` with a re-recorded one); everything that must not
vary between tags is pinned in each run's results.json header.

Self-contained: writes only into `intonation_probe/<tag>/`, never touches
`inputs/`, `outputs/`, or `results.csv`.

Run with the narrator venv, from the repo root:

    .venv-higgs/bin/python bench/intonation_probe.py \
        --voice bench/.voices/ref/adhoc.wav --tag baseline

    .venv-higgs/bin/python bench/intonation_probe.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "intonation_probe"

# The F0 core lives in narrator.prosody (one home — the synth ladder's rise
# selection uses the same measurement this probe validates). The parameters
# and their measured rationale moved there with the code.
from narrator.prosody import (  # noqa: E402
    RISE_THRESHOLD_ST,
    delta_from_f0,
    voiced_f0,
    yes_no_question,
)

HOP_S = 0.010
SLOPE_FRAMES = 70


@dataclass(frozen=True)
class Case:
    id: str
    lang: str
    category: str
    expected: str  # "rise" | "fall" — canonical, not exclusive (see module doc)
    text: str


# Author, sanity-check the Czech before trusting the aggregates: the
# expectations assume these read as natural spoken sentences, and a stilted
# sentence gets stilted prosody regardless of the reference.
CASES = [
    # Czech yes/no — canonical terminal RISE
    Case("cs_yn_meeting", "cs", "cs_yesno", "rise", "Přijdeš zítra na tu schůzku?"),
    Case("cs_yn_prod", "cs", "cs_yesno", "rise", "Máte to už nasazené v produkci?"),
    Case("cs_yn_offline", "cs", "cs_yesno", "rise", "Funguje ti to i bez připojení k internetu?"),
    Case("cs_yn_doc", "cs", "cs_yesno", "rise", "Poslal jsi mu ten dokument včera večer?"),
    Case("cs_yn_short", "cs", "cs_yesno", "rise", "Máš teď chvilku?"),
    # Czech wh — canonical terminal FALL
    Case("cs_wh_when", "cs", "cs_wh", "fall", "Kdy přijdeš zítra na tu schůzku?"),
    Case("cs_wh_where", "cs", "cs_wh", "fall", "Kde jsi nechal ten dokument?"),
    Case("cs_wh_why", "cs", "cs_wh", "fall", "Proč to trvá tak dlouho?"),
    Case("cs_wh_howmany", "cs", "cs_wh", "fall", "Kolik lidí přišlo na tu přednášku?"),
    # Czech declarative controls — matched wording, FALL
    Case("cs_decl_meeting", "cs", "cs_decl", "fall", "Přijdeš zítra na tu schůzku."),
    Case("cs_decl_doc", "cs", "cs_decl", "fall", "Poslal jsi mu ten dokument včera večer."),
    Case("cs_decl_prod", "cs", "cs_decl", "fall", "Už to máte nasazené v produkci."),
    # English yes/no — RISE
    Case("en_yn_meeting", "en", "en_yesno", "rise", "Are you coming to the meeting tomorrow?"),
    Case("en_yn_deploy", "en", "en_yesno", "rise", "Did you deploy the new version to production?"),
    # English wh — FALL
    Case("en_wh_when", "en", "en_wh", "fall", "When are you coming to the meeting tomorrow?"),
    Case("en_wh_where", "en", "en_wh", "fall", "Where did you leave the documentation?"),
    # English declarative control — FALL
    Case("en_decl_meeting", "en", "en_decl", "fall", "You are coming to the meeting tomorrow."),
    # Code-switched: Czech frame, English technical term. lang="cs" per the
    # repo's mixed->cs routing convention.
    Case("mix_yn_pr", "cs", "mix_yesno", "rise", "Nasadil jsi ten pull request do produkce?"),
    Case("mix_yn_deploy", "cs", "mix_yesno", "rise", "Běží ti ten deployment na Kubernetes?"),
    Case("mix_wh_review", "cs", "mix_wh", "fall", "Kdy dorazí ten code review od Marka?"),
    Case("mix_decl_pr", "cs", "mix_decl", "fall", "Nasadil jsem ten pull request do produkce."),
]

CATEGORY_ORDER = [
    "cs_yesno", "cs_wh", "cs_decl",
    "en_yesno", "en_wh", "en_decl",
    "mix_yesno", "mix_wh", "mix_decl",
]


# ------------------------------------------------------------- F0 analysis

def theil_sen_slope_st_per_s(f0: np.ndarray, sample_rate: int) -> float:
    """Median of pairwise slopes, in semitones per second of VOICED time.

    O(n^2) on <= 70 frames is ~2400 pairs — trivial, and it buys immunity to
    the residual outliers a least-squares fit would chase.
    """
    st = 12.0 * np.log2(f0 / f0[0])
    t = np.arange(f0.size) * HOP_S
    slopes = [
        (st[j] - st[i]) / (t[j] - t[i])
        for i in range(f0.size)
        for j in range(i + 1, f0.size)
    ]
    return float(np.median(slopes))


def classify(delta_st: float) -> str:
    if delta_st >= RISE_THRESHOLD_ST:
        return "rise"
    if delta_st <= -RISE_THRESHOLD_ST:
        return "fall"
    return "flat"


def analyze(audio: np.ndarray, sample_rate: int) -> dict:
    """Terminal-contour record for one take. Never raises.

    Any failure in the F0 path (pyin exception, degenerate audio) yields the
    same schema-conforming sentinel — contour "undef", null delta/slope —
    never a missing field, so aggregation and resume always work.
    trim_silence never returns empty (all-silent input comes back unchanged),
    and pure silence simply yields zero voiced frames, landing in the normal
    undef path rather than an exception.
    """
    from narrator.audio import trim_silence

    undef = {"delta_st": None, "slope_st_s": None, "voiced_frames": 0, "contour": "undef"}
    try:
        f0 = voiced_f0(trim_silence(audio, sample_rate), sample_rate)
        n = int(f0.size)
        delta = delta_from_f0(f0)
        if delta is None:
            return {**undef, "voiced_frames": n}
        slope = theil_sen_slope_st_per_s(f0[-min(SLOPE_FRAMES, n):], sample_rate)
        return {
            "delta_st": round(delta, 3),
            "slope_st_s": round(slope, 3),
            "voiced_frames": n,
            "contour": classify(delta),
        }
    except Exception as exc:  # broad on purpose — the sentinel row is the contract
        print(f"    F0 analysis failed ({exc!r}) — recording undef", flush=True)
        return undef


# ---------------------------------------------------------------- selftest

def _harmonic_glide(
    sample_rate: int, f_start: float, f_end: float,
    duration_s: float = 2.0, glide_s: float = 0.3, trailing_silence_s: float = 0.0,
) -> np.ndarray:
    """Sawtooth-like tone (5 harmonics, 1/k amplitudes) with a terminal glide.

    2.0 s at a 10 ms hop is ~200 voiced frames — comfortably above the
    60-frame floor even after pyin edge losses. Harmonic-rich on purpose:
    pyin tracks it more reliably than a pure sine.
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    freq = np.full(n, f_start)
    glide_start = duration_s - glide_s
    in_glide = t >= glide_start
    freq[in_glide] = f_start + (f_end - f_start) * (t[in_glide] - glide_start) / glide_s
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    signal = sum(np.sin(k * phase) / k for k in range(1, 6))
    signal = 0.5 * signal / np.max(np.abs(signal))
    if trailing_silence_s:
        signal = np.concatenate([signal, np.zeros(int(trailing_silence_s * sample_rate))])
    return signal.astype(np.float32)


def selftest() -> int:
    """Classifier check on synthetic glides — no model load, runs in seconds."""
    sr = 24_000
    checks = [
        ("flat 200 Hz", _harmonic_glide(sr, 200, 200), "flat"),
        ("rise 200->300 Hz (+7.0 st)", _harmonic_glide(sr, 200, 300), "rise"),
        ("fall 200->130 Hz (-7.5 st)", _harmonic_glide(sr, 200, 130), "fall"),
        ("rise + 400 ms trailing silence",
         _harmonic_glide(sr, 200, 300, trailing_silence_s=0.4), "rise"),
    ]
    failed = 0
    for name, signal, expected in checks:
        result = analyze(signal, sr)
        ok = result["contour"] == expected
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}: contour={result['contour']} "
              f"(expected {expected}), delta={result['delta_st']} st, "
              f"voiced={result['voiced_frames']}")
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


# -------------------------------------------------------------------- run

def _params(ranked: bool) -> dict:
    """Everything that must not vary between A/B tags, resolved, not echoed.

    Callables in SynthConfig are recorded by presence (bool), not identity —
    JSON cannot hold them and header comparison only needs "same policy".
    """
    from dataclasses import asdict

    from narrator import prosody
    from narrator.backends.higgs import MODEL, SAMPLE_RATE

    synth = {k: (bool(v) if callable(v) else v)
             for k, v in asdict(_synth_config(ranked)).items()}
    return {
        "model": MODEL,
        "sample_rate": SAMPLE_RATE,
        "ranked": ranked,
        "synth": synth,
        "f0": {
            "fmin_hz": prosody.FMIN_HZ, "fmax_hz": prosody.FMAX_HZ,
            "frame_s": prosody.FRAME_S, "hop_s": prosody.HOP_S,
            "min_voiced_frames": prosody.MIN_VOICED_FRAMES,
            "tail_frames": prosody.TAIL_FRAMES, "head_frames": prosody.HEAD_FRAMES,
            "slope_frames": SLOPE_FRAMES,
            "octave_guard_st": prosody.OCTAVE_GUARD_ST,
            "rise_threshold_st": RISE_THRESHOLD_ST,
        },
    }


def _synth_config(ranked: bool):
    """Raw distribution by default; --ranked turns on the ladder's own rise
    selection so the probe measures what the pipeline would actually ship."""
    from narrator.synth import SynthConfig

    return SynthConfig(wants_rise=yes_no_question) if ranked else SynthConfig()


def _load_existing(results_path: Path, header: dict) -> list[dict]:
    """Records from a prior run of this tag, for idempotent resume.

    Refuses to resume across a params/voice change: mixed-provenance rows
    would silently confound the A/B comparison the tag exists to isolate.
    """
    if not results_path.is_file():
        return []
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    for key in ("params", "voice_path", "voice_transcript", "takes"):
        if stored["header"].get(key) != header.get(key):
            sys.exit(
                f"{results_path} was produced with a different {key}; "
                "start a fresh --tag instead of mixing runs."
            )
    return stored["records"]


def _checkpoint(results_path: Path, header: dict, records: list[dict]) -> None:
    """Full rewrite after every take (asr_headtohead pattern): a killed run
    loses at most the in-flight take."""
    payload = {"header": header, "records": records}
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _aggregate(records: list[dict]) -> list[dict]:
    rows = []
    for category in CATEGORY_ORDER:
        takes = [r for r in records if r["category"] == category]
        if not takes:
            continue
        verified = [r for r in takes if r["verified"]]
        counts = {c: sum(1 for r in verified if r["contour"] == c)
                  for c in ("rise", "flat", "fall", "undef")}
        defined = [r for r in verified if r["contour"] != "undef"]
        matches = sum(1 for r in defined if r["contour"] == r["expected"])
        rows.append({
            "category": category,
            "expected": takes[0]["expected"],
            "verified": len(verified),
            "total": len(takes),
            **counts,
            "match_rate": (round(matches / len(defined), 2) if defined else None),
        })
    return rows


def _write_report(out_dir: Path, header: dict, records: list[dict]) -> Path:
    lines = [
        "# Question-intonation probe",
        "",
        f"tag `{header['tag']}` · voice `{header['voice_path']}` · "
        f"takes/case {header['takes']} · model `{header['params']['model']}` · "
        f"threshold ±{header['params']['f0']['rise_threshold_st']} st",
        "",
        "Expected contours are canonical, not exclusive: colloquial Czech",
        "wh-questions can rise (politeness rise), so a wh-rise is noteworthy",
        "rather than an automatic failure. Aggregates count VERIFIED takes only.",
        "",
        "## Per-category aggregate",
        "",
        "| category | expected | rise | flat | fall | undef | verified/total | match |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in _aggregate(records):
        match = ("no verified takes — see appendix" if row["verified"] == 0
                 else ("n/a (all undef)" if row["match_rate"] is None
                       else f"{row['match_rate']:.0%}"))
        lines.append(
            f"| {row['category']} | {row['expected']} | {row['rise']} | {row['flat']} "
            f"| {row['fall']} | {row['undef']} | {row['verified']}/{row['total']} "
            f"| {match} |"
        )
    lines += [
        "",
        "## Per-take detail",
        "",
        "| case | cat | expected | contour | Δ st | slope st/s | voiced | verified (cov) | wav |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        delta = "—" if r["delta_st"] is None else f"{r['delta_st']:+.2f}"
        slope = "—" if r["slope_st_s"] is None else f"{r['slope_st_s']:+.2f}"
        wav = Path(r["wav"]).name if r["wav"] else "—"
        lines.append(
            f"| {r['case']} t{r['take']} | {r['category']} | {r['expected']} "
            f"| {r['contour']} | {delta} | {slope} | {r['voiced_frames']} "
            f"| {'yes' if r['verified'] else 'NO'} ({r['coverage']:.2f}) | {wav} |"
        )
    unverified = [r for r in records if not r["verified"]]
    if unverified:
        lines += ["", "## Appendix — unverified takes (evidence, not excuses)", ""]
        for r in unverified:
            lines += [
                f"- **{r['case']} t{r['take']}** coverage {r['coverage']:.2f}",
                f"  - text: {r['text']}",
                f"  - heard: {r.get('transcript') or '(no transcript)'}",
            ]
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _console_tally(records: list[dict]) -> None:
    print("\n" + "=" * 62)
    for row in _aggregate(records):
        if row["verified"] == 0:
            print(f"{row['category']}: no verified takes ({row['total']} total) — "
                  "see report appendix")
            continue
        got = row[row["expected"]]
        unverified = row["total"] - row["verified"]
        verdict = "OK" if got == row["verified"] else (
            "MISS" if got == 0 else "PARTIAL")
        print(f"{row['category']}: {row['expected']} {got}/{row['verified']} verified "
              f"({unverified} unverified) — expected {row['expected']}, {verdict}")


def run(args: argparse.Namespace) -> int:
    try:
        import librosa  # noqa: F401 — fail before loading the TTS model
        import soundfile as sf

        from narrator.backends.higgs import HiggsBackend
        from narrator.synth import _best_attempt
        from narrator.types import Voice
        from narrator.verify import default_verifier
    except ImportError as exc:
        sys.exit(
            f"missing dependency ({exc}); run with the narrator real-model venv: "
            ".venv-higgs/bin/python bench/intonation_probe.py … "
            "(see bench/README.md)"
        )

    voice_path = args.voice.resolve()
    transcript_path = voice_path.with_suffix(".txt")
    if not voice_path.is_file():
        sys.exit(f"reference clip not found: {voice_path}")
    if not transcript_path.is_file():
        sys.exit(
            f"transcript sidecar not found: {transcript_path} — the exact "
            "spoken text of the clip must sit beside it (see bench/README.md, "
            "Reference clips)"
        )
    transcript = transcript_path.read_text(encoding="utf-8").strip()

    cases = CASES
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.id for c in CASES}
        if unknown:
            sys.exit(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in CASES if c.id in wanted]

    out_dir = OUT_BASE / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    header = {
        "tag": args.tag,
        "voice_path": str(voice_path),
        "voice_transcript": transcript,
        "takes": args.takes,
        "date": time.strftime("%Y-%m-%d"),
        "params": _params(args.ranked),
    }
    records = _load_existing(results_path, header)
    # Idempotent resume: skip (case, take) pairs already recorded, but only
    # while the FULL recorded case matches the current table — an edited
    # case under the same id (text, language, category, or expectation)
    # invalidates its rows, or a tag's numbers stop matching its report.
    records = [
        r for r in records
        if any(
            c.id == r["case"] and c.text == r["text"] and c.lang == r["lang"]
            and c.category == r["category"] and c.expected == r["expected"]
            for c in CASES
        )
    ]
    done = {(r["case"], r["take"]) for r in records}

    backend = HiggsBackend()
    verifier = default_verifier(backend.sample_rate)
    cfg = _synth_config(args.ranked)

    started = time.time()
    total = sum(1 for c in cases for t in range(1, args.takes + 1)
                if (c.id, t) not in done)
    print(f"{len(cases)} cases × {args.takes} takes — {total} to render "
          f"({len(done)} already recorded) → {out_dir}", flush=True)

    for case in cases:
        # Fresh Voice per case so verification runs the case's language;
        # the backend's reference cache is keyed by path, so no re-encoding.
        voice = Voice(voice_path, transcript, case.lang)
        for take in range(1, args.takes + 1):
            if (case.id, take) in done:
                continue
            print(f"[{case.id} t{take}] {case.text}", flush=True)
            row = {
                "case": case.id, "lang": case.lang, "category": case.category,
                "expected": case.expected, "text": case.text, "take": take,
                "wav": None, "verified": False, "coverage": 0.0, "attempts": 0,
                "transcript": "",
                "delta_st": None, "slope_st_s": None, "voiced_frames": 0,
                "contour": "undef", "match": False,
            }
            try:
                attempt = _best_attempt(case.text, backend, verifier, voice, cfg)
            except Exception as exc:  # broad on purpose — one take must not kill the run
                print(f"    synthesis raised {exc!r} — recording failed take", flush=True)
                attempt = None
            if attempt is not None:
                wav_path = out_dir / f"{case.id}__t{take}.wav"
                sf.write(wav_path, attempt.audio, backend.sample_rate, subtype="PCM_16")
                row.update(
                    wav=str(wav_path.relative_to(ROOT)),
                    verified=bool(attempt.ok),
                    coverage=round(attempt.verdict.coverage, 3),
                    attempts=attempt.calls_spent,
                    transcript=attempt.verdict.transcript,
                    **analyze(attempt.audio, backend.sample_rate),
                )
                row["match"] = row["verified"] and row["contour"] == case.expected
                print(f"    contour={row['contour']} delta={row['delta_st']} st "
                      f"verified={row['verified']} cov={row['coverage']}", flush=True)
            records.append(row)
            _checkpoint(results_path, header, records)

    report = _write_report(out_dir, header, records)
    _console_tally(records)
    print(f"\n{len(records)} takes in {time.time() - started:.0f}s → {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--voice", type=Path,
                        help="reference clip; a .txt sidecar with its exact "
                             "transcript must sit beside it")
    parser.add_argument("--tag", default="baseline",
                        help="run label; output goes to intonation_probe/<tag>/ "
                             "(A/B = two runs, two tags)")
    parser.add_argument("--takes", type=int, default=3,
                        help="takes per case — 3 distinguishes 'never rises' "
                             "from 'rises sometimes' at temperature 0.4")
    parser.add_argument("--only", default=None,
                        help="comma-separated case ids (smoke runs, resume)")
    parser.add_argument("--ranked", action="store_true",
                        help="enable the synth ladder's rise selection "
                             "(wants_rise=yes_no_question) — measures what "
                             "ships, not the raw take distribution")
    parser.add_argument("--selftest", action="store_true",
                        help="check the F0 classifier on synthetic glides; "
                             "no model load")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.voice is None:
        parser.error("--voice is required (unless --selftest)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
