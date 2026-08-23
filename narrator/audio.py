"""Assembly and mastering.

The source has no room, no microphone, no rumble, no plosives and no proximity
effect, and its dynamics are already narrow twice over — training corpora are
level-normalised, and mel-spectrogram inversion smooths toward the mean. So most
of a conventional mastering chain is pointless or harmful here:

- **No compression.** It removes the little prosodic variation the model produced,
  and flat prosody is the measurable liability in long-form synthetic speech.
- **No presence boost.** 2-5 kHz is exactly where vocoder artefacts live, so the
  usual "clarity" lift amplifies the model's weakest output. The Speech
  Intelligibility Index does not support it either: clean speech over headphones
  is already at ceiling.
- **No low-mid cut.** Mud comes from proximity effect. There is none.
- **No noise reduction.** Vocoder noise is signal-correlated; denoisers chew into
  the speech.
- **No room-tone bed.** Tried and measured: a -50 dBFS bed against -13.8 dBFS
  speech is audible hiss next to a synthetic engine's true digital silence, and
  masking the worst chunk's own floor (-43 dBFS) would need a bed louder still.
  The masking rationale collapses on inspection.

What is left is a high-pass for DC offset, loudness normalisation, and a limiter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from narrator.types import Audio

TRIM_DB = -42.0
"""Silence threshold, relative to the loudest frame. Peak-relative rather than
absolute so it self-calibrates per chunk — two independent tools converged here."""

TRIM_GUARD_MS = 30
"""Kept either side of detected speech, so a plosive onset is not clipped."""

SILENCE_FRAME_MS = 20
SILENCE_DROP_DB = 35.0
"""How far under its own speech level a stretch must sit to count as silence.

Relative, never absolute, and the counterexample is in this repo: `FakeBackend`
exposes `amplitude` (default 0.1), so `FakeBackend(amplitude=0.001)` renders
entirely correct speech peaking at -60 dBFS. Any fixed dBFS floor near that
number calls the whole utterance silence — and the same shape is legal for a
natively quiet backend, a quiet reference, or the deliberate whisper this
library protects elsewhere (see `Voice.gain_db`). An absolute floor is also
measured in the wrong domain: this runs before `apply_gain` and before
`master` normalises loudness, so it would not refer to the level that ships.

35 dB is deliberately conservative, and the direction of the error is the whole
argument. A review found correct low-level delivery sitting 25-30 dB under the
louder speech in the same chunk; sustained past the threshold that would be
refused as a hole, and a gate that rejects correct renders is worse than the
defect it catches. 35 dB clears every such shape found.

The cost is stated rather than hidden: a whispered chunk's hole shows only ~30 dB
of contrast (-55 speech against a -85 floor), so this will NOT catch one. That is
a fail-open, leaving today's behaviour in place, where a false refusal would
block renders that are correct. The measured normal case is ~42 dB (-35 against
-77) and is caught comfortably.

Relative rather than absolute because scale carries no information here:
multiplying a buffer by any constant cannot change the verdict."""


def longest_silent_run(audio: Audio, sample_rate: int,
                       drop_db: float = SILENCE_DROP_DB) -> float:
    """Longest stretch BETWEEN speech that sits `drop_db` under the speech level.

    Only interior runs count. Leading and trailing silence belongs to
    `trim_silence`, and an utterance that is silent throughout is already caught
    by the duration bounds and by coverage — this measures the hole a render can
    otherwise ship with every word present and a clean report (issue #18: 18.5 s
    of dead air at `failed=0`, `min coverage 1.0`).

    The speech level is a high percentile rather than the maximum, so one loud
    plosive cannot raise the bar, and the hole itself cannot lower it. Measured
    boundaries of that choice, both checked:

    - Speech that merely varies stays safe. A 4 s passage 30 dB under the rest of
      its own chunk does not register at the default drop. Using the maximum
      instead of a percentile would move this the wrong way, since quiet frames
      sit well under a plosive peak.
    - Past ~96% dead air the percentile lands *inside* the hole and this returns
      0.0, failing open. Deliberately not hardened: audio that silent cannot
      have said its words, so the frame cap and the duration ceiling reject it
      first — a 60 s hole on a 12-word chunk is refused by those, and the case
      is pinned by test.
    """
    frame = int(SILENCE_FRAME_MS / 1000 * sample_rate)
    if frame <= 0 or audio.size < frame * 2:
        return 0.0

    count = audio.size // frame
    rms = np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1) + 1e-20)
    speech_level = float(np.percentile(rms, 95))
    if speech_level <= 0.0:
        return 0.0

    loud = rms > speech_level * (10 ** (-drop_db / 20))
    voiced = np.nonzero(loud)[0]
    if voiced.size < 2:
        return 0.0

    # Interior only: measure between the first and last voiced frames.
    gaps = np.diff(voiced) - 1
    return float(gaps.max() * frame / sample_rate) if gaps.size else 0.0


FADE_MS = 8
HPF_HZ = 75.0

TARGET_LUFS = -16.0
PEAK_CEILING_DB = -1.0


@dataclass(frozen=True)
class MasterConfig:
    target_lufs: float = TARGET_LUFS
    peak_ceiling_db: float = PEAK_CEILING_DB
    hpf_hz: float = HPF_HZ
    channels: int = 2
    """Dual-mono by default, and this is not arbitrary.

    BS.1770 integrated loudness sums over channels, so the same waveform measures
    exactly 3.01 dB louder as dual-mono than as mono. Apple asks for -16 LKFS with
    no mono exception; AES TD1008 assigns the correction to players rather than to
    the source. Both readings are self-consistent and the question of whether
    players actually apply it could not be settled from any vendor.

    Shipping dual-mono at -16 dissolves the dispute: measured as delivered it is
    exactly the published target, and perceptually it matches a mono file at -19.
    No assumption about player behaviour is required. Joint-stereo encoding makes
    the second channel nearly free.

    Set channels=1 to match an existing mono back-catalogue — mixing the two in
    one series produces an audible 3 dB step between episodes.
    """



def resample_to_16k(audio: Audio, source_rate: int) -> Audio:
    """Resample to the 16 kHz every recogniser expects.

    The ratio MUST derive from the actual source rate: paired with a 44.1 kHz
    engine while assuming 24 kHz, the resample silently changes time and pitch
    and every verdict becomes unreliable.
    """
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(16_000, source_rate)
    return resample_poly(audio, 16_000 // divisor, source_rate // divisor).astype(np.float32)


def trim_silence(audio: Audio, sample_rate: int) -> Audio:
    """Strip leading and trailing silence, keeping a guard band."""
    frame = int(0.030 * sample_rate)
    hop = int(0.010 * sample_rate)
    if audio.size < frame * 2:
        return audio

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-20)
    loud = np.nonzero(rms > rms.max() * (10 ** (TRIM_DB / 20)))[0]
    if loud.size == 0:
        return audio

    guard = int(TRIM_GUARD_MS / 1000 * sample_rate)
    start = max(0, loud[0] * hop - guard)
    end = min(audio.size, loud[-1] * hop + frame + guard)
    trimmed = audio[start:end]

    # A SECOND pass, speech-relative, because the two thresholds disagree and
    # material lives in the gap between them. This one keeps everything above
    # `TRIM_DB` measured against the chunk's PEAK frame; the interior silence
    # gate calls silent everything below `SILENCE_DROP_DB` measured against the
    # chunk's p95 frame. On speech those two references sit ~0.1 dB apart, so
    # the band between the thresholds is ~7 dB wide — and an edge run inside it
    # survives here (it is above peak-42) while being invisible there (the
    # interior detector deliberately ignores edges, which are this function's
    # job). Measured: 5 s of speech followed by 8 s of tail at p95-38 came
    # through this function untouched at 13.00 s and measured 0.02 s of interior
    # silence, then stitched into a 10 s run in the written file (issue #21).
    #
    # Removal, not refusal: an interior hole cannot be cut without editing the
    # timing between words, which is why that one fails a chunk — but trimming
    # edges is already this function's charter. It also needs no new ordering.
    # The take store keeps raw pre-trim audio precisely so trims are recomputed
    # every run, so every cached take inherits this on its next render.
    head, tail = _edge_runs(trimmed, sample_rate, SILENCE_DROP_DB)
    cut_head = max(0, head - guard) if head / sample_rate >= EDGE_SILENCE_S else 0
    cut_tail = max(0, tail - guard) if tail / sample_rate >= EDGE_SILENCE_S else 0
    return trimmed[cut_head : trimmed.size - cut_tail] if cut_head or cut_tail else trimmed


EDGE_SILENCE_S = 1.0
"""How long an edge run must be before the speech-relative pass removes it.

Generous on purpose. A trailing breath or a decaying final syllable is a few
hundred milliseconds; a second of material sitting 35 dB under the chunk's own
speech is not delivery. The tolerance is what keeps this from eating the quiet
endings `SILENCE_DROP_DB` was calibrated to protect."""


def _edge_runs(audio: Audio, sample_rate: int, drop_db: float) -> tuple[int, int]:
    """Leading and trailing sample counts sitting `drop_db` under the speech."""
    frame = int(SILENCE_FRAME_MS / 1000 * sample_rate)
    if frame <= 0 or audio.size < frame * 2:
        return 0, 0
    count = audio.size // frame
    rms = np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1) + 1e-20)
    voiced = np.nonzero(rms > float(np.percentile(rms, 95)) * (10 ** (-drop_db / 20)))[0]
    # Percentile collapse: if edge material is most of the buffer, p95 lands
    # inside it and this would measure nothing. Same fail-open the interior
    # detector documents, reachable at a lower dead-air fraction because an edge
    # run can dominate without the middle being silent.
    if voiced.size < max(2, count // 20):
        return 0, 0
    return int(voiced[0] * frame), int((count - 1 - voiced[-1]) * frame)


def declick(audio: Audio, sample_rate: int) -> Audio:
    """Short fades at both edges, so a join does not click."""
    n = int(sample_rate * FADE_MS / 1000)
    if audio.size <= 2 * n or n == 0:
        return audio
    out = audio.copy()
    out[:n] *= np.linspace(0, 1, n, dtype=out.dtype)
    out[-n:] *= np.linspace(1, 0, n, dtype=out.dtype)
    return out


def apply_gain(audio: Audio, gain_db: float) -> Audio:
    """Static level correction, as declared on a `Voice`. Identity at 0 dB.

    Static, never dynamic, and never inferred: this is a caller's stated offset
    between reference clips, so it applies equally to every chunk that voice
    speaks and leaves the performance inside a chunk exactly as synthesised.
    See `Voice.gain_db` for why the library refuses to work this out itself.
    """
    if gain_db == 0.0:
        return audio
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)


def concatenate(pieces: list[Audio]) -> Audio:
    """Butt-join in float32. Deliberately no crossfade.

    A crossfade is only safe on untrimmed boundaries. After `trim_silence` leaves
    ~30 ms of guard, a 50 ms crossfade overlaps real speech and eats the last
    phoneme of every chunk. A related trap is worth naming: pydub's
    `AudioSegment.append` defaults to a 100 ms crossfade, which across ~100 joins
    silently deletes about nine seconds of content.
    """
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, dtype=np.float32)


def soft_limit(audio: Audio, ceiling: float, knee_frac: float = 0.8) -> Audio:
    """Peak ceiling that only touches the loudest couple of dB.

    Identity below the knee, asymptotic to the ceiling above it, with continuous
    slope at the knee — so loudness is preserved and |out| < ceiling is guaranteed.
    """
    knee = knee_frac * ceiling
    magnitude = np.abs(audio)
    over = magnitude > knee
    out = magnitude.copy()
    out[over] = knee + (ceiling - knee) * np.tanh((magnitude[over] - knee) / (ceiling - knee))
    # tanh only *asymptotes* to the ceiling, and float32 rounding lands the
    # extremes exactly on it. Shave an epsilon so the strict guarantee holds.
    out = np.minimum(out, ceiling * (1.0 - 1e-6))
    return (np.sign(audio) * out).astype(np.float32)


def master(audio: Audio, sample_rate: int, cfg: MasterConfig = MasterConfig()) -> tuple[Audio, float, float]:
    """High-pass, normalise loudness, limit. Returns (audio, lufs, peak_dbfs).

    **Returns the final channel layout**, already duplicated when `channels == 2`,
    and measures that layout. Normalising the mono signal and duplicating it
    afterwards reintroduced the exact +3 dB BS.1770 offset this config exists to
    avoid: `master` reported -16.00 LUFS while the written file measured -12.99.
    Measuring anything other than what is delivered is how that happens.

    Loudness is applied as a single static gain, not a dynamic pass. ffmpeg's
    `loudnorm` in one-pass mode applies time-varying gain, which on already-flat
    synthetic speech pumps and squashes what prosody exists.
    """
    if cfg.channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {cfg.channels}")
    import pyloudnorm as pyln
    from scipy.signal import butter, sosfilt

    sos = butter(2, cfg.hpf_hz, btype="highpass", fs=sample_rate, output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)

    # BS.1770 needs at least one analysis block. Shorter input cannot be
    # measured, so it is passed through unnormalised rather than crashing — a
    # caller rendering a two-word test clip should not hit a library error.
    # Duplicate FIRST, then measure and normalise the delivered layout.
    laid_out = to_channels(audio, cfg.channels)

    meter = pyln.Meter(sample_rate)
    if laid_out.shape[0] < meter.block_size * sample_rate:
        limited = soft_limit(laid_out, 10 ** (cfg.peak_ceiling_db / 20))
        peak_db = 20 * np.log10(np.max(np.abs(limited)) + 1e-12) if limited.size else -np.inf
        return limited, float("nan"), float(peak_db)

    measured = meter.integrated_loudness(laid_out)
    if np.isfinite(measured):
        laid_out = pyln.normalize.loudness(laid_out, measured, cfg.target_lufs).astype(np.float32)
    laid_out = soft_limit(laid_out, 10 ** (cfg.peak_ceiling_db / 20))

    final = meter.integrated_loudness(laid_out)
    peak = 20 * np.log10(np.max(np.abs(laid_out)) + 1e-12) if laid_out.size else -np.inf
    return laid_out, float(final), float(peak)


def to_channels(audio: Audio, channels: int) -> Audio:
    """Mono stays 1-D; dual-mono duplicates into an (n, 2) array."""
    if channels == 1:
        return audio
    if audio.ndim == 2:
        return audio
    return np.stack([audio, audio], axis=1)
