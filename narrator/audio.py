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
    return audio[start:end]


def declick(audio: Audio, sample_rate: int) -> Audio:
    """Short fades at both edges, so a join does not click."""
    n = int(sample_rate * FADE_MS / 1000)
    if audio.size <= 2 * n or n == 0:
        return audio
    out = audio.copy()
    out[:n] *= np.linspace(0, 1, n, dtype=out.dtype)
    out[-n:] *= np.linspace(1, 0, n, dtype=out.dtype)
    return out


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
