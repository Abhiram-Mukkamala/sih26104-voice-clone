"""
dsp_pipeline.py
================
SIH26104 — Voice Clone Impersonation Detection
Track B: Pure NumPy/SciPy DSP Engine + Track A LFCC Front-End.
100% dependency-free of librosa/numba (fully compatible with Python 3.10–3.14).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window, find_peaks

SAMPLE_RATE = 16000
FRAME_LEN_S = 1.0
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_LEN_S)

N_FFT = 512
HOP_LENGTH = 160          # 10ms hop @ 16kHz
WIN_LENGTH = 400          # 25ms window @ 16kHz
N_LINEAR_FILTERS = 40
N_LFCC = 20


def _linear_filterbank(n_fft: int, sr: int, n_filters: int) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    edges = np.linspace(0, sr / 2, n_filters + 2)

    fbank = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = edges[m - 1], edges[m], edges[m + 1]
        rising = (freqs >= f_left) & (freqs <= f_center)
        falling = (freqs >= f_center) & (freqs <= f_right)
        if f_center > f_left:
            fbank[m - 1, rising] = (freqs[rising] - f_left) / (f_center - f_left)
        if f_right > f_center:
            fbank[m - 1, falling] = (f_right - freqs[falling]) / (f_right - f_center)
    return fbank


_FILTERBANK = _linear_filterbank(N_FFT, SAMPLE_RATE, N_LINEAR_FILTERS)


def _stft_magnitude(audio: np.ndarray) -> np.ndarray:
    """Framed STFT magnitude spectrum. Shape: (n_frames, N_FFT//2 + 1)."""
    window = get_window("hann", WIN_LENGTH)
    n_frames = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH
    if n_frames < 1:
        audio = np.pad(audio, (0, WIN_LENGTH - len(audio)))
        n_frames = 1

    frames = np.stack(
        [
            audio[i * HOP_LENGTH : i * HOP_LENGTH + WIN_LENGTH] * window
            for i in range(n_frames)
        ]
    )
    spectrum = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1))
    return spectrum + 1e-10


def compute_lfcc(audio: np.ndarray, n_lfcc: int = N_LFCC) -> np.ndarray:
    """
    Linear Frequency Cepstral Coefficients.
    Returns shape (n_frames, n_lfcc) — feed directly to the ONNX model as a
    [batch, time, n_lfcc] tensor after batching upstream.
    """
    mag = _stft_magnitude(audio)
    filtered = mag @ _FILTERBANK.T
    log_energy = np.log(filtered + 1e-10)
    lfcc = dct(log_energy, type=2, axis=1, norm="ortho")[:, :n_lfcc]
    return lfcc.astype(np.float32)


def spectral_flatness(audio: np.ndarray) -> float:
    """
    Wiener entropy: geometric_mean(power) / arithmetic_mean(power).
    Near-1.0 = noise-like/flat spectrum; Near-0.0 = strongly tonal.
    """
    mag = _stft_magnitude(audio)
    power = mag ** 2
    geo_mean = np.exp(np.mean(np.log(power), axis=1))
    arith_mean = np.mean(power, axis=1) + 1e-10
    return float(np.mean(geo_mean / arith_mean))


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Fraction of sign changes per sample, averaged over the buffer."""
    signs = np.sign(audio)
    signs[signs == 0] = 1
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def compute_f0_contour(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fast autocorrelation-based pitch extractor. Replaces librosa.pyin
    to eliminate the librosa/numba dependency chain entirely.
    Returns Hz array with NaN for unvoiced frames.
    """
    frame_len = WIN_LENGTH * 2
    hop = HOP_LENGTH
    n_frames = 1 + (len(audio) - frame_len) // hop
    f0 = np.full(n_frames, np.nan, dtype=np.float32)

    min_lag = int(sr / 400.0)  # Max F0 = 400 Hz
    max_lag = int(sr / 65.0)   # Min F0 = 65 Hz

    for i in range(n_frames):
        seg = audio[i * hop : i * hop + frame_len]
        energy = np.sum(seg ** 2)
        if energy < 1e-4:
            continue
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        if max_lag >= len(corr):
            continue
        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue
        peak_idx = np.argmax(search_region) + min_lag
        if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.35:
            f0[i] = float(sr / peak_idx)
    return f0


def f0_contour_variance(f0: np.ndarray) -> float:
    """
    Variance of the *voiced* portion of the F0 contour.
    Synthetic vocoders produce either unnaturally smooth or jumpy contours.
    """
    voiced = f0[~np.isnan(f0)]
    return float(np.var(voiced)) if len(voiced) >= 2 else 0.0


def pitch_jitter(f0: np.ndarray) -> float:
    """
    Jitter (%): average absolute difference between consecutive pitch
    periods, normalized by the mean period.
    """
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    return float(100.0 * np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10))


def amplitude_shimmer(audio: np.ndarray, f0: np.ndarray) -> float:
    """
    Shimmer (%): cycle-to-cycle amplitude perturbation, approximated via
    peak-picking on the waveform envelope at the local pitch period.
    """
    voiced_idx = np.where(~np.isnan(f0))[0]
    if len(voiced_idx) < 3:
        return 0.0
    mean_f0 = np.nanmean(f0)
    if mean_f0 <= 0 or np.isnan(mean_f0):
        return 0.0
    min_dist = max(1, int(SAMPLE_RATE / mean_f0 * 0.7))
    peaks, _ = find_peaks(np.abs(audio), distance=min_dist)
    if len(peaks) < 3:
        return 0.0
    amps = np.abs(audio[peaks])
    return float(100.0 * np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-10))


def respiratory_pause_continuity(audio: np.ndarray, energy_threshold_db: float = -35.0) -> float:
    """
    Scores 0.0-1.0: how "naturally irregular" silence/pause gaps are.
    1.0 = highly irregular (natural), 0.0 = highly regular (synthetic-suspect).
    """
    frame = 160
    n_frames = len(audio) // frame
    if n_frames < 4:
        return 1.0

    rms = np.sqrt([np.mean(audio[i * frame : (i + 1) * frame] ** 2) for i in range(n_frames)])
    energies_db = 20 * np.log10(rms + 1e-10)
    is_silent = energies_db < energy_threshold_db

    runs, curr = [], 0
    for s in is_silent:
        if s:
            curr += 1
        elif curr > 0:
            runs.append(curr)
            curr = 0
    if curr > 0:
        runs.append(curr)

    if len(runs) < 2:
        return 1.0

    cv = np.std(runs) / (np.mean(runs) + 1e-10)
    return float(np.clip(cv / 1.5, 0.0, 1.0))


# =============================================================================
# Convenience: single call that returns everything server.py needs per stride
# =============================================================================

def extract_all_features(audio: np.ndarray) -> dict:
    """
    Runs the full Track A + Track B feature set on one 1.0s buffer.
    Returns a flat dict of scalars/arrays ready for fusion_engine.py.
    Caller MUST discard `audio` immediately after this call (Privacy Plane).
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    f0 = compute_f0_contour(audio)
    return {
        "lfcc": compute_lfcc(audio),
        "spectral_flatness": spectral_flatness(audio),
        "zero_crossing_rate": zero_crossing_rate(audio),
        "f0_variance": f0_contour_variance(f0),
        "jitter_pct": pitch_jitter(f0),
        "shimmer_pct": amplitude_shimmer(audio, f0),
        "pause_naturalness": respiratory_pause_continuity(audio),
        "voiced_ratio": float(np.mean(~np.isnan(f0))) if len(f0) else 0.0,
    }
