"""
speaker_verification.py
========================
SIH26104 — Track C: Speaker Identity Verification.

Produces 192-dimensional L2-normalised speaker embeddings and compares
live audio against enrolled voiceprints.  Two embedding backends:

Heavy path (optional):
    SpeechBrain ``EncoderClassifier`` with the pretrained
    ``speechbrain/spkrec-ecapa-voxceleb`` checkpoint (192-d ECAPA-TDNN).
    Both the ``import`` and the model download are wrapped in try/except
    so the system degrades gracefully when torch, torchaudio, or
    speechbrain are absent or when the network is unavailable.

Fallback path (always available):
    Pure NumPy/SciPy FFT spectral-shape embedding.  Per-frame
    log-magnitude spectrum → 96 log-spaced band energies (60–7600 Hz) →
    temporal mean profile (96) concatenated with temporal delta profile
    (96) = 192-d, L2-normalised.  Deterministic, reuses the STFT params
    from ``dsp_pipeline`` (N_FFT=512, HOP=160, SR=16 kHz).

Privacy Plane: all enrollments are held in-memory only (dict of user_id →
embedding metadata).  No audio is ever persisted to disk.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.signal import get_window, resample_poly

# Reuse STFT geometry from the shared DSP pipeline.
from dsp_pipeline import SAMPLE_RATE as _TARGET_SR, N_FFT, HOP_LENGTH

# ---- Constants --------------------------------------------------------------

EMBEDDING_DIM = 192
_N_BANDS = 96          # half of EMBEDDING_DIM: mean(96) + delta(96) = 192
_BAND_LO_HZ = 60.0
_BAND_HI_HZ = 7600.0
_WIN_LENGTH = 400      # 25 ms @ 16 kHz (matches dsp_pipeline.WIN_LENGTH)

# ---- SpeechBrain heavy path (optional) -------------------------------------

_SPEECHBRAIN_AVAILABLE = False
_sb_model = None

try:
    import torch                    # noqa: F401
    import torchaudio               # noqa: F401
    from speechbrain.inference.speaker import EncoderClassifier

    try:
        _sb_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        _SPEECHBRAIN_AVAILABLE = True
    except Exception:
        # Network down, model cache missing, CUDA mismatch, etc.
        _sb_model = None
except Exception:
    # torch / torchaudio / speechbrain not installed at all.
    pass


# ---- FFT spectral-shape fallback -------------------------------------------

def _log_band_edges(n_bands: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Return *n_bands + 1* edges log-spaced between *lo_hz* and *hi_hz*."""
    return np.logspace(np.log10(lo_hz), np.log10(hi_hz), n_bands + 1)


_BAND_EDGES = _log_band_edges(_N_BANDS, _BAND_LO_HZ, _BAND_HI_HZ)


def _fft_spectral_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Compute a 192-d spectral-shape embedding from a float32 16 kHz signal.

    1. Frame the signal (N_FFT=512, HOP=160, Hann window).
    2. Compute the log-magnitude spectrum per frame.
    3. Aggregate into 96 log-spaced frequency bands (60–7600 Hz).
    4. Take the temporal mean (96-d) and temporal first-order delta (96-d).
    5. Concatenate → 192-d, then L2-normalise.

    Deterministic and dependency-free beyond numpy/scipy.
    """
    window = get_window("hann", _WIN_LENGTH)
    n_frames = 1 + (len(audio) - _WIN_LENGTH) // HOP_LENGTH
    if n_frames < 1:
        audio = np.pad(audio, (0, _WIN_LENGTH - len(audio)))
        n_frames = 1

    # STFT magnitude (n_frames, n_bins)
    frames = np.stack(
        [audio[i * HOP_LENGTH: i * HOP_LENGTH + _WIN_LENGTH] * window
         for i in range(n_frames)]
    )
    mag = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) + 1e-10
    log_mag = np.log(mag)  # log-magnitude spectrum

    # Map FFT bins → 96 log-spaced bands
    n_bins = N_FFT // 2 + 1
    bin_freqs = np.linspace(0, _TARGET_SR / 2, n_bins)
    band_energies = np.zeros((n_frames, _N_BANDS), dtype=np.float64)

    for b in range(_N_BANDS):
        lo, hi = _BAND_EDGES[b], _BAND_EDGES[b + 1]
        mask = (bin_freqs >= lo) & (bin_freqs < hi)
        if mask.any():
            band_energies[:, b] = log_mag[:, mask].mean(axis=1)
        else:
            # Band narrower than one bin — use nearest bin.
            nearest = np.argmin(np.abs(bin_freqs - (lo + hi) / 2))
            band_energies[:, b] = log_mag[:, nearest]

    # Temporal mean profile (96-d)
    mean_profile = band_energies.mean(axis=0)

    # Temporal delta profile (96-d): mean of first-order differences.
    if n_frames >= 2:
        deltas = np.diff(band_energies, axis=0)
        delta_profile = deltas.mean(axis=0)
    else:
        delta_profile = np.zeros(_N_BANDS, dtype=np.float64)

    embedding = np.concatenate([mean_profile, delta_profile]).astype(np.float32)

    # L2-normalise
    norm = np.linalg.norm(embedding) + 1e-10
    return embedding / norm


def _speechbrain_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Produce a 192-d L2-normalised embedding via the SpeechBrain ECAPA-TDNN.
    Caller must ensure ``_SPEECHBRAIN_AVAILABLE`` is True before calling.
    """
    import torch  # already checked at module level

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        emb = _sb_model.encode_batch(waveform).squeeze().cpu().numpy()

    emb = emb.astype(np.float32).ravel()

    # Guarantee exactly 192 dimensions (pad or truncate defensively).
    if emb.shape[0] < EMBEDDING_DIM:
        emb = np.pad(emb, (0, EMBEDDING_DIM - emb.shape[0]))
    elif emb.shape[0] > EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]

    norm = np.linalg.norm(emb) + 1e-10
    return emb / norm


# ---- Resampling helper -----------------------------------------------------

def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample *audio* from *sr* to 16 kHz using polyphase filtering."""
    if sr == _TARGET_SR:
        return audio
    gcd = np.gcd(_TARGET_SR, sr)
    up = _TARGET_SR // gcd
    down = sr // gcd
    resampled = resample_poly(audio.astype(np.float64), up, down)
    return resampled.astype(np.float32)


# ---- Cosine similarity -----------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clipped to [0, 1]."""
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return max(0.0, min(1.0, dot / denom))


# ---- Public API -------------------------------------------------------------

class SpeakerVerifier:
    """
    In-memory speaker identity verifier (Track C).

    Enroll a reference voiceprint, then compare live audio against it.
    Embeddings are always 192-d and stored L2-normalised.  The verifier
    transparently selects the SpeechBrain ECAPA-TDNN heavy path when
    available, otherwise falls back to the deterministic FFT spectral-
    shape embedding.  Embeddings from different backends are **never**
    cross-compared — ``similarity()`` returns ``None`` if the enrolled
    embedding was produced by a different ``model_type`` than the one
    currently active.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    # -- Properties -----------------------------------------------------------

    @property
    def using_fallback(self) -> bool:
        """True when the FFT spectral-shape fallback is active."""
        return not _SPEECHBRAIN_AVAILABLE

    @property
    def enrolled_user_ids(self) -> list[str]:
        """List of currently enrolled user IDs."""
        return list(self._store.keys())

    # -- Core methods ---------------------------------------------------------

    def _current_model_type(self) -> str:
        return "ecapa-tdnn" if _SPEECHBRAIN_AVAILABLE else "fft-spectral"

    def _embed(self, audio_16k: np.ndarray) -> np.ndarray:
        """Produce a 192-d L2-normalised embedding from 16 kHz float32."""
        if _SPEECHBRAIN_AVAILABLE:
            return _speechbrain_embedding(audio_16k)
        return _fft_spectral_embedding(audio_16k)

    def enroll(self, user_id: str, audio_f32: np.ndarray,
               sample_rate: int = _TARGET_SR) -> None:
        """
        Enroll a speaker by computing and storing a single 192-d embedding.

        Parameters
        ----------
        user_id : str
            Unique identifier for this speaker.
        audio_f32 : np.ndarray
            Float32 audio waveform (mono).  Resampled to 16 kHz internally
            if *sample_rate* differs.
        sample_rate : int
            Sample rate of *audio_f32*.
        """
        audio_16k = _resample_to_16k(audio_f32.astype(np.float32), sample_rate)
        embedding = self._embed(audio_16k)

        self._store[user_id] = {
            "embedding": embedding,
            "model_type": self._current_model_type(),
            "created_ms": time.time() * 1000.0,
            "samples_seconds": len(audio_16k) / _TARGET_SR,
        }

    def similarity(self, user_id: str,
                   audio: np.ndarray) -> Optional[float]:
        """
        Cosine similarity between enrolled voiceprint and live audio.

        Returns
        -------
        float in [0, 1] — cosine similarity.
        None — if *user_id* is not enrolled or the enrolled embedding was
               produced by a different backend (model_type mismatch).
        """
        record = self._store.get(user_id)
        if record is None:
            return None
        if record["model_type"] != self._current_model_type():
            return None

        audio_16k = _resample_to_16k(audio.astype(np.float32), _TARGET_SR)
        live_emb = self._embed(audio_16k)
        return _cosine(record["embedding"], live_emb)

    def enrollment(self, user_id: str) -> Optional[dict]:
        """
        Return metadata for an enrolled user, or None.

        Returns a dict with ``model_type``, ``embedding_dim``, and
        ``samples_seconds``.
        """
        record = self._store.get(user_id)
        if record is None:
            return None
        return {
            "model_type": record["model_type"],
            "embedding_dim": EMBEDDING_DIM,
            "samples_seconds": record["samples_seconds"],
        }

    def reset(self) -> None:
        """Clear all enrollments from memory."""
        self._store.clear()
