"""
speaker_encoder.py
===================
SIH26104 — Identity Track: ECAPA-TDNN Speaker Verification.

Wraps an ONNX-exported ECAPA-TDNN model that produces a fixed-dimension
speaker embedding from a 1-second audio window.  When no checkpoint is
available, falls back to a deterministic MFCC-centroid cosine-similarity
heuristic so the rest of the pipeline remains fully functional.

Enrollment is in-memory only (dict of user_id → mean embedding) — nothing
is written to disk, consistent with the Privacy Plane.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.fft import dct

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False


# ---------- DSP constants for the MFCC fallback -----------------------------
_SR = 16000
_N_FFT = 512
_HOP = 160
_WIN = 400
_N_MELS = 40
_N_MFCC = 20


def _mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    """Construct a Mel-scale triangular filterbank (HTK formula)."""
    n_bins = n_fft // 2 + 1
    f_min, f_max = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_indices[m - 1]
        f_center = bin_indices[m]
        f_right = bin_indices[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


_MEL_FILTERBANK = _mel_filterbank(_N_FFT, _SR, _N_MELS)


def _mfcc_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Compute a fixed-dimension MFCC-centroid embedding for speaker
    comparison.  This is a lightweight stand-in for ECAPA-TDNN that
    captures gross spectral-envelope identity without a trained model.

    Returns a 1-D float32 vector of length _N_MFCC.
    """
    from scipy.signal import get_window

    audio = audio.astype(np.float32)
    window = get_window("hann", _WIN)
    n_frames = 1 + (len(audio) - _WIN) // _HOP
    if n_frames < 1:
        audio = np.pad(audio, (0, _WIN - len(audio)))
        n_frames = 1

    frames = np.stack(
        [audio[i * _HOP: i * _HOP + _WIN] * window for i in range(n_frames)]
    )
    mag = np.abs(np.fft.rfft(frames, n=_N_FFT, axis=1)) + 1e-10
    mel_spec = mag @ _MEL_FILTERBANK.T
    log_mel = np.log(mel_spec + 1e-10)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :_N_MFCC]

    # Centroid = temporal mean across all frames → a single embedding vector
    return mfcc.mean(axis=0).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1], clamped to [0, 1] for our use case."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return max(0.0, min(1.0, dot / norm))


class SpeakerEncoder:
    """
    Produces a fixed-dimension speaker embedding from a 1-second audio window.

    With a real ECAPA-TDNN ONNX checkpoint:
        embedding = model(audio)  → 192-dim vector (typical)
    Without (fallback):
        embedding = MFCC centroid → _N_MFCC-dim vector

    Enrollment stores mean embeddings per user_id in memory only.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self.input_name: Optional[str] = None
        self._enrollment_store: dict[str, np.ndarray] = {}

        if model_path and _ORT_AVAILABLE and os.path.exists(model_path):
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

    @property
    def using_fallback(self) -> bool:
        return self.session is None

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        """Compute a speaker embedding for a single audio window."""
        if self.session is not None:
            # ECAPA-TDNN expects (batch, samples) float32 at 16 kHz
            batch = audio[np.newaxis, :].astype(np.float32)
            outputs = self.session.run(None, {self.input_name: batch})
            return outputs[0].squeeze().astype(np.float32)
        return _mfcc_embedding(audio)

    def enroll(self, user_id: str, audio_windows: list[np.ndarray]) -> None:
        """
        Enroll a speaker by averaging embeddings over multiple windows.
        Call with 3-5 representative 1-second windows for best results.
        Audio is not stored — only the resulting embedding vector.
        """
        if not audio_windows:
            return
        embeddings = [self._embed(w) for w in audio_windows]
        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        # Normalise for cosine similarity
        norm = np.linalg.norm(mean_emb) + 1e-10
        self._enrollment_store[user_id] = mean_emb / norm

    def is_enrolled(self, user_id: str) -> bool:
        return user_id in self._enrollment_store

    def compare(self, user_id: str, audio: np.ndarray) -> Optional[float]:
        """
        Compare live audio against an enrolled voiceprint.
        Returns cosine similarity in [0, 1], or None if user is not enrolled.
        """
        if user_id not in self._enrollment_store:
            return None
        live_emb = self._embed(audio)
        norm = np.linalg.norm(live_emb) + 1e-10
        live_emb = live_emb / norm
        return _cosine_similarity(self._enrollment_store[user_id], live_emb)

    def purge_enrollment(self, user_id: str) -> None:
        """Remove an enrollment. Privacy Plane: caller decides when."""
        self._enrollment_store.pop(user_id, None)

    def purge_all(self) -> None:
        """Remove all enrollments."""
        self._enrollment_store.clear()
