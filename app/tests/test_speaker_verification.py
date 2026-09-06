"""
test_speaker_verification.py
==============================
Tests for Track C speaker identity verification (speaker_verification.py).

Validates:
  - Embedding dimension == 192 (both heavy and fallback paths).
  - Self-similarity ≈ 1.0 for identical audio.
  - Cross-signal similarity is measurably lower than self-similarity.
  - None returned for unenrolled users.
  - Pure FFT spectral-shape fallback works without speechbrain installed.
  - Enrollment metadata, reset(), enrolled_user_ids, model_type gating.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from speaker_verification import (
    SpeakerVerifier,
    EMBEDDING_DIM,
    _fft_spectral_embedding,
    _cosine,
    _resample_to_16k,
    _SPEECHBRAIN_AVAILABLE,
)


# ---- Helpers ----------------------------------------------------------------

def _tone(freq: float = 150.0, seconds: float = 1.0,
          sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    """Pure sine tone at *freq* Hz."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rich_tone(freq: float = 220.0, seconds: float = 1.0,
               sr: int = 16000) -> np.ndarray:
    """Tone with harmonics — spectrally distinct from a pure sine."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = (0.30 * np.sin(2 * np.pi * freq * t)
           + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
           + 0.10 * np.sin(2 * np.pi * 3 * freq * t)
           + 0.05 * np.sin(2 * np.pi * 5 * freq * t))
    return sig.astype(np.float32)


# ---- FFT spectral embedding ------------------------------------------------

class TestFFTSpectralEmbedding:
    """Validate the pure-numpy/scipy fallback embedding path."""

    def test_output_dimension_is_192(self):
        emb = _fft_spectral_embedding(_tone())
        assert emb.shape == (EMBEDDING_DIM,)
        assert emb.shape[0] == 192

    def test_output_dtype_float32(self):
        emb = _fft_spectral_embedding(_tone())
        assert emb.dtype == np.float32

    def test_l2_normalised(self):
        emb = _fft_spectral_embedding(_tone())
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_deterministic(self):
        audio = _tone()
        e1 = _fft_spectral_embedding(audio)
        e2 = _fft_spectral_embedding(audio)
        np.testing.assert_array_equal(e1, e2)

    def test_different_signals_differ(self):
        e1 = _fft_spectral_embedding(_tone(freq=100.0))
        e2 = _fft_spectral_embedding(_tone(freq=400.0))
        assert not np.allclose(e1, e2, atol=1e-3)

    def test_short_audio_does_not_crash(self):
        # Audio shorter than one window (400 samples)
        short = np.random.randn(200).astype(np.float32) * 0.1
        emb = _fft_spectral_embedding(short)
        assert emb.shape == (EMBEDDING_DIM,)
        assert np.isfinite(emb).all()

    def test_silence_does_not_crash(self):
        silence = np.zeros(16000, dtype=np.float32)
        emb = _fft_spectral_embedding(silence)
        assert emb.shape == (EMBEDDING_DIM,)
        assert np.isfinite(emb).all()


# ---- Cosine similarity helper -----------------------------------------------

class TestCosine:
    def test_identical_vectors_unity(self):
        v = np.random.randn(192).astype(np.float32)
        v /= np.linalg.norm(v)
        assert abs(_cosine(v, v) - 1.0) < 1e-5

    def test_orthogonal_zero(self):
        a = np.zeros(192, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(192, dtype=np.float32)
        b[1] = 1.0
        assert abs(_cosine(a, b)) < 1e-5

    def test_clipped_to_zero_one(self):
        a = np.ones(4, dtype=np.float32)
        b = -np.ones(4, dtype=np.float32)
        sim = _cosine(a, b)
        assert 0.0 <= sim <= 1.0


# ---- Resampling helper ------------------------------------------------------

class TestResample:
    def test_noop_at_16k(self):
        audio = _tone(sr=16000)
        out = _resample_to_16k(audio, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_upsample_from_8k(self):
        audio_8k = _tone(sr=8000)
        out = _resample_to_16k(audio_8k, 8000)
        expected_len = len(audio_8k) * 2
        assert abs(len(out) - expected_len) <= 1


# ---- SpeakerVerifier: core API -----------------------------------------------

class TestSpeakerVerifierFallback:
    """
    All tests in this class exercise the FFT spectral-shape fallback.
    They pass regardless of whether speechbrain is installed because
    they verify the fallback-path logic directly.
    """

    def test_using_fallback_property(self):
        sv = SpeakerVerifier()
        # In this test environment speechbrain is very likely absent.
        # Either way the property must be a bool.
        assert isinstance(sv.using_fallback, bool)

    def test_unenrolled_returns_none(self):
        sv = SpeakerVerifier()
        result = sv.similarity("nobody", _tone())
        assert result is None

    def test_enrolled_user_ids_empty(self):
        sv = SpeakerVerifier()
        assert sv.enrolled_user_ids == []

    def test_enroll_populates_ids(self):
        sv = SpeakerVerifier()
        sv.enroll("alice", _tone())
        assert "alice" in sv.enrolled_user_ids

    def test_enrollment_metadata(self):
        sv = SpeakerVerifier()
        sv.enroll("alice", _tone(seconds=2.0))
        meta = sv.enrollment("alice")
        assert meta is not None
        assert meta["embedding_dim"] == 192
        assert meta["model_type"] in ("fft-spectral", "ecapa-tdnn")
        assert meta["samples_seconds"] == pytest.approx(2.0, abs=0.1)

    def test_enrollment_none_for_unknown(self):
        sv = SpeakerVerifier()
        assert sv.enrollment("ghost") is None

    def test_self_similarity_near_unity(self):
        sv = SpeakerVerifier()
        audio = _tone(freq=150.0, seconds=1.0)
        sv.enroll("user_a", audio)
        sim = sv.similarity("user_a", audio)
        assert sim is not None
        assert sim > 0.95, f"Self-similarity should be ~1.0, got {sim}"

    def test_cross_signal_similarity_lower(self):
        sv = SpeakerVerifier()
        tone_a = _tone(freq=120.0, seconds=1.5)
        tone_b = _rich_tone(freq=350.0, seconds=1.5)
        sv.enroll("user_a", tone_a)
        self_sim = sv.similarity("user_a", tone_a)
        cross_sim = sv.similarity("user_a", tone_b)
        assert self_sim is not None
        assert cross_sim is not None
        assert self_sim > cross_sim, (
            f"Self-sim ({self_sim:.4f}) should exceed cross-sim ({cross_sim:.4f})"
        )

    def test_embedding_dimension_in_store(self):
        sv = SpeakerVerifier()
        sv.enroll("bob", _tone())
        record = sv._store["bob"]
        assert record["embedding"].shape == (192,)

    def test_embedding_l2_normalised_in_store(self):
        sv = SpeakerVerifier()
        sv.enroll("bob", _tone())
        emb = sv._store["bob"]["embedding"]
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_reset_clears_all(self):
        sv = SpeakerVerifier()
        sv.enroll("a", _tone(100))
        sv.enroll("b", _tone(200))
        assert len(sv.enrolled_user_ids) == 2
        sv.reset()
        assert sv.enrolled_user_ids == []
        assert sv.similarity("a", _tone(100)) is None

    def test_re_enroll_overwrites(self):
        sv = SpeakerVerifier()
        sv.enroll("user", _tone(100))
        emb1 = sv._store["user"]["embedding"].copy()
        sv.enroll("user", _tone(400))
        emb2 = sv._store["user"]["embedding"]
        assert not np.allclose(emb1, emb2, atol=1e-3)

    def test_enroll_with_8k_resamples(self):
        sv = SpeakerVerifier()
        audio_8k = _tone(freq=150.0, sr=8000)
        sv.enroll("user_8k", audio_8k, sample_rate=8000)
        meta = sv.enrollment("user_8k")
        assert meta is not None
        assert meta["embedding_dim"] == 192
        # After resampling 8k → 16k the duration should be approximately the same
        assert meta["samples_seconds"] == pytest.approx(1.0, abs=0.1)

    def test_multiple_users_independent(self):
        sv = SpeakerVerifier()
        sv.enroll("alice", _tone(120))
        sv.enroll("bob", _rich_tone(350))
        sim_alice = sv.similarity("alice", _tone(120))
        sim_bob = sv.similarity("bob", _rich_tone(350))
        assert sim_alice is not None
        assert sim_bob is not None
        assert sim_alice > 0.95
        assert sim_bob > 0.95


# ---- Model-type gating ------------------------------------------------------

class TestModelTypeGating:
    """
    Embeddings enrolled under one backend must not be compared against
    embeddings produced by a different backend.
    """

    def test_mismatched_model_type_returns_none(self):
        sv = SpeakerVerifier()
        sv.enroll("user", _tone())
        # Tamper with the stored model_type to simulate a backend switch.
        sv._store["user"]["model_type"] = "nonexistent-backend"
        result = sv.similarity("user", _tone())
        assert result is None


# ---- Pure FFT fallback isolation --------------------------------------------

class TestPureFFTFallback:
    """
    Directly validate that the FFT fallback works independently of
    whether speechbrain is installed.  These tests call the module-level
    ``_fft_spectral_embedding`` function — they never touch torch.
    """

    def test_dim_192(self):
        emb = _fft_spectral_embedding(_tone())
        assert emb.shape == (192,)

    def test_self_cosine_near_unity(self):
        audio = _rich_tone(freq=200.0)
        emb = _fft_spectral_embedding(audio)
        assert _cosine(emb, emb) > 0.999

    def test_different_spectra_distinguishable(self):
        emb_a = _fft_spectral_embedding(_tone(freq=100.0))
        emb_b = _fft_spectral_embedding(_rich_tone(freq=400.0))
        cross = _cosine(emb_a, emb_b)
        self_a = _cosine(emb_a, emb_a)
        assert self_a > cross

    def test_works_without_speechbrain(self):
        """This test always passes — it only exercises numpy/scipy code."""
        emb = _fft_spectral_embedding(np.random.randn(16000).astype(np.float32))
        assert emb.shape == (192,)
        assert np.isfinite(emb).all()
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
