"""
test_speaker_encoder.py
========================
Unit tests for the ECAPA-TDNN speaker encoder fallback (MFCC-centroid).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from speaker_encoder import SpeakerEncoder, _cosine_similarity, _mfcc_embedding


def _tone(freq: float = 150.0, seconds: float = 1.0, sr: int = 16000, amp: float = 0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _tone_with_harmonics(freq: float = 180.0, seconds: float = 1.0,
                         sr: int = 16000, amp: float = 0.3):
    """A different 'speaker' — different fundamental + harmonics."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = amp * np.sin(2 * np.pi * freq * t)
    sig += 0.15 * np.sin(2 * np.pi * 2 * freq * t)
    sig += 0.08 * np.sin(2 * np.pi * 3 * freq * t)
    return sig.astype(np.float32)


class TestMFCCEmbedding:
    def test_output_shape(self):
        emb = _mfcc_embedding(_tone())
        assert emb.ndim == 1
        assert emb.shape[0] == 20  # _N_MFCC

    def test_dtype(self):
        emb = _mfcc_embedding(_tone())
        assert emb.dtype == np.float32

    def test_deterministic(self):
        audio = _tone()
        e1 = _mfcc_embedding(audio)
        e2 = _mfcc_embedding(audio)
        np.testing.assert_array_equal(e1, e2)

    def test_different_signals_differ(self):
        e1 = _mfcc_embedding(_tone(freq=100.0))
        e2 = _mfcc_embedding(_tone(freq=300.0))
        assert not np.allclose(e1, e2, atol=1e-3)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_clamped_to_zero_one(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([-1.0, -2.0], dtype=np.float32)
        sim = _cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


class TestSpeakerEncoder:
    def test_using_fallback_without_model(self):
        enc = SpeakerEncoder(model_path=None)
        assert enc.using_fallback is True

    def test_not_enrolled_returns_none(self):
        enc = SpeakerEncoder()
        result = enc.compare("unknown_user", _tone())
        assert result is None

    def test_is_enrolled(self):
        enc = SpeakerEncoder()
        assert enc.is_enrolled("user1") is False
        enc.enroll("user1", [_tone()])
        assert enc.is_enrolled("user1") is True

    def test_enroll_and_compare_same_audio(self):
        enc = SpeakerEncoder()
        audio = _tone()
        enc.enroll("user1", [audio, audio, audio])
        sim = enc.compare("user1", audio)
        assert sim is not None
        # Same audio should produce very high similarity
        assert sim > 0.95

    def test_p_identity_formula(self):
        """p_identity = 1.0 - similarity; high sim → low identity risk."""
        enc = SpeakerEncoder()
        audio = _tone()
        enc.enroll("user1", [audio])
        sim = enc.compare("user1", audio)
        p_identity = 1.0 - sim
        assert 0.0 <= p_identity <= 1.0
        # For same audio, identity risk should be very low
        assert p_identity < 0.1

    def test_different_speakers_lower_similarity(self):
        enc = SpeakerEncoder()
        speaker_a = _tone(freq=120.0)
        speaker_b = _tone_with_harmonics(freq=250.0)
        enc.enroll("user_a", [speaker_a])
        sim = enc.compare("user_a", speaker_b)
        assert sim is not None
        # Different signal should have noticeably lower similarity
        # (the MFCC fallback won't perfectly separate, but the similarity
        # should be measurably lower than self-comparison)
        self_sim = enc.compare("user_a", speaker_a)
        assert self_sim > sim

    def test_purge_enrollment(self):
        enc = SpeakerEncoder()
        enc.enroll("user1", [_tone()])
        assert enc.is_enrolled("user1")
        enc.purge_enrollment("user1")
        assert not enc.is_enrolled("user1")
        assert enc.compare("user1", _tone()) is None

    def test_purge_all(self):
        enc = SpeakerEncoder()
        enc.enroll("a", [_tone(100)])
        enc.enroll("b", [_tone(200)])
        enc.purge_all()
        assert not enc.is_enrolled("a")
        assert not enc.is_enrolled("b")

    def test_enroll_empty_list_is_noop(self):
        enc = SpeakerEncoder()
        enc.enroll("user1", [])
        assert not enc.is_enrolled("user1")

    def test_enroll_multiple_windows(self):
        enc = SpeakerEncoder()
        windows = [_tone(freq=150 + i) for i in range(5)]
        enc.enroll("user1", windows)
        assert enc.is_enrolled("user1")
        sim = enc.compare("user1", _tone(freq=152))
        assert sim is not None
        assert sim > 0.5
