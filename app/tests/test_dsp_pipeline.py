import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dsp_pipeline as dsp


def tone(freq=150.0, seconds=1.0, sr=dsp.SAMPLE_RATE, amp=0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_lfcc_shape():
    audio = tone()
    lfcc = dsp.compute_lfcc(audio)
    assert lfcc.shape[1] == dsp.N_LFCC
    assert lfcc.shape[0] > 0


def test_spectral_flatness_bounds():
    audio = tone()
    flat = dsp.spectral_flatness(audio)
    assert 0.0 <= flat <= 1.0


def test_zcr_bounds():
    audio = tone()
    zcr = dsp.zero_crossing_rate(audio)
    assert 0.0 <= zcr <= 1.0


def test_silence_does_not_crash_full_pipeline():
    audio = np.zeros(dsp.FRAME_SAMPLES, dtype=np.float32)
    feats = dsp.extract_all_features(audio)
    assert feats["voiced_ratio"] == 0.0
    assert feats["lfcc"].shape[1] == dsp.N_LFCC


def test_extract_all_features_keys():
    audio = tone()
    feats = dsp.extract_all_features(audio)
    expected = {
        "lfcc", "spectral_flatness", "zero_crossing_rate", "f0_variance",
        "jitter_pct", "shimmer_pct", "pause_naturalness", "voiced_ratio",
    }
    assert expected.issubset(feats.keys())
