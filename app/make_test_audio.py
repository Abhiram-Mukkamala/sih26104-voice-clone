"""
make_test_audio.py
===================
Generates a synthetic 16-bit PCM mono WAV so `client_simulator.py` has
something to stream without requiring a real call recording. Not a voice
clone detector test fixture in the ML sense — just a wobbly-pitch tone
with a few silence gaps so pause-continuity / F0 / jitter features have
non-degenerate input.

Usage:
    python make_test_audio.py [out.wav] [--seconds 6]
"""
from __future__ import annotations

import argparse
import wave

import numpy as np


def make_wav(path: str, seconds: float, sr: int = 16000, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)

    # Wandering F0 around 150Hz with slow vibrato + slow random drift,
    # so jitter/shimmer/F0-variance features see plausible human-ish values.
    f0 = 150 + 10 * np.sin(2 * np.pi * 0.5 * t) + np.cumsum(rng.normal(0, 3, len(t))) * 0.01
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = 0.3 * np.sin(phase) + 0.05 * np.sin(2 * phase) + 0.02 * rng.standard_normal(len(t))

    # A few breathing-pause gaps at irregular offsets.
    for start in np.linspace(1.0, seconds - 1.0, num=max(1, int(seconds // 2))):
        s, e = int(start * sr), int(start * sr) + int(0.15 * sr)
        signal[s:e] *= 0.02

    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", nargs="?", default="sample_call.wav")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    make_wav(args.out_path, args.seconds)
    print(f"wrote {args.out_path} ({args.seconds}s @ 16kHz mono)")
