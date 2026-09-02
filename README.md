# SIH26104 — Real-Time Voice Clone Impersonation Detection

Streaming prototype for the 5-layer architecture: ingestion ring buffer →
dual-track forensic core (Track A deep model, Track B biomechanical DSP) →
temporal fusion/risk engine → decision gateway → zero-retention privacy plane.

## Files

| File | Layer | Responsibility |
|---|---|---|
| `app/dsp_pipeline.py` | Track A + B feature front-end | LFCC, mel-spectrogram, spectral flatness, ZCR, F0 contour, jitter, shimmer, pause continuity |
| `app/acoustic_model.py` | Track A | ONNX INT8 model wrapper with an explainable DSP-heuristic fallback (no checkpoint required to run) |
| `app/fusion_engine.py` | Layer 3 | Composite risk (0.6/0.4 weighting), packet-loss-aware EMA over a 3.0s window, threshold gates, liveness ambiguity band, challenge-response state |
| `app/server.py` | Layers 1, 4, 5 | FastAPI `/ws/stream-verify` WebSocket, ring buffer, G.711/8kHz resampling, liveness challenge issue + resolution, enforcement events, in-place buffer purge |
| `app/client_simulator.py` | demo | Streams a WAV file into the WebSocket like a live call, prints telemetry |
| `app/make_test_audio.py` | demo | Generates a synthetic WAV so the demo runs without a real call recording |
| `app/tests/` | tests | pytest coverage for the fusion engine and DSP feature extraction |

## Run it

```bash
pip install -r requirements.txt
cd app
python make_test_audio.py sample_call.wav   # only needed once, or supply your own WAV
uvicorn server:app --host 0.0.0.0 --port 8000
```

In a second terminal, for a live demo against any 16-bit PCM WAV file:

```bash
cd app
python client_simulator.py sample_call.wav --loss 0.1
```

Run the test suite with:

```bash
cd app
pytest tests/ -q
```

`--loss 0.1` randomly flags 10% of chunks as packet-loss-affected so the
jury can visually see the EMA freeze instead of spiking on network noise.

## Measured latency (this environment, CPU-only, DSP-heuristic fallback)

Cold first stride ~640ms (one-time librosa/JIT warmup inside the process).
Steady-state: **~41-44ms per 250ms stride** — roughly 6x inside the 250ms
budget, leaving headroom for a real ONNX AASIST/RawNet3 checkpoint once
trained (typical INT8 CPU inference for these architectures on a 1s clip:
15-40ms depending on hardware).

## Fixed since last pass

- **Liveness challenge was firing on every stride** while risk stayed in
  the ambiguity band (a fresh challenge every ~50-250ms). It now issues
  one challenge, holds state, and resolves it (`liveness_result: passed`
  or `failed_or_timed_out`) before another can be issued.
- **Server crash on client disconnect** (`RuntimeError: Cannot call
  "send" once a close message has been sent`) — a benign state-check race
  in the WebSocket teardown; now caught rather than propagating as an
  ASGI-level exception.
- **`client_simulator.py` stereo downmix was byte-sliced incorrectly**,
  producing garbage audio for any non-mono WAV — now reshapes to
  `(frames, n_channels)` and averages properly.
- Added `make_test_audio.py` so the documented demo command actually has
  a WAV to stream (none was shipped), and a `tests/` pytest suite for the
  fusion engine and DSP feature extraction (none existed before).

## What's a placeholder vs. production-ready

- **Track A model**: `acoustic_model.py` runs a transparent DSP heuristic
  when no `.onnx` checkpoint is supplied. Swap in a trained AASIST/RawNet3/
  WavLM-spoof INT8 model — the rest of the pipeline (fusion, server,
  WebSocket contract) does not change.
- **F0 extraction**: uses `librosa.pyin` to avoid a compiled dependency in
  this sandbox. Swap for PyWorld DIO/Harvest in production for lower
  per-stride latency (see `dsp_pipeline.compute_f0_contour` docstring).
- **Liveness TTS**: `generate_liveness_challenge()` is a stub; wire it to
  the actual Sarvam Bulbul (or equivalent Indic TTS) API.
- **RTP/WebRTC mirroring**: this prototype takes PCM over the WebSocket
  directly, standing in for the ingestion proxy — swap the transport, not
  the ring buffer or feature/fusion logic.
