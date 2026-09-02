# SIH26104 — Voice Clone Impersonation Detection Gateway

## Project Overview
This project is a real-time Voice Clone Impersonation Detection Gateway built for secure environments (like banking calls). It exposes a bidirectional WebSocket that ingests raw PCM16LE mono audio (simulating a VoIP/RTP stream) and emits real-time telemetry and risk assessments every 250ms. 

It operates using a dual-track detection strategy:
- **Track A (Acoustic/Deep Model):** Uses LFCC features fed into an INT8-quantized ONNX acoustic model (e.g., AASIST/RawNet3) to detect synthetic vocoder artifacts.
- **Track B (Biomechanical/Prosody):** Analyzes pitch (F0), jitter, shimmer, and respiratory pause continuity to detect unnatural speech patterns typical of TTS engines.
- **Fusion Engine:** Combines these scores using an Exponential Moving Average (EMA) and gracefully handles RTP packet loss (masking/freezing the EMA when Packet Loss Concealment artifacts are present).
- **Privacy Plane:** All audio is processed in a 1-second in-memory ring buffer. Audio samples are overwritten with zeroes immediately after feature extraction. Nothing is written to disk.

## Directory Structure
```text
C:\Projects\sih26104\sih26104
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies (numpy, scipy, fastapi, uvicorn, onnxruntime, websockets)
└── app/
    ├── server.py             # FastAPI WebSocket server and coordination layer
    ├── dsp_pipeline.py       # Pure NumPy/SciPy feature extraction (LFCC, F0, Jitter, Shimmer, etc.)
    ├── acoustic_model.py     # ONNX runtime wrapper (with a fallback DSP heuristic if no checkpoint exists)
    ├── fusion_engine.py      # Combines Track A & B probabilities into a unified EMA risk score
    ├── client_simulator.py   # CLI WebSocket client to simulate a VoIP caller streaming a WAV file
    ├── make_test_audio.py    # Utility to generate a synthetic test WAV file (sample_call.wav)
    ├── sample_call.wav       # Generated test audio file
    ├── static/
    │   └── index.html        # Real-time Web UI Dashboard (served at /)
    └── tests/                # Test suite (pytest)
        ├── test_dsp_pipeline.py 
        ├── test_fusion_engine.py
        └── test_server_e2e.py
```

## File Descriptions

### Core Pipeline (`app/`)
* **`server.py`**
  The entry point. Runs a FastAPI server with a WebSocket at `/ws/stream-verify`. Manages the `RingBuffer` (1s capacity, 250ms stride). Handles incoming binary audio and JSON control messages (`handshake`, `rtp_status`), orchestrates the DSP/Acoustic models, and pushes `risk_update`, `liveness_challenge`, and `enforcement_action` JSON events back to the client. Also serves the `index.html` dashboard at `/`.

* **`dsp_pipeline.py`**
  100% pure NumPy/SciPy digital signal processing engine (zero librosa/numba dependencies). Exposes `extract_all_features()` which computes Linear Frequency Cepstral Coefficients (LFCC) for the neural model, and extracts F0 contour (via autocorrelation), spectral flatness, ZCR, jitter, shimmer, and pause continuity for the biomechanical track.

* **`acoustic_model.py`**
  Wraps the ONNX runtime for the deep-learning model. Exposes `predict_vocoder_probability()`. If no `.onnx` checkpoint is provided, it seamlessly falls back to a deterministic, explainable DSP heuristic based on LFCC fine-detail variance so the system remains fully functional for testing.

* **`fusion_engine.py`**
  Contains `FusionEngine` and `SessionRiskState`. Computes a weighted sum of Track A and Track B probabilities, applies an EMA over a 3-second rolling window, and maps the score to risk levels (`SAFE`, `WARNING`, `CRITICAL`). Most importantly, if a stride is flagged as containing packet loss, it *freezes* the EMA to prevent network artifacts from artificially inflating the risk score.

### Frontend (`app/static/`)
* **`index.html`**
  A standalone, dependency-free dashboard. Connects to the WebSocket, allows users to stream their microphone or upload a WAV file, renders real-time waveforms on a canvas, and provides a live gauge and scrolling event log of the server's telemetry.

### Utilities and Tests
* **`make_test_audio.py`** & **`sample_call.wav`**
  Generates a 6-second synthetic 16kHz PCM audio file with wandering pitch and irregular volume gaps to give the DSP pipeline non-degenerate data to analyze.
* **`client_simulator.py`**
  A Python CLI equivalent to the dashboard. Connects to the WebSocket and streams a WAV file in real-time, printing the server's JSON responses to the terminal. Supports simulated packet loss via the `--loss` flag.
* **`tests/`**
  A comprehensive pytest suite validating the DSP math, fusion engine logic (especially the packet-loss freeze behavior), and a fully async end-to-end WebSocket integration test (`test_server_e2e.py`).

## Current Status
- The system is fully operational.
- The librosa dependency was recently removed in favor of a custom NumPy autocorrelation pitch extractor to resolve runtime crashes.
- Tests (both unit and full async E2E) are passing 100%.
- Server observability has been hardened with `traceback.print_exc()` inside the WebSocket loop.
