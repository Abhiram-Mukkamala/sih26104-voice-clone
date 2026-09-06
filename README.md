# SIH26104 — Real-Time Voice Clone Impersonation Detection

Streaming 3-track deep+biomechanical+identity fusion gateway. Zero-tolerance voice-clone detection for secure environments (banking calls, authentication systems). Pitch tracking uses NORMALIZED AUTOCORRELATION (RMS gate + ACF peak voicing + parabolic lag interpolation) — NOT librosa.pyin anywhere.

## Problem Statement + Threat Model + Design Goals

**Threat Model**: Adversarial voice cloning via modern TTS/vocoder systems (AASIST, RawNet3, WavLM-spoof, diffusion TTS). Attacker injects synthetic audio into a live call stream to impersonate a legitimate user.

**Design Goals**:
- **<200ms stride**: Real-time processing with 250ms analysis window for sub-second response
- **Zero retention**: In-memory only processing; audio samples overwritten with zeros immediately after feature extraction
- **In-memory voiceprints**: ECAPA-TDNN embeddings stored only in RAM, purged on session disconnect
- **Network resilience**: Packet-loss-aware with EMA freeze when PLC artifacts detected

---

## Architecture Specification

```
Audio Stream(WS) ─► Ring Buffer(1.0s, 250ms stride)
                      │
                 ┌────┴────┐
                 ▼         ▼
          Track A         Track B
     (Phase Anomalies)  (Autocorrelation Prosody)
   LFCC/phase_consistency   jitter/shimmer/pause
        p_vocoder         p_prosody
                 │         │
                 └────┬────┘
                      ▼
              Track C (Identity)
           ECAPA-TDNN identity :: FFT fallback
            p_identity = 1 - speaker_similarity
                      │
              ┌───────┴───────┐
              ▼               ▼
        3-way fusion     2-way fallback
      (0.45/0.25/0.30)   (0.6/0.4)
              │
        Decaying EMA
       (2.0s window)
              │
     ┌─────────────────┐
     ▼                 ▼
EMA-freeze logic   Decision Window
loss_ratio>0.35 ───► LOW_CONFIDENCE    (8 × 250ms strides)
     │                 │
     └─────────────────┘
              │
        ┌──────┼──────┐
        ▼      ▼      ▼
     ALLOW  CHALLENGE  BLOCK
                     │
             ┌───────┴───────┐
             ▼               ▼
      Mitigation Firewall  Telemetry
         (ALLOW<0.35,     Broadcast
       CHALLENGE<=0.70,  (risk_update,
        BLOCK>0.70)       mitigation_verdict,
                            enforcement_action)
             │
          Buffer Purge
          (in-memory zeroing)
```

**Track A (Phase Anomalies)**: LFCC + phase consistency detection via spectral analysis
**Track B (Autocorrelation Prosody)**: NORMALIZED AUTOCORRELATION pitch tracking (RMS gate + ACF peak voicing + parabolic lag interpolation) → jitter, shimmer, pause continuity analysis
**Track C (Identity)**: ECAPA-TDNN speaker verification with MFCC-centroid cosine-similarity fallback; in-memory voiceprint enrollment

**Fusion**: 3-way weighted (0.45/0.25/0.30) collapses to 2-way (0.6/0.4) when no voiceprint enrolled

**EMA Freeze Logic**: When packet_loss_ratio > 0.35, EMA holds last valid value (no decay toward noisy estimates)

**Decision Window**: Every 8 strides (2.0s) emits mitigation_verdict

**Mitigation Firewall**: ALLOW<0.35, CHALLENGE≤0.70, BLOCK>0.70 with LOW_CONFIDENCE override on high packet loss

---

## Wire Protocol

### Server → Client JSON Event Frames

1. **`session_started`** — Session initiation
   ```json
   {"event": "session_started", "session_id": "uuid-v4", "using_model_fallback": true}
   ```

2. **`risk_update`** — Per-stride (250ms) telemetry
   ```json
   {
     "event": "risk_update",
     "session_id": "uuid-v4",
     "timestamp_ms": 1700000000000,
     "risk_score": 0.2345,
     "composite_ema": 0.2345,
     "composite_raw": 0.2123,
     "risk_level": "SAFE",
     "speaker_similarity": 0.8765,
     "packet_loss": false,
     "packet_loss_ratio": 0.0,
     "processing_latency_ms": 45.2,
     "mitigation_action": "ALLOW",
     "decision_pending": true
   }
   ```

3. **`mitigation_verdict`** — Decision window result (every 2.0s)
   ```json
   {
     "event": "mitigation_verdict",
     "session_id": "uuid-v4",
     "window_index": 0,
     "mitigation_action": "ALLOW",
     "risk_score": 0.2876,
     "window_strides": 8,
     "packet_loss_ratio": 0.0,
     "speaker_similarity": 0.8765,
     "risk_level": "SAFE"
   }
   ```

4. **`liveness_challenge`** — Out-of-band step-up authentication
   ```json
   {
     "event": "liveness_challenge",
     "challenge_id": "uuid-v4",
     "prompt_text": "Please read the following number: 4829",
     "tts_engine": "stub",
     "expires_in_ms": 10000
   }
   ```

5. **`enforcement_action`** — Terminal action for BLOCK
   ```json
   {
     "event": "enforcement_action",
     "session_id": "uuid-v4",
     "action": "SEVER_SESSION",
     "reason": "BLOCK threshold exceeded",
     "risk_score": 0.8234
   }
   ```

### Client → Server Frames

- **Binary frames**: PCM16LE mono audio chunks (any size, typically 20-160ms per RTP packet)
- **Text frames**:
  ```json
  {"type": "handshake", "sample_rate": 16000, "user_id": "string"}
  {"type": "rtp_status", "packet_loss": true}
  ```

---

## Mitigation Semantics

| Action | Behavior | Response |
|--------|----------|----------|
| **ALLOW** | Continue normal processing | Stream continues, no intervention |
| **CHALLENGE** | Trigger out-of-band step-up auth + liveness challenge | Issues `liveness_challenge` frame with random prompt; requires user to respond with correct reading |
| **BLOCK** | Terminal security action | Emits `enforcement_action` with `action:"SEVER_SESSION"` followed by `websocket.close(code=1008)` |

**Low-Confidence Override**: `packet_loss_ratio > 0.35` forces CHALLENGE regardless of risk score

---

## Exact Local Execution

### Setup
```bash
cd /c/Projects/sih26104/sih26104
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR on Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Test Audio
```bash
python app/make_test_audio.py
```

### Start Server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
# OR:
python app/server.py
```

### Run Tests
```bash
pytest app/tests
```

### Stream with Client Simulator (No Identity Track)
```bash
python app/client_simulator.py sample_call.wav --loss 0.15
```

### Stream with Identity Track Enabled
```bash
python app/client_simulator.py sample_call.wav --loss 0.15 --enroll
```

The `--enroll` flag triggers `/enroll` endpoint to create an in-memory voiceprint, enabling Track C (ECAPA-TDNN identity) with 3-way fusion weights (0.45/0.25/0.30). Without `--enroll`, Track C is omitted and weights collapse to 2-way (0.6/0.4).

---

## Docker

### Build (Light - Default)
```bash
docker build -t sih26104 .
```

### Build (Heavy - with ECAPA-TDNN dependencies)
```bash
docker build -t sih26104 --build-arg INSTALL_HEAVY=true .
```

### Run
```bash
docker run -p 8000:8000 sih26104
```

---

## Privacy Plane + Security

### Privacy
- **Zero retention**: Audio processed in 1-second in-memory ring buffer
- **In-place zeroing**: Audio samples overwritten with zeros immediately after feature extraction
- **No disk writes**: Only telemetry (scores, timestamps, flags) retained in SessionRiskState.history
- **Memory-only voiceprints**: ECAPA-TDNN embeddings purged on WebSocket disconnect

### Security
- **Authentication**: JWT-based with `AUTH_SECRET` environment variable
- **Token expiry**: Configurable JWT expiration via standard FastAPI security
- **Secure transport**: WebSocket with authentication headers
- **Codec**: Pure NumPy/SciPy autocorrelation pitch tracking (no external dependencies)
- **No librosa**: NORMALIZED AUTOCORRELATION implementation eliminates librosa/numba dependency chain

### Configuration
- `AUTH_SECRET`: Required environment variable for JWT signing
- `INSTALL_HEAVY=true`: Docker build argument for ECAPA-TDNN runtime
- All thresholds and weights defined as constants in `fusion_engine.py`

---

## Implementation Notes

- **Track A**: ONNX INT8 model wrapper with explainable DSP-heuristic fallback (no checkpoint required)
- **Track B**: Pure NumPy/SciPy DSP — LFCC, spectral flatness, ZCR, NORMALIZED AUTOCORRELATION pitch contour (RMS gate + ACF peak voicing + parabolic lag interpolation), jitter, shimmer, pause continuity
- **Track C**: ECAPA-TDNN ONNX speaker encoder with MFCC-centroid cosine-similarity fallback
- **Fusion**: 3-way weighted sum, decaying EMA over 2.0s window, packet-loss-aware freeze logic
- **Decision**: 8-stride window with ALLOW/CHALLENGE/BLOCK mitigation firewall
- **Enforcement**: BLOCK triggers SEVER_SESSION + close(1008)
- **Telemetry**: All frames broadcast to connected WebSocket clients

All audio processing and risk computation occurs in-memory with zero retention beyond the active session.