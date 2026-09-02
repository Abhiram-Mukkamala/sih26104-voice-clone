"""
server.py
=========
SIH26104 — Ingestion Layer + Decision/Enforcement Gateway.

FastAPI app exposing a bidirectional WebSocket at /ws/stream-verify.

Wire protocol (binary-in, JSON-out):
  Client -> Server: raw PCM16LE mono audio chunks (any chunk size; typically
                     20-160ms per WebRTC/RTP packet), sample rate declared
                     once via the initial JSON handshake message.
  Server -> Client: one JSON telemetry frame per 250ms stride (see
                     StrideResult in fusion_engine.py), plus out-of-band
                     liveness-challenge and MFA-trigger events.

Privacy Plane: the ring buffer is a fixed-size, in-process NumPy array.
After each stride's features are extracted, the *consumed* audio samples
are overwritten with zeros in-place before the buffer slides. Nothing is
written to disk. Only StrideResult telemetry (scores, flags, timestamps —
no waveform, no LFCC, no raw audio) is retained in SessionRiskState.history
for audit, and that dict is dropped entirely on WebSocket disconnect.
"""

from __future__ import annotations

import time
import uuid
import traceback
from dataclasses import asdict
from typing import Optional

import numpy as np

try:
    import audioop  # stdlib, removed in Python 3.13 (PEP 594)
    _AUDIOOP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AUDIOOP_AVAILABLE = False
    from scipy.signal import resample_poly
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from dsp_pipeline import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    extract_all_features,
)
from acoustic_model import AcousticModel
from fusion_engine import FusionEngine, SessionRiskState, RiskLevel, STRIDE_S, AMBIGUITY_LOW

app = FastAPI(title="SIH26104 Voice Clone Detection Gateway")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_S)  # 250ms @ 16kHz = 4000 samples

# Loaded once at process start. Pass a real path once a checkpoint exists;
# falls back to the explainable DSP heuristic otherwise (see acoustic_model.py).
acoustic_model = AcousticModel(model_path=None)
fusion_engine = FusionEngine()


class RingBuffer:
    """
    Fixed-length sliding buffer: 1.0s (16000 samples) window, advanced in
    250ms (4000 sample) strides. Handles arbitrary-sized incoming chunks
    (VoIP/WebRTC packets rarely align to the stride boundary).
    """

    def __init__(self, frame_samples: int = FRAME_SAMPLES, stride_samples: int = STRIDE_SAMPLES):
        self.frame_samples = frame_samples
        self.stride_samples = stride_samples
        self.buffer = np.zeros(frame_samples, dtype=np.float32)
        self.write_pos = 0          # samples written since last stride emission
        self.filled = 0             # total samples ever written, capped at frame_samples

    def push(self, samples: np.ndarray):
        """Append samples, sliding the window left when it overflows."""
        n = len(samples)
        if n >= self.frame_samples:
            self.buffer[:] = samples[-self.frame_samples:]
            self.filled = self.frame_samples
        else:
            self.buffer = np.concatenate([self.buffer[n:], samples])
            self.filled = min(self.filled + n, self.frame_samples)
        self.write_pos += n

    def strides_ready(self) -> int:
        ready = self.write_pos // self.stride_samples
        return ready

    def consume_stride(self) -> np.ndarray:
        """Return a *copy* of the current 1.0s window for feature extraction."""
        self.write_pos -= self.stride_samples
        return self.buffer.copy()

    def purge(self):
        """Privacy Plane: zero the buffer in place. Called on disconnect
        and after the final stride of a session."""
        self.buffer[:] = 0.0


def pcm16_bytes_to_float32(chunk: bytes, input_sample_rate: int) -> np.ndarray:
    """
    Convert raw PCM16LE bytes to a 16kHz mono float32 array in [-1, 1].
    Handles telecom-grade 8kHz G.711-derived PCM by resampling up; see
    Q&A #3 for why 8kHz codecs need explicit handling rather than being
    zero-padded.
    """
    if input_sample_rate != SAMPLE_RATE:
        if _AUDIOOP_AVAILABLE:
            chunk, _ = audioop.ratecv(chunk, 2, 1, input_sample_rate, SAMPLE_RATE, None)
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
        else:
            pcm16_in = np.frombuffer(chunk, dtype=np.int16)
            up = SAMPLE_RATE // np.gcd(SAMPLE_RATE, input_sample_rate)
            down = input_sample_rate // np.gcd(SAMPLE_RATE, input_sample_rate)
            resampled = resample_poly(pcm16_in.astype(np.float32), up, down)
            pcm16 = np.clip(resampled, -32768, 32767).astype(np.int16)
    else:
        pcm16 = np.frombuffer(chunk, dtype=np.int16)
    return (pcm16.astype(np.float32)) / 32768.0


def track_b_probability(features: dict) -> float:
    """
    Maps raw Track B DSP features into P(prosody anomaly) via a hand-
    calibrated logistic combination. Uses .get() for defensive access.
    """
    jitter = features.get("jitter_pct", 0.0)
    shimmer = features.get("shimmer_pct", 0.0)
    pause_naturalness = features.get("pause_naturalness", 1.0)
    voiced_ratio = features.get("voiced_ratio", 0.0)

    # Natural conversational jitter ~0.5-1.5%; shimmer ~3-8%.
    jitter_anomaly = 1.0 - np.exp(-((jitter - 1.0) ** 2) / (2 * 1.2 ** 2))
    shimmer_anomaly = 1.0 - np.exp(-((shimmer - 5.0) ** 2) / (2 * 4.0 ** 2))
    pause_anomaly = 1.0 - pause_naturalness

    if voiced_ratio < 0.05:
        # Almost entirely unvoiced/silent stride: not enough signal to
        # judge prosody — return neutral rather than a false anomaly.
        return 0.5

    score = 0.4 * jitter_anomaly + 0.3 * shimmer_anomaly + 0.3 * pause_anomaly
    return float(np.clip(score, 0.0, 1.0))


def generate_liveness_challenge() -> dict:
    """
    Stub for the out-of-band Dynamic Liveness Challenge. In production
    this calls an Indic TTS engine (e.g. Sarvam Bulbul) to synthesize a
    random phrase server-side, plays it to the caller, and compares the
    caller's *live* repeated utterance against fresh Track A/B scores.
    """
    import random

    challenge_bank = [
        "Please repeat after me: seven, blue, umbrella, forty-two.",
        "Please say your registered branch city and today's weekday.",
        "Please repeat: orange river, twelve stones, quiet morning.",
    ]
    return {
        "challenge_id": str(uuid.uuid4()),
        "prompt_text": random.choice(challenge_bank),
        "tts_engine": "sarvam-bulbul-indic-tts",
        "expires_in_ms": 8000,
    }


@app.websocket("/ws/stream-verify")
async def stream_verify(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    ring_buffer = RingBuffer()
    risk_state = SessionRiskState(session_id=session_id)
    input_sample_rate = SAMPLE_RATE  # updated by optional handshake message
    packet_loss_flag = False         # updated by handshake/control messages

    await websocket.send_json({
        "event": "session_started",
        "session_id": session_id,
        "using_model_fallback": acoustic_model.using_fallback,
    })

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                # Control-plane JSON messages: handshake / RTP-loss signaling.
                import json
                control = json.loads(message["text"])
                if control.get("type") == "handshake":
                    input_sample_rate = int(control.get("sample_rate", SAMPLE_RATE))
                elif control.get("type") == "rtp_status":
                    # Upstream RTP/WebRTC layer reports sequence-number gaps
                    # or PLC (packet loss concealment) activation for the
                    # samples about to be pushed — this is how we
                    # distinguish "network dropped it" from "vocoder made
                    # it" (see Q&A #1). Never inferred from the audio itself.
                    packet_loss_flag = bool(control.get("packet_loss", False))
                continue

            if "bytes" in message and message["bytes"] is not None:
                stride_start = time.perf_counter()
                float_chunk = pcm16_bytes_to_float32(message["bytes"], input_sample_rate)
                ring_buffer.push(float_chunk)

                while ring_buffer.strides_ready() > 0:
                    window = ring_buffer.consume_stride()
                    features = extract_all_features(window)

                    # Privacy Plane: zero consumed samples before freeing.
                    window[:] = 0.0
                    del window

                    p_vocoder = acoustic_model.predict_vocoder_probability(features["lfcc"])
                    p_prosody = track_b_probability(features)

                    result = fusion_engine.process_stride(
                        state=risk_state,
                        p_vocoder=p_vocoder,
                        p_prosody=p_prosody,
                        packet_loss=packet_loss_flag,
                        stride_start_perf_counter=stride_start,
                    )

                    payload = asdict(result)
                    payload["risk_level"] = result.risk_level.value
                    payload["session_id"] = session_id
                    payload["p_vocoder"] = round(p_vocoder, 4)
                    payload["p_prosody"] = round(p_prosody, 4)

                    await websocket.send_json({"event": "risk_update", **payload})

                    now_ms = time.time() * 1000.0

                    if risk_state.challenge_pending:
                        # Resolve an outstanding challenge using the *next*
                        # strides' fused scores rather than firing a fresh
                        # challenge on top of it.
                        if result.composite_ema < AMBIGUITY_LOW:
                            await websocket.send_json({
                                "event": "liveness_result",
                                "session_id": session_id,
                                "outcome": "passed",
                            })
                            risk_state.challenge_pending = False
                        elif now_ms >= risk_state.challenge_deadline_ms:
                            await websocket.send_json({
                                "event": "liveness_result",
                                "session_id": session_id,
                                "outcome": "failed_or_timed_out",
                            })
                            risk_state.challenge_pending = False
                            await websocket.send_json({
                                "event": "enforcement_action",
                                "session_id": session_id,
                                "action": "STEP_UP_MFA_AND_CBS_HOLD",
                                "risk_score": result.composite_ema,
                            })
                    elif result.liveness_challenge_required:
                        challenge = generate_liveness_challenge()
                        risk_state.challenge_pending = True
                        risk_state.challenge_deadline_ms = now_ms + challenge["expires_in_ms"]
                        await websocket.send_json({
                            "event": "liveness_challenge",
                            "session_id": session_id,
                            **challenge,
                        })

                    if result.mfa_required and not risk_state.challenge_pending:
                        await websocket.send_json({
                            "event": "enforcement_action",
                            "session_id": session_id,
                            "action": "STEP_UP_MFA_AND_CBS_HOLD",
                            "risk_score": result.composite_ema,
                        })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Surface the *actual* server-side crash instead of silently
        # closing the socket and leaving the client with a cryptic
        # ConnectionClosedOK (code 1000) error.
        print("\n" + "=" * 60)
        print(f"[SERVER ERROR] session={session_id}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("=" * 60 + "\n")
    finally:
        ring_buffer.purge()
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except RuntimeError:
            # Peer already sent/received a close frame between our state
            # check and the close() call (benign race on fast disconnects).
            pass


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the real-time dashboard."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "using_model_fallback": acoustic_model.using_fallback,
        "sample_rate": SAMPLE_RATE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
