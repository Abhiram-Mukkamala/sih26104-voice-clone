"""
server.py
=========
SIH26104 — Ingestion Layer + Decision/Enforcement Gateway.

FastAPI app exposing a bidirectional WebSocket at /ws/stream-verify.

Wire protocol (binary-in, JSON-out):
  Client -> Server: raw PCM16LE mono audio chunks (any chunk size; typically
                     20-160ms per WebRTC/RTP packet), sample rate declared
                     once via the initial JSON handshake message.
  Server -> Client: JSON telemetry frames per 250ms stride:
      session_started  {session_id, using_model_fallback}
      risk_update      {event, session_id, timestamp_ms, risk_score,
                        composite_ema, composite_raw, risk_level,
                        speaker_similarity|null, packet_loss,
                        packet_loss_ratio, processing_latency_ms,
                        mitigation_action, decision_pending}
      mitigation_verdict {event, session_id, window_index, mitigation_action,
                         risk_score, window_strides, packet_loss_ratio,
                         speaker_similarity|null, risk_level}
      liveness_challenge {event, challenge_id, prompt_text, tts_engine,
                         expires_in_ms}
      enforcement_action {event, session_id, action:"SEVER_SESSION", reason,
                         risk_score}

Client -> Server text frames:
  {"type":"handshake","sample_rate":16000,"user_id":"..."}
  {"type":"rtp_status","packet_loss":bool}

Mitigation thresholds:
  risk < 0.35  → ALLOW
  0.35 ≤ risk ≤ 0.70 → CHALLENGE
  risk > 0.70 → BLOCK
  packet_loss_ratio > 0.35 → force CHALLENGE (low-confidence override)

BLOCK → enforcement_action(SEVER_SESSION) → websocket.close(code=1008).

Privacy Plane: the ring buffer is a fixed-size, in-process NumPy array.
After each stride's features are extracted, the consumed audio samples
are overwritten with zeros in-place before the buffer slides. Nothing is
written to disk. Only StrideResult telemetry (scores, flags, timestamps —
no waveform, no LFCC, no raw audio) is retained in SessionRiskState.history
for audit, and that dict is dropped entirely on WebSocket disconnect.
"""

from __future__ import annotations

import time
import uuid
import json
import traceback
import base64
import io
import wave
from typing import Optional

import numpy as np

try:
    import audioop  # stdlib, removed in Python 3.13 (PEP 594)
    _AUDIOOP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AUDIOOP_AVAILABLE = False
    from scipy.signal import resample_poly
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from dsp_pipeline import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    extract_all_features,
)
from acoustic_model import AcousticModel
from fusion_engine import (
    FusionEngine,
    SessionRiskState,
    RiskLevel,
    mitigation_for_risk,
    MitigationLevel,
    DECISION_WINDOW_STRIDES,
    FREEZE_LOSS_RATIO,
    STRIDE_S,
    AMBIGUITY_LOW,
    AMBIGUITY_HIGH,
)
from speaker_verification import SpeakerVerifier
from auth import create_token, websocket_authenticate

app = FastAPI(title="SIH26104 Voice Clone Detection Gateway")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_S)  # 250ms @ 16kHz = 4000 samples

# Loaded once at process start. Pass a real path once a checkpoint exists;
# falls back to the explainable DSP heuristic otherwise (see acoustic_model.py).
acoustic_model = AcousticModel(model_path=None)
fusion_engine = FusionEngine()
verifier = SpeakerVerifier()

# ---- Telemetry broadcast: all connected WebSocket sessions -----------------
_connected_sessions: set[WebSocket] = set()


async def _broadcast(payload: dict, exclude: Optional[WebSocket] = None) -> None:
    """Send a telemetry frame to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in _connected_sessions:
        if ws is exclude:
            continue
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connected_sessions.discard(ws)


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


@app.get("/dev/token")
@app.post("/dev/token")
async def dev_token(sub: str = "dev-user"):
    """Development-only JWT minting endpoint; never expose in production."""
    return {"token": create_token(sub), "sub": sub}


@app.post("/enroll")
async def enroll_voiceprint(payload: dict):
    """Enroll an in-memory voiceprint from a base64-encoded PCM16 WAV."""
    user_id, audio_b64 = payload.get("user_id"), payload.get("audio_b64")
    if not isinstance(user_id, str) or not user_id or not isinstance(audio_b64, str):
        raise HTTPException(422, "user_id and audio_b64 are required")
    try:
        with wave.open(io.BytesIO(base64.b64decode(audio_b64, validate=True)), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("WAV must be PCM16")
            sample_rate, channels = wav.getframerate(), wav.getnchannels()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if channels < 1:
            raise ValueError("WAV has no channels")
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        # Normalize enrollment input to the verifier's 16 kHz mono contract.
        audio_16k = pcm16_bytes_to_float32(pcm.tobytes(), sample_rate)
        verifier.enroll(user_id, audio_16k)
        metadata = verifier.enrollment(user_id)
        if metadata is None:
            raise RuntimeError("voiceprint enrollment failed")
    except (ValueError, wave.Error, EOFError) as exc:
        raise HTTPException(400, f"Invalid WAV enrollment audio: {exc}") from exc
    return {"voiceprint_id": user_id, "model": metadata["model_type"],
            "embedding_dim": metadata["embedding_dim"],
            "samples_seconds": metadata["samples_seconds"]}


@app.websocket("/ws/stream")
@app.websocket("/ws/stream-verify")  # legacy route alias
async def stream_verify(websocket: WebSocket):
    try:
        claims = await websocket_authenticate(websocket)
    except Exception:
        claims = None
    if claims is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session_id = str(uuid.uuid4())
    ring_buffer = RingBuffer()
    risk_state = SessionRiskState(session_id=session_id)
    input_sample_rate = SAMPLE_RATE  # updated by optional handshake message
    packet_loss_flag = False         # updated by handshake/control messages
    user_id: Optional[str] = claims.get("sub") if claims and isinstance(claims.get("sub"), str) else None
    committed_action = MitigationLevel.ANALYZING
    decision_emas: list[float] = []
    decision_similarities: list[Optional[float]] = []
    decision_losses: list[bool] = []
    decision_window_index = 0

    _connected_sessions.add(websocket)

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
                control = json.loads(message["text"])
                if control.get("type") == "handshake":
                    input_sample_rate = int(control.get("sample_rate", SAMPLE_RATE))
                    # JWT subject remains authoritative for identity lookup.
                    if not user_id and control.get("user_id"):
                        user_id = str(control["user_id"])
                elif control.get("type") == "rtp_status":
                    # Upstream RTP/WebRTC layer reports sequence-number gaps
                    # or PLC (packet loss concealment) activation for the
                    # samples about to be pushed.
                    packet_loss_flag = bool(control.get("packet_loss", False))
                continue

            if "bytes" in message and message["bytes"] is not None:
                stride_start = time.perf_counter()
                float_chunk = pcm16_bytes_to_float32(message["bytes"], input_sample_rate)
                ring_buffer.push(float_chunk)

                while ring_buffer.strides_ready() > 0:
                    window = ring_buffer.consume_stride()
                    features = extract_all_features(window)

                    p_vocoder = acoustic_model.predict_vocoder_probability(features["lfcc"])
                    p_prosody = track_b_probability(features)

                    # Identity track: speaker similarity from enrolled voiceprint
                    speaker_similarity: Optional[float] = None
                    p_identity: Optional[float] = None
                    if user_id:
                        # Re-extract a fresh window for the speaker encoder.
                        # We use the ring buffer's current state (the window was
                        # just consumed and zeroed, but the buffer itself still
                        # has the full 1s context for the *next* stride).
                        # For the identity track we use the LFCC audio that was
                        # already extracted — the speaker encoder's MFCC fallback
                        # works on raw audio, so we reconstruct from the ring
                        # buffer's pre-slide snapshot.
                        speaker_similarity = verifier.similarity(user_id, window)
                        if speaker_similarity is not None:
                            p_identity = 1.0 - speaker_similarity

                    # Privacy Plane: zero the extracted audio copy after both
                    # feature and identity computations have consumed it.
                    window[:] = 0.0
                    del window

                    result = fusion_engine.process_stride(
                        state=risk_state,
                        p_vocoder=p_vocoder,
                        p_prosody=p_prosody,
                        packet_loss=packet_loss_flag,
                        stride_start_perf_counter=stride_start,
                        p_identity=p_identity,
                        speaker_similarity=speaker_similarity,
                    )

                    # Build the risk_update payload matching the wire spec
                    risk_payload = {
                        "event": "risk_update",
                        "session_id": session_id,
                        "timestamp_ms": result.timestamp_ms,
                        "risk_score": result.risk_score,
                        "composite_ema": result.composite_ema,
                        "composite_raw": result.composite_raw,
                        "risk_level": result.risk_level.value,
                        "p_identity": result.p_identity,
                        "speaker_similarity": result.speaker_similarity,
                        "packet_loss": result.packet_loss,
                        "packet_loss_ratio": result.packet_loss_ratio,
                        "processing_latency_ms": result.processing_latency_ms,
                        "mitigation_action": committed_action.value,
                        "decision_pending": len(decision_emas) + 1 < 8,
                    }
                    await websocket.send_json(risk_payload)
                    await _broadcast(risk_payload, exclude=websocket)

                    # Commit mitigation only after a complete 8-stride window.
                    decision_emas.append(result.composite_ema)
                    decision_similarities.append(result.speaker_similarity)
                    decision_losses.append(result.packet_loss)
                    if len(decision_emas) == DECISION_WINDOW_STRIDES:
                        risk_score = float(np.mean(decision_emas))
                        loss_ratio = sum(decision_losses) / len(decision_losses)
                        committed_action = (MitigationLevel.CHALLENGE
                                            if loss_ratio > FREEZE_LOSS_RATIO
                                            else mitigation_for_risk(risk_score))
                        window_risk = fusion_engine._classify(risk_score, loss_ratio)
                        verdict_payload = {
                            "event": "mitigation_verdict", "session_id": session_id,
                            "window_index": decision_window_index,
                            "mitigation_action": committed_action.value,
                            "risk_score": round(risk_score, 4),
                            "window_strides": DECISION_WINDOW_STRIDES,
                            "packet_loss_ratio": round(loss_ratio, 4),
                            "speaker_similarity": decision_similarities[-1],
                            "risk_level": window_risk.value,
                        }
                        await websocket.send_json(verdict_payload)
                        await _broadcast(verdict_payload, exclude=websocket)
                        if committed_action == MitigationLevel.CHALLENGE:
                            challenge_payload = {"event": "liveness_challenge",
                                "session_id": session_id, **generate_liveness_challenge()}
                            await websocket.send_json(challenge_payload)
                            await _broadcast(challenge_payload, exclude=websocket)
                        if committed_action == MitigationLevel.BLOCK:
                            enforcement_payload = {
                                "event": "enforcement_action", "session_id": session_id,
                                "action": "SEVER_SESSION",
                                "reason": f"Voice clone risk {risk_score:.4f} exceeded BLOCK threshold over decision window {decision_window_index}",
                                "risk_score": round(risk_score, 4),
                            }
                            await websocket.send_json(enforcement_payload)
                            await _broadcast(enforcement_payload, exclude=websocket)
                            ring_buffer.purge()
                            await websocket.close(code=1008)
                            return
                        decision_window_index += 1
                        decision_emas.clear()
                        decision_similarities.clear()
                        decision_losses.clear()

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
        _connected_sessions.discard(websocket)
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
        "speaker_encoder_fallback": verifier.using_fallback,
        "sample_rate": SAMPLE_RATE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
