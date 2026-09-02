"""
test_server_e2e.py
===================
End-to-end WebSocket integration tests against a live server at
localhost:8000.  Validates the full pipeline: handshake → audio stream →
risk_update events → clean disconnect.

Requires the server to be running: python server.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
import wave
import numpy as np
import websockets
import pytest

WS_URL = "ws://localhost:8000/ws/stream-verify"
HEALTHZ_URL = "http://localhost:8000/healthz"


def _make_pcm16_chunk(freq=150.0, duration_s=0.2, sr=16000, amp=0.3):
    """Generate a short PCM16LE mono chunk."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = amp * np.sin(2 * np.pi * freq * t)
    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    return pcm16.tobytes(), sr


# ---- Test 1: Health endpoint ------------------------------------------------

def test_healthz():
    """GET /healthz returns 200 with expected keys."""
    import urllib.request
    resp = urllib.request.urlopen(HEALTHZ_URL)
    assert resp.status == 200
    data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert "using_model_fallback" in data
    assert data["sample_rate"] == 16000


# ---- Test 2: Root serves dashboard ------------------------------------------

def test_root_serves_html():
    """GET / returns 200 with HTML content."""
    import urllib.request
    resp = urllib.request.urlopen("http://localhost:8000/")
    assert resp.status == 200
    html = resp.read().decode()
    assert "SIH26104" in html
    assert "Voice Clone Detection" in html


# ---- Test 3: WebSocket session lifecycle -------------------------------------

@pytest.mark.asyncio
async def test_ws_session_started():
    """Connect and receive session_started event."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        msg = json.loads(await ws.recv())
        assert msg["event"] == "session_started"
        assert "session_id" in msg
        assert "using_model_fallback" in msg


# ---- Test 4: Handshake + single audio chunk → risk_update --------------------

@pytest.mark.asyncio
async def test_ws_handshake_and_risk_update():
    """Send handshake + enough audio to trigger at least one stride → get risk_update."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        # Receive session_started
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        # Send handshake
        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        # Send enough audio chunks to fill a stride (250ms = 4000 samples).
        # Each chunk is 200ms = 3200 samples, so 2 chunks = 6400 samples → 1 stride.
        chunk_bytes, sr = _make_pcm16_chunk(duration_s=0.25)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
        await ws.send(chunk_bytes)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
        await ws.send(chunk_bytes)

        # Collect events until we get a risk_update (with timeout)
        risk_update = None
        for _ in range(20):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                evt = json.loads(raw)
                if evt["event"] == "risk_update":
                    risk_update = evt
                    break
            except asyncio.TimeoutError:
                break

        assert risk_update is not None, "Never received a risk_update event"
        assert "composite_ema" in risk_update
        assert "composite_raw" in risk_update
        assert "risk_level" in risk_update
        assert risk_update["risk_level"] in ("SAFE", "WARNING", "CRITICAL", "LOW_CONFIDENCE")
        assert "p_vocoder" in risk_update
        assert "p_prosody" in risk_update
        assert "latency_ms" in risk_update
        assert risk_update["latency_ms"] > 0
        assert "session_id" in risk_update


# ---- Test 5: Multiple strides produce increasing stride count ----------------

@pytest.mark.asyncio
async def test_ws_multiple_strides():
    """Stream enough audio for multiple strides and verify we get multiple risk_updates."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        # Send 1.5s of audio (6 strides worth at 250ms each)
        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        for _ in range(6):
            await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
            await ws.send(chunk_bytes)

        risk_count = 0
        for _ in range(30):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                evt = json.loads(raw)
                if evt["event"] == "risk_update":
                    risk_count += 1
            except asyncio.TimeoutError:
                break

        assert risk_count >= 3, f"Expected >=3 risk_updates, got {risk_count}"


# ---- Test 6: Packet loss flag is echoed back ---------------------------------

@pytest.mark.asyncio
async def test_ws_packet_loss_flag():
    """When packet_loss=True is sent, the risk_update reflects it."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        # Send with packet_loss=True
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": True}))
        await ws.send(chunk_bytes)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": True}))
        await ws.send(chunk_bytes)

        for _ in range(20):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                evt = json.loads(raw)
                if evt["event"] == "risk_update":
                    assert evt["packet_loss_flagged"] is True
                    break
            except asyncio.TimeoutError:
                break


# ---- Test 7: Sample WAV file streams correctly ------------------------------

@pytest.mark.asyncio
async def test_ws_stream_wav_file():
    """Stream the generated sample_call.wav through the WebSocket and verify output."""
    wav_path = os.path.join(os.path.dirname(__file__), "..", "sample_call.wav")
    if not os.path.exists(wav_path):
        pytest.skip("sample_call.wav not found — run make_test_audio.py first")

    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    chunk_frames = int(sr * 0.2)  # 200ms chunks

    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        await ws.send(json.dumps({"type": "handshake", "sample_rate": sr}))

        chunks_sent = 0
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
            await ws.send(frames)
            chunks_sent += 1

        wf.close()

        # Collect all risk_updates
        risk_updates = []
        for _ in range(50):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                evt = json.loads(raw)
                if evt["event"] == "risk_update":
                    risk_updates.append(evt)
            except asyncio.TimeoutError:
                break

        assert len(risk_updates) >= 5, f"Expected >=5 risk_updates from 6s WAV, got {len(risk_updates)}"
        # All EMA values should be between 0 and 1
        for ru in risk_updates:
            assert 0.0 <= ru["composite_ema"] <= 1.0
            assert 0.0 <= ru["composite_raw"] <= 1.0
