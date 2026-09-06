"""
test_server_e2e.py
===================
End-to-end WebSocket integration tests against a live server at
localhost:8000.  Validates the full pipeline: handshake → audio stream →
risk_update events (with all wire-spec fields) → mitigation_verdict →
enforcement_action → clean disconnect.

Requires the server to be running: python server.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
import numpy as np
import websockets
import websockets.exceptions
import pytest

WS_URL = "ws://localhost:8000/ws/stream-verify"
HEALTHZ_URL = "http://localhost:8000/healthz"


def _server_is_reachable():
    try:
        with urllib.request.urlopen(HEALTHZ_URL, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


pytestmark = pytest.mark.skipif(
    not _server_is_reachable(), reason="server offline"
)


def _authenticated_ws_url(path="/ws/stream-verify"):
    token_url = "http://localhost:8000/dev/token?" + urllib.parse.urlencode(
        {"sub": "e2e-caller"}
    )
    with urllib.request.urlopen(token_url, timeout=5) as response:
        token = json.loads(response.read().decode("utf-8"))["token"]
    return "ws://localhost:8000" + path + "?" + urllib.parse.urlencode(
        {"token": token}
    )


WS_URL = _authenticated_ws_url() if _server_is_reachable() else ""


def _make_pcm16_chunk(freq=150.0, duration_s=0.2, sr=16000, amp=0.3):
    """Generate a short PCM16LE mono chunk."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = amp * np.sin(2 * np.pi * freq * t)
    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    return pcm16.tobytes(), sr


async def _recv_events(ws, max_iters=40, timeout=5.0):
    """Receive events from a WebSocket, gracefully handling server-side closes.
    Returns a list of parsed JSON event dicts."""
    events = []
    for _ in range(max_iters):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            events.append(json.loads(raw))
        except asyncio.TimeoutError:
            break
        except websockets.exceptions.ConnectionClosed:
            break
    return events


# ---- Test 1: Health endpoint ------------------------------------------------

def test_healthz():
    """GET /healthz returns 200 with expected keys."""
    import urllib.request
    resp = urllib.request.urlopen(HEALTHZ_URL)
    assert resp.status == 200
    data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert "using_model_fallback" in data
    assert "speaker_encoder_fallback" in data
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


# ---- Test 4: Unauthenticated WebSocket connections are rejected -------------

@pytest.mark.asyncio
async def test_ws_rejects_unauthenticated():
    """Both WebSocket routes reject connections without a JWT."""
    for path in ("/ws/stream-verify", "/ws/stream"):
        with pytest.raises(
            (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatus)
        ) as exc_info:
            async with websockets.connect(
                "ws://localhost:8000" + path, max_size=None
            ) as ws:
                await ws.recv()
        if isinstance(exc_info.value, websockets.exceptions.ConnectionClosed):
            assert exc_info.value.code == 4401


# ---- Test 4: Handshake with user_id + risk_update wire-spec fields -----------

@pytest.mark.asyncio
async def test_ws_handshake_and_risk_update_fields():
    """Send handshake + audio → get risk_update with all wire-spec fields."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        # Handshake with user_id
        await ws.send(json.dumps({
            "type": "handshake",
            "sample_rate": 16000,
            "user_id": "test_user_001",
        }))

        chunk_bytes, sr = _make_pcm16_chunk(duration_s=0.25)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
        await ws.send(chunk_bytes)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
        await ws.send(chunk_bytes)

        events = await _recv_events(ws, max_iters=20)
        risk_updates = [e for e in events if e["event"] == "risk_update"]

        assert len(risk_updates) > 0, "Never received a risk_update event"
        risk_update = risk_updates[0]

        # Validate all wire-spec fields
        assert "session_id" in risk_update
        assert "timestamp_ms" in risk_update
        assert "risk_score" in risk_update
        assert "composite_ema" in risk_update
        assert "composite_raw" in risk_update
        assert "risk_level" in risk_update
        assert risk_update["risk_level"] in ("SAFE", "WARNING", "CRITICAL", "LOW_CONFIDENCE")
        assert "speaker_similarity" in risk_update  # may be null
        assert "packet_loss" in risk_update
        assert isinstance(risk_update["packet_loss"], bool)
        assert "packet_loss_ratio" in risk_update
        assert "processing_latency_ms" in risk_update
        assert risk_update["processing_latency_ms"] > 0
        assert "mitigation_action" in risk_update
        assert risk_update["mitigation_action"] in ("ANALYZING", "ALLOW", "CHALLENGE", "BLOCK")
        assert "decision_pending" in risk_update
        assert isinstance(risk_update["decision_pending"], bool)

        # risk_score should equal composite_ema
        assert risk_update["risk_score"] == risk_update["composite_ema"]


# ---- Test 5: Multiple strides produce risk_updates ---------------------------

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
            try:
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
                await ws.send(chunk_bytes)
            except websockets.exceptions.ConnectionClosed:
                break

        events = await _recv_events(ws, max_iters=30)
        risk_count = sum(1 for e in events if e["event"] == "risk_update")
        # We should get at least 3 risk_updates, or the server may have
        # BLOCKed the session (which is correct enforcement behavior).
        enforced = any(
            e.get("event") == "enforcement_action" for e in events
        )
        assert risk_count >= 3 or enforced, (
            f"Expected >=3 risk_updates or enforcement, got {risk_count} updates"
        )


# ---- Test 6: Packet loss flag is echoed back ---------------------------------

@pytest.mark.asyncio
async def test_ws_packet_loss_flag():
    """When packet_loss=True is sent, the risk_update reflects it."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": True}))
        await ws.send(chunk_bytes)
        await ws.send(json.dumps({"type": "rtp_status", "packet_loss": True}))
        await ws.send(chunk_bytes)

        events = await _recv_events(ws, max_iters=20)
        risk_updates = [e for e in events if e["event"] == "risk_update"]

        assert len(risk_updates) > 0
        assert risk_updates[0]["packet_loss"] is True
        assert risk_updates[0]["packet_loss_ratio"] > 0


# ---- Test 7: Decision window emits mitigation_verdict ------------------------

@pytest.mark.asyncio
async def test_ws_mitigation_verdict():
    """Stream 10+ strides (2.5s) and verify a mitigation_verdict is emitted."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        # 10 chunks of 250ms = 10 strides → should produce at least 1 verdict
        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        for _ in range(10):
            try:
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
                await ws.send(chunk_bytes)
            except websockets.exceptions.ConnectionClosed:
                break

        events = await _recv_events(ws, max_iters=50)
        verdicts = [e for e in events if e["event"] == "mitigation_verdict"]

        assert len(verdicts) >= 1, (
            f"Never received a mitigation_verdict event. "
            f"Events received: {[e['event'] for e in events]}"
        )

        verdict = verdicts[0]
        assert verdict["window_index"] == 0
        assert verdict["mitigation_action"] in ("ALLOW", "CHALLENGE", "BLOCK")
        assert "risk_score" in verdict
        assert verdict["window_strides"] == 8
        assert "packet_loss_ratio" in verdict
        assert "speaker_similarity" in verdict
        assert "risk_level" in verdict


# ---- Test 8: BLOCK enforcement triggers SEVER_SESSION + close(1008) ---------

@pytest.mark.asyncio
async def test_ws_block_enforcement():
    """If risk is high enough to BLOCK, the server sends enforcement_action
    with action=SEVER_SESSION and closes with code 1008."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session_started"

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        # Send enough strides to trigger a decision window (8 strides).
        # The synthetic tone will likely produce high risk via the DSP
        # heuristic, potentially triggering BLOCK.
        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        for _ in range(10):
            try:
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
                await ws.send(chunk_bytes)
            except websockets.exceptions.ConnectionClosed:
                break

        events = await _recv_events(ws, max_iters=50)
        enforcements = [
            e for e in events
            if e.get("event") == "enforcement_action"
        ]
        verdicts = [e for e in events if e["event"] == "mitigation_verdict"]

        # The test audio may or may not trigger BLOCK depending on the
        # fallback heuristic output.  If a verdict was BLOCK, we must
        # have an enforcement_action with SEVER_SESSION.
        block_verdicts = [v for v in verdicts
                          if v["mitigation_action"] == "BLOCK"]
        if block_verdicts:
            assert len(enforcements) >= 1, (
                "BLOCK verdict without enforcement_action"
            )
            assert enforcements[0]["action"] == "SEVER_SESSION"
            assert "reason" in enforcements[0]
            assert "risk_score" in enforcements[0]


# ---- Test 9: Sample WAV file streams correctly ------------------------------

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
            try:
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
                await ws.send(frames)
            except websockets.exceptions.ConnectionClosed:
                break
            chunks_sent += 1

        wf.close()

        events = await _recv_events(ws, max_iters=80, timeout=3.0)
        risk_updates = [e for e in events if e["event"] == "risk_update"]
        verdicts = [e for e in events if e["event"] == "mitigation_verdict"]
        enforcements = [e for e in events if e.get("event") == "enforcement_action"]

        # We should get risk_updates (at least a few before any potential BLOCK)
        assert len(risk_updates) >= 3, (
            f"Expected >=3 risk_updates from WAV, got {len(risk_updates)}"
        )
        # All EMA values should be between 0 and 1
        for ru in risk_updates:
            assert 0.0 <= ru["composite_ema"] <= 1.0
            assert 0.0 <= ru["composite_raw"] <= 1.0
            assert ru["mitigation_action"] in ("ANALYZING", "ALLOW", "CHALLENGE", "BLOCK")

        # Should get at least 1 verdict (8 strides in a 6s file = ~24 strides)
        # unless the session was severed early by a BLOCK
        assert len(verdicts) >= 1 or len(enforcements) >= 1, (
            f"Expected >=1 verdict or enforcement, got "
            f"{len(verdicts)} verdicts, {len(enforcements)} enforcements"
        )


# ---- Test 10: Mitigation action in risk_update is valid ---------------------

@pytest.mark.asyncio
async def test_ws_mitigation_action_in_risk_update():
    """Verify mitigation_action field appears in every risk_update."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        hello = json.loads(await ws.recv())

        await ws.send(json.dumps({"type": "handshake", "sample_rate": 16000}))

        chunk_bytes, _ = _make_pcm16_chunk(duration_s=0.25)
        for _ in range(4):
            try:
                await ws.send(json.dumps({"type": "rtp_status", "packet_loss": False}))
                await ws.send(chunk_bytes)
            except websockets.exceptions.ConnectionClosed:
                break

        events = await _recv_events(ws, max_iters=20)
        risk_updates = [e for e in events if e["event"] == "risk_update"]

        assert len(risk_updates) > 0
        for ru in risk_updates:
            assert "mitigation_action" in ru
            assert ru["mitigation_action"] in ("ANALYZING", "ALLOW", "CHALLENGE", "BLOCK")
            assert "decision_pending" in ru


# ---- Full sample-call stream test -------------------------------------------

def test_stream_sample_call_full(tmp_path):
    """Stream a complete six-second call and validate all decision telemetry."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wav_path = os.path.join(repo_root, "sample_call.wav")
    if not os.path.exists(wav_path):
        from make_test_audio import make_wav

        wav_path = str(tmp_path / "sample_call.wav")
        make_wav(wav_path, seconds=6.0, sr=16000)

    token_url = "http://localhost:8000/dev/token?" + urllib.parse.urlencode(
        {"sub": "test-caller"}
    )
    with urllib.request.urlopen(token_url, timeout=5) as response:
        token = json.loads(response.read().decode("utf-8"))["token"]

    async def stream_and_collect():
        events = []
        ws_url = "ws://localhost:8000/ws/stream?" + urllib.parse.urlencode(
            {"token": token}
        )
        async with websockets.connect(ws_url, max_size=None) as ws:
            events.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({
                "type": "handshake",
                "sample_rate": 16000,
                "user_id": "test-caller",
            }))

            with wave.open(wav_path, "rb") as wav_file:
                assert wav_file.getframerate() == 16000
                assert wav_file.getnchannels() == 1
                chunk_frames = int(wav_file.getframerate() * 0.2)
                while True:
                    frames = wav_file.readframes(chunk_frames)
                    if not frames:
                        break
                    try:
                        await ws.send(json.dumps({
                            "type": "rtp_status",
                            "packet_loss": False,
                        }))
                        await ws.send(frames)
                        await asyncio.sleep(0.005)
                    except websockets.exceptions.ConnectionClosed:
                        break

            deadline = asyncio.get_running_loop().time() + 1.5
            while asyncio.get_running_loop().time() < deadline:
                timeout = deadline - asyncio.get_running_loop().time()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    break
                events.append(json.loads(raw))
        return events

    events = asyncio.run(stream_and_collect())
    assert any(event.get("event") == "session_started" for event in events)

    risk_updates = [event for event in events if event.get("event") == "risk_update"]
    assert risk_updates, "No risk_update received from the complete WAV stream"
    required_keys = {
        "risk_score",
        "speaker_similarity",
        "packet_loss",
        "processing_latency_ms",
        "mitigation_action",
        "decision_pending",
    }
    valid_actions = {"ANALYZING", "ALLOW", "CHALLENGE", "BLOCK"}
    for risk_update in risk_updates:
        assert required_keys.issubset(risk_update)
        assert risk_update["speaker_similarity"] is None or isinstance(
            risk_update["speaker_similarity"], float
        )
        assert risk_update["mitigation_action"] in valid_actions

    verdicts = [
        event for event in events if event.get("event") == "mitigation_verdict"
    ]
    assert verdicts, "No mitigation_verdict received from the six-second stream"
    assert any(verdict["mitigation_action"] in valid_actions for verdict in verdicts)
    risk_actions = {risk_update["mitigation_action"] for risk_update in risk_updates}
    assert any(
        verdict["mitigation_action"] in risk_actions for verdict in verdicts
    )
