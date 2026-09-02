"""
client_simulator.py
====================
Jury-demo utility: streams a local WAV file into /ws/stream-verify as if
it were a live VoIP call, printing risk telemetry as it arrives.

Usage:
    python client_simulator.py path/to/audio.wav [--rate 16000] [--loss 0.1]

--loss simulates RTP packet loss by randomly setting the `packet_loss`
control flag on a fraction of chunks, to demonstrate the EMA-freeze
behavior described in fusion_engine.py / Q&A #1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import wave

import numpy as np
import websockets


async def stream_file(path: str, ws_url: str, chunk_ms: int, loss_prob: float):
    wf = wave.open(path, "rb")
    sample_rate = wf.getframerate()
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()

    if sampwidth != 2:
        raise ValueError("Simulator expects 16-bit PCM WAV input.")

    chunk_frames = int(sample_rate * chunk_ms / 1000)

    async with websockets.connect(ws_url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        print(f"[session] {hello}")

        await ws.send(json.dumps({"type": "handshake", "sample_rate": sample_rate}))

        async def receiver():
            async for message in ws:
                evt = json.loads(message)
                if evt["event"] == "risk_update":
                    print(
                        f"[{evt['risk_level']:>13}] ema={evt['composite_ema']:.3f} "
                        f"raw={evt['composite_raw']:.3f} loss={evt['packet_loss_flagged']} "
                        f"latency={evt['latency_ms']}ms"
                    )
                elif evt["event"] == "liveness_challenge":
                    print(f"  -> LIVENESS CHALLENGE: \"{evt['prompt_text']}\"")
                elif evt["event"] == "enforcement_action":
                    print(f"  !! ENFORCEMENT: {evt['action']} (score={evt['risk_score']})")

        recv_task = asyncio.create_task(receiver())

        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            if n_channels > 1:
                # Downmix to mono by averaging channels. (The previous
                # byte-stride slice here was wrong for 16-bit samples: it
                # sliced raw bytes instead of 2-byte sample frames, which
                # interleaved high/low bytes from different channels into
                # garbage samples instead of isolating channel 0.)
                samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, n_channels)
                mono = samples.mean(axis=1).astype(np.int16)
                frames = mono.tobytes()

            packet_loss = random.random() < loss_prob
            await ws.send(json.dumps({"type": "rtp_status", "packet_loss": packet_loss}))
            await ws.send(frames)
            await asyncio.sleep(chunk_ms / 1000.0)  # pace like a real-time call

        await asyncio.sleep(1.0)
        recv_task.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream-verify")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--loss", type=float, default=0.0, help="simulated packet loss probability per chunk")
    args = parser.parse_args()

    asyncio.run(stream_file(args.wav_path, args.url, args.chunk_ms, args.loss))


if __name__ == "__main__":
    main()
