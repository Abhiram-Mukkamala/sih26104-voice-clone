# Claude Context — SIH26104 Voice Clone Detection Gateway

Read `README.md` and `claude_context.md` first (architecture, wire protocol, thresholds, run commands, and the hard constraints that are not in the README). This file tracks what has happened so far — the verified state of the project as of the latest check.

## Project & Workflow

- Project root: `C:\Projects\sih26104\sih26104` (parent `C:\Projects\sih26104` is the VSC workspace).
- Code is agent-generated. The user writes code themselves in Antigravity/VSC — deliver findings and prompts; do NOT edit files directly unless explicitly asked.
- Before/after changes, re-verify the baseline below. E2E tests SKIP (do NOT fail) when the server is offline.

## Verified Baseline (current ground truth — 06-09-2026)

- Unit tests: **64 passed** (`test_dsp_pipeline` 5, `test_fusion_engine` 11, `test_speaker_encoder` 17, `test_speaker_verification` 31).
- E2E tests: **12 passed** against a live server — includes `test_ws_rejects_unauthenticated`, `test_ws_block_enforcement`, `test_ws_stream_wav_file`, `test_stream_sample_call_full`. Total = **76 passed**.
- Live client simulator verified: `session_started` → `risk_update` → `mitigation_verdict` every 8 strides → `liveness_challenge`; risk ~0.56 (WARNING/CHALLENGE); packet-loss freeze/override fire correctly.
- Server currently on `127.0.0.1:8000`, `/healthz` OK: `{"status":"ok","using_model_fallback":true,"speaker_encoder_fallback":true,"sample_rate":16000}`.

## What Has Happened / Completed

1. **Initial agent-written codebase** produced at `C:\Projects\sih26104\sih26104`: mitigation firewall (`app/server.py`), dashboard (`app/static/index.html`), authenticated client simulator (`app/client_simulator.py`), 8th deliverable e2e test, Dockerfile (`INSTALL_HEAVY`, non-root, HEALTHCHECK), README (autocorrelation not librosa.pyin). Unit 84 + e2e 11 at review time.
2. **Full review completed.** Findings issued as fix prompts:
   - FIX 1 — Dashboard had no red animated "Voice Cloning Detected" banner and no `mitigation_verdict` handler.
   - FIX 2 — Legacy `/ws/stream-verify` alias skipped JWT auth (path-conditional); no auth-rejection test existed.
   - FIX 3 — Duplicate decision window: `fusion_engine.process_stride` kept a verdict accumulator that `server.py` discarded (`result, _ =`) while `server.py` computed its own `decision_emas`. Shared thresholds (0.35 / 0.70 / freeze 0.35 / 8 strides) were duplicated constants.
   - FIX 4 — Dashboard field-name mismatch: read `data.latency_ms` / `data.packet_loss_flagged` while server sends `processing_latency_ms` / `packet_loss` → `undefined.toFixed()` TypeError killed the event feed.
3. **Fixes applied and re-verified**: FIX 1–4 are DONE in the repo.
   - `#cloneBanner` exists (`index.html:476`), CSS shake/pulse animation (`index.html:68`), triggered on `mitigation_verdict === BLOCK` and `enforcement_action === SEVER_SESSION` (`index.html:813,823`).
   - `mitigation_verdict` handler added (`index.html:820`).
   - Metrics grid reworked to 6 tiles: EMA Score, Raw Score, **Speaker Sim**, **Loss Ratio**, Latency, Pkt Loss (`index.html:589-594`).
   - `test_ws_rejects_unauthenticated` added (`test_server_e2e.py:121`) → e2e went 11 → 12.
   - Fusion engine now returns stride-only results (`test_fusion_engine.py:35`), shared policy definitions verified (`test_fusion_engine.py:15`) → unit count 84 → 64.
4. **`claude_context.md`** created at project root (conventions + constraints). This file documents the full trail.

## Architecture (implemented, unchanged)

- Ring buffer 1.0s / 250ms stride; Track A LFCC/spectral anomalies; Track B normalized-autocorrelation prosody (RMS gate + ACF peak voicing + parabolic lag interpolation, no librosa.pyin); Track C identity: ECAPA-TDNN with FFT-spectral fallback (192-d L2-normalised, in-memory, model-type mismatch → None).
- Fusion 3-way 0.45/0.25/0.30 ⇒ 2-way 0.6/0.4 when unenrolled; `p_identity = 1 − similarity`.
- EMA freeze at `packet_loss_ratio > 0.35`; decision window 2.0s = 8 × 250ms strides; ALLOW < 0.35, CHALLENGE ≤ 0.70, BLOCK > 0.70; BLOCK → `enforcement_action {action:"SEVER_SESSION"}` then `close(code=1008)`.
- Auth: stdlib HS256 JWT (`AUTH_SECRET`), alg pinned, `nbf`/`exp` enforced, timing-safe compare; WS accepts `?token=` or `Authorization: Bearer` on EVERY route.
- Zero retention: audio zeroed after feature extraction, no disk writes, in-memory voiceprints purged on disconnect.

## Wire Protocol (server → client)

| Event | Payload fields |
|---|---|
| `session_started` | `session_id`, `using_model_fallback` |
| `risk_update` | `risk_score`, `composite_ema`, `composite_raw`, `risk_level`, `p_identity`, `speaker_similarity`, `packet_loss`, `packet_loss_ratio`, `processing_latency_ms`, `mitigation_action`, `decision_pending` |
| `mitigation_verdict` | `session_id`, `window_index`, `mitigation_action`, `risk_score`, `window_strides`, `packet_loss_ratio`, `speaker_similarity`, `risk_level` |
| `liveness_challenge` | `challenge_id`, `prompt_text`, `tts_engine`, `expires_in_ms` |
| `enforcement_action` | `action` (`SEVER_SESSION`), `reason`, `risk_score` |

Wire field names are FROZEN — `index.html` and `client_simulator.py` depend on them.

## Notes / Known Caveats

- Fallback DSP path is what runs locally (no torch/speechbrain); tests must not require heavy deps.
- Python 3.11, websockets 15+ (`additional_headers` in `websockets.connect`, NOT `extra_headers`).
- Earlier `.pytest_cache` failures were from running e2e against a stale server instance; not real test failures.