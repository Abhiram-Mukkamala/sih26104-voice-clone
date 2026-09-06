# Claude Context — SIH26104 Voice Clone Detection Gateway

Read `README.md` and `claude_context.md` first (architecture, wire protocol, thresholds, run commands, and the hard constraints that are not in the README). This file tracks what has happened so far — the verified state of the project as of the latest check.

## Project & Workflow

- Project root: `C:\Projects\sih26104\sih26104` (parent `C:\Projects\sih26104` is the VSC workspace).
- Code is agent-generated. The user writes code themselves in Antigravity/VSC — deliver findings and prompts; do NOT edit files directly unless explicitly asked.
- Before/after changes, re-verify the baseline below. E2E tests SKIP (do NOT fail) when the server is offline.

## Verified Baseline (current ground truth — 06-09-2026)

- Unit tests: **64 passed** (`test_dsp_pipeline` 5, `test_fusion_engine` 11, `test_speaker_encoder` 17, `test_speaker_verification` 31).
- E2E tests: **17 passed** against a live server. Total = **81 passed**.
  - `test_ws_telemetry_is_session_private`: caller only receives telemetry from their own session (no cross-session leaks).
  - `test_enroll_rejects_missing_token`: unauthenticated `POST /enroll` rejected with HTTP 401.
  - `test_enroll_succeeds_with_valid_token`: authenticated `POST /enroll` with bearer token succeeds with HTTP 200 and valid voiceprint metadata.
  - `test_enroll_rejects_identity_mismatch`: enrollment rejected with HTTP 403 when payload `user_id` does not match token `sub` (identity binding).
  - `test_enroll_succeeds_with_query_token`: confirms `POST /enroll?token=<jwt>` query-parameter auth path succeeds with HTTP 200.
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
5. **Security & Session Scoping Hardening (Agents 1–3)**:
   - Secured `POST /enroll`: dual-path auth (Bearer header or query token), token validated before payload parsing, HTTP 401 on missing/invalid/expired token.
   - Enforced strict identity binding: `user_id` in body must match token `sub` claim (HTTP 403 on mismatch) to prevent cross-user voiceprint overwriting.
   - Scoped WebSocket telemetry to active caller sessions (verified session-private).
   - E2E suite expanded from 12 → 17 tests (unit 64 + e2e 17 = 81 passed).

## Architecture (implemented, unchanged)

- Ring buffer 1.0s / 250ms stride; Track A LFCC/spectral anomalies; Track B normalized-autocorrelation prosody (RMS gate + ACF peak voicing + parabolic lag interpolation, no librosa.pyin); Track C identity: ECAPA-TDNN with FFT-spectral fallback (192-d L2-normalised, in-memory, model-type mismatch → None).
- Fusion 3-way 0.45/0.25/0.30 ⇒ 2-way 0.6/0.4 when unenrolled; `p_identity = 1 − similarity`.
- EMA freeze at `packet_loss_ratio > 0.35`; decision window 2.0s = 8 × 250ms strides; ALLOW < 0.35, CHALLENGE ≤ 0.70, BLOCK > 0.70; BLOCK → `enforcement_action {action:"SEVER_SESSION"}` then `close(code=1008)`.
- Auth: stdlib HS256 JWT (`AUTH_SECRET`), alg pinned, `nbf`/`exp` enforced, timing-safe compare; WS accepts `?token=` or `Authorization: Bearer` on EVERY route; `POST /enroll` enforces bearer auth and identity binding (`sub == user_id`).
- Zero retention: audio zeroed after feature extraction, no disk writes, in-memory voiceprints purged on disconnect.

## Wire Protocol (server → client)

| Event | Payload fields |
|---|---|
| `session_started` | `event`, `session_id`, `using_model_fallback` |
| `risk_update` | `event`, `session_id`, `timestamp_ms`, `risk_score`, `composite_ema`, `composite_raw`, `risk_level`, `p_identity`, `speaker_similarity`, `packet_loss`, `packet_loss_ratio`, `processing_latency_ms`, `mitigation_action`, `decision_pending` |
| `mitigation_verdict` | `event`, `session_id`, `window_index`, `mitigation_action`, `risk_score`, `window_strides`, `packet_loss_ratio`, `speaker_similarity`, `risk_level` |
| `liveness_challenge` | `event`, `session_id`, `challenge_id`, `prompt_text`, `tts_engine`, `expires_in_ms` |
| `enforcement_action` | `event`, `session_id`, `action` (`SEVER_SESSION`), `reason`, `risk_score` |

Wire field names are FROZEN — `index.html` and `client_simulator.py` depend on them.

## Notes / Known Caveats

- Fallback DSP path is what runs locally (no torch/speechbrain); tests must not require heavy deps.
- Python 3.11, websockets 15+ (`additional_headers` in `websockets.connect`, NOT `extra_headers`).
- Earlier `.pytest_cache` failures were from running e2e against a stale server instance; not real test failures.

## Known Limitations for the Sept 10 Presentation

1. **Lightweight Mode DSP Heuristic Fallback (Track A)**: In the standard test/demo environment without an external `.onnx` deepfake checkpoint (`model_path=None`), `acoustic_model.py` defaults to an explainable DSP heuristic based on LFCC fine-detail variance. For final production deployment, an ONNX-exported vocoder detector model (such as RawNet3 or AASIST) should be supplied.
2. **Track C SpeechBrain ECAPA-TDNN is Optional**: SpeechBrain + PyTorch are kept in `requirements-heavy.txt` to keep the base image lightweight and fast to boot. In environments without SpeechBrain, Track C gracefully and deterministically falls back to pure NumPy/SciPy 192-d FFT spectral profile embeddings (96 temporal mean + 96 delta band energies, L2-normalized).
3. **Liveness Challenge is Orchestration Metadata Only**: When mitigation triggers `CHALLENGE`, the server emits a `liveness_challenge` JSON frame containing challenge ID, expiry (8000ms), and a prompt targeting `sarvam-bulbul-indic-tts`. Full end-to-end audio playback over an out-of-band channel and automated verification of the caller's spoken response are architectural stubs for integration with an IVR/telephony gateway.
4. **Development Token Minting (`/dev/token`)**: `/dev/token` provides instant HS256 JWT minting for demo ease. In production, tokens must be supplied by an external enterprise authentication gateway / IdP.
5. **Client Simulator `--enroll` Flag Requires Auth Token**: `client_simulator.py:83` currently posts to `/enroll` without forwarding the bearer token, triggering HTTP 401 when `--enroll` is passed unless the URL or headers are updated to pass `?token={token}`.