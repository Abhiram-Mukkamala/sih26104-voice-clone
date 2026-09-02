"""
fusion_engine.py
=================
SIH26104 — Temporal Fusion & Risk Engine.

Combines Track A (vocoder/deep-model probability) and Track B (prosody
anomaly probability) into a single composite risk score, smooths it with
an EMA over a rolling window, and exposes threshold-gated decisions.

Key defensive design point (see jury Q&A #1): network jitter/packet loss
must NOT be allowed to masquerade as vocoder risk. This engine is
explicitly packet-loss-aware — it is handed a `packet_loss` flag per
stride and FREEZES (holds last EMA value, does not decay toward the new
noisy estimate) rather than integrating a stride that was computed from a
buffer contaminated by concealment/PLC (packet-loss-concealment) audio.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


# ---- Static configuration --------------------------------------------------
W_VOCODER = 0.6
W_PROSODY = 0.4

STRIDE_S = 0.25
ROLLING_WINDOW_S = 3.0
ROLLING_WINDOW_STRIDES = int(ROLLING_WINDOW_S / STRIDE_S)  # 12 strides

EMA_ALPHA = 2.0 / (ROLLING_WINDOW_STRIDES + 1)  # standard EMA-from-window-length mapping

SAFE_MAX = 0.40
WARNING_MAX = 0.70
AMBIGUITY_LOW = 0.45
AMBIGUITY_HIGH = 0.65

# If more than this fraction of strides in the trailing window were
# packet-loss-affected, we downgrade confidence rather than emit a hard
# score — a call that's mostly PLC-reconstructed audio can't be trusted
# for either "safe" or "critical".
MAX_TOLERABLE_LOSS_RATIO = 0.35


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # too much packet loss to trust the score


@dataclass
class StrideResult:
    """One 250ms stride's fused output."""
    timestamp_ms: float
    composite_raw: float
    composite_ema: float
    risk_level: RiskLevel
    liveness_challenge_required: bool
    mfa_required: bool
    packet_loss_flagged: bool
    latency_ms: float


@dataclass
class SessionRiskState:
    """
    Per-call (per-WebSocket-session) state. One instance lives for the
    duration of a single verified call and is discarded on disconnect —
    no persistence beyond the session, per the Privacy Plane.
    """
    session_id: str
    ema_value: float = 0.0
    initialized: bool = False
    loss_flags: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW_STRIDES))
    history: deque = field(default_factory=lambda: deque(maxlen=200))  # audit telemetry only
    # Liveness challenge-response state (Layer 4). A challenge, once issued,
    # must be resolved (pass/fail) before another can be issued — without
    # this, every stride that lands in the ambiguity band re-fires a new
    # challenge, spamming the caller several times a second.
    challenge_pending: bool = False
    challenge_deadline_ms: float = 0.0

    def loss_ratio(self) -> float:
        if not self.loss_flags:
            return 0.0
        return sum(self.loss_flags) / len(self.loss_flags)


class FusionEngine:
    """
    Stateless computation, statefulness lives entirely in the
    SessionRiskState object the caller passes in — lets server.py hold one
    engine instance and many concurrent per-call states safely.
    """

    @staticmethod
    def composite_probability(p_vocoder: float, p_prosody: float) -> float:
        p_vocoder = min(max(p_vocoder, 0.0), 1.0)
        p_prosody = min(max(p_prosody, 0.0), 1.0)
        return W_VOCODER * p_vocoder + W_PROSODY * p_prosody

    @staticmethod
    def _classify(ema_value: float, loss_ratio: float) -> RiskLevel:
        if loss_ratio > MAX_TOLERABLE_LOSS_RATIO:
            return RiskLevel.LOW_CONFIDENCE
        if ema_value < SAFE_MAX:
            return RiskLevel.SAFE
        if ema_value < WARNING_MAX:
            return RiskLevel.WARNING
        return RiskLevel.CRITICAL

    def process_stride(
        self,
        state: SessionRiskState,
        p_vocoder: float,
        p_prosody: float,
        packet_loss: bool,
        stride_start_perf_counter: float,
    ) -> StrideResult:
        """
        Ingest one stride's dual-track probabilities and update the
        session's EMA. This is the single call site server.py needs per
        250ms stride.
        """
        state.loss_flags.append(packet_loss)
        raw = self.composite_probability(p_vocoder, p_prosody)

        if packet_loss:
            # Freeze: don't let a PLC-reconstructed stride pull the EMA in
            # either direction. We still record it in history for audit,
            # tagged as loss-affected, but the EMA carries forward unchanged.
            ema = state.ema_value if state.initialized else raw
        else:
            if not state.initialized:
                ema = raw
                state.initialized = True
            else:
                ema = EMA_ALPHA * raw + (1 - EMA_ALPHA) * state.ema_value

        state.ema_value = ema
        loss_ratio = state.loss_ratio()
        risk_level = self._classify(ema, loss_ratio)

        liveness_required = (
            AMBIGUITY_LOW <= ema <= AMBIGUITY_HIGH
            and risk_level != RiskLevel.LOW_CONFIDENCE
        )
        mfa_required = risk_level == RiskLevel.CRITICAL

        latency_ms = (time.perf_counter() - stride_start_perf_counter) * 1000.0

        result = StrideResult(
            timestamp_ms=time.time() * 1000.0,
            composite_raw=round(raw, 4),
            composite_ema=round(ema, 4),
            risk_level=risk_level,
            liveness_challenge_required=liveness_required,
            mfa_required=mfa_required,
            packet_loss_flagged=packet_loss,
            latency_ms=round(latency_ms, 2),
        )
        state.history.append(result)
        return result
