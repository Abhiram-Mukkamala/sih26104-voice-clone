"""
fusion_engine.py
=================
SIH26104 — Temporal Fusion & Risk Engine.

Combines Track A (vocoder/deep-model probability), Track B (prosody anomaly
probability), and Track C (identity/speaker impersonation probability) into
a single composite risk score, smooths it with a decaying EMA over a 2.0s
rolling window (8 × 250ms strides), and exposes threshold-gated mitigation
decisions through a decision-window firewall.

3-way fusion weights (when enrolled voiceprint available):
    W_VOCODER=0.45, W_PROSODY=0.25, W_IDENTITY=0.30
2-way fallback weights (no voiceprint):
    W_VOCODER=0.60, W_PROSODY=0.40

Mitigation thresholds:
    risk < 0.35           → ALLOW
    0.35 ≤ risk ≤ 0.70    → CHALLENGE
    risk > 0.70            → BLOCK

Key defensive design point: network jitter/packet loss must NOT masquerade
as vocoder risk.  The engine is explicitly packet-loss-aware — it FREEZES
the EMA (holds last value, does not decay toward the new noisy estimate)
rather than integrating a stride computed from PLC-contaminated audio.

Low-confidence override: if packet_loss_ratio > 0.35 over the trailing
window, the mitigation action is forced to CHALLENGE regardless of score.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---- Static configuration --------------------------------------------------
W_VOCODER_3WAY = 0.45
W_PROSODY_3WAY = 0.25
W_IDENTITY_3WAY = 0.30

W_VOCODER_2WAY = 0.60
W_PROSODY_2WAY = 0.40

# Public two-track weights retained for callers of the original 2-way API.
W_VOCODER = W_VOCODER_2WAY
W_PROSODY = W_PROSODY_2WAY

STRIDE_S = 0.25
ROLLING_WINDOW_S = 2.0
ROLLING_WINDOW_STRIDES = int(ROLLING_WINDOW_S / STRIDE_S)  # 8 strides

EMA_ALPHA = 2.0 / (ROLLING_WINDOW_STRIDES + 1)  # standard EMA span mapping

# Shared decision-policy definitions.  The server owns the decision window,
# but imports these values so scoring and enforcement cannot drift apart.
THRESHOLD_ALLOW = 0.35
THRESHOLD_BLOCK = 0.70
FREEZE_LOSS_RATIO = 0.35
DECISION_WINDOW_STRIDES = 8

# Compatibility aliases for existing callers.
ALLOW_MAX = THRESHOLD_ALLOW
BLOCK_MIN = THRESHOLD_BLOCK
MITIGATE_ALLOW_MAX = THRESHOLD_ALLOW
MITIGATE_CHALLENGE_MAX = THRESHOLD_BLOCK

# Risk-level thresholds are intentionally independent of mitigation policy.
# Keep these values and the RiskLevel boundary behavior stable for 2-way users.
SAFE_MAX = 0.40
WARNING_MAX = 0.70

# Ambiguity band for liveness challenge issuance (within the CHALLENGE zone)
AMBIGUITY_LOW = 0.45
AMBIGUITY_HIGH = 0.65

# If more than this fraction of strides in the trailing window were
# packet-loss-affected, force CHALLENGE rather than trusting the score.
MAX_TOLERABLE_LOSS_RATIO = FREEZE_LOSS_RATIO


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class MitigationLevel(str, Enum):
    """Mitigation state exposed to policy/UI consumers."""
    ANALYZING = "ANALYZING"
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


def mitigation_for_risk(risk: float) -> MitigationLevel:
    """Return the score-only mitigation level (packet-loss policy excluded)."""
    if risk < MITIGATE_ALLOW_MAX:
        return MitigationLevel.ALLOW
    if risk <= MITIGATE_CHALLENGE_MAX:
        return MitigationLevel.CHALLENGE
    return MitigationLevel.BLOCK


@dataclass
class StrideResult:
    """One 250ms stride's fused output — matches the risk_update wire frame."""
    timestamp_ms: float
    composite_raw: float
    composite_ema: float
    risk_level: RiskLevel
    risk_score: float              # same as composite_ema, explicit per wire spec
    speaker_similarity: Optional[float] = field(default=None, kw_only=True)
    packet_loss: bool              # per-stride flag
    packet_loss_ratio: float       # trailing window ratio
    processing_latency_ms: float
    p_identity: Optional[float] = field(default=None, kw_only=True)


@dataclass
class SessionRiskState:
    """
    Per-call (per-WebSocket-session) state.  One instance lives for the
    duration of a single verified call and is discarded on disconnect —
    no persistence beyond the session, per the Privacy Plane.
    """
    session_id: str
    ema_value: float = 0.0
    initialized: bool = False
    loss_flags: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW_STRIDES))
    history: deque = field(default_factory=lambda: deque(maxlen=200))

    # Liveness challenge-response state (Layer 4).
    challenge_pending: bool = False
    challenge_deadline_ms: float = 0.0

    def loss_ratio(self) -> float:
        if not self.loss_flags:
            return 0.0
        return sum(self.loss_flags) / len(self.loss_flags)


class FusionEngine:
    """
    Stateless computation; statefulness lives entirely in the
    SessionRiskState object the caller passes in — lets server.py hold one
    engine instance and many concurrent per-call states safely.
    """

    @staticmethod
    def composite_probability(
        p_vocoder: float,
        p_prosody: float,
        p_identity: Optional[float] = None,
    ) -> float:
        """
        Weighted fusion of track probabilities.
        3-way when identity is available, 2-way fallback otherwise.
        """
        p_vocoder = min(max(p_vocoder, 0.0), 1.0)
        p_prosody = min(max(p_prosody, 0.0), 1.0)

        if p_identity is not None:
            p_identity = min(max(p_identity, 0.0), 1.0)
            return (W_VOCODER_3WAY * p_vocoder
                    + W_PROSODY_3WAY * p_prosody
                    + W_IDENTITY_3WAY * p_identity)
        return W_VOCODER_2WAY * p_vocoder + W_PROSODY_2WAY * p_prosody

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
        p_identity: Optional[float] = None,
        speaker_similarity: Optional[float] = None,
    ) -> StrideResult:
        """
        Ingest one stride's dual/triple-track probabilities and update the
        session's EMA and return only per-stride scoring primitives.  Decision
        windows and mitigation verdicts are deliberately owned by server.py.
        """
        state.loss_flags.append(packet_loss)
        raw = self.composite_probability(p_vocoder, p_prosody, p_identity)

        if packet_loss:
            # Freeze: don't let a PLC-reconstructed stride pull the EMA in
            # either direction.
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
        latency_ms = (time.perf_counter() - stride_start_perf_counter) * 1000.0

        result = StrideResult(
            timestamp_ms=time.time() * 1000.0,
            composite_raw=round(raw, 4),
            composite_ema=round(ema, 4),
            risk_level=risk_level,
            risk_score=round(ema, 4),
            speaker_similarity=(round(speaker_similarity, 4)
                                if speaker_similarity is not None else None),
            packet_loss=packet_loss,
            packet_loss_ratio=round(loss_ratio, 4),
            processing_latency_ms=round(latency_ms, 2),
            p_identity=(round(p_identity, 4) if p_identity is not None else None),
        )
        state.history.append(result)
        return result
