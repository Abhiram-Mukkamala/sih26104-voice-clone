"""Unit tests for per-stride fusion primitives (no decision-window policy)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_engine import (
    DECISION_WINDOW_STRIDES, FREEZE_LOSS_RATIO, FusionEngine, MitigationLevel,
    RiskLevel, SessionRiskState, THRESHOLD_ALLOW, THRESHOLD_BLOCK,
    W_IDENTITY_3WAY, W_PROSODY, W_PROSODY_3WAY, W_VOCODER, W_VOCODER_3WAY,
    mitigation_for_risk,
)

def state(): return SessionRiskState(session_id="test")

def test_shared_policy_definitions():
    assert (THRESHOLD_ALLOW, THRESHOLD_BLOCK, FREEZE_LOSS_RATIO, DECISION_WINDOW_STRIDES) == (0.35, 0.70, 0.35, 8)

def test_two_way_weights_remain_public_and_compatible():
    assert FusionEngine().composite_probability(1, 0) == W_VOCODER
    assert FusionEngine().composite_probability(0, 1) == W_PROSODY

def test_three_way_weighting():
    engine = FusionEngine()
    assert engine.composite_probability(1, 0, 0) == W_VOCODER_3WAY
    assert engine.composite_probability(0, 1, 0) == W_PROSODY_3WAY
    assert engine.composite_probability(0, 0, 1) == W_IDENTITY_3WAY

def test_none_identity_collapses_to_two_way():
    engine = FusionEngine()
    assert engine.composite_probability(.5, .5, None) == engine.composite_probability(.5, .5)

def test_probability_inputs_are_clamped():
    assert FusionEngine().composite_probability(2, -1, 3) == 0.75

def test_process_stride_returns_only_stride_result():
    result = FusionEngine().process_stride(state(), .1, .2, False, 0.0)
    assert not isinstance(result, tuple)
    assert result.composite_raw == .14
    assert result.p_identity is None
    assert not hasattr(result, "mitigation_action")
    assert not hasattr(result, "decision_pending")

def test_ema_freezes_on_packet_loss_and_resumes():
    engine, session = FusionEngine(), state()
    first = engine.process_stride(session, .1, .1, False, 0.0)
    frozen = engine.process_stride(session, .99, .99, True, 0.0)
    resumed = engine.process_stride(session, .9, .9, False, 0.0)
    assert frozen.composite_ema == first.composite_ema
    assert resumed.composite_ema > frozen.composite_ema

def test_loss_ratio_and_low_confidence_classification():
    engine, session = FusionEngine(), state()
    for _ in range(DECISION_WINDOW_STRIDES):
        result = engine.process_stride(session, .1, .1, True, 0.0)
    assert result.packet_loss_ratio == 1.0
    assert result.risk_level == RiskLevel.LOW_CONFIDENCE

def test_risk_level_thresholds_are_independent_from_mitigation():
    engine = FusionEngine()
    assert engine._classify(.39, 0) == RiskLevel.SAFE
    assert engine._classify(.40, 0) == RiskLevel.WARNING
    assert engine._classify(.70, 0) == RiskLevel.CRITICAL

def test_mitigation_threshold_boundaries():
    assert mitigation_for_risk(.34) == MitigationLevel.ALLOW
    assert mitigation_for_risk(.35) == MitigationLevel.CHALLENGE
    assert mitigation_for_risk(.70) == MitigationLevel.CHALLENGE
    assert mitigation_for_risk(.71) == MitigationLevel.BLOCK

def test_identity_and_similarity_are_forwarded():
    result = FusionEngine().process_stride(state(), .5, .5, False, 0.0, .2, .8)
    assert result.p_identity == .2
    assert result.speaker_similarity == .8
