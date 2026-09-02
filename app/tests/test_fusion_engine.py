import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_engine import FusionEngine, SessionRiskState, RiskLevel, SAFE_MAX, WARNING_MAX


def make_state():
    return SessionRiskState(session_id="test")


def test_composite_probability_weighting():
    fe = FusionEngine()
    assert abs(fe.composite_probability(1.0, 0.0) - 0.6) < 1e-9
    assert abs(fe.composite_probability(0.0, 1.0) - 0.4) < 1e-9


def test_packet_loss_freezes_ema():
    fe = FusionEngine()
    state = make_state()
    r1 = fe.process_stride(state, p_vocoder=0.1, p_prosody=0.1, packet_loss=False,
                            stride_start_perf_counter=0.0)
    frozen_value = r1.composite_ema
    # A wildly different score arrives, but flagged as packet-loss —
    # the EMA must not move.
    r2 = fe.process_stride(state, p_vocoder=0.99, p_prosody=0.99, packet_loss=True,
                            stride_start_perf_counter=0.0)
    assert r2.composite_ema == frozen_value
    assert r2.packet_loss_flagged is True


def test_risk_level_thresholds():
    fe = FusionEngine()
    state = make_state()
    r = fe.process_stride(state, p_vocoder=0.05, p_prosody=0.05, packet_loss=False,
                           stride_start_perf_counter=0.0)
    assert r.composite_ema < SAFE_MAX
    assert r.risk_level == RiskLevel.SAFE

    state2 = make_state()
    r2 = fe.process_stride(state2, p_vocoder=0.95, p_prosody=0.95, packet_loss=False,
                            stride_start_perf_counter=0.0)
    assert r2.composite_ema >= WARNING_MAX
    assert r2.risk_level == RiskLevel.CRITICAL
    assert r2.mfa_required is True


def test_low_confidence_on_high_loss_ratio():
    fe = FusionEngine()
    state = make_state()
    # Flood the loss window past MAX_TOLERABLE_LOSS_RATIO.
    for _ in range(12):
        r = fe.process_stride(state, p_vocoder=0.1, p_prosody=0.1, packet_loss=True,
                               stride_start_perf_counter=0.0)
    assert r.risk_level == RiskLevel.LOW_CONFIDENCE
