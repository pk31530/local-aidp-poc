"""Phase 5: versioned ensemble policy (guide section 16). No database,
Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from src.fraud_intel.ensemble.policy import (
    EnsemblePolicy,
    band_priority,
    compute_operational_priority_score,
    load_ensemble_policy,
    max_priority_band,
)


def _policy(**overrides) -> EnsemblePolicy:
    base = dict(
        channel="online_banking", policy_version="v1",
        weight_rule=0.35, weight_gbm=0.40, weight_anomaly=0.15, weight_graph=0.10,
        rule_score_cap=1.0, high_threshold=0.75, medium_threshold=0.40,
    )
    base.update(overrides)
    return EnsemblePolicy(**base)


# ---- policy loading/validation --------------------------------------------------


def test_ensemble_policy_loads_and_validates_the_real_yaml_file():
    policy = load_ensemble_policy("online_banking")
    assert policy.channel == "online_banking"
    assert policy.policy_version == "v1"
    assert policy.rule_score_cap > 0


@pytest.mark.parametrize(
    "channel", sorted({"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"})
)
def test_ensemble_policy_loads_and_validates_for_every_channel(channel):
    """Phase 7A: all 7 channels now have a real ensemble_policy_<channel>.yaml
    file, not just online_banking."""
    policy = load_ensemble_policy(channel)
    assert policy.channel == channel
    assert policy.policy_version == "v1"
    assert policy.rule_score_cap > 0
    if channel != "online_banking":
        # Phase 7A decision 4: every NEW channel's policy must explicitly
        # declare its POC-default status -- online_banking's own policy
        # predates these fields and is exempt.
        assert policy.calibration_status == "UNVALIDATED_POC_DEFAULT"
        assert policy.promotion_note


def test_ensemble_policy_rejects_medium_threshold_above_high():
    with pytest.raises(ValidationError):
        _policy(high_threshold=0.5, medium_threshold=0.6)


def test_ensemble_policy_rejects_negative_weight():
    with pytest.raises(ValidationError):
        _policy(weight_rule=-0.1)


def test_ensemble_policy_rejects_non_positive_rule_score_cap():
    with pytest.raises(ValidationError):
        _policy(rule_score_cap=0.0)


# ---- LR never in the ensemble signature -----------------------------------------


def test_compute_operational_priority_score_has_no_lr_probability_parameter():
    sig = inspect.signature(compute_operational_priority_score)
    param_names = set(sig.parameters.keys())
    assert "lr_probability" not in param_names
    assert not any("lr" in name.lower() for name in param_names)


# ---- ensemble formula -------------------------------------------------------------


def test_score_is_weighted_sum_of_all_four_normalized_components():
    policy = _policy(weight_rule=0.25, weight_gbm=0.25, weight_anomaly=0.25, weight_graph=0.25, rule_score_cap=1.0)
    score = compute_operational_priority_score(
        rule_score_contribution=1.0, calibrated_gbm_probability=1.0, anomaly_score=1.0, graph_risk_score=1.0, policy=policy
    )
    assert score == pytest.approx(1.0)


def test_score_bounds_never_exceed_zero_to_one_even_with_extreme_inputs():
    policy = _policy(weight_rule=1.0, weight_gbm=1.0, weight_anomaly=1.0, weight_graph=1.0, rule_score_cap=1.0)
    score = compute_operational_priority_score(
        rule_score_contribution=100.0, calibrated_gbm_probability=1.0, anomaly_score=1.0, graph_risk_score=1.0, policy=policy
    )
    assert score == pytest.approx(1.0)
    assert 0.0 <= score <= 1.0


def test_zero_inputs_produce_zero_score():
    policy = _policy()
    score = compute_operational_priority_score(
        rule_score_contribution=0.0, calibrated_gbm_probability=0.0, anomaly_score=0.0, graph_risk_score=0.0, policy=policy
    )
    assert score == 0.0


# ---- rule_score_cap boundary tests (Phase 5 decision 7) ---------------------------


def test_rule_score_cap_negative_contribution_clamped_to_zero():
    policy = _policy(weight_rule=1.0, weight_gbm=0, weight_anomaly=0, weight_graph=0, rule_score_cap=1.0)
    score = compute_operational_priority_score(
        rule_score_contribution=-5.0, calibrated_gbm_probability=0, anomaly_score=0, graph_risk_score=0, policy=policy
    )
    assert score == 0.0


def test_rule_score_cap_at_cap_normalizes_to_one():
    policy = _policy(weight_rule=1.0, weight_gbm=0, weight_anomaly=0, weight_graph=0, rule_score_cap=0.5)
    score = compute_operational_priority_score(
        rule_score_contribution=0.5, calibrated_gbm_probability=0, anomaly_score=0, graph_risk_score=0, policy=policy
    )
    assert score == pytest.approx(1.0)


def test_rule_score_cap_above_cap_is_clamped_not_exceeded():
    policy = _policy(weight_rule=1.0, weight_gbm=0, weight_anomaly=0, weight_graph=0, rule_score_cap=0.5)
    at_cap = compute_operational_priority_score(
        rule_score_contribution=0.5, calibrated_gbm_probability=0, anomaly_score=0, graph_risk_score=0, policy=policy
    )
    above_cap = compute_operational_priority_score(
        rule_score_contribution=50.0, calibrated_gbm_probability=0, anomaly_score=0, graph_risk_score=0, policy=policy
    )
    assert at_cap == above_cap == pytest.approx(1.0)


def test_rule_score_below_cap_scales_linearly():
    policy = _policy(weight_rule=1.0, weight_gbm=0, weight_anomaly=0, weight_graph=0, rule_score_cap=1.0)
    score = compute_operational_priority_score(
        rule_score_contribution=0.25, calibrated_gbm_probability=0, anomaly_score=0, graph_risk_score=0, policy=policy
    )
    assert score == pytest.approx(0.25)


# ---- priority band / mandatory minimum enforcement --------------------------------


def test_mandatory_review_hit_forces_high_regardless_of_score():
    policy = _policy()
    band = band_priority(0.0, policy, mandatory_review_hit=True, provider_status="OK", minimum_priority_band=None)
    assert band == "HIGH"


def test_thresholds_determine_band_when_no_override():
    policy = _policy(high_threshold=0.75, medium_threshold=0.40)
    assert band_priority(0.9, policy, mandatory_review_hit=False, provider_status="OK", minimum_priority_band=None) == "HIGH"
    assert band_priority(0.5, policy, mandatory_review_hit=False, provider_status="OK", minimum_priority_band=None) == "MEDIUM"
    assert band_priority(0.1, policy, mandatory_review_hit=False, provider_status="OK", minimum_priority_band=None) == "LOW"


def test_low_band_is_real_and_retained_not_dropped():
    policy = _policy(high_threshold=0.99, medium_threshold=0.99)
    band = band_priority(0.0, policy, mandatory_review_hit=False, provider_status="OK", minimum_priority_band=None)
    assert band == "LOW"


def test_provider_failure_floors_band_at_minimum_never_lower():
    policy = _policy(high_threshold=0.75, medium_threshold=0.40)
    band = band_priority(0.0, policy, mandatory_review_hit=False, provider_status="ERROR", minimum_priority_band="MEDIUM")
    assert band == "MEDIUM"


def test_provider_failure_floor_never_lowers_an_already_higher_band():
    policy = _policy(high_threshold=0.75, medium_threshold=0.40)
    band = band_priority(0.9, policy, mandatory_review_hit=False, provider_status="ERROR", minimum_priority_band="MEDIUM")
    assert band == "HIGH"


def test_provider_ok_ignores_minimum_priority_band():
    policy = _policy(high_threshold=0.75, medium_threshold=0.40)
    band = band_priority(0.0, policy, mandatory_review_hit=False, provider_status="OK", minimum_priority_band="HIGH")
    assert band == "LOW"


def test_max_priority_band_ordering():
    assert max_priority_band("LOW", "MEDIUM") == "MEDIUM"
    assert max_priority_band("HIGH", "MEDIUM") == "HIGH"
    assert max_priority_band("LOW", "LOW") == "LOW"
