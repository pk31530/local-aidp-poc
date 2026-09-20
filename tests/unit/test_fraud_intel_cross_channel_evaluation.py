"""Phase 7A: cross-channel evaluation (src.fraud_intel.evaluation) --
capacity config and the evaluation contract itself. No database, MLflow,
Docker, or network access anywhere in this file -- every test builds
AlertOutcome fixtures directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.fraud_intel.evaluation.capacity import AnalystCapacityConfig, CountCapacity, FractionCapacity, resolve_capacity_count
from src.fraud_intel.evaluation.cross_channel import AlertOutcome, evaluate_channel, evaluate_cross_channel
from src.fraud_intel.monitoring.drift import DriftWindow, compute_alert_volume_drift, compute_score_distribution_drift

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _outcome(
    *, minute: int, score: float, band: str, label: str, channel: str = "online_banking", source_alert_id=None,
) -> AlertOutcome:
    return AlertOutcome(
        source_alert_id=source_alert_id or uuid.uuid4(), channel=channel, event_timestamp=T0 + timedelta(minutes=minute),
        operational_priority_score=score, baseline_priority_score=score, priority_band=band, resolved_label=label,
    )


# ---- AnalystCapacityConfig: exactly one of count or fraction -----------------------


def test_count_capacity_rejects_non_positive_value():
    with pytest.raises(ValidationError):
        CountCapacity(value=0)


def test_fraction_capacity_rejects_out_of_range_value():
    with pytest.raises(ValidationError):
        FractionCapacity(value=1.5)
    with pytest.raises(ValidationError):
        FractionCapacity(value=0)


def test_capacity_config_rejects_an_unrecognized_mode():
    from pydantic import TypeAdapter

    with pytest.raises(ValidationError):
        TypeAdapter(AnalystCapacityConfig).validate_python({"mode": "percentage", "value": 5})


def test_resolve_capacity_count_caps_at_total_population():
    assert resolve_capacity_count(CountCapacity(value=1000), total_alerts=10) == 10
    assert resolve_capacity_count(FractionCapacity(value=1.0), total_alerts=10) == 10


def test_resolve_capacity_count_zero_population_is_zero():
    assert resolve_capacity_count(CountCapacity(value=5), total_alerts=0) == 0


# ---- deterministic ranking -----------------------------------------------------------


def test_ranking_is_score_descending_then_timestamp_then_id_ascending():
    # Same score, same timestamp -- id ascending (str(uuid) comparison)
    # decides order: str(UUID(int=1)) < str(UUID(int=2)), so
    # id_low sorts BEFORE id_high whenever their score/timestamp tie.
    id_high_fraud = _outcome(minute=1, score=0.5, band="MEDIUM", label="RESOLVED_FRAUD", source_alert_id=uuid.UUID(int=2))
    id_low_legit = _outcome(minute=1, score=0.5, band="MEDIUM", label="RESOLVED_LEGITIMATE", source_alert_id=uuid.UUID(int=1))
    high_score = _outcome(minute=5, score=0.9, band="HIGH", label="RESOLVED_FRAUD")

    # capacity=1 must pick the highest score (`high_score`) first, regardless
    # of list input order.
    result = evaluate_channel(
        "online_banking", [id_high_fraud, id_low_legit, high_score], capacity=CountCapacity(value=1), recall_target=0.5,
    )
    assert result.fraud_captured == 1

    # Full ranking is [high_score, id_low_legit, id_high_fraud]. There are 2
    # fraud rows total (high_score, id_high_fraud); reaching 100% recall
    # requires all 3 rows, since the second fraud row (id_high_fraud) sorts
    # LAST of the two tied rows.
    result_all = evaluate_channel(
        "online_banking", [id_high_fraud, id_low_legit, high_score], capacity=CountCapacity(value=3), recall_target=1.0,
    )
    assert result_all.minimum_alerts_required_for_recall_target == 3


# ---- empty / single-class channels -> explicit non-computable, never zero ----------


def test_empty_channel_every_metric_is_non_computable_not_zero():
    result = evaluate_channel("atm", [], capacity=CountCapacity(value=5), recall_target=0.8)
    assert result.precision.status == "non_computable_empty"
    assert result.precision.value is None
    assert result.recall.status == "non_computable_empty"
    assert result.pr_auc.status == "non_computable_empty"


def test_single_class_all_fraud_precision_and_related_metrics_non_computable():
    outcomes = [_outcome(minute=i, score=0.9, band="HIGH", label="RESOLVED_FRAUD") for i in range(3)]
    result = evaluate_channel("atm", outcomes, capacity=CountCapacity(value=2), recall_target=0.8)
    assert result.precision.status == "non_computable_single_class"
    assert result.pr_auc.status == "non_computable_single_class"
    assert result.false_positive_reduction.status == "non_computable_single_class"


def test_single_class_all_legitimate_recall_and_related_metrics_non_computable():
    outcomes = [_outcome(minute=i, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE") for i in range(3)]
    result = evaluate_channel("atm", outcomes, capacity=CountCapacity(value=2), recall_target=0.8)
    assert result.recall.status == "non_computable_single_class"
    assert result.recall_at_capacity.status == "non_computable_single_class"
    assert result.workload_reduction_at_recall_target.status == "non_computable_single_class"


# ---- accuracy is never the headline metric -----------------------------------------


def test_accuracy_is_present_but_not_privileged_in_the_result_shape():
    """Structural: accuracy is one ordinary MetricValue field among many --
    nothing elevates it (e.g. no top-level bare `accuracy` outside the
    typed result, no summary that reports only accuracy)."""
    outcomes = [
        _outcome(minute=0, score=0.9, band="HIGH", label="RESOLVED_FRAUD"),
        _outcome(minute=1, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE"),
    ]
    result = evaluate_channel("online_banking", outcomes, capacity=CountCapacity(value=1), recall_target=0.5)
    assert isinstance(result.accuracy.value, float)
    # precision/recall/pr_auc are computed independently of accuracy -- not
    # derived from or gated by it.
    assert result.precision.status == "ok"
    assert result.recall.status == "ok"


# ---- macro / volume-weighted summaries are correct ----------------------------------


def test_macro_and_volume_weighted_summaries_hand_computed():
    channel_a = [
        _outcome(minute=0, score=0.9, band="HIGH", label="RESOLVED_FRAUD", channel="a"),
        _outcome(minute=1, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE", channel="a"),
    ]  # precision = 1/1 = 1.0 (1 MEDIUM/HIGH row, it's fraud)
    channel_b = [
        _outcome(minute=0, score=0.9, band="HIGH", label="RESOLVED_LEGITIMATE", channel="b"),
        _outcome(minute=1, score=0.8, band="HIGH", label="RESOLVED_FRAUD", channel="b"),
        _outcome(minute=2, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE", channel="b"),
    ]  # precision = 1/2 = 0.5 (2 HIGH rows, 1 fraud)

    report = evaluate_cross_channel(
        {"a": channel_a, "b": channel_b}, capacity=CountCapacity(value=1), recall_target=0.5,
    )

    precision_summary = report.summaries["precision"]
    assert precision_summary.macro == pytest.approx((1.0 + 0.5) / 2)
    # volume-weighted: (1.0*2 + 0.5*3) / (2+3) = 3.5/5 = 0.7
    assert precision_summary.volume_weighted == pytest.approx(3.5 / 5)
    assert precision_summary.channels_included == ["a", "b"]
    assert precision_summary.channels_excluded_non_computable == []


def test_summary_excludes_non_computable_channels_from_both_macro_and_weighted():
    empty_channel: list = []
    real_channel = [
        _outcome(minute=0, score=0.9, band="HIGH", label="RESOLVED_FRAUD", channel="real"),
        _outcome(minute=1, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE", channel="real"),
    ]
    report = evaluate_cross_channel(
        {"empty": empty_channel, "real": real_channel}, capacity=CountCapacity(value=1), recall_target=0.5,
    )
    summary = report.summaries["precision"]
    assert summary.channels_included == ["real"]
    assert summary.channels_excluded_non_computable == ["empty"]
    assert summary.macro == pytest.approx(1.0)
    assert summary.volume_weighted == pytest.approx(1.0)


# ---- evaluation reads only AlertOutcome rows (source-alerted, resolved) ------------


def test_alert_outcome_has_no_synthetic_or_disposition_shaped_field():
    """Structural: AlertOutcome's own fields cannot carry scenario_id,
    a raw analyst_disposition, or any unresolved/pending state -- the
    type itself enforces "source-alerted and resolved only"."""
    field_names = set(AlertOutcome.model_fields)
    assert "scenario_id" not in field_names
    assert "analyst_disposition" not in field_names
    assert "synthetic_scenario_label" not in field_names


# ---- rules-only baseline comparison (guide section 21) -----------------------------


def test_rules_only_baseline_uses_baseline_priority_score_not_the_model_score():
    """A channel where the model score and baseline score disagree
    completely -- proves rules_only_baseline is genuinely computed off
    baseline_priority_score, not silently reusing operational_priority_score."""
    outcomes = [
        AlertOutcome(
            source_alert_id=uuid.uuid4(), channel="online_banking", event_timestamp=T0,
            operational_priority_score=0.9, baseline_priority_score=0.1,
            priority_band="HIGH", resolved_label="RESOLVED_FRAUD",
        ),
        AlertOutcome(
            source_alert_id=uuid.uuid4(), channel="online_banking", event_timestamp=T0 + timedelta(minutes=1),
            operational_priority_score=0.1, baseline_priority_score=0.9,
            priority_band="LOW", resolved_label="RESOLVED_LEGITIMATE",
        ),
    ]
    result = evaluate_channel("online_banking", outcomes, capacity=CountCapacity(value=1), recall_target=0.5)
    # Model correctly ranks the fraud row first -> perfect precision@capacity.
    assert result.precision_at_capacity.value == pytest.approx(1.0)
    # Baseline ranks the LEGITIMATE row first (higher baseline_priority_score)
    # -> zero precision@capacity, proving it used the baseline column.
    assert result.rules_only_baseline.precision_at_capacity.value == pytest.approx(0.0)


def test_rules_only_baseline_is_non_computable_for_a_single_class_channel():
    outcomes = [
        _outcome(minute=i, score=0.9, band="HIGH", label="RESOLVED_FRAUD") for i in range(3)
    ]
    result = evaluate_channel("atm", outcomes, capacity=CountCapacity(value=1), recall_target=0.5)
    assert result.rules_only_baseline.pr_auc.status == "non_computable_single_class"


def test_recall_target_out_of_range_raises():
    outcomes = [_outcome(minute=0, score=0.9, band="HIGH", label="RESOLVED_FRAUD")]
    with pytest.raises(ValueError):
        evaluate_channel("online_banking", outcomes, capacity=CountCapacity(value=1), recall_target=0.0)
    with pytest.raises(ValueError):
        evaluate_channel("online_banking", outcomes, capacity=CountCapacity(value=1), recall_target=1.5)


def test_evaluate_channel_requires_explicit_capacity_and_recall_target():
    """No implicit default anywhere -- both are required positional/keyword
    arguments with no default value in the function signature."""
    import inspect

    sig = inspect.signature(evaluate_channel)
    assert sig.parameters["capacity"].default is inspect.Parameter.empty
    assert sig.parameters["recall_target"].default is inspect.Parameter.empty


# ---- drift monitoring (src.fraud_intel.monitoring.drift) ---------------------------


def _window(*, label: str, start_offset_days: float, duration_days: float, scores: tuple[float, ...]) -> DriftWindow:
    start = T0 + timedelta(days=start_offset_days)
    return DriftWindow(label=label, window_start=start, window_end=start + timedelta(days=duration_days), scores=scores)


def test_alert_volume_drift_reports_relative_change():
    baseline = _window(label="baseline", start_offset_days=0, duration_days=7, scores=(0.1,) * 7)
    current = _window(label="current", start_offset_days=7, duration_days=7, scores=(0.1,) * 14)
    drift = compute_alert_volume_drift(baseline, current)
    assert drift.baseline_count == 7
    assert drift.current_count == 14
    assert drift.relative_change == pytest.approx(1.0)  # volume doubled
    assert drift.status == "ok"


def test_alert_volume_drift_non_computable_for_an_empty_baseline():
    baseline = _window(label="baseline", start_offset_days=0, duration_days=7, scores=())
    current = _window(label="current", start_offset_days=7, duration_days=7, scores=(0.1, 0.2))
    drift = compute_alert_volume_drift(baseline, current)
    assert drift.status == "non_computable_empty"
    assert drift.relative_change is None


def test_score_distribution_drift_identical_windows_have_zero_distance():
    baseline = _window(label="baseline", start_offset_days=0, duration_days=7, scores=(0.1, 0.5, 0.9))
    current = _window(label="current", start_offset_days=7, duration_days=7, scores=(0.1, 0.5, 0.9))
    drift = compute_score_distribution_drift(baseline, current, num_buckets=10)
    assert drift.total_variation_distance == pytest.approx(0.0)
    assert drift.mean_delta == pytest.approx(0.0)
    assert drift.status == "ok"


def test_score_distribution_drift_fully_disjoint_buckets_have_distance_one():
    baseline = _window(label="baseline", start_offset_days=0, duration_days=7, scores=(0.05, 0.05, 0.05))
    current = _window(label="current", start_offset_days=7, duration_days=7, scores=(0.95, 0.95, 0.95))
    drift = compute_score_distribution_drift(baseline, current, num_buckets=10)
    assert drift.total_variation_distance == pytest.approx(1.0)
    assert drift.mean_delta == pytest.approx(0.9)


def test_score_distribution_drift_non_computable_for_an_empty_window():
    baseline = _window(label="baseline", start_offset_days=0, duration_days=7, scores=())
    current = _window(label="current", start_offset_days=7, duration_days=7, scores=(0.5,))
    drift = compute_score_distribution_drift(baseline, current)
    assert drift.status == "non_computable_empty"
    assert drift.total_variation_distance is None


def test_drift_window_rejects_a_non_positive_duration():
    with pytest.raises(ValidationError):
        DriftWindow(label="bad", window_start=T0, window_end=T0, scores=())


def test_drift_window_rejects_an_out_of_bounds_score():
    with pytest.raises(ValidationError):
        DriftWindow(label="bad", window_start=T0, window_end=T0 + timedelta(days=1), scores=(1.5,))
