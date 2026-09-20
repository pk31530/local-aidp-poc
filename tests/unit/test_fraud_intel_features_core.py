"""Phase 2: shared feature core -- FeatureComputationContext validation,
hand-computed feature values, as-of-time fail-fast behavior, deterministic
ordering, and no-history neutral defaults. No database, Docker, or network
access anywhere in this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
    ordered_feature_vector,
)

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ob_event(
    *,
    event_id: uuid.UUID | None = None,
    event_timestamp: datetime = T0,
    amount_minor_units: int = 10_000,
    customer_id: str = CUSTOMER,
    account_id: str = ACCOUNT,
    device_id: str | None = "DEV1",
    ip_address: str | None = None,
    transaction_type: str = "transfer",
    mfa_used_flag: bool = True,
) -> FraudEvent:
    return FraudEvent(
        event_id=event_id or uuid.uuid4(),
        channel="online_banking",
        customer_id=customer_id,
        account_id=account_id,
        event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units,
        direction="debit",
        device_id=device_id,
        ip_address=ip_address,
        channel_payload=OnlineBankingPayload(
            session_id="SESS1",
            login_method="password",
            mfa_used_flag=mfa_used_flag,
            transaction_type=transaction_type,
            target_account="TGT1",
        ),
    )


def _source_alert(*, event_id: uuid.UUID, created_at: datetime) -> SourceAlertContext:
    return SourceAlertContext(
        source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id,
        source_alert_created_at=created_at,
        source_rule_ids=["RULE1"],
        source_rule_version="v1",
        source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        created_at=created_at,
    )


# ---- FeatureComputationContext validation --------------------------------------


def test_context_accepts_valid_input():
    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    assert ctx.as_of_time == current.event_timestamp


def test_as_of_time_must_equal_current_event_timestamp():
    current = _ob_event()
    with pytest.raises(ValidationError, match="as_of_time must equal current_event.event_timestamp"):
        FeatureComputationContext(
            current_event=current,
            historical_events=(),
            source_alert_history=(),
            as_of_time=current.event_timestamp + timedelta(seconds=1),
        )


def test_current_event_excluded_from_its_own_history_raises():
    current = _ob_event()
    with pytest.raises(ValidationError, match="must not include the current event"):
        FeatureComputationContext(
            current_event=current,
            historical_events=(current,),
            source_alert_history=(),
            as_of_time=current.event_timestamp,
        )


def test_boundary_historical_event_at_exact_as_of_time_raises():
    current = _ob_event()
    boundary_record = _ob_event(event_timestamp=current.event_timestamp)  # exactly at T, distinct event_id
    with pytest.raises(ValidationError, match="future or boundary records"):
        FeatureComputationContext(
            current_event=current,
            historical_events=(boundary_record,),
            source_alert_history=(),
            as_of_time=current.event_timestamp,
        )


def test_future_historical_event_raises():
    current = _ob_event()
    future_record = _ob_event(event_timestamp=current.event_timestamp + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="future or boundary records"):
        FeatureComputationContext(
            current_event=current,
            historical_events=(future_record,),
            source_alert_history=(),
            as_of_time=current.event_timestamp,
        )


def test_boundary_source_alert_at_exact_as_of_time_raises():
    current = _ob_event()
    alert = _source_alert(event_id=uuid.uuid4(), created_at=current.event_timestamp)
    with pytest.raises(ValidationError, match="future or boundary source alerts"):
        FeatureComputationContext(
            current_event=current, historical_events=(), source_alert_history=(alert,), as_of_time=current.event_timestamp
        )


def test_future_source_alert_raises():
    current = _ob_event()
    alert = _source_alert(event_id=uuid.uuid4(), created_at=current.event_timestamp + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="future or boundary source alerts"):
        FeatureComputationContext(
            current_event=current, historical_events=(), source_alert_history=(alert,), as_of_time=current.event_timestamp
        )


def test_current_events_own_source_alert_cannot_be_counted_as_prior():
    """A source alert fired ON the current event itself cannot have been
    created strictly before the current event's own timestamp -- so it is
    always rejected by the boundary/future check, proving it can never
    sneak into prior_alert_count."""
    current = _ob_event()
    current_events_alert = _source_alert(event_id=current.event_id, created_at=current.event_timestamp)
    with pytest.raises(ValidationError, match="future or boundary source alerts"):
        FeatureComputationContext(
            current_event=current,
            historical_events=(),
            source_alert_history=(current_events_alert,),
            as_of_time=current.event_timestamp,
        )


def test_context_is_frozen():
    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    with pytest.raises(ValidationError):
        ctx.as_of_time = current.event_timestamp + timedelta(days=1)


# ---- compute_shared_features: hand-computed values -----------------------------


def test_no_history_neutral_defaults():
    current = _ob_event(amount_minor_units=10_000, device_id="DEV1")
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    f = compute_shared_features(ctx)
    assert f["amount_vs_entity_average"] == 1.0
    assert f["amount_zscore"] == 0.0
    assert f["is_new_device"] is True
    assert f["is_new_ip"] is False  # no ip_address on current event -> not applicable, not flagged
    assert f["events_last_10m"] == 0
    assert f["events_last_1h"] == 0
    assert f["events_last_24h"] == 0
    assert f["prior_alert_count"] == 0
    assert f["customer_observed_tenure_days"] == 0
    assert f["account_observed_tenure_days"] == 0
    assert f["counterparty_risk_score"] == pytest.approx(0.2)


def test_amount_vs_entity_average_and_zscore_hand_computed():
    history = tuple(
        _ob_event(event_timestamp=T0 - timedelta(hours=h), amount_minor_units=amt)
        for h, amt in [(1, 1000), (2, 2000), (3, 3000)]
    )
    current = _ob_event(amount_minor_units=4000)
    ctx = FeatureComputationContext(
        current_event=current, historical_events=history, source_alert_history=(), as_of_time=current.event_timestamp
    )
    f = compute_shared_features(ctx)
    mean = 2000.0
    stddev = (((1000 - mean) ** 2 + (2000 - mean) ** 2 + (3000 - mean) ** 2) / 3) ** 0.5
    assert f["amount_vs_entity_average"] == pytest.approx(4000 / mean)
    assert f["amount_zscore"] == pytest.approx((4000 - mean) / stddev)


def test_is_new_device_false_when_device_seen_before():
    history = (_ob_event(event_timestamp=T0 - timedelta(hours=1), device_id="DEV1"),)
    current = _ob_event(device_id="DEV1")
    ctx = FeatureComputationContext(
        current_event=current, historical_events=history, source_alert_history=(), as_of_time=current.event_timestamp
    )
    f = compute_shared_features(ctx)
    assert f["is_new_device"] is False


def test_velocity_windows_hand_computed():
    now = T0
    history = (
        _ob_event(event_timestamp=now - timedelta(minutes=5)),   # in 10m, 1h, 24h
        _ob_event(event_timestamp=now - timedelta(minutes=40)),  # in 1h, 24h
        _ob_event(event_timestamp=now - timedelta(hours=20)),    # in 24h
        _ob_event(event_timestamp=now - timedelta(hours=30)),    # outside all
    )
    current = _ob_event(event_timestamp=now)
    ctx = FeatureComputationContext(
        current_event=current, historical_events=history, source_alert_history=(), as_of_time=now
    )
    f = compute_shared_features(ctx)
    assert f["events_last_10m"] == 1
    assert f["events_last_1h"] == 2
    assert f["events_last_24h"] == 3


def test_current_event_is_excluded_from_velocity_by_construction():
    """A history entry at exactly T (the current event's own time) is
    rejected at context construction -- proving the current event cannot
    inflate its own velocity counts."""
    current = _ob_event()
    would_be_self = _ob_event(event_timestamp=current.event_timestamp)
    with pytest.raises(ValidationError):
        FeatureComputationContext(
            current_event=current,
            historical_events=(would_be_self,),
            source_alert_history=(),
            as_of_time=current.event_timestamp,
        )


def test_prior_alert_count_counts_only_strictly_prior_alerts():
    alerts = (
        _source_alert(event_id=uuid.uuid4(), created_at=T0 - timedelta(hours=1)),
        _source_alert(event_id=uuid.uuid4(), created_at=T0 - timedelta(days=2)),
    )
    current = _ob_event(event_timestamp=T0)
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=alerts, as_of_time=T0
    )
    f = compute_shared_features(ctx)
    assert f["prior_alert_count"] == 2


def test_observed_tenure_derived_from_earliest_eligible_event():
    history = (
        _ob_event(event_timestamp=T0 - timedelta(days=10), customer_id=CUSTOMER, account_id=ACCOUNT),
        _ob_event(event_timestamp=T0 - timedelta(days=30), customer_id=CUSTOMER, account_id=ACCOUNT),  # earliest
        _ob_event(event_timestamp=T0 - timedelta(days=5), customer_id=CUSTOMER, account_id=ACCOUNT),
    )
    current = _ob_event(event_timestamp=T0)
    ctx = FeatureComputationContext(
        current_event=current, historical_events=history, source_alert_history=(), as_of_time=T0
    )
    f = compute_shared_features(ctx)
    assert f["customer_observed_tenure_days"] == 30
    assert f["account_observed_tenure_days"] == 30


def test_observed_tenure_scoped_separately_for_customer_and_account():
    """A history event that matches the customer but a DIFFERENT account
    must not extend the account-tenure feature, and vice versa."""
    other_account_record = _ob_event(
        event_timestamp=T0 - timedelta(days=100), customer_id=CUSTOMER, account_id="FIA-OTHER"
    )
    same_account_record = _ob_event(event_timestamp=T0 - timedelta(days=7), customer_id=CUSTOMER, account_id=ACCOUNT)
    current = _ob_event(event_timestamp=T0, customer_id=CUSTOMER, account_id=ACCOUNT)
    ctx = FeatureComputationContext(
        current_event=current,
        historical_events=(other_account_record, same_account_record),
        source_alert_history=(),
        as_of_time=T0,
    )
    f = compute_shared_features(ctx)
    assert f["customer_observed_tenure_days"] == 100  # other_account_record still matches the customer
    assert f["account_observed_tenure_days"] == 7      # but only same_account_record matches the account


def test_counterparty_risk_score_uses_lookup_when_key_provided():
    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    risk = CounterpartyRiskLookups(risk_by_key={"TGT1": 0.9}, default_risk=0.2)
    f = compute_shared_features(ctx, counterparty_key="TGT1", counterparty_risk=risk)
    assert f["counterparty_risk_score"] == pytest.approx(0.9)

    f_unseen = compute_shared_features(ctx, counterparty_key="UNSEEN", counterparty_risk=risk)
    assert f_unseen["counterparty_risk_score"] == pytest.approx(0.2)


# ---- deterministic ordered output -----------------------------------------------


def test_ordered_feature_vector_matches_shared_feature_columns_order():
    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    f = compute_shared_features(ctx)
    vector = ordered_feature_vector(f, SHARED_FEATURE_COLUMNS)
    assert vector == [f[c] for c in SHARED_FEATURE_COLUMNS]
    assert len(vector) == len(SHARED_FEATURE_COLUMNS)


def test_ordered_feature_vector_is_deterministic_across_repeated_calls():
    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    v1 = ordered_feature_vector(compute_shared_features(ctx), SHARED_FEATURE_COLUMNS)
    v2 = ordered_feature_vector(compute_shared_features(ctx), SHARED_FEATURE_COLUMNS)
    assert v1 == v2


def test_ordered_feature_vector_raises_on_missing_column():
    with pytest.raises(ValueError, match="missing required columns"):
        ordered_feature_vector({"amount_zscore": 0.0}, SHARED_FEATURE_COLUMNS)


def test_feature_schema_version_is_defined_and_stable():
    assert FEATURE_SCHEMA_VERSION == "v1"


# ---- JSON-safety -----------------------------------------------------------------


def test_shared_feature_values_are_json_safe_primitives():
    import json

    current = _ob_event()
    ctx = FeatureComputationContext(
        current_event=current, historical_events=(), source_alert_history=(), as_of_time=current.event_timestamp
    )
    f = compute_shared_features(ctx)
    for key, value in f.items():
        assert type(value) in (float, int, bool), f"{key} has non-JSON-safe type {type(value)}"
    json.dumps(f)  # must not raise
