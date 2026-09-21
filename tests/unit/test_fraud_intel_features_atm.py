"""Phase 7A: ATM channel feature adapter. No database, Docker, or
network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.features.channels.atm import ATM_FEATURE_COLUMNS, ATM_FEATURE_SCHEMA_VERSION, compute_atm_features
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _atm_event(
    *,
    event_timestamp: datetime = T0,
    amount_minor_units: int = 10_000,
    atm_geo_bucket: str = "domestic",
    atm_id: str = "ATM1",
) -> FraudEvent:
    return FraudEvent(
        channel="atm", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units, direction="debit",
        channel_payload=ATMPayload(atm_id=atm_id, atm_geo_bucket=atm_geo_bucket, transaction_type="withdrawal", card_present_flag=True),
    )


def _ach_event(*, event_timestamp: datetime = T0) -> FraudEvent:
    return FraudEvent(
        channel="ach", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=5_000, direction="debit",
        channel_payload=ACHPayload(sec_code="PPD", originating_routing_number="123456789", receiving_routing_number="987654321", batch_id="B1", effective_entry_date=event_timestamp.date(), company_id="C1"),
    )


def _ctx(current: FraudEvent, history: tuple[FraudEvent, ...] = ()) -> FeatureComputationContext:
    return FeatureComputationContext(current_event=current, historical_events=history, source_alert_history=(), as_of_time=current.event_timestamp)


# ---- channel mismatch rejection --------------------------------------------------


def test_compute_atm_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='atm'"):
        compute_atm_features(ctx)


# ---- is_international_geo_flag -----------------------------------------------------


def test_is_international_geo_flag_true_for_international_bucket():
    f = compute_atm_features(_ctx(_atm_event(atm_geo_bucket="international")))
    assert f["is_international_geo_flag"] is True


def test_is_international_geo_flag_false_for_domestic_bucket():
    f = compute_atm_features(_ctx(_atm_event(atm_geo_bucket="domestic")))
    assert f["is_international_geo_flag"] is False


# ---- geo_velocity_distinct_buckets_1h ----------------------------------------------


def test_geo_velocity_counts_distinct_buckets_within_window():
    prior = _atm_event(event_timestamp=T0 - timedelta(minutes=30), atm_geo_bucket="domestic")
    current = _atm_event(event_timestamp=T0, atm_geo_bucket="international")
    f = compute_atm_features(_ctx(current, history=(prior,)))
    assert f["geo_velocity_distinct_buckets_1h"] == 2


def test_geo_velocity_ignores_events_outside_window():
    prior = _atm_event(event_timestamp=T0 - timedelta(hours=2), atm_geo_bucket="international")
    current = _atm_event(event_timestamp=T0, atm_geo_bucket="domestic")
    f = compute_atm_features(_ctx(current, history=(prior,)))
    assert f["geo_velocity_distinct_buckets_1h"] == 1


# ---- cash_out_ratio_vs_average ------------------------------------------------------


def test_cash_out_ratio_neutral_default_with_no_history():
    f = compute_atm_features(_ctx(_atm_event(amount_minor_units=10_000)))
    assert f["cash_out_ratio_vs_average"] == pytest.approx(1.0)


def test_cash_out_ratio_reflects_a_spike_above_average():
    prior1 = _atm_event(event_timestamp=T0 - timedelta(hours=1), amount_minor_units=10_000)
    prior2 = _atm_event(event_timestamp=T0 - timedelta(hours=2), amount_minor_units=10_000)
    current = _atm_event(event_timestamp=T0, amount_minor_units=40_000)
    f = compute_atm_features(_ctx(current, history=(prior1, prior2)))
    assert f["cash_out_ratio_vs_average"] == pytest.approx(4.0)


# ---- entity extraction --------------------------------------------------------------


def test_extract_entities_includes_atm_id():
    from src.fraud_intel.features.channels.atm import extract_entities

    event = _atm_event(atm_id="ATM-XYZ")
    entities = extract_entities(event)
    assert ("atm", "ATM-XYZ") in entities


# ---- deterministic ordering / schema version --------------------------------------


def test_atm_feature_columns_ordered_vector():
    current = _atm_event()
    f = compute_atm_features(_ctx(current))
    vector = ordered_feature_vector(f, ATM_FEATURE_COLUMNS)
    assert vector == [f[c] for c in ATM_FEATURE_COLUMNS]
    assert set(ATM_FEATURE_COLUMNS) == set(f.keys())


def test_atm_feature_schema_version_defined():
    assert ATM_FEATURE_SCHEMA_VERSION == "v1"
