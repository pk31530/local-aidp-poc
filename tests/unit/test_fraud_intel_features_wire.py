"""Phase 7A: Wire channel feature adapter. No database, Docker, or
network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.features.channels.wire import WIRE_FEATURE_COLUMNS, WIRE_FEATURE_SCHEMA_VERSION, compute_wire_features
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _wire_event(
    *,
    event_timestamp: datetime = T0,
    amount_minor_units: int = 10_000,
    wire_type: str = "domestic",
    beneficiary_account: str = "BEN1",
) -> FraudEvent:
    return FraudEvent(
        channel="wire", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units, direction="debit",
        channel_payload=WirePayload(wire_type=wire_type, beneficiary_bank_id="BANK1", beneficiary_account=beneficiary_account, purpose_code="GDS"),
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


def test_compute_wire_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='wire'"):
        compute_wire_features(ctx)


# ---- is_international_flag ---------------------------------------------------------


def test_is_international_flag_true_for_international_wire():
    f = compute_wire_features(_ctx(_wire_event(wire_type="international")))
    assert f["is_international_flag"] is True


def test_is_international_flag_false_for_domestic_wire():
    f = compute_wire_features(_ctx(_wire_event(wire_type="domestic")))
    assert f["is_international_flag"] is False


# ---- first_time_beneficiary_flag ---------------------------------------------------


def test_first_time_beneficiary_flag_true_with_no_history():
    f = compute_wire_features(_ctx(_wire_event(beneficiary_account="BEN1")))
    assert f["first_time_beneficiary_flag"] is True


def test_first_time_beneficiary_flag_false_when_beneficiary_seen_before():
    prior = _wire_event(event_timestamp=T0 - timedelta(hours=1), beneficiary_account="BEN1")
    current = _wire_event(event_timestamp=T0, beneficiary_account="BEN1")
    f = compute_wire_features(_ctx(current, history=(prior,)))
    assert f["first_time_beneficiary_flag"] is False


# ---- amount_vs_historical_max_ratio ------------------------------------------------


def test_amount_vs_historical_max_ratio_neutral_default_with_no_history():
    f = compute_wire_features(_ctx(_wire_event(amount_minor_units=10_000)))
    assert f["amount_vs_historical_max_ratio"] == pytest.approx(1.0)


def test_amount_vs_historical_max_ratio_reflects_a_spike_above_history():
    prior = _wire_event(event_timestamp=T0 - timedelta(hours=1), amount_minor_units=10_000)
    current = _wire_event(event_timestamp=T0, amount_minor_units=50_000)
    f = compute_wire_features(_ctx(current, history=(prior,)))
    assert f["amount_vs_historical_max_ratio"] == pytest.approx(5.0)


# ---- deterministic ordering / schema version --------------------------------------


def test_wire_feature_columns_ordered_vector():
    current = _wire_event()
    f = compute_wire_features(_ctx(current))
    vector = ordered_feature_vector(f, WIRE_FEATURE_COLUMNS)
    assert vector == [f[c] for c in WIRE_FEATURE_COLUMNS]
    assert set(WIRE_FEATURE_COLUMNS) == set(f.keys())


def test_wire_feature_schema_version_defined():
    assert WIRE_FEATURE_SCHEMA_VERSION == "v1"
