"""Phase 7A: ACH channel feature adapter. No database, Docker, or network
access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.features.channels.ach import ACH_FEATURE_COLUMNS, ACH_FEATURE_SCHEMA_VERSION, compute_ach_features
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ach_event(
    *,
    event_timestamp: datetime = T0,
    sec_code: str = "PPD",
    receiving_routing_number: str = "987654321",
    company_id: str = "COMP1",
) -> FraudEvent:
    return FraudEvent(
        channel="ach", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=5_000, direction="debit",
        channel_payload=ACHPayload(
            sec_code=sec_code, originating_routing_number="123456789", receiving_routing_number=receiving_routing_number,
            batch_id="BATCH1", effective_entry_date=event_timestamp.date(), company_id=company_id,
        ),
    )


def _ob_event(*, event_timestamp: datetime = T0) -> FraudEvent:
    return FraudEvent(
        channel="online_banking", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=5_000, direction="debit",
        channel_payload=OnlineBankingPayload(session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"),
    )


def _ctx(current: FraudEvent, history: tuple[FraudEvent, ...] = ()) -> FeatureComputationContext:
    return FeatureComputationContext(current_event=current, historical_events=history, source_alert_history=(), as_of_time=current.event_timestamp)


# ---- channel mismatch rejection --------------------------------------------------


def test_compute_ach_features_rejects_wrong_channel():
    ctx = _ctx(_ob_event())
    with pytest.raises(ValueError, match="requires channel='ach'"):
        compute_ach_features(ctx)


# ---- first_time_receiving_routing_number_flag -------------------------------------


def test_first_time_receiving_routing_number_flag_true_with_no_history():
    current = _ach_event()
    f = compute_ach_features(_ctx(current))
    assert f["first_time_receiving_routing_number_flag"] is True


def test_first_time_receiving_routing_number_flag_false_when_seen_before():
    prior = _ach_event(event_timestamp=T0 - timedelta(hours=1), receiving_routing_number="987654321")
    current = _ach_event(event_timestamp=T0, receiving_routing_number="987654321")
    f = compute_ach_features(_ctx(current, history=(prior,)))
    assert f["first_time_receiving_routing_number_flag"] is False


# ---- same_day_mixed_sec_code_flag --------------------------------------------------


def test_same_day_mixed_sec_code_flag_true_when_sec_code_differs_same_day():
    prior = _ach_event(event_timestamp=T0 - timedelta(hours=2), sec_code="CCD")
    current = _ach_event(event_timestamp=T0, sec_code="PPD")
    f = compute_ach_features(_ctx(current, history=(prior,)))
    assert f["same_day_mixed_sec_code_flag"] is True


def test_same_day_mixed_sec_code_flag_false_when_sec_code_matches():
    prior = _ach_event(event_timestamp=T0 - timedelta(hours=2), sec_code="PPD")
    current = _ach_event(event_timestamp=T0, sec_code="PPD")
    f = compute_ach_features(_ctx(current, history=(prior,)))
    assert f["same_day_mixed_sec_code_flag"] is False


def test_same_day_mixed_sec_code_flag_false_when_prior_is_a_different_day():
    prior = _ach_event(event_timestamp=T0 - timedelta(days=1), sec_code="CCD")
    current = _ach_event(event_timestamp=T0, sec_code="PPD")
    f = compute_ach_features(_ctx(current, history=(prior,)))
    assert f["same_day_mixed_sec_code_flag"] is False


# ---- first_time_company_id_flag ----------------------------------------------------


def test_first_time_company_id_flag_false_when_company_seen_before():
    prior = _ach_event(event_timestamp=T0 - timedelta(hours=1), company_id="COMP1")
    current = _ach_event(event_timestamp=T0, company_id="COMP1")
    f = compute_ach_features(_ctx(current, history=(prior,)))
    assert f["first_time_company_id_flag"] is False


# ---- deterministic ordering / schema version --------------------------------------


def test_ach_feature_columns_ordered_vector():
    current = _ach_event()
    f = compute_ach_features(_ctx(current))
    vector = ordered_feature_vector(f, ACH_FEATURE_COLUMNS)
    assert vector == [f[c] for c in ACH_FEATURE_COLUMNS]
    assert set(ACH_FEATURE_COLUMNS) == set(f.keys())


def test_ach_feature_schema_version_defined():
    assert ACH_FEATURE_SCHEMA_VERSION == "v1"
