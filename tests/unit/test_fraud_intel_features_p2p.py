"""Phase 7A: P2P/Instant Payment channel feature adapter. No database,
Docker, or network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.features.channels.p2p import (
    P2P_FEATURE_COLUMNS,
    P2P_FEATURE_SCHEMA_VERSION,
    compute_p2p_features,
    extract_entities,
)
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _p2p_event(
    *,
    event_timestamp: datetime = T0,
    amount_minor_units: int = 15_000,
    recipient_is_new_flag: bool = False,
    memo_present_flag: bool = True,
    recipient_handle: str = "@recipient1",
) -> FraudEvent:
    return FraudEvent(
        channel="p2p", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units, direction="debit",
        channel_payload=P2PPayload(recipient_handle=recipient_handle, network="zelle", memo_present_flag=memo_present_flag, recipient_is_new_flag=recipient_is_new_flag),
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


def test_compute_p2p_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='p2p'"):
        compute_p2p_features(ctx)


# ---- direct passthrough flags ------------------------------------------------------


def test_recipient_is_new_flag_passthrough():
    f = compute_p2p_features(_ctx(_p2p_event(recipient_is_new_flag=True)))
    assert f["recipient_is_new_flag"] is True


def test_no_memo_flag_true_when_memo_absent():
    f = compute_p2p_features(_ctx(_p2p_event(memo_present_flag=False)))
    assert f["no_memo_flag"] is True


def test_no_memo_flag_false_when_memo_present():
    f = compute_p2p_features(_ctx(_p2p_event(memo_present_flag=True)))
    assert f["no_memo_flag"] is False


# ---- rapid_sequential_transfer_flag ------------------------------------------------


def test_rapid_sequential_transfer_flag_true_at_threshold():
    prior1 = _p2p_event(event_timestamp=T0 - timedelta(minutes=2))
    prior2 = _p2p_event(event_timestamp=T0 - timedelta(minutes=5))
    current = _p2p_event(event_timestamp=T0)
    f = compute_p2p_features(_ctx(current, history=(prior1, prior2)))
    assert f["rapid_sequential_transfer_flag"] is True


def test_rapid_sequential_transfer_flag_false_below_threshold():
    prior = _p2p_event(event_timestamp=T0 - timedelta(minutes=2))
    current = _p2p_event(event_timestamp=T0)
    f = compute_p2p_features(_ctx(current, history=(prior,)))
    assert f["rapid_sequential_transfer_flag"] is False


# ---- round_dollar_amount_flag -------------------------------------------------------


def test_round_dollar_amount_flag_true_for_round_amount():
    f = compute_p2p_features(_ctx(_p2p_event(amount_minor_units=20_000)))  # $200.00
    assert f["round_dollar_amount_flag"] is True


def test_round_dollar_amount_flag_false_for_non_round_amount():
    f = compute_p2p_features(_ctx(_p2p_event(amount_minor_units=20_037)))
    assert f["round_dollar_amount_flag"] is False


# ---- entity extraction --------------------------------------------------------------


def test_extract_entities_includes_recipient_handle():
    event = _p2p_event(recipient_handle="@mule99")
    entities = extract_entities(event)
    assert ("recipient", "@mule99") in entities


# ---- deterministic ordering / schema version --------------------------------------


def test_p2p_feature_columns_ordered_vector():
    current = _p2p_event()
    f = compute_p2p_features(_ctx(current))
    vector = ordered_feature_vector(f, P2P_FEATURE_COLUMNS)
    assert vector == [f[c] for c in P2P_FEATURE_COLUMNS]
    assert set(P2P_FEATURE_COLUMNS) == set(f.keys())


def test_p2p_feature_schema_version_defined():
    assert P2P_FEATURE_SCHEMA_VERSION == "v1"
