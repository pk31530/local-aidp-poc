"""Phase 7A: Debit Card channel feature adapter. No database, Docker, or
network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.features.channels.debit_card import (
    DEBIT_CARD_FEATURE_COLUMNS,
    DEBIT_CARD_FEATURE_SCHEMA_VERSION,
    compute_debit_card_features,
    extract_entities,
)
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _debit_card_event(
    *,
    event_timestamp: datetime = T0,
    device_id: str | None = "DEV1",
    card_present_flag: bool = True,
    cross_border_flag: bool = False,
    merchant_id: str = "MER1",
    card_token: str | None = "CARDTOK1",
) -> FraudEvent:
    return FraudEvent(
        channel="debit_card", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=10_000, direction="debit", device_id=device_id,
        channel_payload=DebitCardPayload(
            merchant_id=merchant_id, mcc_code="5411", pos_entry_mode="chip", card_present_flag=card_present_flag,
            cross_border_flag=cross_border_flag, card_token=card_token,
        ),
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


def test_compute_debit_card_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='debit_card'"):
        compute_debit_card_features(ctx)


# ---- card_not_present_flag ----------------------------------------------------------


def test_card_not_present_flag_true_when_card_not_present():
    f = compute_debit_card_features(_ctx(_debit_card_event(card_present_flag=False)))
    assert f["card_not_present_flag"] is True


def test_card_not_present_flag_false_when_card_present():
    f = compute_debit_card_features(_ctx(_debit_card_event(card_present_flag=True)))
    assert f["card_not_present_flag"] is False


# ---- cross_border_new_device_combo_flag --------------------------------------------


def test_cross_border_new_device_combo_flag_true_for_cross_border_new_device():
    f = compute_debit_card_features(_ctx(_debit_card_event(cross_border_flag=True, device_id="DEV-NEW")))
    assert f["cross_border_new_device_combo_flag"] is True


def test_cross_border_new_device_combo_flag_false_when_device_is_known():
    prior = _debit_card_event(event_timestamp=T0 - timedelta(hours=1), device_id="DEV-KNOWN")
    current = _debit_card_event(event_timestamp=T0, cross_border_flag=True, device_id="DEV-KNOWN")
    f = compute_debit_card_features(_ctx(current, history=(prior,)))
    assert f["cross_border_new_device_combo_flag"] is False


# ---- distinct_merchant_count_1h ------------------------------------------------------


def test_distinct_merchant_count_counts_distinct_merchants_within_window():
    prior = _debit_card_event(event_timestamp=T0 - timedelta(minutes=30), merchant_id="MER-A")
    current = _debit_card_event(event_timestamp=T0, merchant_id="MER-B")
    f = compute_debit_card_features(_ctx(current, history=(prior,)))
    assert f["distinct_merchant_count_1h"] == 2


def test_distinct_merchant_count_ignores_events_outside_window():
    prior = _debit_card_event(event_timestamp=T0 - timedelta(hours=2), merchant_id="MER-A")
    current = _debit_card_event(event_timestamp=T0, merchant_id="MER-B")
    f = compute_debit_card_features(_ctx(current, history=(prior,)))
    assert f["distinct_merchant_count_1h"] == 1


# ---- entity extraction: CARD (Phase 7A additive correction) -----------------------


def test_extract_entities_emits_card_when_token_present():
    event = _debit_card_event(card_token="CARDTOK99")
    entities = extract_entities(event)
    assert ("card", "CARDTOK99") in entities


def test_extract_entities_omits_card_when_token_absent():
    event = _debit_card_event(card_token=None)
    entities = extract_entities(event)
    assert not any(entity_type == "card" for entity_type, _ in entities)


# ---- deterministic ordering / schema version --------------------------------------


def test_debit_card_feature_columns_ordered_vector():
    current = _debit_card_event()
    f = compute_debit_card_features(_ctx(current))
    vector = ordered_feature_vector(f, DEBIT_CARD_FEATURE_COLUMNS)
    assert vector == [f[c] for c in DEBIT_CARD_FEATURE_COLUMNS]
    assert set(DEBIT_CARD_FEATURE_COLUMNS) == set(f.keys())


def test_debit_card_feature_schema_version_defined():
    assert DEBIT_CARD_FEATURE_SCHEMA_VERSION == "v1"
