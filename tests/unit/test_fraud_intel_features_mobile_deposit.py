"""Phase 7A: Mobile Check Deposit channel feature adapter. No database,
Docker, or network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.features.channels.mobile_deposit import (
    MOBILE_DEPOSIT_FEATURE_COLUMNS,
    MOBILE_DEPOSIT_FEATURE_SCHEMA_VERSION,
    compute_mobile_deposit_features,
)
from src.fraud_intel.features.channels.mobile_deposit import extract_entities
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _mobile_deposit_event(
    *,
    event_timestamp: datetime = T0,
    duplicate_image_hash_flag: bool = False,
    car_lar_mismatch_flag: bool = False,
    image_quality_score: float = 0.9,
    check_payee_token: str | None = "PAYEE1",
) -> FraudEvent:
    return FraudEvent(
        channel="mobile_deposit", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=10_000, direction="credit",
        channel_payload=MobileDepositPayload(
            duplicate_image_hash_flag=duplicate_image_hash_flag, car_lar_mismatch_flag=car_lar_mismatch_flag,
            signature_verification_flag=True, endorsement_present_flag=True, micr_consistency_flag=True,
            image_quality_score=image_quality_score, check_payee_token=check_payee_token,
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


def test_compute_mobile_deposit_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='mobile_deposit'"):
        compute_mobile_deposit_features(ctx)


# ---- direct passthrough flags ------------------------------------------------------


def test_duplicate_image_hash_flag_passthrough():
    f = compute_mobile_deposit_features(_ctx(_mobile_deposit_event(duplicate_image_hash_flag=True)))
    assert f["duplicate_image_hash_flag"] is True


def test_car_lar_mismatch_flag_passthrough():
    f = compute_mobile_deposit_features(_ctx(_mobile_deposit_event(car_lar_mismatch_flag=True)))
    assert f["car_lar_mismatch_flag"] is True


# ---- low_image_quality_flag ---------------------------------------------------------


def test_low_image_quality_flag_true_below_threshold():
    f = compute_mobile_deposit_features(_ctx(_mobile_deposit_event(image_quality_score=0.5)))
    assert f["low_image_quality_flag"] is True


def test_low_image_quality_flag_false_at_or_above_threshold():
    f = compute_mobile_deposit_features(_ctx(_mobile_deposit_event(image_quality_score=0.6)))
    assert f["low_image_quality_flag"] is False


# ---- rapid_resubmission_flag ---------------------------------------------------------


def test_rapid_resubmission_flag_true_within_window():
    prior = _mobile_deposit_event(event_timestamp=T0 - timedelta(minutes=5))
    current = _mobile_deposit_event(event_timestamp=T0)
    f = compute_mobile_deposit_features(_ctx(current, history=(prior,)))
    assert f["rapid_resubmission_flag"] is True


def test_rapid_resubmission_flag_false_outside_window():
    prior = _mobile_deposit_event(event_timestamp=T0 - timedelta(hours=2))
    current = _mobile_deposit_event(event_timestamp=T0)
    f = compute_mobile_deposit_features(_ctx(current, history=(prior,)))
    assert f["rapid_resubmission_flag"] is False


# ---- entity extraction: CHECK_PAYEE (Phase 7A additive correction) -----------------


def test_extract_entities_emits_check_payee_when_token_present():
    event = _mobile_deposit_event(check_payee_token="PAYEETOK123")
    entities = extract_entities(event)
    assert ("check_payee", "PAYEETOK123") in entities


def test_extract_entities_omits_check_payee_when_token_absent():
    event = _mobile_deposit_event(check_payee_token=None)
    entities = extract_entities(event)
    assert not any(entity_type == "check_payee" for entity_type, _ in entities)


# ---- deterministic ordering / schema version --------------------------------------


def test_mobile_deposit_feature_columns_ordered_vector():
    current = _mobile_deposit_event()
    f = compute_mobile_deposit_features(_ctx(current))
    vector = ordered_feature_vector(f, MOBILE_DEPOSIT_FEATURE_COLUMNS)
    assert vector == [f[c] for c in MOBILE_DEPOSIT_FEATURE_COLUMNS]
    assert set(MOBILE_DEPOSIT_FEATURE_COLUMNS) == set(f.keys())


def test_mobile_deposit_feature_schema_version_defined():
    assert MOBILE_DEPOSIT_FEATURE_SCHEMA_VERSION == "v1"
