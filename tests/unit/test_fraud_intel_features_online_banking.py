"""Phase 2: Online/Mobile Banking reference channel feature adapter, and
the six remaining channel stubs. No database, Docker, or network access.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.features.channels.ach import compute_ach_features
from src.fraud_intel.features.channels.atm import compute_atm_features
from src.fraud_intel.features.channels.debit_card import compute_debit_card_features
from src.fraud_intel.features.channels.mobile_deposit import compute_mobile_deposit_features
from src.fraud_intel.features.channels.online_banking import (
    ONLINE_BANKING_FEATURE_COLUMNS,
    ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
    PROFILE_CHANGE_TRANSACTION_TYPE,
    TRANSFER_TRANSACTION_TYPE,
    compute_online_banking_features,
)
from src.fraud_intel.features.channels.p2p import compute_p2p_features
from src.fraud_intel.features.channels.wire import compute_wire_features
from src.fraud_intel.features.core import FeatureComputationContext, ordered_feature_vector

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ob_event(
    *,
    event_timestamp: datetime = T0,
    amount_minor_units: int = 10_000,
    device_id: str | None = "DEV1",
    transaction_type: str = TRANSFER_TRANSACTION_TYPE,
    mfa_used_flag: bool = True,
) -> FraudEvent:
    return FraudEvent(
        channel="online_banking",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units,
        direction="debit",
        device_id=device_id,
        channel_payload=OnlineBankingPayload(
            session_id="SESS1",
            login_method="password",
            mfa_used_flag=mfa_used_flag,
            transaction_type=transaction_type,
            target_account="TGT1",
        ),
    )


def _ach_event(*, event_timestamp: datetime = T0) -> FraudEvent:
    return FraudEvent(
        channel="ach",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=event_timestamp,
        amount_minor_units=5_000,
        direction="debit",
        channel_payload=ACHPayload(
            sec_code="PPD",
            originating_routing_number="123456789",
            receiving_routing_number="987654321",
            batch_id="BATCH1",
            effective_entry_date=event_timestamp.date(),
            company_id="COMP1",
        ),
    )


def _ctx(current: FraudEvent, history: tuple[FraudEvent, ...] = ()) -> FeatureComputationContext:
    return FeatureComputationContext(
        current_event=current, historical_events=history, source_alert_history=(), as_of_time=current.event_timestamp
    )


# ---- channel mismatch rejection --------------------------------------------------


def test_compute_online_banking_features_rejects_wrong_channel():
    ctx = _ctx(_ach_event())
    with pytest.raises(ValueError, match="requires channel='online_banking'"):
        compute_online_banking_features(ctx)


# ---- profile-change-then-transfer -------------------------------------------------


def test_profile_change_then_transfer_detected_within_lookback():
    profile_change = _ob_event(event_timestamp=T0 - timedelta(hours=2), transaction_type=PROFILE_CHANGE_TRANSACTION_TYPE)
    current = _ob_event(event_timestamp=T0, transaction_type=TRANSFER_TRANSACTION_TYPE)
    ctx = _ctx(current, history=(profile_change,))
    f = compute_online_banking_features(ctx)
    assert f["profile_change_then_transfer_flag"] is True


def test_profile_change_outside_lookback_does_not_trigger():
    stale_profile_change = _ob_event(
        event_timestamp=T0 - timedelta(hours=48), transaction_type=PROFILE_CHANGE_TRANSACTION_TYPE
    )
    current = _ob_event(event_timestamp=T0, transaction_type=TRANSFER_TRANSACTION_TYPE)
    ctx = _ctx(current, history=(stale_profile_change,))
    f = compute_online_banking_features(ctx, profile_change_lookback=timedelta(hours=24))
    assert f["profile_change_then_transfer_flag"] is False


def test_profile_change_without_a_following_transfer_does_not_trigger():
    profile_change = _ob_event(event_timestamp=T0 - timedelta(hours=1), transaction_type=PROFILE_CHANGE_TRANSACTION_TYPE)
    current = _ob_event(event_timestamp=T0, transaction_type="login")  # not a transfer
    ctx = _ctx(current, history=(profile_change,))
    f = compute_online_banking_features(ctx)
    assert f["profile_change_then_transfer_flag"] is False


def test_configurable_lookback_window_is_respected():
    profile_change = _ob_event(event_timestamp=T0 - timedelta(hours=10), transaction_type=PROFILE_CHANGE_TRANSACTION_TYPE)
    current = _ob_event(event_timestamp=T0, transaction_type=TRANSFER_TRANSACTION_TYPE)
    ctx = _ctx(current, history=(profile_change,))
    # A shorter, explicit lookback excludes an event that the default would include.
    f_short = compute_online_banking_features(ctx, profile_change_lookback=timedelta(hours=5))
    f_default = compute_online_banking_features(ctx)
    assert f_short["profile_change_then_transfer_flag"] is False
    assert f_default["profile_change_then_transfer_flag"] is True


# ---- other channel-specific features ----------------------------------------------


def test_new_device_high_value_combo_flag():
    current = _ob_event(amount_minor_units=600_000, device_id="DEV-NEW")
    ctx = _ctx(current)
    f = compute_online_banking_features(ctx)
    assert f["new_device_high_value_combo_flag"] is True


def test_new_device_high_value_combo_flag_false_when_amount_below_threshold():
    current = _ob_event(amount_minor_units=1_000, device_id="DEV-NEW")
    ctx = _ctx(current)
    f = compute_online_banking_features(ctx)
    assert f["new_device_high_value_combo_flag"] is False


def test_mfa_bypass_flag_true_when_mfa_not_used():
    current = _ob_event(mfa_used_flag=False)
    ctx = _ctx(current)
    f = compute_online_banking_features(ctx)
    assert f["mfa_bypass_flag"] is True


def test_session_velocity_counts_only_online_banking_history_in_window():
    ob_history = _ob_event(event_timestamp=T0 - timedelta(minutes=30))
    ach_history = _ach_event(event_timestamp=T0 - timedelta(minutes=30))
    current = _ob_event(event_timestamp=T0)
    ctx = _ctx(current, history=(ob_history, ach_history))
    f = compute_online_banking_features(ctx)
    assert f["session_velocity"] == 1  # ach_history must not count


# ---- deterministic ordering / schema version --------------------------------------


def test_online_banking_feature_columns_ordered_vector():
    current = _ob_event()
    ctx = _ctx(current)
    f = compute_online_banking_features(ctx)
    vector = ordered_feature_vector(f, ONLINE_BANKING_FEATURE_COLUMNS)
    assert vector == [f[c] for c in ONLINE_BANKING_FEATURE_COLUMNS]
    assert set(ONLINE_BANKING_FEATURE_COLUMNS) == set(f.keys())


def test_online_banking_feature_schema_version_defined():
    assert ONLINE_BANKING_FEATURE_SCHEMA_VERSION == "v1"


# ---- the six remaining channel stubs ----------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [compute_ach_features, compute_wire_features, compute_mobile_deposit_features, compute_atm_features, compute_debit_card_features, compute_p2p_features],
)
def test_remaining_channel_stub_adapters_raise_not_implemented(fn):
    current = _ob_event()
    ctx = _ctx(current)
    with pytest.raises(NotImplementedError):
        fn(ctx)
