"""Online/Mobile Banking reference channel feature adapter (guide section
9's channel-specific feature group; guide section 25 Phase 2). Built
entirely on top of src.fraud_intel.features.core -- no database, Docker,
or network access.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from src.common.features import WINDOW_1H
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)

ONLINE_BANKING_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

ONLINE_BANKING_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "new_device_high_value_combo_flag",
    "mfa_bypass_flag",
    "session_velocity",
    "profile_change_then_transfer_flag",
]

# $5,000.00 in minor units -- the same threshold value the guide's own
# section 10 RuleProvider YAML example uses for NEW_DEVICE_HIGH_AMOUNT,
# reused here for consistency between the feature and rule layers.
HIGH_VALUE_AMOUNT_MINOR_UNITS_THRESHOLD = 500_000

# transaction_type is free text on OnlineBankingPayload (guide section 6);
# these are the two values this adapter looks for. Not a schema change.
PROFILE_CHANGE_TRANSACTION_TYPE = "profile_change"
TRANSFER_TRANSACTION_TYPE = "transfer"

# How far back a prior profile-change event may be and still count toward
# profile_change_then_transfer_flag. Explicit and configurable via the
# `lookback` parameter below -- never a silent, hardcoded assumption.
DEFAULT_PROFILE_CHANGE_LOOKBACK = timedelta(hours=24)

SESSION_VELOCITY_WINDOW = WINDOW_1H


def compute_online_banking_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
    profile_change_lookback: timedelta = DEFAULT_PROFILE_CHANGE_LOOKBACK,
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "online_banking":
        raise ValueError(
            f"compute_online_banking_features requires channel='online_banking', "
            f"got {ctx.current_event.channel!r}"
        )
    if not isinstance(ctx.current_event.channel_payload, OnlineBankingPayload):
        raise ValueError("current_event.channel_payload must be an OnlineBankingPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_online = tuple(
        r
        for r in ctx.same_customer_history()
        if r.channel == "online_banking" and isinstance(r.channel_payload, OnlineBankingPayload)
    )

    new_device_high_value_combo_flag = bool(shared["is_new_device"]) and (
        ctx.current_event.amount_minor_units > HIGH_VALUE_AMOUNT_MINOR_UNITS_THRESHOLD
    )
    mfa_bypass_flag = not payload.mfa_used_flag

    session_velocity = sum(
        1 for r in same_customer_online if ctx.as_of_time - r.event_timestamp <= SESSION_VELOCITY_WINDOW
    )

    profile_change_then_transfer_flag = False
    if payload.transaction_type == TRANSFER_TRANSACTION_TYPE:
        lookback_start = ctx.as_of_time - profile_change_lookback
        profile_change_then_transfer_flag = any(
            r.channel_payload.transaction_type == PROFILE_CHANGE_TRANSACTION_TYPE
            and r.event_timestamp >= lookback_start
            for r in same_customer_online
        )

    channel_features: dict[str, float | int | bool] = {
        "new_device_high_value_combo_flag": bool(new_device_high_value_combo_flag),
        "mfa_bypass_flag": bool(mfa_bypass_flag),
        "session_velocity": int(session_velocity),
        "profile_change_then_transfer_flag": bool(profile_change_then_transfer_flag),
    }
    return {**shared, **channel_features}
