"""Mobile Check Deposit channel feature/entity adapter (Phase 7A, guide
section 25), built entirely on top of src.fraud_intel.features.core -- no
database, Docker, or network access.
"""
from __future__ import annotations

from typing import Optional

from src.common.features import WINDOW_10M
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

MOBILE_DEPOSIT_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

# Same threshold MOBILE_DEPOSIT_LOW_IMAGE_QUALITY (rules_mobile_deposit.yaml,
# Phase 3) already uses, reused here for consistency between the feature and
# rule layers -- same convention online_banking.py's
# HIGH_VALUE_AMOUNT_MINOR_UNITS_THRESHOLD already establishes.
LOW_IMAGE_QUALITY_THRESHOLD = 0.6

MOBILE_DEPOSIT_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "duplicate_image_hash_flag",
    "car_lar_mismatch_flag",
    "low_image_quality_flag",
    "rapid_resubmission_flag",
]


def compute_mobile_deposit_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "mobile_deposit":
        raise ValueError(
            f"compute_mobile_deposit_features requires channel='mobile_deposit', got {ctx.current_event.channel!r}"
        )
    if not isinstance(ctx.current_event.channel_payload, MobileDepositPayload):
        raise ValueError("current_event.channel_payload must be a MobileDepositPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_account_deposits = tuple(
        r
        for r in ctx.same_account_history()
        if r.channel == "mobile_deposit" and isinstance(r.channel_payload, MobileDepositPayload)
    )
    rapid_resubmission_flag = any(
        ctx.as_of_time - r.event_timestamp <= WINDOW_10M for r in same_account_deposits
    )

    channel_features: dict[str, float | int | bool] = {
        "duplicate_image_hash_flag": bool(payload.duplicate_image_hash_flag),
        "car_lar_mismatch_flag": bool(payload.car_lar_mismatch_flag),
        "low_image_quality_flag": bool(payload.image_quality_score < LOW_IMAGE_QUALITY_THRESHOLD),
        "rapid_resubmission_flag": bool(rapid_resubmission_flag),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, MobileDepositPayload) and event.channel_payload.check_payee_token:
        entities.append(("check_payee", event.channel_payload.check_payee_token))
    return entities
