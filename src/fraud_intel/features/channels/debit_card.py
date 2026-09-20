"""Debit Card channel feature/entity adapter (Phase 7A, guide section
25), built entirely on top of src.fraud_intel.features.core -- no
database, Docker, or network access.
"""
from __future__ import annotations

from typing import Optional

from src.common.features import WINDOW_1H
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

DEBIT_CARD_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

DEBIT_CARD_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "card_not_present_flag",
    "cross_border_new_device_combo_flag",
    "distinct_merchant_count_1h",
]


def compute_debit_card_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "debit_card":
        raise ValueError(
            f"compute_debit_card_features requires channel='debit_card', got {ctx.current_event.channel!r}"
        )
    if not isinstance(ctx.current_event.channel_payload, DebitCardPayload):
        raise ValueError("current_event.channel_payload must be a DebitCardPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_debit = tuple(
        r
        for r in ctx.same_customer_history()
        if r.channel == "debit_card" and isinstance(r.channel_payload, DebitCardPayload)
    )

    card_not_present_flag = not payload.card_present_flag
    cross_border_new_device_combo_flag = payload.cross_border_flag and bool(shared["is_new_device"])

    recent = tuple(r for r in same_customer_debit if ctx.as_of_time - r.event_timestamp <= WINDOW_1H)
    distinct_merchant_count_1h = len({r.channel_payload.merchant_id for r in recent} | {payload.merchant_id})

    channel_features: dict[str, float | int | bool] = {
        "card_not_present_flag": bool(card_not_present_flag),
        "cross_border_new_device_combo_flag": bool(cross_border_new_device_combo_flag),
        "distinct_merchant_count_1h": int(distinct_merchant_count_1h),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, DebitCardPayload) and event.channel_payload.card_token:
        entities.append(("card", event.channel_payload.card_token))
    return entities
