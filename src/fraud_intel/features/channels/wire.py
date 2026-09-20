"""Wire channel feature/entity adapter (Phase 7A, guide section 25),
built entirely on top of src.fraud_intel.features.core -- no database,
Docker, or network access.
"""
from __future__ import annotations

from typing import Optional

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

WIRE_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

WIRE_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "is_international_flag",
    "first_time_beneficiary_flag",
    "amount_vs_historical_max_ratio",
]


def compute_wire_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "wire":
        raise ValueError(f"compute_wire_features requires channel='wire', got {ctx.current_event.channel!r}")
    if not isinstance(ctx.current_event.channel_payload, WirePayload):
        raise ValueError("current_event.channel_payload must be a WirePayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_wire = tuple(
        r for r in ctx.same_customer_history() if r.channel == "wire" and isinstance(r.channel_payload, WirePayload)
    )

    is_international_flag = payload.wire_type == "international"
    first_time_beneficiary_flag = not any(
        r.channel_payload.beneficiary_account == payload.beneficiary_account for r in same_customer_wire
    )

    prior_amounts = [r.amount_minor_units for r in same_customer_wire]
    historical_max = max(prior_amounts) if prior_amounts else None
    # Neutral default of 1.0 (no elevated-vs-history signal) when there is
    # no prior wire history at all -- same "no history yet" convention
    # compute_shared_features() itself uses for amount_vs_entity_average.
    amount_vs_historical_max_ratio = (
        ctx.current_event.amount_minor_units / historical_max if historical_max else 1.0
    )

    channel_features: dict[str, float | int | bool] = {
        "is_international_flag": bool(is_international_flag),
        "first_time_beneficiary_flag": bool(first_time_beneficiary_flag),
        "amount_vs_historical_max_ratio": float(amount_vs_historical_max_ratio),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, WirePayload) and event.channel_payload.beneficiary_account:
        entities.append(("beneficiary", event.channel_payload.beneficiary_account))
    return entities
