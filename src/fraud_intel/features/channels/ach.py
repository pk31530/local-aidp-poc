"""ACH channel feature/entity adapter (Phase 7A, guide section 25),
built entirely on top of src.fraud_intel.features.core -- no database,
Docker, or network access.

Phase 7A decision 3: originating/receiving routing numbers and
company_id are validated direct/channel features (and may participate in
frequency/novelty features below) but are deliberately NOT graph nodes --
extract_entities() below emits only the approved shared "actor" entities
(customer/account/device/ip_address), never a routing-number- or
company-id-derived entity. This is a deliberate semantic decision, not a
missing implementation: no BANK_ROUTING or ORIGINATOR_COMPANY EntityType
exists yet.
"""
from __future__ import annotations

from typing import Optional

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

ACH_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

ACH_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "first_time_receiving_routing_number_flag",
    "same_day_mixed_sec_code_flag",
    "first_time_company_id_flag",
]


def compute_ach_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "ach":
        raise ValueError(f"compute_ach_features requires channel='ach', got {ctx.current_event.channel!r}")
    if not isinstance(ctx.current_event.channel_payload, ACHPayload):
        raise ValueError("current_event.channel_payload must be an ACHPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_ach = tuple(
        r for r in ctx.same_customer_history() if r.channel == "ach" and isinstance(r.channel_payload, ACHPayload)
    )

    first_time_receiving_routing_number_flag = not any(
        r.channel_payload.receiving_routing_number == payload.receiving_routing_number for r in same_customer_ach
    )
    first_time_company_id_flag = not any(r.channel_payload.company_id == payload.company_id for r in same_customer_ach)

    same_day = tuple(
        r for r in same_customer_ach if r.event_timestamp.date() == ctx.current_event.event_timestamp.date()
    )
    same_day_mixed_sec_code_flag = any(r.channel_payload.sec_code != payload.sec_code for r in same_day)

    channel_features: dict[str, float | int | bool] = {
        "first_time_receiving_routing_number_flag": bool(first_time_receiving_routing_number_flag),
        "same_day_mixed_sec_code_flag": bool(same_day_mixed_sec_code_flag),
        "first_time_company_id_flag": bool(first_time_company_id_flag),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    """Phase 7A decision 3: customer/account/device/ip_address only --
    NEVER a routing-number- or company-id-derived entity."""
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    return entities
