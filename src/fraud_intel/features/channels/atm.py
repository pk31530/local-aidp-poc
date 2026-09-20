"""ATM channel feature/entity adapter (Phase 7A, guide section 25), built
entirely on top of src.fraud_intel.features.core -- no database, Docker,
or network access.
"""
from __future__ import annotations

from typing import Optional

from src.common.features import WINDOW_1H
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

ATM_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

INTERNATIONAL_GEO_BUCKET = "international"

ATM_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "is_international_geo_flag",
    "geo_velocity_distinct_buckets_1h",
    "cash_out_ratio_vs_average",
]


def compute_atm_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "atm":
        raise ValueError(f"compute_atm_features requires channel='atm', got {ctx.current_event.channel!r}")
    if not isinstance(ctx.current_event.channel_payload, ATMPayload):
        raise ValueError("current_event.channel_payload must be an ATMPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_atm = tuple(
        r for r in ctx.same_customer_history() if r.channel == "atm" and isinstance(r.channel_payload, ATMPayload)
    )

    is_international_geo_flag = payload.atm_geo_bucket == INTERNATIONAL_GEO_BUCKET

    recent = tuple(r for r in same_customer_atm if ctx.as_of_time - r.event_timestamp <= WINDOW_1H)
    distinct_buckets = {r.channel_payload.atm_geo_bucket for r in recent} | {payload.atm_geo_bucket}
    geo_velocity_distinct_buckets_1h = len(distinct_buckets)

    prior_amounts = [r.amount_minor_units for r in same_customer_atm]
    mean_amount = (sum(prior_amounts) / len(prior_amounts)) if prior_amounts else None
    cash_out_ratio_vs_average = (
        ctx.current_event.amount_minor_units / mean_amount if mean_amount else 1.0
    )

    channel_features: dict[str, float | int | bool] = {
        "is_international_geo_flag": bool(is_international_geo_flag),
        "geo_velocity_distinct_buckets_1h": int(geo_velocity_distinct_buckets_1h),
        "cash_out_ratio_vs_average": float(cash_out_ratio_vs_average),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, ATMPayload) and event.channel_payload.atm_id:
        entities.append(("atm", event.channel_payload.atm_id))
    return entities
