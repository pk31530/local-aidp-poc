"""P2P/Instant Payment channel feature/entity adapter (Phase 7A, guide
section 25), built entirely on top of src.fraud_intel.features.core -- no
database, Docker, or network access.
"""
from __future__ import annotations

from typing import Optional

from src.common.features import WINDOW_10M
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.features.core import (
    FEATURE_SCHEMA_VERSION,
    SHARED_FEATURE_COLUMNS,
    CounterpartyRiskLookups,
    FeatureComputationContext,
    compute_shared_features,
)
from src.fraud_intel.graph.entity_graph import EntityKey

P2P_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

# $100.00 in minor units (cents) -- a common money-mule heuristic (round
# transfer amounts are disproportionately used to move funds quickly).
ROUND_DOLLAR_AMOUNT_MINOR_UNITS = 10_000
RAPID_SEQUENTIAL_TRANSFER_THRESHOLD = 2

P2P_FEATURE_COLUMNS: list[str] = SHARED_FEATURE_COLUMNS + [
    "recipient_is_new_flag",
    "no_memo_flag",
    "rapid_sequential_transfer_flag",
    "round_dollar_amount_flag",
]


def compute_p2p_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    if ctx.current_event.channel != "p2p":
        raise ValueError(f"compute_p2p_features requires channel='p2p', got {ctx.current_event.channel!r}")
    if not isinstance(ctx.current_event.channel_payload, P2PPayload):
        raise ValueError("current_event.channel_payload must be a P2PPayload")

    shared = compute_shared_features(ctx, counterparty_key=counterparty_key, counterparty_risk=counterparty_risk)

    payload = ctx.current_event.channel_payload
    same_customer_p2p = tuple(
        r for r in ctx.same_customer_history() if r.channel == "p2p" and isinstance(r.channel_payload, P2PPayload)
    )

    recent_count = sum(1 for r in same_customer_p2p if ctx.as_of_time - r.event_timestamp <= WINDOW_10M)
    rapid_sequential_transfer_flag = recent_count >= RAPID_SEQUENTIAL_TRANSFER_THRESHOLD

    channel_features: dict[str, float | int | bool] = {
        "recipient_is_new_flag": bool(payload.recipient_is_new_flag),
        "no_memo_flag": bool(not payload.memo_present_flag),
        "rapid_sequential_transfer_flag": bool(rapid_sequential_transfer_flag),
        "round_dollar_amount_flag": bool(ctx.current_event.amount_minor_units % ROUND_DOLLAR_AMOUNT_MINOR_UNITS == 0),
    }
    return {**shared, **channel_features}


def extract_entities(event: FraudEvent) -> list[EntityKey]:
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, P2PPayload) and event.channel_payload.recipient_handle:
        entities.append(("recipient", event.channel_payload.recipient_handle))
    return entities
