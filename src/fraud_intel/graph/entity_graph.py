"""In-process entity graph (guide section 15). Hand-rolled adjacency
structure, rebuilt per scoring call from the caller's already as-of-time-
validated event history -- no NetworkX, no graph database, no persistence.

`ResolvedFraudEntityEvidence` makes the "mature, training-eligible,
RESOLVED_FRAUD, strictly earlier than the scored event" requirement
(guide section 15) structurally enforceable, not just documented (Phase 5
decision 5): it has no disposition-shaped field at all, so a raw
`analyst_disposition` object or string cannot be passed here even by
mistake, and `validate_resolved_fraud_evidence()` raises `GraphLeakageError`
before the graph is ever built if any evidence item is not strictly prior
to the scored event -- equal-boundary or future evidence is never silently
filtered.
"""
from __future__ import annotations

import functools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from src.common.config import PROJECT_ROOT
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload

GRAPH_POLICY_CONFIG_DIR = PROJECT_ROOT / "config" / "fraud_intel"

EntityType = Literal[
    "customer", "account", "device", "ip_address", "beneficiary", "recipient", "card", "atm", "check_payee"
]
EntityKey = tuple[str, str]  # (entity_type, entity_id)

GRAPH_HISTORY_TRUNCATED_REASON_CODE = "GRAPH_HISTORY_TRUNCATED"

# The "other side of the transaction" entity types (Phase 7A) -- used by
# graph_counterparty_key() to pick, generically across every channel, the
# ONE entity that plays the role src.fraud_intel.scoring.orchestrator
# previously hardcoded as online_banking's own "recipient_key" (fed to
# compute_graph_risk_score()'s fan-in signal). "customer"/"account"/
# "device"/"ip_address" are never counterparty entities -- they identify
# the ACTOR, not the other party.
COUNTERPARTY_ENTITY_TYPES: frozenset[str] = frozenset({"beneficiary", "recipient", "card", "atm", "check_payee"})


def graph_counterparty_key(entities: Sequence[EntityKey]) -> Optional[EntityKey]:
    """The first counterparty-typed entity in `entities` (an adapter's own
    extract_entities() output), or None if the event has no counterparty
    entity at all -- e.g. ACH, whose routing numbers/company_id are
    deliberately NOT graph nodes (Phase 7A decision 3)."""
    for entity in entities:
        if entity[0] in COUNTERPARTY_ENTITY_TYPES:
            return entity
    return None


def _reject_naive_or_non_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("must be UTC")
    return value


class GraphLeakageError(ValueError):
    """A ResolvedFraudEntityEvidence entry's resolved_fraud_at is not
    strictly before the scored event's event_timestamp, or a historical
    event is not strictly before it -- raised before the graph is built,
    never silently filtered."""


class ResolvedFraudEntityEvidence(BaseModel):
    """Typed, immutable proof that one entity is fraud-linked. NOT an
    analyst_disposition -- has no disposition-shaped field, so a raw
    disposition value cannot be passed here even by mistake. Phase 6 will
    build the real label_assessments-backed adapter that PRODUCES these;
    Phase 5's own fixtures construct them directly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_type: EntityType
    entity_id: str
    label_assessment_id: str
    resolved_fraud_at: datetime
    eligibility_policy_version: str
    label_source: Literal["SYNTHETIC_GENERATOR", "ANALYST_DISPOSITION", "EXTERNAL_CONFIRMATION"]

    @field_validator("resolved_fraud_at")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _reject_naive_or_non_utc(value)

    @property
    def entity_key(self) -> EntityKey:
        return (self.entity_type, self.entity_id)


def validate_resolved_fraud_evidence(
    evidence: Sequence[ResolvedFraudEntityEvidence], *, current_event_timestamp: datetime
) -> frozenset[EntityKey]:
    """Guide section 15 / Phase 5 decision 5: every evidence item's
    resolved_fraud_at must be strictly before current_event_timestamp.
    Equal-boundary or future evidence raises GraphLeakageError -- the
    current event's own label, and any future label, can never influence
    its own graph score."""
    for item in evidence:
        if item.resolved_fraud_at >= current_event_timestamp:
            raise GraphLeakageError(
                f"ResolvedFraudEntityEvidence for {item.entity_key} has resolved_fraud_at="
                f"{item.resolved_fraud_at!r} >= current_event_timestamp={current_event_timestamp!r} -- "
                "future or boundary fraud-link evidence is not permitted"
            )
    return frozenset(item.entity_key for item in evidence)


class GraphPolicy(BaseModel):
    """Versioned, config-driven graph policy (config/fraud_intel/
    graph_policy_<channel>.yaml). No arbitrary expressions -- every field
    is a plain, validated numeric bound or weight."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    graph_policy_version: str
    shared_device_cap: int = Field(gt=0)
    shared_device_weight: float = Field(ge=0)
    fan_in_cap: int = Field(gt=0)
    fan_in_weight: float = Field(ge=0)
    fan_out_cap: int = Field(gt=0)
    fan_out_weight: float = Field(ge=0)
    shortest_path_weight: float = Field(ge=0)
    graph_max_history_events: int = Field(gt=0)
    max_nodes: int = Field(gt=0)
    max_edges: int = Field(gt=0)

    # Phase 7A decision 4 (applied consistently to graph policy alongside
    # ensemble policy, since graph weights are equally untuned defaults for
    # every new channel): optional so online_banking's existing policy file
    # -- which predates these fields -- stays valid unchanged.
    calibration_status: Optional[str] = None
    promotion_note: Optional[str] = None

    @model_validator(mode="after")
    def _weights_bounded(self) -> "GraphPolicy":
        total = self.shared_device_weight + self.fan_in_weight + self.fan_out_weight + self.shortest_path_weight
        if total > 1.0 + 1e-6:
            raise ValueError(f"graph_policy weights must sum to at most 1.0, got {total}")
        return self


@functools.lru_cache
def load_graph_policy(channel: str) -> GraphPolicy:
    path = GRAPH_POLICY_CONFIG_DIR / f"graph_policy_{channel}.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return GraphPolicy.model_validate(raw)


class EdgeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    other: EntityKey
    event_timestamp: datetime
    event_id: str


class EntityGraph(BaseModel):
    """Immutable once built. `truncated` and `truncation_reason_code`
    expose the stable, testable signal that a resource bound was hit --
    never unbounded growth, never silent data loss."""

    model_config = ConfigDict(frozen=True)

    adjacency: dict[EntityKey, tuple[EdgeRecord, ...]]
    truncated: bool
    truncation_reason_code: Optional[str]
    node_count: int
    edge_count: int


def extract_entities_online_banking(event: FraudEvent) -> list[EntityKey]:
    """Reference-channel (online_banking) entity extraction -- also
    build_entity_graph()'s default `entity_extractor`, so every existing
    caller/test that doesn't pass one keeps this exact behavior unchanged.
    Phase 7A's other six channels each define their own extract_entities()
    in their feature-adapter module (src.fraud_intel.features.channels.*),
    registered via src.fraud_intel.registry, on top of this same
    EntityGraph core."""
    entities: list[EntityKey] = [("customer", event.customer_id), ("account", event.account_id)]
    if event.device_id:
        entities.append(("device", event.device_id))
    if event.ip_address:
        entities.append(("ip_address", event.ip_address))
    if isinstance(event.channel_payload, OnlineBankingPayload) and event.channel_payload.target_account:
        entities.append(("recipient", event.channel_payload.target_account))
    return entities


def build_entity_graph(
    *,
    historical_events: Sequence[FraudEvent],
    current_event_timestamp: datetime,
    policy: GraphPolicy,
    entity_extractor: Callable[[FraudEvent], list[EntityKey]] = extract_entities_online_banking,
) -> EntityGraph:
    """Builds the graph from `historical_events` only -- respects the
    as-of-time contract by construction (every event must be strictly
    before current_event_timestamp; violated by construction raises
    GraphLeakageError rather than silently dropping the offending row).

    Deterministic, bounded accumulation: events are considered most-
    recent-first (after a deterministic (event_timestamp, event_id) sort),
    and accumulation stops the instant adding the next (older) event would
    exceed policy.max_nodes or policy.max_edges -- so the graph always
    retains the most recent eligible activity, never an arbitrary or
    unbounded slice.
    """
    ordered = sorted(historical_events, key=lambda event: (event.event_timestamp, str(event.event_id)))
    for event in ordered:
        if event.event_timestamp >= current_event_timestamp:
            raise GraphLeakageError(
                f"historical event {event.event_id} has event_timestamp {event.event_timestamp!r} >= "
                f"current_event_timestamp {current_event_timestamp!r} -- future or boundary history is not permitted"
            )

    history_capped = ordered[-policy.graph_max_history_events :] if len(ordered) > policy.graph_max_history_events else ordered
    truncated_by_history_cap = len(history_capped) < len(ordered)

    adjacency: dict[EntityKey, list[EdgeRecord]] = {}
    truncated_by_resource_cap = False
    total_edges = 0

    for event in reversed(history_capped):
        entities = entity_extractor(event)
        new_edges = len(entities) * (len(entities) - 1)
        prospective_nodes = set(adjacency.keys()) | set(entities)
        prospective_edges = total_edges + new_edges
        if len(prospective_nodes) > policy.max_nodes or prospective_edges > policy.max_edges:
            truncated_by_resource_cap = True
            break
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                if i == j:
                    continue
                adjacency.setdefault(a, []).append(
                    EdgeRecord(other=b, event_timestamp=event.event_timestamp, event_id=str(event.event_id))
                )
        total_edges += new_edges

    truncated = truncated_by_history_cap or truncated_by_resource_cap
    return EntityGraph(
        adjacency={key: tuple(edges) for key, edges in adjacency.items()},
        truncated=truncated,
        truncation_reason_code=GRAPH_HISTORY_TRUNCATED_REASON_CODE if truncated else None,
        node_count=len(adjacency),
        edge_count=sum(len(edges) for edges in adjacency.values()),
    )


def shared_device_across_distinct_customers_count(graph: EntityGraph, device_key: EntityKey) -> int:
    neighbors = graph.adjacency.get(device_key, ())
    return len({edge.other[1] for edge in neighbors if edge.other[0] == "customer"})


def beneficiary_fan_in_count(graph: EntityGraph, beneficiary_key: EntityKey) -> int:
    neighbors = graph.adjacency.get(beneficiary_key, ())
    return len({edge.other[1] for edge in neighbors if edge.other[0] == "customer"})


def sender_fan_out_count(graph: EntityGraph, customer_key: EntityKey) -> int:
    neighbors = graph.adjacency.get(customer_key, ())
    return len({edge.other for edge in neighbors if edge.other[0] in ("recipient", "beneficiary")})


def shortest_path_to_fraud_linked_entity(
    graph: EntityGraph, start: EntityKey, fraud_linked_entities: frozenset[EntityKey]
) -> Optional[int]:
    """Deterministic BFS -- neighbors visited in sorted order at every
    step, so traversal is fully reproducible even though the returned
    distance itself is order-independent."""
    if start in fraud_linked_entities:
        return 0
    visited = {start}
    frontier = [start]
    distance = 0
    while frontier:
        distance += 1
        next_frontier: list[EntityKey] = []
        for node in frontier:
            neighbors = sorted({edge.other for edge in graph.adjacency.get(node, ())})
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if neighbor in fraud_linked_entities:
                    return distance
                visited.add(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
    return None


def compute_graph_risk_score(
    *,
    graph: EntityGraph,
    customer_key: EntityKey,
    device_key: Optional[EntityKey],
    recipient_key: Optional[EntityKey],
    fraud_linked_entities: frozenset[EntityKey],
    policy: GraphPolicy,
) -> float:
    shared_device = shared_device_across_distinct_customers_count(graph, device_key) if device_key else 0
    fan_in = beneficiary_fan_in_count(graph, recipient_key) if recipient_key else 0
    fan_out = sender_fan_out_count(graph, customer_key)
    shortest_path = shortest_path_to_fraud_linked_entity(graph, customer_key, fraud_linked_entities)

    score = (
        policy.shared_device_weight * min(shared_device / policy.shared_device_cap, 1.0)
        + policy.fan_in_weight * min(fan_in / policy.fan_in_cap, 1.0)
        + policy.fan_out_weight * min(fan_out / policy.fan_out_cap, 1.0)
        + policy.shortest_path_weight * (1.0 / (1 + shortest_path) if shortest_path is not None else 0.0)
    )
    return float(min(max(score, 0.0), 1.0))
