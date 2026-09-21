"""Phase 5: in-process entity graph (guide section 15). No database,
Docker, network access, or NetworkX/graph-database dependency anywhere.
"""
from __future__ import annotations

import ast
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.graph import entity_graph as graph_module
from src.fraud_intel.graph.entity_graph import (
    GraphLeakageError,
    GraphPolicy,
    ResolvedFraudEntityEvidence,
    beneficiary_fan_in_count,
    build_entity_graph,
    compute_graph_risk_score,
    sender_fan_out_count,
    shared_device_across_distinct_customers_count,
    shortest_path_to_fraud_linked_entity,
    validate_resolved_fraud_evidence,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _policy(**overrides) -> GraphPolicy:
    base = dict(
        channel="online_banking", graph_policy_version="v1",
        shared_device_cap=5, shared_device_weight=0.3,
        fan_in_cap=10, fan_in_weight=0.3,
        fan_out_cap=10, fan_out_weight=0.2,
        shortest_path_weight=0.2,
        graph_max_history_events=5000, max_nodes=20000, max_edges=200000,
    )
    base.update(overrides)
    return GraphPolicy(**base)


def _event(*, customer_id: str, account_id: str, event_timestamp: datetime, device_id: str | None = None, target_account: str | None = None) -> FraudEvent:
    return FraudEvent(
        channel="online_banking", customer_id=customer_id, account_id=account_id,
        event_timestamp=event_timestamp, amount_minor_units=10_000, direction="debit",
        device_id=device_id,
        channel_payload=OnlineBankingPayload(
            session_id="S", login_method="password", mfa_used_flag=True,
            transaction_type="transfer", target_account=target_account or "TGT-DEFAULT",
        ),
    )


def _evidence(*, entity_type: str, entity_id: str, resolved_fraud_at: datetime) -> ResolvedFraudEntityEvidence:
    return ResolvedFraudEntityEvidence(
        entity_type=entity_type, entity_id=entity_id, label_assessment_id=str(uuid.uuid4()),
        resolved_fraud_at=resolved_fraud_at, eligibility_policy_version="v1", label_source="SYNTHETIC_GENERATOR",
    )


# ---- graph policy validation -------------------------------------------------------


def test_graph_policy_loads_and_validates_the_real_yaml_file():
    from src.fraud_intel.graph.entity_graph import load_graph_policy

    policy = load_graph_policy("online_banking")
    assert policy.channel == "online_banking"
    assert policy.graph_policy_version == "v1"
    assert policy.graph_max_history_events > 0


@pytest.mark.parametrize(
    "channel", sorted({"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"})
)
def test_graph_policy_loads_and_validates_for_every_channel(channel):
    """Phase 7A: all 7 channels now have a real graph_policy_<channel>.yaml
    file, not just online_banking."""
    from src.fraud_intel.graph.entity_graph import load_graph_policy

    policy = load_graph_policy(channel)
    assert policy.channel == channel
    assert policy.graph_policy_version == "v1"
    assert policy.graph_max_history_events > 0
    if channel != "online_banking":
        # Phase 7A decision 4: every NEW channel's policy must explicitly
        # declare its POC-default status -- online_banking's own policy
        # predates these fields and is exempt.
        assert policy.calibration_status == "UNVALIDATED_POC_DEFAULT"
        assert policy.promotion_note


def test_graph_policy_rejects_weights_summing_above_one():
    with pytest.raises(ValidationError):
        _policy(shared_device_weight=0.5, fan_in_weight=0.5, fan_out_weight=0.5, shortest_path_weight=0.5)


def test_graph_policy_rejects_extra_fields():
    with pytest.raises(ValidationError):
        GraphPolicy.model_validate(
            {
                "channel": "online_banking", "graph_policy_version": "v1", "shared_device_cap": 5,
                "shared_device_weight": 0.3, "fan_in_cap": 10, "fan_in_weight": 0.3, "fan_out_cap": 10,
                "fan_out_weight": 0.2, "shortest_path_weight": 0.2, "graph_max_history_events": 5000,
                "max_nodes": 100, "max_edges": 100, "arbitrary_expression": "eval('1+1')",
            }
        )


# ---- ResolvedFraudEntityEvidence / as-of enforcement --------------------------------


def test_evidence_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        ResolvedFraudEntityEvidence(
            entity_type="customer", entity_id="C1", label_assessment_id="A1",
            resolved_fraud_at=datetime(2026, 1, 1), eligibility_policy_version="v1", label_source="SYNTHETIC_GENERATOR",
        )


def test_evidence_is_frozen():
    ev = _evidence(entity_type="customer", entity_id="C1", resolved_fraud_at=T0 - timedelta(days=1))
    with pytest.raises(ValidationError):
        ev.entity_id = "C2"


def test_validate_resolved_fraud_evidence_accepts_strictly_prior_evidence():
    ev = _evidence(entity_type="customer", entity_id="C1", resolved_fraud_at=T0 - timedelta(days=1))
    entities = validate_resolved_fraud_evidence([ev], current_event_timestamp=T0)
    assert ("customer", "C1") in entities


def test_current_events_own_label_cannot_affect_its_graph_score():
    """Evidence exactly at the current event's timestamp -- the current
    event's own label -- must raise, never be silently included."""
    ev = _evidence(entity_type="customer", entity_id="C1", resolved_fraud_at=T0)
    with pytest.raises(GraphLeakageError):
        validate_resolved_fraud_evidence([ev], current_event_timestamp=T0)


def test_future_label_cannot_affect_graph_score():
    ev = _evidence(entity_type="customer", entity_id="C1", resolved_fraud_at=T0 + timedelta(seconds=1))
    with pytest.raises(GraphLeakageError):
        validate_resolved_fraud_evidence([ev], current_event_timestamp=T0)


def test_raw_analyst_disposition_structurally_rejected():
    """ResolvedFraudEntityEvidence has no disposition-shaped field at all
    -- a raw disposition string cannot be smuggled through any of its
    fields."""
    with pytest.raises(ValidationError):
        ResolvedFraudEntityEvidence.model_validate(
            {
                "entity_type": "customer", "entity_id": "C1", "label_assessment_id": "A1",
                "resolved_fraud_at": T0 - timedelta(days=1), "eligibility_policy_version": "v1",
                "label_source": "SYNTHETIC_GENERATOR", "analyst_disposition": "CONFIRMED_FRAUD",
            }
        )


def test_graph_module_code_never_uses_disposition_identifiers():
    tree = ast.parse(inspect.getsource(graph_module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    forbidden = {"analyst_disposition", "AnalystDisposition", "SyntheticGroundTruthLabel", "outcome_status"}
    assert forbidden.isdisjoint(names)


# ---- graph as-of-time isolation (historical_events) ---------------------------------


def test_build_entity_graph_rejects_a_future_or_boundary_historical_event():
    boundary_event = _event(customer_id="C1", account_id="A1", event_timestamp=T0)
    with pytest.raises(GraphLeakageError):
        build_entity_graph(historical_events=[boundary_event], current_event_timestamp=T0, policy=_policy())


def test_build_entity_graph_accepts_strictly_prior_events():
    prior_event = _event(customer_id="C1", account_id="A1", event_timestamp=T0 - timedelta(hours=1))
    graph = build_entity_graph(historical_events=[prior_event], current_event_timestamp=T0, policy=_policy())
    assert graph.node_count > 0


def test_build_entity_graph_uses_extract_entities_online_banking_by_default():
    """Phase 7A: build_entity_graph()'s default entity_extractor is
    extract_entities_online_banking -- omitting the parameter (as every
    pre-Phase-7A caller/test does) must keep producing IDENTICAL graphs."""
    from src.fraud_intel.graph.entity_graph import extract_entities_online_banking

    prior_event = _event(customer_id="C1", account_id="A1", event_timestamp=T0 - timedelta(hours=1), device_id="DEV1")
    default_graph = build_entity_graph(historical_events=[prior_event], current_event_timestamp=T0, policy=_policy())
    explicit_graph = build_entity_graph(
        historical_events=[prior_event], current_event_timestamp=T0, policy=_policy(),
        entity_extractor=extract_entities_online_banking,
    )
    assert default_graph == explicit_graph


def test_build_entity_graph_respects_a_custom_entity_extractor():
    """A channel-specific extract_entities() (e.g. ACH's, which never
    emits a routing-number-derived entity) changes the resulting graph --
    proves entity_extractor is genuinely wired through, not ignored."""
    def _customer_and_marker_extractor(event):
        return [("customer", event.customer_id), ("beneficiary", "MARKER")]

    prior_event = _event(customer_id="C1", account_id="A1", event_timestamp=T0 - timedelta(hours=1), device_id="DEV1")
    graph = build_entity_graph(
        historical_events=[prior_event], current_event_timestamp=T0, policy=_policy(),
        entity_extractor=_customer_and_marker_extractor,
    )
    customer_neighbors = {edge.other for edge in graph.adjacency[("customer", "C1")]}
    assert ("beneficiary", "MARKER") in customer_neighbors
    assert ("account", "A1") not in graph.adjacency
    assert ("device", "DEV1") not in graph.adjacency


# ---- empty history ------------------------------------------------------------------


def test_empty_history_produces_a_neutral_empty_graph():
    graph = build_entity_graph(historical_events=[], current_event_timestamp=T0, policy=_policy())
    assert graph.node_count == 0
    assert graph.edge_count == 0
    assert graph.truncated is False
    score = compute_graph_risk_score(
        graph=graph, customer_key=("customer", "C1"), device_key=None, recipient_key=None,
        fraud_linked_entities=frozenset(), policy=_policy(),
    )
    assert score == 0.0


# ---- connected-risk / shared-entity calculations -------------------------------------


def test_shared_device_across_distinct_customers_count():
    events = [
        _event(customer_id="C1", account_id="A1", device_id="DEV1", event_timestamp=T0 - timedelta(hours=3)),
        _event(customer_id="C2", account_id="A2", device_id="DEV1", event_timestamp=T0 - timedelta(hours=2)),
        _event(customer_id="C3", account_id="A3", device_id="DEV1", event_timestamp=T0 - timedelta(hours=1)),
    ]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    assert shared_device_across_distinct_customers_count(graph, ("device", "DEV1")) == 3


def test_beneficiary_fan_in_count():
    events = [
        _event(customer_id="C1", account_id="A1", target_account="TGT-SHARED", event_timestamp=T0 - timedelta(hours=3)),
        _event(customer_id="C2", account_id="A2", target_account="TGT-SHARED", event_timestamp=T0 - timedelta(hours=2)),
    ]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    assert beneficiary_fan_in_count(graph, ("recipient", "TGT-SHARED")) == 2


def test_sender_fan_out_count():
    events = [
        _event(customer_id="C1", account_id="A1", target_account="TGT-A", event_timestamp=T0 - timedelta(hours=3)),
        _event(customer_id="C1", account_id="A1", target_account="TGT-B", event_timestamp=T0 - timedelta(hours=2)),
    ]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    assert sender_fan_out_count(graph, ("customer", "C1")) == 2


def test_shortest_path_to_fraud_linked_entity_direct_link():
    events = [_event(customer_id="C1", account_id="A1", device_id="DEV-BAD", event_timestamp=T0 - timedelta(hours=1))]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    fraud_linked = frozenset({("device", "DEV-BAD")})
    assert shortest_path_to_fraud_linked_entity(graph, ("customer", "C1"), fraud_linked) == 1


def test_shortest_path_returns_none_when_unreachable():
    events = [_event(customer_id="C1", account_id="A1", event_timestamp=T0 - timedelta(hours=1))]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    fraud_linked = frozenset({("device", "NEVER_SEEN")})
    assert shortest_path_to_fraud_linked_entity(graph, ("customer", "C1"), fraud_linked) is None


def test_shortest_path_deterministic_traversal_across_repeated_calls():
    events = [
        _event(customer_id="C1", account_id="A1", device_id="DEV1", event_timestamp=T0 - timedelta(hours=3)),
        _event(customer_id="C2", account_id="A2", device_id="DEV1", target_account="TGT-BAD", event_timestamp=T0 - timedelta(hours=2)),
    ]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    fraud_linked = frozenset({("recipient", "TGT-BAD")})
    d1 = shortest_path_to_fraud_linked_entity(graph, ("customer", "C1"), fraud_linked)
    d2 = shortest_path_to_fraud_linked_entity(graph, ("customer", "C1"), fraud_linked)
    assert d1 == d2 == 2  # C1 -> DEV1 -> C2 -> ... actually via shared device then C2's own edges


# ---- graph score normalization -------------------------------------------------------


def test_graph_risk_score_is_bounded_zero_to_one():
    events = [
        _event(customer_id=f"C{i}", account_id=f"A{i}", device_id="DEV1", event_timestamp=T0 - timedelta(hours=i + 1))
        for i in range(20)
    ]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    score = compute_graph_risk_score(
        graph=graph, customer_key=("customer", "C0"), device_key=("device", "DEV1"), recipient_key=None,
        fraud_linked_entities=frozenset(), policy=_policy(),
    )
    assert 0.0 <= score <= 1.0


def test_graph_risk_score_increases_with_shared_device_count():
    def _graph_with_n_customers(n):
        events = [
            _event(customer_id=f"C{i}", account_id=f"A{i}", device_id="DEV1", event_timestamp=T0 - timedelta(hours=i + 1))
            for i in range(n)
        ]
        return build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())

    policy = _policy(fan_in_weight=0, fan_out_weight=0, shortest_path_weight=0, shared_device_weight=1.0)
    small = compute_graph_risk_score(
        graph=_graph_with_n_customers(1), customer_key=("customer", "C0"), device_key=("device", "DEV1"),
        recipient_key=None, fraud_linked_entities=frozenset(), policy=policy,
    )
    large = compute_graph_risk_score(
        graph=_graph_with_n_customers(5), customer_key=("customer", "C0"), device_key=("device", "DEV1"),
        recipient_key=None, fraud_linked_entities=frozenset(), policy=policy,
    )
    assert large > small


# ---- resource bounds (Phase 5 decision 8) --------------------------------------------


def test_graph_max_history_events_truncates_deterministically_keeping_most_recent():
    events = [
        _event(customer_id=f"C{i}", account_id=f"A{i}", device_id="DEV1", event_timestamp=T0 - timedelta(hours=100 - i))
        for i in range(100)
    ]
    policy = _policy(graph_max_history_events=10)
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=policy)
    assert graph.truncated is True
    assert graph.truncation_reason_code == "GRAPH_HISTORY_TRUNCATED"
    # The most recent 10 customers (C90..C99) should be present; the
    # oldest (C0) should not.
    assert ("customer", "C99") in graph.adjacency
    assert ("customer", "C0") not in graph.adjacency


def test_max_nodes_bound_is_never_exceeded():
    events = [
        _event(customer_id=f"C{i}", account_id=f"A{i}", event_timestamp=T0 - timedelta(hours=200 - i))
        for i in range(200)
    ]
    policy = _policy(graph_max_history_events=5000, max_nodes=50, max_edges=200000)
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=policy)
    assert graph.node_count <= 50
    assert graph.truncated is True


def test_max_edges_bound_is_never_exceeded():
    # A single event with many entities creates many edges; force a tiny
    # max_edges to prove the bound is honored.
    events = [
        _event(customer_id=f"C{i}", account_id=f"A{i}", device_id="DEV1", target_account="TGT-SHARED", event_timestamp=T0 - timedelta(hours=50 - i))
        for i in range(50)
    ]
    policy = _policy(graph_max_history_events=5000, max_nodes=100000, max_edges=20)
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=policy)
    assert graph.edge_count <= 20
    assert graph.truncated is True


def test_no_truncation_when_within_bounds():
    events = [_event(customer_id="C1", account_id="A1", event_timestamp=T0 - timedelta(hours=1))]
    graph = build_entity_graph(historical_events=events, current_event_timestamp=T0, policy=_policy())
    assert graph.truncated is False
    assert graph.truncation_reason_code is None
