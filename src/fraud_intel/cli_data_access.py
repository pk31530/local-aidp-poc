"""Minimal, real Postgres data access backing the v1.3 CLI's `generate`/
`train`/`score` commands (Phase 6, generalized to all 7 channels in Phase
7A). Reviewed as SQL, not exercised by any unit test -- every CLI test
monkeypatches these functions directly, same precedent as every other
"_Postgres*Store" in this codebase.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import psycopg2.extras

from src.common.db import get_connection
from src.fraud_intel.ensemble.policy import compute_operational_priority_score, load_ensemble_policy
from src.fraud_intel.evaluation.cross_channel import AlertOutcome
from src.fraud_intel.evaluation.shadow_candidate import CandidateScoringInput
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.generator.ach import generate_ach_events
from src.fraud_intel.generator.atm import generate_atm_events
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.generator.debit_card import generate_debit_card_events
from src.fraud_intel.generator.mobile_deposit import generate_mobile_deposit_events
from src.fraud_intel.generator.online_banking import generate_online_banking_events
from src.fraud_intel.generator.p2p import generate_p2p_events
from src.fraud_intel.generator.wire import generate_wire_events
from src.fraud_intel.registry import get_channel_adapter

_GENERATORS = {
    "ach": generate_ach_events,
    "wire": generate_wire_events,
    "mobile_deposit": generate_mobile_deposit_events,
    "online_banking": generate_online_banking_events,
    "atm": generate_atm_events,
    "debit_card": generate_debit_card_events,
    "p2p": generate_p2p_events,
}


def generate_and_write(*, channel: str, count: int, seed: int, database: str, reference_date: date) -> dict[str, Any]:
    if channel not in _GENERATORS:
        raise ValueError(f"unknown channel {channel!r}")
    generation_run_id = f"genrun-{uuid.uuid4()}"
    dataset_version = f"dsv-{uuid.uuid4().hex[:16]}"
    customers = generate_customers(n=max(count // 5, 1), seed=seed, reference_date=reference_date)
    results = _GENERATORS[channel](
        seed=seed, n=count, reference_date=reference_date, customers=customers,
        generation_run_id=generation_run_id, dataset_version=dataset_version,
    )
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor() as cur:
                for event, source_alert, label in results:
                    cur.execute(
                        "INSERT INTO channel_events (event_id, channel, customer_id, account_id, event_timestamp, "
                        "amount_minor_units, direction, device_id, ip_address, channel_payload, scenario_id, "
                        "schema_version, generation_run_id, dataset_version) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                        (
                            str(event.event_id), event.channel, event.customer_id, event.account_id,
                            event.event_timestamp, event.amount_minor_units, event.direction, event.device_id,
                            event.ip_address, psycopg2.extras.Json(event.channel_payload.model_dump(mode="json")),
                            event.scenario_id, event.schema_version, generation_run_id, dataset_version,
                        ),
                    )
                    if source_alert is not None:
                        cur.execute(
                            "INSERT INTO source_alerts (source_alert_id, source_system, event_id, "
                            "source_alert_created_at, source_rule_ids, source_rule_version, source_alert_score, "
                            "source_alert_reason_codes, generation_run_id, dataset_version) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT (source_system, source_alert_id) DO NOTHING",
                            (
                                str(source_alert.source_alert_id), source_alert.source_system, str(event.event_id),
                                source_alert.source_alert_created_at, psycopg2.extras.Json(source_alert.source_rule_ids),
                                source_alert.source_rule_version, source_alert.source_alert_score,
                                psycopg2.extras.Json(source_alert.source_alert_reason_codes),
                                generation_run_id, dataset_version,
                            ),
                        )
                    cur.execute(
                        "INSERT INTO synthetic_event_labels (event_id, scenario_id, synthetic_scenario_label, "
                        "scenario_type, generation_run_id, dataset_version) VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (event_id) DO NOTHING",
                        (
                            str(event.event_id), label.scenario_id, label.synthetic_scenario_label,
                            label.scenario_type, generation_run_id, dataset_version,
                        ),
                    )
    finally:
        conn.close()
    return {"channel": channel, "count": len(results), "generation_run_id": generation_run_id, "dataset_version": dataset_version}


def load_channel_population(
    channel: str, database: str,
) -> tuple[list[FraudEvent], list[SourceAlertContext], list[SyntheticGroundTruthLabel]]:
    """Phase 7A: generalized from Phase 6's load_online_banking_population()
    -- the channel's own registered payload class (src.fraud_intel.registry)
    reconstructs channel_payload, so this one function serves all 7
    channels instead of one hardcoded to online_banking. get_channel_adapter()
    itself is the channel-validity check."""
    payload_class = get_channel_adapter(channel).payload_class
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM channel_events WHERE channel = %s", (channel,))
                events = [
                    FraudEvent(
                        event_id=row["event_id"], channel=row["channel"], customer_id=row["customer_id"],
                        account_id=row["account_id"], event_timestamp=row["event_timestamp"],
                        amount_minor_units=row["amount_minor_units"], direction=row["direction"],
                        device_id=row["device_id"], ip_address=row["ip_address"], scenario_id=row["scenario_id"],
                        schema_version=row["schema_version"], channel_payload=payload_class(**row["channel_payload"]),
                    )
                    for row in cur.fetchall()
                ]
                event_ids = [str(e.event_id) for e in events]
                if not event_ids:
                    return [], [], []

                cur.execute("SELECT * FROM source_alerts WHERE event_id = ANY(%s)", (event_ids,))
                source_alerts = [
                    SourceAlertContext(
                        source_alert_id=row["source_alert_id"], source_system=row["source_system"],
                        event_id=row["event_id"], source_alert_created_at=row["source_alert_created_at"],
                        source_rule_ids=row["source_rule_ids"], source_rule_version=row["source_rule_version"],
                        source_alert_score=row["source_alert_score"],
                        source_alert_reason_codes=row["source_alert_reason_codes"],
                        generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                        created_at=row["created_at"],
                    )
                    for row in cur.fetchall()
                ]

                cur.execute("SELECT * FROM synthetic_event_labels WHERE event_id = ANY(%s)", (event_ids,))
                labels = [
                    SyntheticGroundTruthLabel(
                        event_id=row["event_id"], scenario_id=row["scenario_id"],
                        synthetic_scenario_label=row["synthetic_scenario_label"], scenario_type=row["scenario_type"],
                        generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                        generated_at=row["generated_at"],
                    )
                    for row in cur.fetchall()
                ]
    finally:
        conn.close()
    return events, source_alerts, labels


# ---- evaluation data access (Phase 7A corrective pass) --------------------------------


def load_resolved_alert_outcomes(channel: str, database: str) -> list[AlertOutcome]:
    """Real Postgres read backing `aidp fraud-intel evaluate`. Reviewed as
    SQL, not exercised by any unit test -- every CLI/evaluation test
    supplies fixture AlertOutcome rows directly. One row per fraud_alerts
    row that has both a latest alert_evidence row and a latest,
    ELIGIBLE, RESOLVED label_assessments row -- never an unresolved or
    ineligible alert. `baseline_priority_score` is recomputed for real
    from the stored evidence's own rule_result.score_contribution via the
    SAME compute_operational_priority_score() mechanism used everywhere
    else in this codebase, weight_gbm=weight_anomaly=weight_graph=0 on the
    channel's real EnsemblePolicy -- never a separately-implemented
    formula (guide section 21)."""
    policy = load_ensemble_policy(channel)
    baseline_policy = policy.model_copy(update={"weight_gbm": 0.0, "weight_anomaly": 0.0, "weight_graph": 0.0})

    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT fa.source_alert_id, fa.channel, ev.rule_result, ev.operational_priority_score,
                           ev.priority_band, ev.event_time, la.resolved_label
                    FROM fraud_alerts fa
                    JOIN LATERAL (
                        SELECT * FROM alert_evidence WHERE alert_id = fa.alert_id
                        ORDER BY scored_at DESC, evidence_id DESC LIMIT 1
                    ) ev ON true
                    JOIN LATERAL (
                        SELECT * FROM label_assessments WHERE alert_id = fa.alert_id
                        ORDER BY evaluated_at DESC, assessment_id DESC LIMIT 1
                    ) la ON true
                    WHERE fa.channel = %s AND la.eligibility_result = true
                      AND la.resolved_label IN ('RESOLVED_FRAUD', 'RESOLVED_LEGITIMATE')
                    """,
                    (channel,),
                )
                rows = cur.fetchall()
    finally:
        conn.close()

    outcomes = []
    for row in rows:
        rule_score_contribution = (row["rule_result"] or {}).get("score_contribution", 0.0)
        baseline_score = compute_operational_priority_score(
            rule_score_contribution=rule_score_contribution, calibrated_gbm_probability=0.0,
            anomaly_score=0.0, graph_risk_score=0.0, policy=baseline_policy,
        )
        outcomes.append(
            AlertOutcome(
                source_alert_id=row["source_alert_id"], channel=row["channel"], event_timestamp=row["event_time"],
                operational_priority_score=row["operational_priority_score"], baseline_priority_score=baseline_score,
                priority_band=row["priority_band"], resolved_label=row["resolved_label"],
            )
        )
    return outcomes


_MAX_HISTORICAL_EVENTS_PER_ALERT = 1000
_MAX_SOURCE_ALERT_HISTORY_PER_ALERT = 200
_MAX_RESOLVED_FRAUD_EVIDENCE_ROWS = 500


def load_resolved_alert_scoring_contexts(channel: str, database: str) -> list[CandidateScoringInput]:
    """Real Postgres read backing the OPTIONAL shadow-candidate
    comparison. Reviewed as SQL, not exercised. For every RESOLVED,
    ELIGIBLE alert in `channel` (the same population
    load_resolved_alert_outcomes() reads), reconstructs its full
    event-time scoring context -- event, source_alert,
    FeatureComputationContext (historical_events/source_alert_history),
    resolved_fraud_evidence -- the exact same shape and query pattern
    src.fraud_intel.scoring.dispatch._PostgresScoringDataAccess.list_pending()
    already builds for PENDING alerts, here for ALREADY-resolved ones.

    This function only READS. The candidate bundle is re-scored against
    these contexts in memory, via src.fraud_intel.evaluation.
    shadow_candidate.score_candidate_shadow() (the pure score_source_alert()
    path) -- never persisted anywhere. There used to be a
    load_candidate_shadow_scores() here that queried alert_evidence for a
    channel_model_bundle_id tag no writer has ever produced; it has been
    removed and replaced by this real, in-memory-scoring-oriented read."""
    from src.fraud_intel.graph.entity_graph import ResolvedFraudEntityEvidence

    payload_class = get_channel_adapter(channel).payload_class
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT fa.source_alert_id, fa.event_id, ce.channel AS ce_channel, ce.customer_id AS ce_customer_id,
                           ce.account_id AS ce_account_id, ce.event_timestamp AS ce_event_timestamp,
                           ce.amount_minor_units AS ce_amount_minor_units, ce.direction AS ce_direction,
                           ce.device_id AS ce_device_id, ce.ip_address AS ce_ip_address,
                           ce.channel_payload AS ce_channel_payload, ce.scenario_id AS ce_scenario_id,
                           ce.schema_version AS ce_schema_version,
                           sa.source_system, sa.source_alert_created_at, sa.source_rule_ids, sa.source_rule_version,
                           sa.source_alert_score, sa.source_alert_reason_codes, sa.generation_run_id,
                           sa.dataset_version, sa.created_at
                    FROM fraud_alerts fa
                    JOIN channel_events ce ON ce.event_id = fa.event_id
                    JOIN source_alerts sa ON sa.source_system = fa.source_system AND sa.source_alert_id = fa.source_alert_id
                    JOIN LATERAL (
                        SELECT * FROM label_assessments WHERE alert_id = fa.alert_id
                        ORDER BY evaluated_at DESC, assessment_id DESC LIMIT 1
                    ) la ON true
                    WHERE fa.channel = %s AND la.eligibility_result = true
                      AND la.resolved_label IN ('RESOLVED_FRAUD', 'RESOLVED_LEGITIMATE')
                    """,
                    (channel,),
                )
                alert_rows = cur.fetchall()

                items: list[CandidateScoringInput] = []
                for row in alert_rows:
                    event = FraudEvent(
                        event_id=row["event_id"], channel=row["ce_channel"], customer_id=row["ce_customer_id"],
                        account_id=row["ce_account_id"], event_timestamp=row["ce_event_timestamp"],
                        amount_minor_units=row["ce_amount_minor_units"], direction=row["ce_direction"],
                        device_id=row["ce_device_id"], ip_address=row["ce_ip_address"], scenario_id=row["ce_scenario_id"],
                        schema_version=row["ce_schema_version"], channel_payload=payload_class(**row["ce_channel_payload"]),
                    )
                    source_alert = SourceAlertContext(
                        source_alert_id=row["source_alert_id"], source_system=row["source_system"],
                        event_id=row["event_id"], source_alert_created_at=row["source_alert_created_at"],
                        source_rule_ids=row["source_rule_ids"], source_rule_version=row["source_rule_version"],
                        source_alert_score=row["source_alert_score"], source_alert_reason_codes=row["source_alert_reason_codes"],
                        generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                        created_at=row["created_at"],
                    )

                    cur.execute(
                        "SELECT * FROM channel_events WHERE customer_id = %s AND event_timestamp < %s "
                        "ORDER BY event_timestamp DESC LIMIT %s",
                        (event.customer_id, event.event_timestamp, _MAX_HISTORICAL_EVENTS_PER_ALERT),
                    )
                    historical_events = tuple(
                        FraudEvent(
                            event_id=h["event_id"], channel=h["channel"], customer_id=h["customer_id"],
                            account_id=h["account_id"], event_timestamp=h["event_timestamp"],
                            amount_minor_units=h["amount_minor_units"], direction=h["direction"],
                            device_id=h["device_id"], ip_address=h["ip_address"], scenario_id=h["scenario_id"],
                            schema_version=h["schema_version"],
                            channel_payload=get_channel_adapter(h["channel"]).payload_class(**h["channel_payload"]),
                        )
                        for h in cur.fetchall()
                    )

                    cur.execute(
                        "SELECT sa2.* FROM source_alerts sa2 JOIN channel_events ce2 ON ce2.event_id = sa2.event_id "
                        "WHERE ce2.customer_id = %s AND sa2.source_alert_created_at < %s "
                        "ORDER BY sa2.source_alert_created_at DESC LIMIT %s",
                        (event.customer_id, event.event_timestamp, _MAX_SOURCE_ALERT_HISTORY_PER_ALERT),
                    )
                    source_alert_history = tuple(
                        SourceAlertContext(
                            source_alert_id=h["source_alert_id"], source_system=h["source_system"],
                            event_id=h["event_id"], source_alert_created_at=h["source_alert_created_at"],
                            source_rule_ids=h["source_rule_ids"], source_rule_version=h["source_rule_version"],
                            source_alert_score=h["source_alert_score"], source_alert_reason_codes=h["source_alert_reason_codes"],
                            generation_run_id=h["generation_run_id"], dataset_version=h["dataset_version"],
                            created_at=h["created_at"],
                        )
                        for h in cur.fetchall()
                    )

                    context = FeatureComputationContext(
                        current_event=event, historical_events=historical_events,
                        source_alert_history=source_alert_history, as_of_time=event.event_timestamp,
                    )

                    cur.execute(
                        "SELECT la.assessment_id, la.policy_version, la.basis_timestamp, la.resolved_label_source, "
                        "fa2.customer_id, fa2.account_id FROM label_assessments la "
                        "JOIN fraud_alerts fa2 ON fa2.alert_id = la.alert_id "
                        "WHERE la.resolved_label = 'RESOLVED_FRAUD' AND la.eligibility_result = true "
                        "AND la.basis_timestamp < %s ORDER BY la.evaluated_at DESC LIMIT %s",
                        (event.event_timestamp, _MAX_RESOLVED_FRAUD_EVIDENCE_ROWS),
                    )
                    resolved_fraud_evidence: list[ResolvedFraudEntityEvidence] = []
                    for fraud_row in cur.fetchall():
                        common = dict(
                            label_assessment_id=str(fraud_row["assessment_id"]), resolved_fraud_at=fraud_row["basis_timestamp"],
                            eligibility_policy_version=fraud_row["policy_version"], label_source=fraud_row["resolved_label_source"],
                        )
                        resolved_fraud_evidence.append(
                            ResolvedFraudEntityEvidence(entity_type="customer", entity_id=fraud_row["customer_id"], **common)
                        )
                        resolved_fraud_evidence.append(
                            ResolvedFraudEntityEvidence(entity_type="account", entity_id=fraud_row["account_id"], **common)
                        )

                    items.append(
                        CandidateScoringInput(
                            source_alert_id=row["source_alert_id"], event=event, source_alert=source_alert,
                            context=context, resolved_fraud_evidence=tuple(resolved_fraud_evidence),
                        )
                    )
    finally:
        conn.close()
    return items
