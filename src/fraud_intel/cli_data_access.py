"""Minimal, real Postgres data access backing the v1.3 CLI's `generate`/
`train`/`score` commands (Phase 6, generalized to all 7 channels in Phase
7A). Reviewed as SQL, not exercised by any unit test -- every CLI test
monkeypatches these functions directly, same precedent as every other
"_Postgres*Store" in this codebase.
"""
from __future__ import annotations

import hashlib
import json
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
from src.fraud_intel.labels.eligibility import AlertLabelBasis, resolve_label_from_disposition
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

# Bump only when a channel generator's own logic changes such that the
# IDENTICAL (channel, count, seed, reference_date) inputs would now
# produce different events -- this is folded into the generation identity
# hash below specifically so that scenario, not the per-event payload
# schema_version (src.fraud_intel.events.base.FraudEvent.schema_version,
# which describes the payload shape, not the generator's own behavior)
# also invalidates old identities.
GENERATION_SPEC_VERSION = "v1"


class GenerationIdentityConflictError(ValueError):
    """The about-to-be-generated batch would silently collide, via
    channel_events' own ON CONFLICT (event_id) DO NOTHING, with rows
    already stored under a DIFFERENT generation_run_id -- e.g. the same
    (channel, seed, count) regenerated with a different reference_date.
    Event ids are deterministic per (seed, channel) alone (see
    src.fraud_intel.generator._shared.channel_rng) and are NOT themselves
    a function of reference_date, so this check exists precisely to catch
    that case explicitly rather than letting it be silently discarded."""


def _generation_identity(*, channel: str, count: int, seed: int, reference_date: date) -> tuple[str, str]:
    """Deterministic generation_run_id/dataset_version, derived from the
    complete generation specification (channel, count, seed,
    reference_date, GENERATION_SPEC_VERSION) -- NOT a fresh uuid4() per
    call. An exact retry (identical inputs) therefore always recomputes
    the SAME identifiers as whatever is already stored, rather than
    minting a new, never-actually-persisted "phantom" id while every
    insert is silently skipped by ON CONFLICT DO NOTHING."""
    canonical = json.dumps(
        {
            "channel": channel, "count": count, "seed": seed,
            "reference_date": reference_date.isoformat(),
            "generation_spec_version": GENERATION_SPEC_VERSION,
        },
        sort_keys=True, separators=(",", ":"),
    )
    spec_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"genrun-{spec_hash}", f"dsv-{spec_hash}"


def generate_and_write(*, channel: str, count: int, seed: int, database: str, reference_date: date) -> dict[str, Any]:
    if channel not in _GENERATORS:
        raise ValueError(f"unknown channel {channel!r}")

    generation_run_id, dataset_version = _generation_identity(channel=channel, count=count, seed=seed, reference_date=reference_date)
    customers = generate_customers(n=max(count // 5, 1), seed=seed, reference_date=reference_date)
    results = _GENERATORS[channel](
        seed=seed, n=count, reference_date=reference_date, customers=customers,
        generation_run_id=generation_run_id, dataset_version=dataset_version,
    )
    event_ids = [str(event.event_id) for event, _, _ in results]

    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as dict_cur:
                existing_run_id_by_event: dict[str, str] = {}
                if event_ids:
                    dict_cur.execute(
                        "SELECT event_id, generation_run_id FROM channel_events WHERE event_id = ANY(%s::uuid[])",
                        (event_ids,),
                    )
                    existing_run_id_by_event = {str(row["event_id"]): row["generation_run_id"] for row in dict_cur.fetchall()}

                foreign = {
                    eid: existing_run_id
                    for eid, existing_run_id in existing_run_id_by_event.items()
                    if existing_run_id != generation_run_id
                }
                if foreign:
                    sample_event_id, sample_run_id = next(iter(foreign.items()))
                    raise GenerationIdentityConflictError(
                        f"generating channel={channel!r} seed={seed} count={count} "
                        f"reference_date={reference_date.isoformat()!r} (generation_run_id={generation_run_id!r}) "
                        f"would collide with {len(foreign)} event_id(s) already stored under a DIFFERENT "
                        f"generation_run_id (e.g. event_id={sample_event_id!r} belongs to "
                        f"generation_run_id={sample_run_id!r}). event_ids are deterministic per (seed, channel) "
                        "alone, so the same (channel, seed, count) combination has already been generated with a "
                        "different reference_date or generator version -- refusing to silently discard those rows. "
                        "Use a different seed, or remove the conflicting generation_run_id's rows first."
                    )

                existing_event_count = len(existing_run_id_by_event)

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

    return {
        "channel": channel,
        "requested_count": count,
        "inserted_event_count": len(results) - existing_event_count,
        "existing_event_count": existing_event_count,
        "source_alert_count": sum(1 for _, source_alert, _ in results if source_alert is not None),
        "label_count": len(results),
        "generation_run_id": generation_run_id,
        "dataset_version": dataset_version,
        "reference_date": reference_date.isoformat(),
        "seed": seed,
    }


class UnknownGenerationRunError(ValueError):
    """No channel_events row exists for this generation_run_id at all --
    it was never generated (or a real writer failure left nothing
    persisted), so there is structurally nothing to train on."""


class GenerationRunChannelMismatchError(ValueError):
    """generation_run_id exists, but for a DIFFERENT channel than
    requested. Since generation_run_id is now a deterministic hash of
    the full generation spec (src.fraud_intel.cli_data_access.
    _generation_identity), including channel, this should never happen
    for a run_id this codebase itself minted -- it indicates a copy-paste
    error in the value passed on the command line."""


class GenerationRunDatasetVersionError(ValueError):
    """A generation_run_id row set spans more than one dataset_version
    for the requested channel -- the generation-identity invariant
    (exactly one dataset_version per generation_run_id) has been
    violated, e.g. by a hand-edited row. Training must refuse rather
    than silently pick one."""


def load_channel_population(
    channel: str, database: str, *, generation_run_id: str,
) -> tuple[list[FraudEvent], list[SourceAlertContext], list[SyntheticGroundTruthLabel]]:
    """Phase 7A: generalized from Phase 6's load_online_banking_population()
    -- the channel's own registered payload class (src.fraud_intel.registry)
    reconstructs channel_payload, so this one function serves all 7
    channels instead of one hardcoded to online_banking. get_channel_adapter()
    itself is the channel-validity check.

    Phase 7B Stage 2 corrective pass: `generation_run_id` is now a
    required keyword -- this function must NEVER silently load every row
    ever generated for a channel; it loads exactly the one generation
    run's population, and refuses (rather than training on an unintended
    mix of runs) if that run_id is unknown, belongs to a different
    channel, or spans more than one dataset_version for this channel."""
    payload_class = get_channel_adapter(channel).payload_class
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT DISTINCT channel FROM channel_events WHERE generation_run_id = %s", (generation_run_id,)
                )
                existing_channels = {row["channel"] for row in cur.fetchall()}
                if not existing_channels:
                    raise UnknownGenerationRunError(
                        f"no channel_events rows found for generation_run_id {generation_run_id!r} -- it was never "
                        "generated, or generation itself failed and rolled back"
                    )
                if existing_channels != {channel}:
                    raise GenerationRunChannelMismatchError(
                        f"generation_run_id {generation_run_id!r} belongs to channel(s) {sorted(existing_channels)!r}, "
                        f"not {channel!r}"
                    )

                cur.execute(
                    "SELECT count(DISTINCT dataset_version) AS n FROM channel_events "
                    "WHERE channel = %s AND generation_run_id = %s",
                    (channel, generation_run_id),
                )
                dataset_version_count = cur.fetchone()["n"]
                if dataset_version_count != 1:
                    raise GenerationRunDatasetVersionError(
                        f"generation_run_id {generation_run_id!r} has {dataset_version_count} distinct "
                        f"dataset_version value(s) for channel {channel!r} -- expected exactly 1"
                    )

                cur.execute(
                    "SELECT * FROM channel_events WHERE channel = %s AND generation_run_id = %s",
                    (channel, generation_run_id),
                )
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
                    # Defensive -- the distinct-channel check above already
                    # guarantees at least one row exists for this exact
                    # (channel, generation_run_id) pair.
                    return [], [], []

                cur.execute("SELECT * FROM source_alerts WHERE event_id = ANY(%s::uuid[])", (event_ids,))
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

                cur.execute("SELECT * FROM synthetic_event_labels WHERE event_id = ANY(%s::uuid[])", (event_ids,))
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


def load_alert_label_bases(channel: str, database: str) -> list[AlertLabelBasis]:
    """Real Postgres read backing `aidp fraud-intel labels assess`
    (Phase 7B Stage 0). Reviewed as SQL, not exercised by any unit test --
    every eligibility/CLI test supplies fixture AlertLabelBasis rows
    directly. One row per fraud_alerts row in `channel`: its latest
    analyst_dispositions row when one exists (ANALYST_DISPOSITION,
    basis_timestamp = disposed_at -- the evidentiary basis, never
    event_timestamp, per assess_label()'s own contract), else the
    channel_event's own synthetic_event_labels row (SYNTHETIC_GENERATOR,
    basis_timestamp = event_timestamp -- ground truth is known instantly
    at generation time for synthetic data, so it matures immediately
    through assess_label()'s SYNTHETIC_IMMEDIATE_MATURITY path)."""
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT fa.alert_id, ce.event_timestamp, sel.synthetic_scenario_label,
                           ad.disposition_id, ad.disposition, ad.disposed_at
                    FROM fraud_alerts fa
                    JOIN channel_events ce ON ce.event_id = fa.event_id
                    JOIN synthetic_event_labels sel ON sel.event_id = fa.event_id
                    LEFT JOIN LATERAL (
                        SELECT * FROM analyst_dispositions WHERE alert_id = fa.alert_id
                        ORDER BY disposed_at DESC, disposition_id DESC LIMIT 1
                    ) ad ON true
                    WHERE fa.channel = %s
                    """,
                    (channel,),
                )
                rows = cur.fetchall()
    finally:
        conn.close()

    bases: list[AlertLabelBasis] = []
    for row in rows:
        if row["disposition_id"] is not None:
            bases.append(
                AlertLabelBasis(
                    alert_id=row["alert_id"],
                    label_source="ANALYST_DISPOSITION",
                    basis_timestamp=row["disposed_at"],
                    source_disposition_id=row["disposition_id"],
                    resolved_label=resolve_label_from_disposition(row["disposition"]),
                )
            )
        else:
            bases.append(
                AlertLabelBasis(
                    alert_id=row["alert_id"],
                    label_source="SYNTHETIC_GENERATOR",
                    basis_timestamp=row["event_timestamp"],
                    source_disposition_id=None,
                    resolved_label="RESOLVED_FRAUD" if row["synthetic_scenario_label"] else "RESOLVED_LEGITIMATE",
                )
            )
    return bases
