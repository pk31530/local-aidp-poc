"""Alert queue (guide section 18, Phase 6): idempotent alert creation,
idempotent evidence persistence, and concurrency-safe append-only
disposition capture with status transitions. No real database contact in
this module's own code -- every unit test supplies `_FakeAlertQueueStore`;
the real, Postgres-backed store is reviewed as SQL, never exercised, same
precedent as every other "_Postgres*Store" in this codebase.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

import psycopg2.extras
from pydantic import BaseModel, ConfigDict

from src.common.db import get_connection
from src.common.logging import get_logger
from src.control_plane.provenance import redact_credentials, truncate_text
from src.fraud_intel.ensemble.policy import EnsemblePolicy
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy, ResolvedFraudEntityEvidence
from src.fraud_intel.reason_codes.builder import ReasonCode
from src.fraud_intel.rules.provider import RuleProvider
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle, ScoredAlert, score_source_alert

log = get_logger(__name__)

MAX_NOTES_LENGTH = 2000
SCORING_UNAVAILABLE_REASON_CODE = "SCORING_UNAVAILABLE"

# Phase 7B corrective pass (stale alert-list read-path fix): the single,
# canonical "current state" selection rule for one alert's evidence --
# the most recently scored row, ties broken by the higher evidence_id,
# with NO bundle filter (the repository's existing, already-shipped
# contract is "absolute latest evidence", not "current OPERATIONAL bundle
# evidence" -- see get_latest_evidence()'s pre-existing real-store SQL,
# unchanged here). Both get_latest_evidence() (single-alert) and
# list_alerts_with_current_state() (batch, avoids N+1 via one real SQL
# query with a LATERAL join) interpolate this SAME string, so they cannot
# silently drift apart the way _list_alerts()/get_latest_evidence() did
# before this fix -- a structural test asserts both real-store SQL
# strings contain it.
LATEST_EVIDENCE_ORDER_SQL = "scored_at DESC, evidence_id DESC"


def _latest_evidence_sort_key(evidence: "AlertEvidenceRecord") -> tuple:
    """Pure form of LATEST_EVIDENCE_ORDER_SQL, for in-memory selection
    (_FakeAlertQueueStore) -- sort ascending by this key and take the
    LAST element to get the same row the real SQL's ORDER BY ... LIMIT 1
    would return. `str(evidence_id)` matches Postgres UUID text ordering
    closely enough for the tie-break's purpose (distinguishing rows with
    equal scored_at, never used to compare across unrelated UUIDs for any
    other reason)."""
    return (evidence.scored_at, str(evidence.evidence_id))


_STATUS_BY_DISPOSITION = {
    "CONFIRMED_FRAUD": "CLOSED",
    "CONFIRMED_LEGITIMATE": "CLOSED",
    "NEEDS_MORE_INFO": "IN_REVIEW",
    "ESCALATED": "IN_REVIEW",
}


class InvalidAlertTransitionError(ValueError):
    """CLOSED is terminal in this MVP -- no disposition may reopen it
    (guide/Phase 6 decision 6)."""


# ---- typed records ------------------------------------------------------------------


class FraudAlertRecord(BaseModel):
    """`initial_operational_priority_score`/`initial_priority_band`/
    `initial_ensemble_policy_version` are set ONCE, at first-scoring time,
    and never updated by a later rescore (Phase 6 decision 2) -- the
    current/latest scoring result always comes from the latest
    `AlertEvidenceRecord` instead."""

    model_config = ConfigDict(frozen=True)

    alert_id: int
    event_id: uuid.UUID
    source_alert_id: uuid.UUID
    source_system: str
    channel: str
    customer_id: str
    account_id: str
    amount_minor_units: int
    initial_operational_priority_score: float
    initial_priority_band: str
    initial_ensemble_policy_version: str
    status: str
    created_at: datetime


class AlertEvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: uuid.UUID
    alert_id: int
    score_execution_id: uuid.UUID
    rule_result: Optional[dict]
    gbm_probability: Optional[float]
    lr_probability: Optional[float]
    anomaly_score: Optional[float]
    graph_risk_score: Optional[float]
    operational_priority_score: Optional[float]
    priority_band: Optional[str]
    degraded: bool
    component_statuses: dict
    reason_codes: list
    channel_model_bundle_id: int
    gbm_model_version: str
    lr_model_version: str
    anomaly_model_version: str
    preprocessing_artifact_version: str
    feature_schema_version: str
    rule_set_version: Optional[str]
    graph_policy_version: str
    ensemble_policy_version: str
    reason_code_version: str
    config_hash: str
    git_sha: Optional[str]
    event_time: datetime
    scored_at: datetime


class AlertListItem(BaseModel):
    """One row for `aidp alerts list` -- combines the alert's immutable
    identity/`initial_*` audit fields (Phase 6 decision 2, unchanged) with
    its CURRENT state, sourced from the latest `AlertEvidenceRecord`
    (same selection rule as `get_latest_evidence()`:
    LATEST_EVIDENCE_ORDER_SQL). `current_*` fields are explicitly None,
    never a fabricated or fallback-to-initial value, for the rare alert
    that has no evidence row at all (a partial write on the
    catastrophic-failure path of score_and_record_alert() -- see its own
    docstring). `initial_*` field NAMES are unchanged from FraudAlertRecord
    -- already unambiguous, never presented as current."""

    model_config = ConfigDict(frozen=True)

    alert_id: int
    event_id: uuid.UUID
    source_alert_id: uuid.UUID
    source_system: str
    channel: str
    customer_id: str
    account_id: str
    amount_minor_units: int
    status: str
    created_at: datetime

    initial_operational_priority_score: float
    initial_priority_band: str
    initial_ensemble_policy_version: str

    current_evidence_id: Optional[uuid.UUID]
    current_channel_model_bundle_id: Optional[int]
    current_priority_band: Optional[str]
    current_operational_priority_score: Optional[float]
    current_scored_at: Optional[datetime]


class AnalystDispositionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    disposition_id: int
    alert_id: int
    analyst_id: str
    disposition: str
    notes: Optional[str]
    disposed_at: datetime


# ---- store protocol -------------------------------------------------------------------


class AlertQueueStore(Protocol):
    def create_alert_if_new(
        self,
        *,
        event: FraudEvent,
        source_alert: SourceAlertContext,
        initial_operational_priority_score: float,
        initial_priority_band: str,
        initial_ensemble_policy_version: str,
    ) -> FraudAlertRecord: ...

    def record_evidence(self, *, alert_id: int, score_execution_id: uuid.UUID, **fields: Any) -> AlertEvidenceRecord: ...

    def get_alert(self, alert_id: int) -> FraudAlertRecord: ...

    def get_latest_evidence(self, alert_id: int) -> Optional[AlertEvidenceRecord]: ...

    def list_alerts_with_current_state(
        self,
        *,
        channel: Optional[str] = None,
        status: Optional[str] = None,
        current_priority_band: Optional[str] = None,
        limit: int = 100,
    ) -> list["AlertListItem"]: ...

    def list_dispositions(self, alert_id: int) -> list["AnalystDispositionRecord"]: ...

    def record_disposition_and_update_status(
        self, *, alert_id: int, analyst_id: str, disposition: str, notes: Optional[str], new_status: str
    ) -> AnalystDispositionRecord: ...


# ---- evidence-field reconciliation (Phase 6 decision 3/5) ------------------------------


def _evidence_fields_from_scored(scored: ScoredAlert, bundle: LoadedChannelBundle) -> dict[str, Any]:
    """channel_model_bundle_id/component-version/policy-version fields all
    come from `bundle` (its own PINNED provenance), never from `scored` --
    see LoadedChannelBundle's docstring for why."""
    return dict(
        score_execution_id=scored.score_execution_id,
        rule_result=scored.rule_result.model_dump(mode="json"),
        gbm_probability=scored.calibrated_gbm_probability,
        lr_probability=scored.lr_probability,
        anomaly_score=scored.anomaly_score,
        graph_risk_score=scored.graph_risk_score,
        operational_priority_score=scored.operational_priority_score,
        priority_band=scored.priority_band,
        degraded=scored.degraded,
        component_statuses={name: {"status": cs.status, "error_code": cs.error_code} for name, cs in scored.component_statuses.items()},
        reason_codes=[rc.model_dump(mode="json") for rc in scored.reason_codes],
        channel_model_bundle_id=bundle.bundle_id,
        gbm_model_version=bundle.gbm_model_version,
        lr_model_version=bundle.lr_model_version,
        anomaly_model_version=bundle.anomaly_model_version,
        preprocessing_artifact_version=bundle.preprocessing_artifact_version,
        feature_schema_version=bundle.feature_schema_version,
        rule_set_version=bundle.rule_set_version,
        graph_policy_version=bundle.graph_policy_version,
        ensemble_policy_version=bundle.ensemble_policy_version,
        reason_code_version=bundle.reason_code_version,
        config_hash=scored.config_hash,
        git_sha=scored.git_sha,
        event_time=scored.event_time,
        scored_at=scored.scored_at,
    )


def _catastrophic_evidence_fields(
    *, score_execution_id: uuid.UUID, bundle: LoadedChannelBundle, config_hash: str,
    git_sha: Optional[str], event: FraudEvent,
) -> dict[str, Any]:
    """Guide/Phase 6 decision 1 & 3: everything derivable from the CALL
    ARGUMENTS (bundle's own pinned provenance, config_hash, git_sha,
    event.event_timestamp) is populated even though scoring itself never
    completed -- only genuine scoring OUTPUTS (rule_result,
    operational_priority_score, priority_band, rule_set_version is the one
    exception since it is bundle-pinned too) are null. Never the raw
    exception text anywhere in this row."""
    return dict(
        score_execution_id=score_execution_id,
        rule_result=None,
        gbm_probability=None,
        lr_probability=None,
        anomaly_score=None,
        graph_risk_score=None,
        operational_priority_score=None,
        priority_band=None,
        degraded=True,
        component_statuses={"orchestrator": {"status": "ERROR", "error_code": SCORING_UNAVAILABLE_REASON_CODE}},
        reason_codes=[
            ReasonCode(
                code=SCORING_UNAVAILABLE_REASON_CODE,
                text="Scoring could not be completed for this alert.",
                layer="orchestrator",
                severity="informational",
            ).model_dump(mode="json")
        ],
        channel_model_bundle_id=bundle.bundle_id,
        gbm_model_version=bundle.gbm_model_version,
        lr_model_version=bundle.lr_model_version,
        anomaly_model_version=bundle.anomaly_model_version,
        preprocessing_artifact_version=bundle.preprocessing_artifact_version,
        feature_schema_version=bundle.feature_schema_version,
        rule_set_version=bundle.rule_set_version,
        graph_policy_version=bundle.graph_policy_version,
        ensemble_policy_version=bundle.ensemble_policy_version,
        reason_code_version=bundle.reason_code_version,
        config_hash=config_hash,
        git_sha=git_sha,
        event_time=event.event_timestamp,
        scored_at=datetime.now(timezone.utc),
    )


# ---- top-level orchestration -----------------------------------------------------------


def score_and_record_alert(
    *,
    event: FraudEvent,
    source_alert: SourceAlertContext,
    context: FeatureComputationContext,
    bundle: LoadedChannelBundle,
    rule_provider: RuleProvider,
    ensemble_policy: EnsemblePolicy,
    graph_policy: GraphPolicy,
    resolved_fraud_evidence: Sequence[ResolvedFraudEntityEvidence],
    config_hash: str,
    git_sha: Optional[str],
    store: AlertQueueStore,
    score_execution_id: Optional[uuid.UUID] = None,
) -> tuple[FraudAlertRecord, Optional[AlertEvidenceRecord]]:
    """Guarantees exactly one fraud_alerts row per valid SourceAlertContext
    -- created at FIRST-scoring time (guide section 4's architecture
    order), even when scoring is fully degraded or fails catastrophically.
    Never persists anything if `score_source_alert()` was never even
    called with valid, well-formed arguments (a wrong-channel event, for
    example, still raises immediately -- that is a caller bug, not a
    scoring-degradation case this function is responsible for
    absorbing).

    `score_execution_id` is optional (Phase 6 corrective pass): omitting
    it mints a fresh id, exactly as before. A caller that needs to safely
    RETRY the same logical scoring attempt (e.g. after a transient store
    failure) may instead pass the same id it used on the prior attempt --
    `alert_evidence`'s own UNIQUE(alert_id, score_execution_id) dedup
    (store.record_evidence()) then returns the existing row instead of
    writing a duplicate, rather than the retry silently producing a second,
    distinct evidence row for what is really one execution."""
    # Minted BEFORE scoring begins (Phase 6 decision 2) -- the SAME id
    # identifies both a genuine ScoredAlert and a catastrophic-failure
    # record for this one logical scoring attempt.
    if score_execution_id is None:
        score_execution_id = uuid.uuid4()

    try:
        scored = score_source_alert(
            event=event,
            source_alert=source_alert,
            context=context,
            bundle=bundle,
            rule_provider=rule_provider,
            ensemble_policy=ensemble_policy,
            graph_policy=graph_policy,
            resolved_fraud_evidence=resolved_fraud_evidence,
            score_execution_id=score_execution_id,
            config_hash=config_hash,
            git_sha=git_sha,
        )
    except Exception as exc:
        # Best-effort provenance block (Phase 6 decision 2): BOTH the
        # fallback alert creation and the catastrophic-evidence write are
        # inside this try -- if EITHER database operation fails, it is
        # logged (never exposed to normal output) and the ORIGINAL
        # scoring exception `exc` is re-raised regardless, never masked
        # by a secondary persistence failure.
        try:
            alert = store.create_alert_if_new(
                event=event,
                source_alert=source_alert,
                initial_operational_priority_score=1.0,
                initial_priority_band="HIGH",
                initial_ensemble_policy_version=ensemble_policy.policy_version,
            )
            store.record_evidence(
                alert_id=alert.alert_id,
                **_catastrophic_evidence_fields(
                    score_execution_id=score_execution_id,
                    bundle=bundle,
                    config_hash=config_hash,
                    git_sha=git_sha,
                    event=event,
                ),
            )
        except Exception:
            log.error(
                "catastrophic_alert_persistence_failed",
                event_id=str(event.event_id),
                source_alert_id=str(source_alert.source_alert_id),
                exc_info=True,
            )
        raise

    alert = store.create_alert_if_new(
        event=event,
        source_alert=source_alert,
        initial_operational_priority_score=scored.operational_priority_score,
        initial_priority_band=scored.priority_band,
        initial_ensemble_policy_version=ensemble_policy.policy_version,
    )
    evidence = store.record_evidence(alert_id=alert.alert_id, **_evidence_fields_from_scored(scored, bundle))
    return alert, evidence


def record_disposition(
    *, alert_id: int, analyst_id: str, disposition: str, notes: Optional[str], store: AlertQueueStore
) -> AnalystDispositionRecord:
    """`store.record_disposition_and_update_status()` performs the SELECT
    ... FOR UPDATE + status validation + insert + status update as ONE
    transaction (Phase 6 decision 4) -- this function never validates
    status itself, so there is no window between a read and a write for a
    race to exploit."""
    safe_notes = truncate_text(redact_credentials(notes), MAX_NOTES_LENGTH) if notes else None
    target_status = _STATUS_BY_DISPOSITION[disposition]
    return store.record_disposition_and_update_status(
        alert_id=alert_id, analyst_id=analyst_id, disposition=disposition, notes=safe_notes, new_status=target_status
    )


# ---- real, Postgres-backed store (reviewed as SQL, not exercised by any unit test) -----


class _PostgresAlertQueueStore:
    """Not exercised by any Phase 6 unit test -- reviewed as SQL instead,
    same precedent as every other "_Postgres*Store" in this codebase. Not
    used anywhere until Phase 7B applies migration 004 and wires real
    scoring runs to it."""

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def create_alert_if_new(
        self, *, event, source_alert, initial_operational_priority_score, initial_priority_band, initial_ensemble_policy_version
    ) -> FraudAlertRecord:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO fraud_alerts
                            (event_id, source_alert_id, source_system, channel, customer_id, account_id,
                             amount_minor_units, initial_operational_priority_score, initial_priority_band,
                             initial_ensemble_policy_version)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_system, source_alert_id) DO NOTHING
                        RETURNING alert_id, event_id, source_alert_id, source_system, channel, customer_id,
                                  account_id, amount_minor_units, initial_operational_priority_score,
                                  initial_priority_band, initial_ensemble_policy_version, status, created_at
                        """,
                        (
                            str(event.event_id), str(source_alert.source_alert_id), source_alert.source_system,
                            event.channel, event.customer_id, event.account_id, event.amount_minor_units,
                            initial_operational_priority_score, initial_priority_band, initial_ensemble_policy_version,
                        ),
                    )
                    row = cur.fetchone()
                    if row is None:
                        cur.execute(
                            "SELECT alert_id, event_id, source_alert_id, source_system, channel, customer_id, "
                            "account_id, amount_minor_units, initial_operational_priority_score, "
                            "initial_priority_band, initial_ensemble_policy_version, status, created_at "
                            "FROM fraud_alerts WHERE source_system = %s AND source_alert_id = %s",
                            (source_alert.source_system, str(source_alert.source_alert_id)),
                        )
                        row = cur.fetchone()
                    return FraudAlertRecord(**row)
        finally:
            conn.close()

    def record_evidence(self, *, alert_id, score_execution_id, **fields) -> AlertEvidenceRecord:
        columns = ["evidence_id", "alert_id", "score_execution_id"] + list(fields.keys())
        values = [str(uuid.uuid4()), alert_id, str(score_execution_id)] + [
            psycopg2.extras.Json(v) if isinstance(v, (dict, list)) else v for v in fields.values()
        ]
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    placeholders = ", ".join(["%s"] * len(columns))
                    cur.execute(
                        f"INSERT INTO alert_evidence ({', '.join(columns)}) VALUES ({placeholders}) "
                        f"ON CONFLICT (alert_id, score_execution_id) DO NOTHING "
                        f"RETURNING {', '.join(columns)}",
                        values,
                    )
                    row = cur.fetchone()
                    if row is None:
                        cur.execute(
                            f"SELECT {', '.join(columns)} FROM alert_evidence "
                            "WHERE alert_id = %s AND score_execution_id = %s",
                            (alert_id, str(score_execution_id)),
                        )
                        row = cur.fetchone()
                    return AlertEvidenceRecord(**row)
        finally:
            conn.close()

    def get_alert(self, alert_id: int) -> FraudAlertRecord:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM fraud_alerts WHERE alert_id = %s", (alert_id,))
                    return FraudAlertRecord(**cur.fetchone())
        finally:
            conn.close()

    def get_latest_evidence(self, alert_id: int) -> Optional[AlertEvidenceRecord]:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT * FROM alert_evidence WHERE alert_id = %s "
                        f"ORDER BY {LATEST_EVIDENCE_ORDER_SQL} LIMIT 1",
                        (alert_id,),
                    )
                    row = cur.fetchone()
                    return AlertEvidenceRecord(**row) if row else None
        finally:
            conn.close()

    def list_alerts_with_current_state(
        self,
        *,
        channel: Optional[str] = None,
        status: Optional[str] = None,
        current_priority_band: Optional[str] = None,
        limit: int = 100,
    ) -> list[AlertListItem]:
        """One parameterized query, no N+1: a LEFT JOIN LATERAL picks each
        alert's own latest evidence row via LATEST_EVIDENCE_ORDER_SQL --
        the SAME ordering get_latest_evidence() uses, so `alerts list` and
        `alerts show` can never again disagree about what "current" means
        (Phase 7B corrective pass: this replaced the old _list_alerts(),
        which read fraud_alerts.initial_priority_band only and never
        reflected a rescore under a later bundle). LEFT JOIN (not INNER)
        keeps an alert with zero evidence rows visible, with every
        current_* field None -- an explicit null representation, never a
        silent drop or a fabricated fallback to the initial_* values.
        `current_priority_band` filters on the JOINED (current) band, not
        `fraud_alerts.initial_priority_band`."""
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    clauses, params = [], []
                    if channel:
                        clauses.append("fa.channel = %s")
                        params.append(channel)
                    if status:
                        clauses.append("fa.status = %s")
                        params.append(status)
                    if current_priority_band:
                        clauses.append("le.priority_band = %s")
                        params.append(current_priority_band)
                    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                    cur.execute(
                        f"""
                        SELECT fa.*,
                               le.evidence_id AS current_evidence_id,
                               le.channel_model_bundle_id AS current_channel_model_bundle_id,
                               le.priority_band AS current_priority_band,
                               le.operational_priority_score AS current_operational_priority_score,
                               le.scored_at AS current_scored_at
                        FROM fraud_alerts fa
                        LEFT JOIN LATERAL (
                            SELECT evidence_id, channel_model_bundle_id, priority_band,
                                   operational_priority_score, scored_at
                            FROM alert_evidence ae
                            WHERE ae.alert_id = fa.alert_id
                            ORDER BY {LATEST_EVIDENCE_ORDER_SQL}
                            LIMIT 1
                        ) le ON true
                        {where}
                        ORDER BY fa.created_at DESC, fa.alert_id DESC
                        LIMIT %s
                        """,
                        params + [limit],
                    )
                    return [AlertListItem(**row) for row in cur.fetchall()]
        finally:
            conn.close()

    def list_dispositions(self, alert_id: int) -> list[AnalystDispositionRecord]:
        """Phase 8 (read-only dashboard): the full analyst disposition
        history for one alert, oldest first -- never written to by this
        method. Disposition CAPTURE remains exclusively
        record_disposition_and_update_status() (CLI-only, per Phase 6
        decision 4); this is purely an additional getter, mirroring
        get_alert()/get_latest_evidence()'s existing read-only pattern."""
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT disposition_id, alert_id, analyst_id, disposition, notes, disposed_at "
                        "FROM analyst_dispositions WHERE alert_id = %s ORDER BY disposed_at ASC, disposition_id ASC",
                        (alert_id,),
                    )
                    return [AnalystDispositionRecord(**row) for row in cur.fetchall()]
        finally:
            conn.close()

    def record_disposition_and_update_status(self, *, alert_id, analyst_id, disposition, notes, new_status) -> AnalystDispositionRecord:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # SELECT ... FOR UPDATE + validation + insert + update,
                    # all inside this one transaction (Phase 6 decision 4).
                    cur.execute("SELECT status FROM fraud_alerts WHERE alert_id = %s FOR UPDATE", (alert_id,))
                    row = cur.fetchone()
                    if row is None:
                        raise LookupError(f"no fraud_alerts row for alert_id={alert_id}")
                    if row["status"] == "CLOSED":
                        raise InvalidAlertTransitionError(f"alert {alert_id} is CLOSED (terminal); disposition rejected")
                    cur.execute(
                        "INSERT INTO analyst_dispositions (alert_id, analyst_id, disposition, notes) "
                        "VALUES (%s, %s, %s, %s) RETURNING disposition_id, alert_id, analyst_id, disposition, notes, disposed_at",
                        (alert_id, analyst_id, disposition, notes),
                    )
                    disposition_row = cur.fetchone()
                    cur.execute("UPDATE fraud_alerts SET status = %s WHERE alert_id = %s", (new_status, alert_id))
                    return AnalystDispositionRecord(**disposition_row)
        finally:
            conn.close()


def create_default_alert_queue_store(database: Optional[str] = None) -> AlertQueueStore:
    return _PostgresAlertQueueStore(database)


# ---- fake, in-memory store (every Phase 6 unit test uses this) -------------------------


class _FakeAlertQueueStore:
    def __init__(self) -> None:
        self.alerts_by_key: dict[tuple[str, str], FraudAlertRecord] = {}
        self.alerts_by_id: dict[int, FraudAlertRecord] = {}
        self.evidence_by_key: dict[tuple[int, str], AlertEvidenceRecord] = {}
        self.evidence_by_alert: dict[int, list[AlertEvidenceRecord]] = {}
        self.dispositions: list[AnalystDispositionRecord] = []
        self._next_alert_id = 1
        self._next_disposition_id = 1

    def create_alert_if_new(
        self, *, event, source_alert, initial_operational_priority_score, initial_priority_band, initial_ensemble_policy_version
    ) -> FraudAlertRecord:
        key = (source_alert.source_system, str(source_alert.source_alert_id))
        existing = self.alerts_by_key.get(key)
        if existing is not None:
            return existing
        alert = FraudAlertRecord(
            alert_id=self._next_alert_id,
            event_id=event.event_id,
            source_alert_id=source_alert.source_alert_id,
            source_system=source_alert.source_system,
            channel=event.channel,
            customer_id=event.customer_id,
            account_id=event.account_id,
            amount_minor_units=event.amount_minor_units,
            initial_operational_priority_score=initial_operational_priority_score,
            initial_priority_band=initial_priority_band,
            initial_ensemble_policy_version=initial_ensemble_policy_version,
            status="OPEN",
            created_at=datetime.now(timezone.utc),
        )
        self._next_alert_id += 1
        self.alerts_by_key[key] = alert
        self.alerts_by_id[alert.alert_id] = alert
        return alert

    def record_evidence(self, *, alert_id, score_execution_id, **fields) -> AlertEvidenceRecord:
        key = (alert_id, str(score_execution_id))
        existing = self.evidence_by_key.get(key)
        if existing is not None:
            return existing
        evidence = AlertEvidenceRecord(
            evidence_id=uuid.uuid4(), alert_id=alert_id, score_execution_id=score_execution_id, **fields
        )
        self.evidence_by_key[key] = evidence
        self.evidence_by_alert.setdefault(alert_id, []).append(evidence)
        return evidence

    def get_alert(self, alert_id: int) -> FraudAlertRecord:
        return self.alerts_by_id[alert_id]

    def get_latest_evidence(self, alert_id: int) -> Optional[AlertEvidenceRecord]:
        rows = self.evidence_by_alert.get(alert_id, [])
        if not rows:
            return None
        return sorted(rows, key=_latest_evidence_sort_key, reverse=True)[0]

    def list_alerts_with_current_state(
        self,
        *,
        channel: Optional[str] = None,
        status: Optional[str] = None,
        current_priority_band: Optional[str] = None,
        limit: int = 100,
    ) -> list[AlertListItem]:
        items: list[AlertListItem] = []
        for alert in self.alerts_by_id.values():
            if channel is not None and alert.channel != channel:
                continue
            if status is not None and alert.status != status:
                continue
            latest = self.get_latest_evidence(alert.alert_id)
            if current_priority_band is not None and (latest is None or latest.priority_band != current_priority_band):
                continue
            items.append(
                AlertListItem(
                    alert_id=alert.alert_id, event_id=alert.event_id, source_alert_id=alert.source_alert_id,
                    source_system=alert.source_system, channel=alert.channel, customer_id=alert.customer_id,
                    account_id=alert.account_id, amount_minor_units=alert.amount_minor_units, status=alert.status,
                    created_at=alert.created_at,
                    initial_operational_priority_score=alert.initial_operational_priority_score,
                    initial_priority_band=alert.initial_priority_band,
                    initial_ensemble_policy_version=alert.initial_ensemble_policy_version,
                    current_evidence_id=latest.evidence_id if latest else None,
                    current_channel_model_bundle_id=latest.channel_model_bundle_id if latest else None,
                    current_priority_band=latest.priority_band if latest else None,
                    current_operational_priority_score=latest.operational_priority_score if latest else None,
                    current_scored_at=latest.scored_at if latest else None,
                )
            )
        items.sort(key=lambda i: (i.created_at, i.alert_id), reverse=True)
        return items[:limit]

    def list_dispositions(self, alert_id: int) -> list[AnalystDispositionRecord]:
        return sorted(
            (d for d in self.dispositions if d.alert_id == alert_id),
            key=lambda d: (d.disposed_at, d.disposition_id),
        )

    def record_disposition_and_update_status(self, *, alert_id, analyst_id, disposition, notes, new_status) -> AnalystDispositionRecord:
        alert = self.alerts_by_id[alert_id]
        if alert.status == "CLOSED":
            raise InvalidAlertTransitionError(f"alert {alert_id} is CLOSED (terminal); disposition rejected")
        record = AnalystDispositionRecord(
            disposition_id=self._next_disposition_id,
            alert_id=alert_id,
            analyst_id=analyst_id,
            disposition=disposition,
            notes=notes,
            disposed_at=datetime.now(timezone.utc),
        )
        self._next_disposition_id += 1
        self.dispositions.append(record)
        updated_alert = alert.model_copy(update={"status": new_status})
        self.alerts_by_id[alert_id] = updated_alert
        self.alerts_by_key[(alert.source_system, str(alert.source_alert_id))] = updated_alert
        return record
