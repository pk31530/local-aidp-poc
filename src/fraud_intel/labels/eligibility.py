"""Real, versioned label-eligibility policy (guide sections 8, 19; Phase 6)
-- supersedes Phase 4's interim `_phase4_interim_training_eligible()`.
Append-only: every re-evaluation produces a NEW `label_assessments` row,
never an update to an existing one. No real database contact in this
module's own code -- every unit test supplies `_FakeLabelAssessmentStore`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict

from src.common.db import get_connection
from src.control_plane.runs import RunLifecycle
import psycopg2.extras

LABEL_ELIGIBILITY_POLICY_VERSION = "v1"

# Phase 7B Stage 7 corrective pass: the fixed, AiDP-specific namespace half
# of a per-alert transaction-scoped advisory lock, serializing
# append_if_changed() so an exact-retry comparison and its conditional
# INSERT can never race against a concurrent attempt for the SAME alert.
# alert_id is BIGINT (fraud_alerts.alert_id is BIGSERIAL), so -- unlike
# src.fraud_intel.models.bundle._BUNDLE_VERSION_LOCK_NAMESPACE, which locks
# a hashtext(channel) STRING key via the two-int-key pg_advisory_xact_lock
# (int, int) overload -- the lock key here is built from the already-
# numeric, already-unique alert_id directly (no hashtext, no collision
# risk), packed into the single-bigint pg_advisory_xact_lock(bigint)
# overload: high 32 bits = this fixed namespace, low 32 bits = alert_id.
# Arbitrary but MUST stay fixed forever once assigned -- derived once as
# the low 31 bits of sha256(b"aidp_fraud_intel_label_assessments") purely
# so the value is reproducible and documented, never recomputed at
# runtime.
_LABEL_ASSESSMENT_LOCK_NAMESPACE = 0x6BDC2C5F

SYNTHETIC_IMMEDIATE_MATURITY = "SYNTHETIC_IMMEDIATE_MATURITY"
MATURITY_WINDOW_ELAPSED = "MATURITY_WINDOW_ELAPSED"
MATURITY_WINDOW_NOT_ELAPSED = "MATURITY_WINDOW_NOT_ELAPSED"
MATURE_BUT_UNRESOLVED = "MATURE_BUT_UNRESOLVED"

# Only exercised via constructed fixtures in this POC -- no real
# disposition-to-label maturity pipeline exists yet (Phase 6 scope note).
ANALYST_MATURITY_WINDOW_DAYS = 14

LabelSource = Literal["SYNTHETIC_GENERATOR", "ANALYST_DISPOSITION", "EXTERNAL_CONFIRMATION"]
ResolvedLabel = Literal["RESOLVED_FRAUD", "RESOLVED_LEGITIMATE", "UNRESOLVED"]


class AlertLabelBasis(BaseModel):
    """One fraud_alerts row's label basis, resolved by its data-access
    loader (src.fraud_intel.cli_data_access.load_alert_label_bases) to
    whichever evidentiary source actually exists for it -- the latest
    analyst_dispositions row when one exists, else the channel_event's own
    synthetic_event_labels row. Shaped as exactly assess_label()'s own
    non-`now` keyword arguments, so a basis unpacks straight into it."""

    model_config = ConfigDict(frozen=True)

    alert_id: int
    label_source: LabelSource
    basis_timestamp: datetime
    source_disposition_id: Optional[int]
    resolved_label: Optional[ResolvedLabel]

# CONFIRMED_FRAUD/CONFIRMED_LEGITIMATE resolve; NEEDS_MORE_INFO/ESCALATED
# are intentionally absent here and therefore always resolve to UNRESOLVED
# (guide/Phase 6 decision 5).
_RESOLVING_DISPOSITION_TO_LABEL: dict[str, ResolvedLabel] = {
    "CONFIRMED_FRAUD": "RESOLVED_FRAUD",
    "CONFIRMED_LEGITIMATE": "RESOLVED_LEGITIMATE",
}


class LabelAssessmentInputError(ValueError):
    """The supplied inputs violate the eligibility contract -- e.g. an
    analyst/external-derived assessment missing its source_disposition_id,
    or a synthetic one supplying one."""


def resolve_label_from_disposition(disposition: str) -> ResolvedLabel:
    return _RESOLVING_DISPOSITION_TO_LABEL.get(disposition, "UNRESOLVED")


def assess_label(
    *,
    alert_id: int,
    label_source: LabelSource,
    basis_timestamp: datetime,
    source_disposition_id: Optional[int],
    resolved_label: Optional[ResolvedLabel],
    now: datetime,
) -> dict[str, Any]:
    """Pure function -- no I/O.

    Guide section 8's critical rule, made concrete (Phase 6 decision 5):
    eligibility_result is true only when maturity has ALSO been reached
    AND the label actually resolved to fraud/legitimate -- UNRESOLVED or
    missing labels stay ineligible even once mature (MATURE_BUT_UNRESOLVED).

    For synthetic labels, basis_timestamp is the event's own
    event_timestamp (ground truth is known instantly at generation time --
    no real-world observation delay). For analyst/external-derived labels,
    basis_timestamp must be the actual evidentiary basis (e.g.
    disposition.disposed_at), never event_timestamp -- maturity for those
    is measured from when the evidence itself arrived, not from the
    original event.
    """
    if label_source == "SYNTHETIC_GENERATOR":
        if source_disposition_id is not None:
            raise LabelAssessmentInputError("synthetic assessments must have source_disposition_id = None")
        maturity_due_at = basis_timestamp
        maturity_status, maturity_reason = "MATURE", SYNTHETIC_IMMEDIATE_MATURITY
    else:
        if source_disposition_id is None:
            raise LabelAssessmentInputError(f"{label_source} assessments require a source_disposition_id")
        maturity_due_at = basis_timestamp + timedelta(days=ANALYST_MATURITY_WINDOW_DAYS)
        if now >= maturity_due_at:
            maturity_status, maturity_reason = "MATURE", MATURITY_WINDOW_ELAPSED
        else:
            maturity_status, maturity_reason = "IMMATURE", MATURITY_WINDOW_NOT_ELAPSED

    if maturity_status != "MATURE":
        eligibility_result = False
        eligibility_reason_code = maturity_reason
    elif resolved_label in ("RESOLVED_FRAUD", "RESOLVED_LEGITIMATE"):
        eligibility_result = True
        eligibility_reason_code = maturity_reason
    else:
        eligibility_result = False
        eligibility_reason_code = MATURE_BUT_UNRESOLVED

    return dict(
        alert_id=alert_id,
        policy_version=LABEL_ELIGIBILITY_POLICY_VERSION,
        source_disposition_id=source_disposition_id,
        basis_timestamp=basis_timestamp,
        maturity_due_at=maturity_due_at,
        maturity_status=maturity_status,
        eligibility_result=eligibility_result,
        eligibility_reason_code=eligibility_reason_code,
        resolved_label=resolved_label,
        resolved_label_source=label_source,
    )


# ---- typed record / store protocol -------------------------------------------------


class LabelAssessmentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    assessment_id: int
    alert_id: int
    policy_version: str
    evaluated_at: datetime
    source_disposition_id: Optional[int]
    basis_timestamp: datetime
    maturity_due_at: datetime
    maturity_status: str
    eligibility_result: bool
    eligibility_reason_code: str
    resolved_label: Optional[str]
    resolved_label_source: Optional[str]


# The complete set of LabelAssessmentRecord fields that define WHY/HOW a
# label was resolved -- i.e. everything except the two generated,
# per-insert identity fields (assessment_id, evaluated_at). Computed from
# the model itself (never a hand-maintained duplicate list) so it can
# never silently drift out of sync with LabelAssessmentRecord's real
# fields.
_LABEL_ASSESSMENT_GENERATED_FIELDS = frozenset({"assessment_id", "evaluated_at"})


@dataclass(frozen=True)
class AppendIfChangedResult:
    """Outcome of LabelAssessmentStore.append_if_changed(): the row now
    representing this alert's latest assessment (a freshly inserted one,
    or -- when the proposed assessment is semantically identical to what
    is already there -- the pre-existing one, unchanged), plus whether a
    new row was actually inserted."""

    record: LabelAssessmentRecord
    inserted: bool


def _semantic_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k not in _LABEL_ASSESSMENT_GENERATED_FIELDS}


class LabelAssessmentStore(Protocol):
    def append_assessment(self, **fields: Any) -> LabelAssessmentRecord: ...
    def append_if_changed(self, **fields: Any) -> AppendIfChangedResult: ...
    def get_latest_assessment(self, alert_id: int) -> Optional[LabelAssessmentRecord]: ...


class _PostgresLabelAssessmentStore:
    """Not exercised by any Phase 6 unit test -- reviewed as SQL instead.
    Not used anywhere until Phase 7B."""

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def append_assessment(self, **fields: Any) -> LabelAssessmentRecord:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    columns = list(fields.keys())
                    placeholders = ", ".join(["%s"] * len(columns))
                    cur.execute(
                        f"INSERT INTO label_assessments ({', '.join(columns)}) VALUES ({placeholders}) "
                        f"RETURNING assessment_id, {', '.join(columns)}, evaluated_at",
                        list(fields.values()),
                    )
                    return LabelAssessmentRecord(**cur.fetchone())
        finally:
            conn.close()

    def append_if_changed(self, **fields: Any) -> AppendIfChangedResult:
        """Phase 7B Stage 7 corrective pass: `append_assessment()` above is
        an unconditional INSERT -- calling it twice with an unchanged basis
        (e.g. an accidental retry of `labels assess`) silently doubles
        label_assessments forever, since there is no uniqueness constraint
        at the schema level either. This method makes an EXACT retry a
        no-op while still preserving genuine append-only history (a
        changed disposition, policy version, maturity/eligibility/resolved
        result always still appends a new row -- see this module's
        docstring and the Stage 7 test suite).

        Lock-then-read-then-write, all on one connection/transaction (same
        pg_advisory_xact_lock convention as
        src.fraud_intel.models.bundle._PostgresChannelModelBundleStore.
        register_candidate() -- released automatically at COMMIT/ROLLBACK,
        never an explicit unlock): the lock is acquired FIRST, before the
        latest-row SELECT, so a concurrent append_if_changed() for the
        SAME alert_id always serializes (one call's compare-then-insert
        completes, and commits, before the other's SELECT can run);
        concurrent calls for DIFFERENT alert_ids use different lock keys
        and proceed independently. Never UPDATEs or DELETEs an existing
        row -- the only write here is a plain INSERT, exactly like
        append_assessment()."""
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock((%s::bigint << 32) | (%s::bigint & 4294967295))",
                        (_LABEL_ASSESSMENT_LOCK_NAMESPACE, fields["alert_id"]),
                    )

                    cur.execute(
                        "SELECT * FROM label_assessments WHERE alert_id = %s "
                        "ORDER BY evaluated_at DESC, assessment_id DESC LIMIT 1",
                        (fields["alert_id"],),
                    )
                    row = cur.fetchone()
                    latest = LabelAssessmentRecord(**row) if row else None

                    if latest is not None:
                        existing_semantic = _semantic_fields(latest.model_dump())
                        proposed_semantic = _semantic_fields(fields)
                        if existing_semantic == proposed_semantic:
                            return AppendIfChangedResult(record=latest, inserted=False)

                    columns = list(fields.keys())
                    placeholders = ", ".join(["%s"] * len(columns))
                    cur.execute(
                        f"INSERT INTO label_assessments ({', '.join(columns)}) VALUES ({placeholders}) "
                        f"RETURNING assessment_id, {', '.join(columns)}, evaluated_at",
                        list(fields.values()),
                    )
                    return AppendIfChangedResult(record=LabelAssessmentRecord(**cur.fetchone()), inserted=True)
        finally:
            conn.close()

    def get_latest_assessment(self, alert_id: int) -> Optional[LabelAssessmentRecord]:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM label_assessments WHERE alert_id = %s "
                        "ORDER BY evaluated_at DESC, assessment_id DESC LIMIT 1",
                        (alert_id,),
                    )
                    row = cur.fetchone()
                    return LabelAssessmentRecord(**row) if row else None
        finally:
            conn.close()


def create_default_label_assessment_store(database: Optional[str] = None) -> LabelAssessmentStore:
    return _PostgresLabelAssessmentStore(database)


class _FakeLabelAssessmentStore:
    def __init__(self) -> None:
        self.rows: list[LabelAssessmentRecord] = []
        self._next_id = 1

    def append_assessment(self, **fields: Any) -> LabelAssessmentRecord:
        record = LabelAssessmentRecord(assessment_id=self._next_id, evaluated_at=datetime.now(timezone.utc), **fields)
        self._next_id += 1
        self.rows.append(record)  # append-only -- no update/delete path exists anywhere
        return record

    def append_if_changed(self, **fields: Any) -> AppendIfChangedResult:
        """In-memory mirror of _PostgresLabelAssessmentStore.append_if_changed
        -- no lock needed (single-threaded fakes), same compare-then-insert
        semantics."""
        latest = self.get_latest_assessment(fields["alert_id"])
        if latest is not None:
            existing_semantic = _semantic_fields(latest.model_dump())
            proposed_semantic = _semantic_fields(fields)
            if existing_semantic == proposed_semantic:
                return AppendIfChangedResult(record=latest, inserted=False)
        return AppendIfChangedResult(record=self.append_assessment(**fields), inserted=True)

    def get_latest_assessment(self, alert_id: int) -> Optional[LabelAssessmentRecord]:
        matching = [r for r in self.rows if r.alert_id == alert_id]
        if not matching:
            return None
        return sorted(matching, key=lambda r: (r.evaluated_at, r.assessment_id), reverse=True)[0]


# ---- explicit label-eligibility command orchestration (Phase 7B Stage 0) -----------------


LoadAlertLabelBases = Callable[[str, str], tuple[str, list[AlertLabelBasis]]]


def assess_channel_labels(
    *,
    channel: str,
    generation_run_id: str,
    lifecycle: RunLifecycle,
    load_label_bases: LoadAlertLabelBases,
    store: LabelAssessmentStore,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """One `label_eligibility` pipeline run (guide sections 8/19; the
    command this command is FOR is Stage 0's explicit
    `aidp fraud-intel labels assess`, never scoring itself -- scoring
    creates fraud_alerts/alert_evidence, never a label_assessments row).

    Phase 7B Stage 7 corrective pass: `generation_run_id` is now a
    required keyword -- `load_label_bases` must never silently load every
    row ever generated for a channel (same principle as
    src.fraud_intel.cli_data_access.load_channel_population()), and now
    returns the generation's own single dataset_version alongside its
    bases, recorded below as provenance. Every re-run appends a NEW
    assessment per alert only when its complete basis/result actually
    changed, via store.append_if_changed() -- an EXACT retry (identical
    basis/result) is a no-op, never a duplicate row, while a genuine
    reassessment (a changed disposition, policy version, or
    maturity/eligibility/resolved-label transition) always still appends,
    preserving history exactly as before (assess_label() itself remains
    pure and append-only). Never trains or promotes a model; a STRUCTURAL
    failure (the label-basis load itself failing) aborts the whole run
    and is recorded via fail_from_exception(), which always re-raises."""
    run = lifecycle.begin("label_eligibility", trigger_source="cli")
    effective_now = now if now is not None else datetime.now(timezone.utc)
    try:
        source_dataset_version, bases = load_label_bases(channel, generation_run_id)
        mature_count = 0
        immature_count = 0
        eligible_count = 0
        unresolved_count = 0
        resolved_fraud_count = 0
        resolved_legitimate_count = 0
        assessments_inserted = 0
        assessments_unchanged = 0
        for basis in bases:
            result = assess_label(
                alert_id=basis.alert_id,
                label_source=basis.label_source,
                basis_timestamp=basis.basis_timestamp,
                source_disposition_id=basis.source_disposition_id,
                resolved_label=basis.resolved_label,
                now=effective_now,
            )
            outcome = store.append_if_changed(**result)
            record = outcome.record
            if outcome.inserted:
                assessments_inserted += 1
            else:
                assessments_unchanged += 1
            if record.maturity_status == "MATURE":
                mature_count += 1
            else:
                immature_count += 1
            if record.eligibility_result:
                eligible_count += 1
            if record.resolved_label == "RESOLVED_FRAUD":
                resolved_fraud_count += 1
            elif record.resolved_label == "RESOLVED_LEGITIMATE":
                resolved_legitimate_count += 1
            if record.resolved_label in (None, "UNRESOLVED"):
                unresolved_count += 1
    except Exception as exc:
        lifecycle.fail_from_exception(run.run_id, exc)
        raise

    summary = {
        "channel": channel,
        "generation_run_id": generation_run_id,
        "source_dataset_version": source_dataset_version,
        "run_id": run.run_id,
        "policy_version": LABEL_ELIGIBILITY_POLICY_VERSION,
        "bases_evaluated": len(bases),
        "assessments_inserted": assessments_inserted,
        "assessments_unchanged": assessments_unchanged,
        "mature_count": mature_count,
        "immature_count": immature_count,
        "eligible_count": eligible_count,
        "resolved_fraud_count": resolved_fraud_count,
        "resolved_legitimate_count": resolved_legitimate_count,
        "unresolved_count": unresolved_count,
    }
    lifecycle.succeed(
        run.run_id,
        records_processed=len(bases),
        dataset_version=source_dataset_version,
        artifacts={
            "channel": channel,
            "generation_run_id": generation_run_id,
            "source_dataset_version": source_dataset_version,
            "bases_evaluated": len(bases),
            "assessments_inserted": assessments_inserted,
            "assessments_unchanged": assessments_unchanged,
            "eligible_count": eligible_count,
            "resolved_fraud_count": resolved_fraud_count,
            "resolved_legitimate_count": resolved_legitimate_count,
            "unresolved_count": unresolved_count,
        },
    )
    return summary
