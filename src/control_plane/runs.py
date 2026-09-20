"""Central run lifecycle/provenance service for AiDP v1.2 (Phase 2).

Not yet wired into src/processing/pipeline.py, src/ml/train.py or
src/ingestion/consumer.py (Phase 3). Deliberately independent of those
modules for the same reason src/control_plane/config.py is: Phase 3 will
have them depend on this module, not the other way around.

`RunLifecycle` never talks to Postgres directly — it delegates to an
injectable store (an instance of `_PostgresRunStore` by default). This keeps
the lifecycle/transition logic itself unit-testable without a live database,
which matters in this phase specifically because the migration that adds the
columns this module writes has not been applied to any running database yet.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2.extras
from pydantic import BaseModel, ConfigDict

from src.common.db import get_connection
from src.common.logging import get_logger
from src.control_plane.config import redact_secret_keys
from src.control_plane.provenance import summarize_exception, truncate_text

log = get_logger(__name__)

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

STATUSES = {PENDING, RUNNING, SUCCESS, FAILED, CANCELLED}
TERMINAL_STATUSES = {SUCCESS, FAILED, CANCELLED}

PIPELINE_NAMES = {
    "batch",
    "train",
    "stream",
    # v1.3 Phase 6 (guide section 11): additive only. Valid only once
    # migration 004's pipeline_runs.pipeline_name CHECK widen has also
    # been applied -- schema and this set must land together.
    "fraud_score",
    "label_eligibility",
    "model_promotion",
    "fraud_evaluation",
}
TRIGGER_SOURCES = {"cli", "legacy", "github_actions", "api", "test"}

# PENDING -> SUCCESS is deliberately absent: a run must pass through RUNNING
# before it can succeed.
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    PENDING: {RUNNING, FAILED, CANCELLED},
    RUNNING: {SUCCESS, FAILED, CANCELLED},
}

MAX_ERROR_MESSAGE_LENGTH = 2000
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 500

_COLUMNS = (
    "run_id",
    "pipeline_name",
    "status",
    "trigger_source",
    "git_sha",
    "config_snapshot",
    "config_hash",
    "dataset_version",
    "model_version",
    "records_processed",
    "records_rejected",
    "artifacts",
    "error_type",
    "error_message",
    "started_at",
    "heartbeat_at",
    "completed_at",
)


class RunNotFoundError(LookupError):
    """No pipeline_runs row exists for the given run_id."""


class InvalidRunTransitionError(ValueError):
    """The requested status change is not a legal transition from the run's
    current status."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: int
    pipeline_name: str
    status: str
    trigger_source: Optional[str] = None
    git_sha: Optional[str] = None
    config_snapshot: Optional[dict[str, Any]] = None
    config_hash: Optional[str] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    records_processed: int = 0
    records_rejected: int = 0
    artifacts: dict[str, Any] = {}
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    started_at: datetime
    heartbeat_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def _adapt(value: Any) -> Any:
    """Wrap dict values for JSONB columns; pass everything else through."""
    return psycopg2.extras.Json(value) if isinstance(value, dict) else value


class _PostgresRunStore:
    """The real, Postgres-backed persistence layer. Not exercised by Phase 2
    unit tests (see module docstring) — reviewed as SQL instead."""

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def insert(self, row: dict[str, Any]) -> RunRecord:
        columns = list(row.keys())
        placeholders = ", ".join(["%s"] * len(columns))
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        INSERT INTO pipeline_runs ({", ".join(columns)})
                        VALUES ({placeholders})
                        RETURNING {", ".join(_COLUMNS)}
                        """,
                        [_adapt(row[c]) for c in columns],
                    )
                    return RunRecord(**cur.fetchone())
        finally:
            conn.close()

    def compare_and_set(self, run_id: int, allowed_from: set[str], updates: dict[str, Any]) -> Optional[RunRecord]:
        columns = list(updates.keys())
        set_clause = ", ".join(f"{c} = %s" for c in columns)
        params = [_adapt(updates[c]) for c in columns] + [run_id, list(allowed_from)]
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        UPDATE pipeline_runs
                        SET {set_clause}
                        WHERE run_id = %s AND status = ANY(%s)
                        RETURNING {", ".join(_COLUMNS)}
                        """,
                        params,
                    )
                    row = cur.fetchone()
                    return RunRecord(**row) if row else None
        finally:
            conn.close()

    def get(self, run_id: int) -> Optional[RunRecord]:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT {', '.join(_COLUMNS)} FROM pipeline_runs WHERE run_id = %s",
                        (run_id,),
                    )
                    row = cur.fetchone()
                    return RunRecord(**row) if row else None
        finally:
            conn.close()

    def list(
        self, *, pipeline_name: Optional[str] = None, status: Optional[str] = None, limit: int = DEFAULT_LIST_LIMIT
    ) -> list[RunRecord]:
        clauses = []
        params: list[Any] = []
        if pipeline_name is not None:
            clauses.append("pipeline_name = %s")
            params.append(pipeline_name)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT {', '.join(_COLUMNS)} FROM pipeline_runs
                        {where}
                        ORDER BY started_at DESC
                        LIMIT %s
                        """,
                        params,
                    )
                    return [RunRecord(**row) for row in cur.fetchall()]
        finally:
            conn.close()


class RunLifecycle:
    """Begin, heartbeat, finalise and query pipeline_runs rows.

    `store` is injectable purely for testing; production callers should omit
    it and let a `_PostgresRunStore(database)` be constructed automatically.
    """

    def __init__(self, database: Optional[str] = None, store: Optional[Any] = None):
        self._store = store if store is not None else _PostgresRunStore(database)

    def begin(
        self,
        pipeline_name: str,
        *,
        status: str = RUNNING,
        trigger_source: Optional[str] = None,
        git_sha: Optional[str] = None,
        config_snapshot: Optional[dict[str, Any]] = None,
        config_hash: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> RunRecord:
        if pipeline_name not in PIPELINE_NAMES:
            raise ValueError(f"unknown pipeline_name {pipeline_name!r}; expected one of {sorted(PIPELINE_NAMES)}")
        if status not in (PENDING, RUNNING):
            raise ValueError(f"a run must begin as PENDING or RUNNING, not {status!r}")
        if trigger_source is not None and trigger_source not in TRIGGER_SOURCES:
            raise ValueError(f"unknown trigger_source {trigger_source!r}; expected one of {sorted(TRIGGER_SOURCES)}")

        row = {
            "pipeline_name": pipeline_name,
            "status": status,
            "trigger_source": trigger_source,
            "git_sha": git_sha,
            # Defence in depth: re-redact even an already-redacted snapshot.
            "config_snapshot": redact_secret_keys(config_snapshot) if config_snapshot is not None else None,
            "config_hash": config_hash,
            "dataset_version": dataset_version,
            "records_processed": 0,
            "records_rejected": 0,
            "artifacts": {},
        }
        return self._store.insert(row)

    def heartbeat(self, run_id: int) -> RunRecord:
        return self._transition(run_id, target=RUNNING, allowed_from={RUNNING}, updates={"heartbeat_at": _now()})

    def succeed(
        self,
        run_id: int,
        *,
        records_processed: Optional[int] = None,
        records_rejected: Optional[int] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> RunRecord:
        updates: dict[str, Any] = {"completed_at": _now()}
        if records_processed is not None:
            updates["records_processed"] = records_processed
        if records_rejected is not None:
            updates["records_rejected"] = records_rejected
        if dataset_version is not None:
            updates["dataset_version"] = dataset_version
        if model_version is not None:
            updates["model_version"] = model_version
        if artifacts is not None:
            updates["artifacts"] = redact_secret_keys(artifacts)
        return self._transition(run_id, target=SUCCESS, allowed_from={RUNNING}, updates=updates)

    def fail(
        self,
        run_id: int,
        *,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        records_processed: Optional[int] = None,
        records_rejected: Optional[int] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> RunRecord:
        updates: dict[str, Any] = {
            "completed_at": _now(),
            "error_type": error_type,
            "error_message": truncate_text(error_message, MAX_ERROR_MESSAGE_LENGTH),
        }
        if records_processed is not None:
            updates["records_processed"] = records_processed
        if records_rejected is not None:
            updates["records_rejected"] = records_rejected
        if dataset_version is not None:
            updates["dataset_version"] = dataset_version
        if model_version is not None:
            updates["model_version"] = model_version
        if artifacts is not None:
            updates["artifacts"] = redact_secret_keys(artifacts)
        return self._transition(run_id, target=FAILED, allowed_from={PENDING, RUNNING}, updates=updates)

    def fail_from_exception(
        self,
        run_id: int,
        exc: BaseException,
        *,
        records_processed: Optional[int] = None,
        records_rejected: Optional[int] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> Optional[RunRecord]:
        """Best-effort FAILED recording that never raises.

        Intended Phase 3 usage:
            except Exception as exc:
                lifecycle.fail_from_exception(run_id, exc)
                raise

        If the provenance write itself fails (e.g. the database is down),
        that secondary failure is logged and swallowed here so the caller's
        `raise` always re-propagates the *original* workload exception,
        never a provenance-layer one.
        """
        error_type, error_message = summarize_exception(exc)
        try:
            return self.fail(
                run_id,
                error_type=error_type,
                error_message=error_message,
                records_processed=records_processed,
                records_rejected=records_rejected,
                dataset_version=dataset_version,
                model_version=model_version,
                artifacts=artifacts,
            )
        except Exception:
            log.error("provenance_failed_status_write_failed", run_id=run_id, exc_info=True)
            return None

    def cancel(
        self,
        run_id: int,
        *,
        reason: Optional[str] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
    ) -> RunRecord:
        updates: dict[str, Any] = {
            "completed_at": _now(),
            "error_message": truncate_text(reason, MAX_ERROR_MESSAGE_LENGTH),
        }
        if dataset_version is not None:
            updates["dataset_version"] = dataset_version
        if model_version is not None:
            updates["model_version"] = model_version
        if artifacts is not None:
            updates["artifacts"] = redact_secret_keys(artifacts)
        return self._transition(run_id, target=CANCELLED, allowed_from={PENDING, RUNNING}, updates=updates)

    def get(self, run_id: int) -> RunRecord:
        record = self._store.get(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    def list(
        self, *, pipeline_name: Optional[str] = None, status: Optional[str] = None, limit: int = DEFAULT_LIST_LIMIT
    ) -> list[RunRecord]:
        if not (1 <= limit <= MAX_LIST_LIMIT):
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}, got {limit}")
        return self._store.list(pipeline_name=pipeline_name, status=status, limit=limit)

    def _transition(self, run_id: int, *, target: str, allowed_from: set[str], updates: dict[str, Any]) -> RunRecord:
        full_updates = {**updates, "status": target}
        result = self._store.compare_and_set(run_id, allowed_from, full_updates)
        if result is not None:
            return result

        current = self._store.get(run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        if current.status == target:
            # Safe idempotent duplicate finalisation: the first call already
            # wrote the terminal row, so return it unchanged rather than
            # re-applying (possibly different) arguments from this call.
            return current
        raise InvalidRunTransitionError(f"cannot move run {run_id} from {current.status!r} to {target!r}")
