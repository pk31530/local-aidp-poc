from datetime import datetime, timezone

import pytest

from src.control_plane.runs import (
    CANCELLED,
    FAILED,
    PENDING,
    RUNNING,
    SUCCESS,
    InvalidRunTransitionError,
    RunLifecycle,
    RunNotFoundError,
    RunRecord,
)


def _now():
    return datetime.now(timezone.utc)


class _FakeRunStore:
    """In-memory stand-in for _PostgresRunStore. No database, no network —
    this is what lets Phase 2's lifecycle tests run without the migration
    that adds these columns having been applied anywhere."""

    def __init__(self):
        self._rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row: dict) -> RunRecord:
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id,
            "trigger_source": None,
            "git_sha": None,
            "config_snapshot": None,
            "config_hash": None,
            "dataset_version": None,
            "model_version": None,
            "error_type": None,
            "error_message": None,
            "started_at": _now(),
            "heartbeat_at": None,
            "completed_at": None,
            **row,
        }
        self._rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self._rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self._rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        rows = list(self._rows.values())
        if pipeline_name is not None:
            rows = [r for r in rows if r["pipeline_name"] == pipeline_name]
        if status is not None:
            rows = [r for r in rows if r["status"] == status]
        rows = sorted(rows, key=lambda r: r["started_at"], reverse=True)
        return [RunRecord(**r) for r in rows[:limit]]


class _RaisingStore(_FakeRunStore):
    """A store whose compare_and_set always fails, to simulate a database
    that's down during failure finalisation."""

    def compare_and_set(self, run_id, allowed_from, updates):
        raise RuntimeError("db down")


@pytest.fixture
def lifecycle():
    return RunLifecycle(store=_FakeRunStore())


# ---- begin ------------------------------------------------------------------------


def test_begin_defaults_to_running(lifecycle):
    run = lifecycle.begin("batch")
    assert run.status == RUNNING
    assert run.pipeline_name == "batch"
    assert run.records_processed == 0
    assert run.records_rejected == 0


def test_begin_with_pending_status(lifecycle):
    run = lifecycle.begin("train", status=PENDING)
    assert run.status == PENDING


def test_begin_rejects_unknown_pipeline_name(lifecycle):
    with pytest.raises(ValueError):
        lifecycle.begin("not_a_real_pipeline")


def test_begin_rejects_unknown_trigger_source(lifecycle):
    with pytest.raises(ValueError):
        lifecycle.begin("batch", trigger_source="not_a_real_source")


def test_begin_rejects_terminal_initial_status(lifecycle):
    with pytest.raises(ValueError):
        lifecycle.begin("batch", status=SUCCESS)


def test_begin_redacts_config_snapshot_defensively(lifecycle):
    run = lifecycle.begin("batch", config_snapshot={"postgres_dsn": "postgresql://u:p@h/db", "input_path": "x.csv"})
    assert run.config_snapshot["postgres_dsn"] == "***REDACTED***"
    assert run.config_snapshot["input_path"] == "x.csv"


# ---- successful / failed lifecycle -------------------------------------------------


def test_successful_lifecycle(lifecycle):
    run = lifecycle.begin("batch")
    lifecycle.heartbeat(run.run_id)
    finished = lifecycle.succeed(run.run_id, records_processed=100, records_rejected=2)
    assert finished.status == SUCCESS
    assert finished.records_processed == 100
    assert finished.records_rejected == 2
    assert finished.completed_at is not None


def test_failed_lifecycle_records_error_type_and_message(lifecycle):
    run = lifecycle.begin("stream")
    failed = lifecycle.fail(run.run_id, error_type="RuntimeError", error_message="boom")
    assert failed.status == FAILED
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "boom"
    assert failed.completed_at is not None


def test_fail_truncates_long_error_message(lifecycle):
    run = lifecycle.begin("stream")
    failed = lifecycle.fail(run.run_id, error_type="RuntimeError", error_message="x" * 3000)
    assert len(failed.error_message) <= 2000


def test_fail_records_dataset_and_model_version_and_artifacts(lifecycle):
    run = lifecycle.begin("train")
    failed = lifecycle.fail(
        run.run_id,
        error_type="RuntimeError",
        error_message="boom",
        dataset_version="abc123",
        model_version="7",
        artifacts={"postgres_dsn": "postgresql://u:p@h/db", "note": "ok"},
    )
    assert failed.dataset_version == "abc123"
    assert failed.model_version == "7"
    assert failed.artifacts["postgres_dsn"] == "***REDACTED***"
    assert failed.artifacts["note"] == "ok"


def test_cancel_records_dataset_and_model_version_and_artifacts(lifecycle):
    run = lifecycle.begin("stream")
    cancelled = lifecycle.cancel(
        run.run_id,
        reason="operator stopped it",
        dataset_version="abc123",
        model_version="7",
        artifacts={"raw_bucket": "aidp-raw"},
    )
    assert cancelled.dataset_version == "abc123"
    assert cancelled.model_version == "7"
    assert cancelled.artifacts == {"raw_bucket": "aidp-raw"}


# ---- state-transition rules ---------------------------------------------------------


def test_heartbeat_on_terminal_run_raises(lifecycle):
    run = lifecycle.begin("batch")
    lifecycle.succeed(run.run_id)
    with pytest.raises(InvalidRunTransitionError):
        lifecycle.heartbeat(run.run_id)


def test_succeed_after_failed_raises(lifecycle):
    run = lifecycle.begin("batch")
    lifecycle.fail(run.run_id, error_type="X", error_message="x")
    with pytest.raises(InvalidRunTransitionError):
        lifecycle.succeed(run.run_id)


def test_fail_after_succeeded_raises(lifecycle):
    run = lifecycle.begin("batch")
    lifecycle.succeed(run.run_id)
    with pytest.raises(InvalidRunTransitionError):
        lifecycle.fail(run.run_id, error_type="X", error_message="x")


def test_pending_cannot_succeed_directly(lifecycle):
    run = lifecycle.begin("batch", status=PENDING)
    with pytest.raises(InvalidRunTransitionError):
        lifecycle.succeed(run.run_id)


def test_pending_can_fail_directly(lifecycle):
    run = lifecycle.begin("batch", status=PENDING)
    failed = lifecycle.fail(run.run_id, error_type="SetupError", error_message="setup failed")
    assert failed.status == FAILED


def test_pending_can_be_cancelled(lifecycle):
    run = lifecycle.begin("train", status=PENDING)
    cancelled = lifecycle.cancel(run.run_id, reason="superseded")
    assert cancelled.status == CANCELLED


def test_running_can_be_cancelled(lifecycle):
    run = lifecycle.begin("stream")
    cancelled = lifecycle.cancel(run.run_id, reason="operator stopped it")
    assert cancelled.status == CANCELLED


def test_cancelled_is_terminal(lifecycle):
    run = lifecycle.begin("stream")
    lifecycle.cancel(run.run_id)
    with pytest.raises(InvalidRunTransitionError):
        lifecycle.succeed(run.run_id)


# ---- duplicate finalisation is a safe, non-mutating no-op ---------------------------


def test_duplicate_succeed_is_idempotent_and_does_not_mutate(lifecycle):
    run = lifecycle.begin("batch")
    first = lifecycle.succeed(run.run_id, records_processed=10)
    second = lifecycle.succeed(run.run_id, records_processed=999)
    assert second.records_processed == 10
    assert second.completed_at == first.completed_at


def test_duplicate_fail_is_idempotent_and_does_not_mutate(lifecycle):
    run = lifecycle.begin("batch")
    first = lifecycle.fail(run.run_id, error_type="A", error_message="first")
    second = lifecycle.fail(run.run_id, error_type="B", error_message="second")
    assert second.error_type == "A"
    assert second.error_message == "first"
    assert second.completed_at == first.completed_at


# ---- get / list ------------------------------------------------------------------------


def test_get_missing_run_raises(lifecycle):
    with pytest.raises(RunNotFoundError):
        lifecycle.get(999)


def test_transition_on_missing_run_raises(lifecycle):
    with pytest.raises(RunNotFoundError):
        lifecycle.heartbeat(999)


@pytest.mark.parametrize("limit", [0, -1, 501, 10_000])
def test_list_rejects_out_of_bounds_limit(lifecycle, limit):
    with pytest.raises(ValueError):
        lifecycle.list(limit=limit)


def test_list_respects_limit_and_filters(lifecycle):
    for _ in range(3):
        lifecycle.begin("batch")
    for _ in range(2):
        lifecycle.begin("stream")
    assert len(lifecycle.list(pipeline_name="batch", limit=50)) == 3
    assert len(lifecycle.list(limit=2)) == 2


# ---- fail_from_exception: never hides the original exception ------------------------


class _OriginalWorkloadError(Exception):
    pass


def test_fail_from_exception_records_failure_on_healthy_store(lifecycle):
    run = lifecycle.begin("batch")
    result = lifecycle.fail_from_exception(run.run_id, ValueError("bad input"))
    assert result is not None
    assert result.status == FAILED
    assert result.error_type == "ValueError"
    assert result.error_message == "bad input"


def test_fail_from_exception_swallows_provenance_write_failure():
    lifecycle = RunLifecycle(store=_RaisingStore())
    run = lifecycle.begin("batch")
    result = lifecycle.fail_from_exception(run.run_id, ValueError("bad input"))
    assert result is None  # provenance write failed, but no exception escaped


def test_fail_from_exception_never_hides_the_original_exception_in_caller_pattern():
    """Exercises the exact except/raise pattern Phase 3 will use, with a
    store that fails while recording FAILED, and proves the original
    exception is still what a caller observes."""
    lifecycle = RunLifecycle(store=_RaisingStore())
    run = lifecycle.begin("batch")

    def _do_work():
        try:
            raise _OriginalWorkloadError("the real problem")
        except Exception as exc:
            lifecycle.fail_from_exception(run.run_id, exc)
            raise

    with pytest.raises(_OriginalWorkloadError, match="the real problem"):
        _do_work()
