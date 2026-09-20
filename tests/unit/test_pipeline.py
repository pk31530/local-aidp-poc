import csv
import inspect
from datetime import datetime, timezone

import polars as pl
import pytest

from src.control_plane.config import BatchRunConfig
from src.control_plane.runs import RunRecord
from src.processing import pipeline as pipeline_module

VALID_ROW = {
    "transaction_id": "TX1",
    "customer_id": "C1",
    "transaction_timestamp": "2026-06-01T10:00:00+05:30",
    "amount": "100.50",
    "merchant": "Grocery",
    "country": "India",
    "device_id": "DEV1",
    "payment_method": "CARD",
    "is_fraud": "false",
}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class _FakeRunStore:
    """In-memory stand-in for _PostgresRunStore — no database touched, same
    contract used by tests/unit/test_control_plane_runs.py."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
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
            "started_at": datetime.now(timezone.utc),
            "heartbeat_at": None,
            "completed_at": None,
            **row,
        }
        self.rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self.rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self.rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        return [RunRecord(**r) for r in list(self.rows.values())[:limit]]


@pytest.fixture(autouse=True)
def _fake_lifecycle_class(monkeypatch):
    """Replaces pipeline_module.RunLifecycle with a factory that always
    returns a lifecycle backed by a fresh in-memory store — no real
    database connection is ever opened by these tests."""
    from src.control_plane.runs import RunLifecycle as RealRunLifecycle

    store_holder = {"store": _FakeRunStore()}

    def _factory(*args, **kwargs):
        return RealRunLifecycle(store=store_holder["store"])

    monkeypatch.setattr(pipeline_module, "RunLifecycle", _factory)
    yield store_holder


def _redirect_output_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline_module, "MODELS_DIR", tmp_path / "models")


def _redirect_customer_data(monkeypatch, tmp_path):
    """Points customers/failed-attempts lookups at tiny synthetic fixtures
    instead of the real, gitignored, locally-generated seed data — a
    pipeline unit test must not depend on files nothing in the repo
    guarantees exist."""
    customers_path = tmp_path / "customers.parquet"
    pl.DataFrame(
        {
            "customer_id": ["C1"],
            "avg_transaction_amount": [100.0],
            "stddev_transaction_amount": [20.0],
            "known_devices": [["DEV1"]],
            "known_countries": [["India"]],
            "home_country": ["India"],
        }
    ).write_parquet(customers_path)
    monkeypatch.setattr(pipeline_module, "CUSTOMERS_PATH", customers_path)
    monkeypatch.setattr(pipeline_module, "FAILED_ATTEMPTS_PATH", tmp_path / "no_failed_attempts.csv")


def _write_valid_dataset(path, n=20):
    """20 rows, half fraud/half not, so the stratified 70/15/15 split in
    src.common.splits.assign_split has enough of each class to succeed."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "transaction_id": f"TX{i}",
                "customer_id": "C1",
                "transaction_timestamp": f"2026-06-01T{10 + i % 12:02d}:00:00+05:30",
                "amount": "100.50",
                "merchant": "Grocery",
                "country": "India",
                "device_id": "DEV1",
                "payment_method": "CARD",
                "is_fraud": "true" if i % 2 == 0 else "false",
            }
        )
    _write_csv(path, rows)


# ---- legacy signature is unchanged --------------------------------------------------


def test_run_pipeline_signature_unchanged():
    sig = inspect.signature(pipeline_module.run_pipeline)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["input_path", "upload_to_minio"]
    assert params[0].default is inspect.Parameter.empty  # input_path stays required
    assert params[1].default is True


_ORDINARY_DEFAULT_TYPES = (type(None), str, bool, int, float)


def test_no_sentinel_object_in_public_signatures():
    for fn in (pipeline_module.run_pipeline, pipeline_module.run_pipeline_configured):
        for param in inspect.signature(fn).parameters.values():
            if param.default is inspect.Parameter.empty:
                continue
            assert isinstance(param.default, _ORDINARY_DEFAULT_TYPES), (
                f"{fn.__name__}'s {param.name!r} default {param.default!r} is not an ordinary value"
            )


# ---- legacy wrapper builds the correct typed config, delegates once ----------------


def test_legacy_wrapper_builds_expected_config_and_delegates_once(monkeypatch, tmp_path):
    captured = {}

    def _fake_configured(config, *, trigger_source="legacy"):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(pipeline_module, "run_pipeline_configured", _fake_configured)

    input_path = tmp_path / "custom.csv"
    result = pipeline_module.run_pipeline(input_path, upload_to_minio=False)

    assert result == {"ok": True}
    assert captured["config"] == BatchRunConfig(input_path=input_path, upload_to_minio=False)
    assert captured["trigger_source"] == "legacy"


# ---- successful run: begin before work, dataset_version/artifacts recorded --------


def test_successful_run_records_success_with_counts_and_artifacts(monkeypatch, tmp_path, _fake_lifecycle_class):
    _redirect_output_dirs(monkeypatch, tmp_path)
    _redirect_customer_data(monkeypatch, tmp_path)
    input_path = tmp_path / "batch.csv"
    _write_valid_dataset(input_path)

    config = BatchRunConfig(input_path=input_path, upload_to_minio=False)
    summary = pipeline_module.run_pipeline_configured(config, trigger_source="test")

    assert summary["raw_valid"] == 20
    store = _fake_lifecycle_class["store"]
    assert len(store.rows) == 1  # exactly one run record for this invocation
    record = next(iter(store.rows.values()))
    assert record["status"] == "SUCCESS"
    assert record["trigger_source"] == "test"
    assert record["dataset_version"] is not None
    assert record["artifacts"]["raw_path"]
    assert "password" not in str(record["config_snapshot"])


# ---- failure inside the protected block: FAILED recorded, original error re-raised -


def test_run_pipeline_records_failed_status_and_reraises_on_stage_exception(monkeypatch, tmp_path, _fake_lifecycle_class):
    _redirect_output_dirs(monkeypatch, tmp_path)
    input_path = tmp_path / "batch.csv"
    _write_csv(input_path, [VALID_ROW])

    def _boom(*args, **kwargs):
        raise RuntimeError("clean stage exploded")

    monkeypatch.setattr(pipeline_module, "clean_transactions", _boom)

    config = BatchRunConfig(input_path=input_path, upload_to_minio=False)
    with pytest.raises(RuntimeError, match="clean stage exploded"):
        pipeline_module.run_pipeline_configured(config)

    store = _fake_lifecycle_class["store"]
    assert len(store.rows) == 1
    record = next(iter(store.rows.values()))
    assert record["status"] == "FAILED"
    assert record["records_processed"] == 0
    assert record["error_type"] == "RuntimeError"
    assert "clean stage exploded" in record["error_message"]


# ---- a missing input file must still record FAILED, not crash before begin() ------


def test_missing_input_file_records_failed_not_a_bare_crash(monkeypatch, tmp_path, _fake_lifecycle_class):
    _redirect_output_dirs(monkeypatch, tmp_path)
    missing_path = tmp_path / "does_not_exist.csv"

    config = BatchRunConfig(input_path=missing_path, upload_to_minio=False)
    with pytest.raises(FileNotFoundError):
        pipeline_module.run_pipeline_configured(config)

    store = _fake_lifecycle_class["store"]
    assert len(store.rows) == 1
    record = next(iter(store.rows.values()))
    assert record["status"] == "FAILED"
    assert record["dataset_version"] is None  # never computed — the file read failed first


# ---- legacy call path still ends in exactly one run record -------------------------


def test_legacy_call_creates_exactly_one_run_record(monkeypatch, tmp_path, _fake_lifecycle_class):
    _redirect_output_dirs(monkeypatch, tmp_path)
    _redirect_customer_data(monkeypatch, tmp_path)
    input_path = tmp_path / "batch.csv"
    _write_valid_dataset(input_path)

    pipeline_module.run_pipeline(input_path, upload_to_minio=False)

    store = _fake_lifecycle_class["store"]
    assert len(store.rows) == 1
    assert next(iter(store.rows.values()))["trigger_source"] == "legacy"
