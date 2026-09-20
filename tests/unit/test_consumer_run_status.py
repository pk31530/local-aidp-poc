import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.control_plane.config import StreamRunConfig
from src.control_plane.runs import RunLifecycle as RealRunLifecycle
from src.control_plane.runs import RunRecord
from src.ingestion import consumer as consumer_module

VALID_TX_PAYLOAD = {
    "transaction_id": "TX1",
    "customer_id": "C1",
    "transaction_timestamp": "2026-06-01T10:00:00+05:30",
    "amount": 100.50,
    "merchant": "Grocery",
    "country": "India",
    "device_id": "DEV1",
    "payment_method": "CARD",
    "schema_version": 1,
}


class _FakeMsg:
    def __init__(self, err=None, value=None):
        self._err = err
        self._value = value

    def error(self):
        return self._err

    def value(self):
        return self._value


class _FakeBrokerErrorConsumer:
    """Always returns a message carrying a broker error, so run()'s
    `if msg.error(): raise KafkaException(...)` fires on the first poll."""

    def __init__(self, config):
        self.config = config

    def subscribe(self, topics):
        pass

    def poll(self, timeout=1.0):
        return _FakeMsg(err="simulated broker error")

    def commit(self, msg):
        pass

    def close(self):
        pass


class _FakeAlwaysMessageConsumer:
    """Always returns the same valid, schema-compliant message — combined
    with a short `duration`, the loop processes it repeatedly for a brief
    real-time window and then exits gracefully."""

    def __init__(self, config):
        self.config = config

    def subscribe(self, topics):
        pass

    def poll(self, timeout=1.0):
        return _FakeMsg(value=json.dumps(VALID_TX_PAYLOAD).encode("utf-8"))

    def commit(self, msg):
        pass

    def close(self):
        pass


class _FakeInterruptingConsumer:
    """Raises KeyboardInterrupt on the first poll, simulating an operator
    stopping the consumer with Ctrl+C."""

    def __init__(self, config):
        self.config = config

    def subscribe(self, topics):
        pass

    def poll(self, timeout=1.0):
        raise KeyboardInterrupt()

    def commit(self, msg):
        pass

    def close(self):
        pass


class _FakeProducer:
    def __init__(self, config):
        self.config = config

    def produce(self, topic, value=None):
        pass

    def poll(self, timeout=0):
        pass

    def flush(self, timeout=None):
        pass


class _FakeScoreResult:
    fraud_probability = 0.1
    decision = "APPROVE"


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


class _HeartbeatAlwaysFailsLifecycle(RealRunLifecycle):
    def heartbeat(self, run_id):
        raise RuntimeError("heartbeat db down")


def _patch_common(monkeypatch, consumer_cls):
    monkeypatch.setattr(consumer_module, "_load_model_or_raise", lambda: ("fake-model", "v1"))
    # never a real risk_lookups.json on disk for this test, regardless of repo state
    monkeypatch.setattr(consumer_module, "RISK_LOOKUPS_PATH", Path("/nonexistent/risk_lookups.json"))
    monkeypatch.setattr(consumer_module, "Consumer", consumer_cls)
    monkeypatch.setattr(consumer_module, "Producer", _FakeProducer)
    monkeypatch.setattr(consumer_module, "_score_with_retry", lambda *a, **k: _FakeScoreResult())
    monkeypatch.setattr(consumer_module, "upload_file", lambda *a, **k: None)


@pytest.fixture
def fake_lifecycle(monkeypatch):
    store = _FakeRunStore()
    monkeypatch.setattr(consumer_module, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store))
    return store


# ---- legacy signature unchanged, no sentinel in public API -------------------------


def test_run_signature_unchanged():
    sig = inspect.signature(consumer_module.run)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == [
        "duration",
        "from_beginning",
        "topic",
        "dlq_topic",
        "database",
        "group_id",
        "raw_bucket",
    ]
    assert params[0].default is inspect.Parameter.empty  # duration stays required
    assert params[1].default is inspect.Parameter.empty  # from_beginning stays required
    assert params[2].default is None
    assert params[3].default is None
    assert params[4].default is None
    assert params[5].default == "aidp-consumer"
    assert params[6].default == "aidp-raw"


_ORDINARY_DEFAULT_TYPES = (type(None), str, bool, int, float)


def test_no_sentinel_object_in_public_signatures():
    for fn in (consumer_module.run, consumer_module.run_configured):
        for param in inspect.signature(fn).parameters.values():
            if param.default is inspect.Parameter.empty:
                continue
            assert isinstance(param.default, _ORDINARY_DEFAULT_TYPES)


def test_legacy_wrapper_builds_expected_config_and_delegates_once(monkeypatch):
    captured = {}

    def _fake_configured(config, *, trigger_source="legacy"):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(consumer_module, "run_configured", _fake_configured)

    result = consumer_module.run(5, False, database="aidp_test", group_id="g1")

    assert result == {"ok": True}
    assert captured["config"] == StreamRunConfig(duration=5, from_beginning=False, database="aidp_test", group_id="g1")
    assert captured["trigger_source"] == "legacy"


# ---- unexpected failure: FAILED recorded, original exception re-raised ------------


def test_run_records_failed_status_and_reraises_on_broker_error(monkeypatch, fake_lifecycle):
    _patch_common(monkeypatch, _FakeBrokerErrorConsumer)

    with pytest.raises(consumer_module.KafkaException):
        consumer_module.run_configured(StreamRunConfig(duration=5, from_beginning=False))

    assert len(fake_lifecycle.rows) == 1
    record = next(iter(fake_lifecycle.rows.values()))
    assert record["status"] == "FAILED"
    assert record["records_processed"] == 0
    assert record["records_rejected"] == 0


# ---- graceful duration expiry -> SUCCESS -------------------------------------------


def test_duration_expiry_records_success(monkeypatch, fake_lifecycle):
    _patch_common(monkeypatch, _FakeAlwaysMessageConsumer)

    summary = consumer_module.run_configured(StreamRunConfig(duration=0.05, from_beginning=False), trigger_source="test")

    assert summary["processed"] >= 1
    assert len(fake_lifecycle.rows) == 1  # exactly one run record for this invocation
    record = next(iter(fake_lifecycle.rows.values()))
    assert record["status"] == "SUCCESS"
    assert record["trigger_source"] == "test"
    assert record["model_version"] == "v1"


# ---- operator interrupt (Ctrl+C) -> SUCCESS, same as graceful duration expiry -----


def test_keyboard_interrupt_records_success_not_stuck_running(monkeypatch, fake_lifecycle):
    _patch_common(monkeypatch, _FakeInterruptingConsumer)

    summary = consumer_module.run_configured(StreamRunConfig(duration=None, from_beginning=False))

    assert summary["processed"] == 0
    assert len(fake_lifecycle.rows) == 1
    record = next(iter(fake_lifecycle.rows.values()))
    assert record["status"] == "SUCCESS"  # never left RUNNING


# ---- heartbeat failures never affect message processing ---------------------------


def test_heartbeat_failure_does_not_disrupt_processing_or_final_status(monkeypatch):
    _patch_common(monkeypatch, _FakeAlwaysMessageConsumer)
    monkeypatch.setattr(consumer_module, "HEARTBEAT_EVERY_N_MESSAGES", 1)

    store = _FakeRunStore()
    monkeypatch.setattr(consumer_module, "RunLifecycle", lambda *a, **k: _HeartbeatAlwaysFailsLifecycle(store=store))

    summary = consumer_module.run_configured(StreamRunConfig(duration=0.05, from_beginning=False))

    assert summary["processed"] >= 1  # heartbeat raising every time never blocked processing
    record = next(iter(store.rows.values()))
    assert record["status"] == "SUCCESS"
    assert record["heartbeat_at"] is None  # every heartbeat attempt failed and was swallowed


# ---- legacy call path still ends in exactly one run record -------------------------


def test_legacy_call_creates_exactly_one_run_record(monkeypatch, fake_lifecycle):
    _patch_common(monkeypatch, _FakeAlwaysMessageConsumer)

    consumer_module.run(0.05, False)

    assert len(fake_lifecycle.rows) == 1
    assert next(iter(fake_lifecycle.rows.values()))["trigger_source"] == "legacy"


# ---- no secrets in recorded config snapshot ----------------------------------------


def test_config_snapshot_contains_no_secrets(monkeypatch, fake_lifecycle):
    _patch_common(monkeypatch, _FakeAlwaysMessageConsumer)

    consumer_module.run_configured(StreamRunConfig(duration=0.05, from_beginning=False, database="aidp_test"))

    record = next(iter(fake_lifecycle.rows.values()))
    assert "password" not in json.dumps(record["config_snapshot"])
