from pathlib import Path

import pytest

from src.ingestion import consumer as consumer_module


class _FakeMsg:
    def __init__(self, err):
        self._err = err

    def error(self):
        return self._err


class _FakeConsumer:
    """Always returns a message carrying a broker error, so run()'s
    `if msg.error(): raise KafkaException(...)` fires on the first poll."""

    def __init__(self, config):
        self.config = config

    def subscribe(self, topics):
        pass

    def poll(self, timeout=1.0):
        return _FakeMsg("simulated broker error")

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


def _patch_common(monkeypatch):
    monkeypatch.setattr(consumer_module, "_load_model_or_raise", lambda: ("fake-model", "v1"))
    # never a real risk_lookups.json on disk for this test, regardless of repo state
    monkeypatch.setattr(consumer_module, "RISK_LOOKUPS_PATH", Path("/nonexistent/risk_lookups.json"))
    monkeypatch.setattr(consumer_module, "Consumer", _FakeConsumer)
    monkeypatch.setattr(consumer_module, "Producer", _FakeProducer)
    monkeypatch.setattr(consumer_module, "_record_run_start", lambda database=None: 42)


def test_run_records_failed_status_and_reraises_on_broker_error(monkeypatch):
    _patch_common(monkeypatch)

    calls = []
    monkeypatch.setattr(
        consumer_module,
        "_record_run_end",
        lambda run_id, processed, rejected, database=None, status="SUCCESS": calls.append(
            (run_id, processed, rejected, status)
        ),
    )

    with pytest.raises(consumer_module.KafkaException):
        consumer_module.run(duration=5, from_beginning=False)

    assert len(calls) == 1
    run_id, processed, rejected, status = calls[0]
    assert run_id == 42
    assert status == "FAILED"
    assert processed == 0
    assert rejected == 0
