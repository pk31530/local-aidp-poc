import pytest

from src.common.retry import transient_retry


def test_transient_retry_recovers_after_transient_failures():
    """Simulates Postgres being briefly down: the wrapped call fails twice,
    then succeeds — proves the retry path actually retries and returns the
    eventual success, not just catches-and-gives-up."""
    calls = {"count": 0}

    @transient_retry()
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("simulated transient Postgres outage")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_transient_retry_exhausts_and_reraises():
    """Simulates Postgres staying down for the whole retry budget: after
    max_attempts (5, per config/settings.yaml) the original exception is
    re-raised, not swallowed — this is what lets the consumer route the
    message to the DLQ instead of silently losing it (fix H3)."""
    calls = {"count": 0}

    @transient_retry()
    def always_fails():
        calls["count"] += 1
        raise ConnectionError("Postgres never came back")

    with pytest.raises(ConnectionError, match="Postgres never came back"):
        always_fails()

    assert calls["count"] == 5  # max_attempts from config/settings.yaml
