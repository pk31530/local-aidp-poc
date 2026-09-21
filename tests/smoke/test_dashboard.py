"""Dashboard smoke tests — require the local infrastructure stack running
(./scripts/start.sh; at minimum Postgres reachable at 127.0.0.1:5432).

Runs the actual Streamlit script headlessly (via Streamlit's own AppTest
framework, no browser needed) against the real running stack, and checks it
executes cleanly. This is a real execution of src/dashboard/app.py end to
end, not a mock — Streamlit runs every tab's code on each script run
regardless of which tab is visible, and src/dashboard/data.py opens a real
Postgres connection. Not part of tests/unit and not run in CI for that
reason — see RUNBOOK.md's "Running the tests" section.

Phase 8 corrective pass: every database connection reachable from the
dashboard (src/dashboard/app.py, src/dashboard/data.py,
src/dashboard/fraud_intel_tab.py) must resolve to the configured
`settings.postgres_test_db` (aidp_test by default) -- never a
zero-argument get_connection() call, which would silently resolve to
`settings.postgres_db`'s default ("aidp"). This was a real defect: the
platform health check's PostgreSQL probe (`get_platform_health()`) called
get_connection() with no argument, so every dashboard smoke-test run
(including the full-app AppTest runs above) briefly connected to `aidp`
even though this whole subsystem is meant to be aidp_test-only. Fixed by
threading `settings.postgres_test_db` through every dashboard connection
call explicitly.
"""
import src.common.db as db_module
from src.common.config import get_settings
from src.dashboard import data as dashboard_data
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_dashboard_shows_platform_health_metrics():
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=30)
    labels = [m.label for m in at.metric]
    for service in ("PostgreSQL", "MinIO", "Redpanda", "MLflow", "FastAPI"):
        assert service in labels


# ---- Phase 8 corrective pass: explicit, configured database only ----------------------


def _trace_get_connection(monkeypatch, module):
    """Wraps `module.get_connection` (already imported by name into that
    module, per its own `from src.common.db import get_connection`) with a
    tracer that records every `database` argument actually passed, then
    delegates to the real function -- proves what was ACTUALLY called,
    not just what the source appears to call."""
    calls: list[str | None] = []
    real = module.get_connection

    def traced(database=None):
        calls.append(database)
        return real(database)

    monkeypatch.setattr(module, "get_connection", traced)
    return calls


def test_get_platform_health_uses_the_configured_database(monkeypatch):
    calls = _trace_get_connection(monkeypatch, dashboard_data)
    dashboard_data.get_platform_health()
    assert calls, "get_platform_health() made no PostgreSQL connection attempt at all"
    assert all(c == get_settings().postgres_test_db for c in calls), calls
    assert all(c is not None for c in calls), "a zero-argument get_connection() call occurred"


def test_query_df_never_calls_get_connection_without_an_explicit_database(monkeypatch):
    calls = _trace_get_connection(monkeypatch, dashboard_data)
    dashboard_data.get_active_model()
    dashboard_data.get_live_transactions(5)
    dashboard_data.get_fraud_analysis()
    assert calls
    assert all(c == get_settings().postgres_test_db for c in calls), calls


def test_full_dashboard_path_never_makes_a_zero_argument_get_connection_call(monkeypatch):
    """Runs the entire app.py (all six tabs' code, same as AppTest always
    does) with BOTH src/dashboard/data.py's and
    src/dashboard/fraud_intel_tab.py's own `get_connection` names traced,
    and asserts every single real connection attempt across the whole app
    explicitly used aidp_test -- proving aidp was not contacted anywhere
    in the dashboard, not just in the one function already fixed."""
    from src.dashboard import fraud_intel_tab as fit_module

    data_calls = _trace_get_connection(monkeypatch, dashboard_data)
    fit_calls = _trace_get_connection(monkeypatch, fit_module)

    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0

    all_calls = data_calls + fit_calls
    assert all_calls, "no database connection was attempted at all -- test setup is broken"
    assert all(c is not None for c in all_calls), f"a zero-argument get_connection() call occurred: {all_calls}"
    assert all(c == "aidp_test" for c in all_calls), f"a non-aidp_test database was contacted: {all_calls}"
    assert all(c == get_settings().postgres_test_db for c in all_calls)


def _legacy_table_snapshot():
    with db_module.get_connection("aidp_test") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM transactions")
            transactions = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM fraud_decisions")
            fraud_decisions = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM model_versions")
            model_versions = cur.fetchone()[0]
    return {"transactions": transactions, "fraud_decisions": fraud_decisions, "model_versions": model_versions}


def test_full_dashboard_run_makes_zero_writes_anywhere_including_legacy_tables():
    """Broader than the fraud-intelligence-specific no-write proof in
    tests/smoke/test_fraud_intel_dashboard.py: covers the legacy v1.1
    tables too, since the full app.py (all six tabs) runs on every
    AppTest.run() regardless of which tab is "active"."""
    before = _legacy_table_snapshot()
    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0
    after = _legacy_table_snapshot()
    assert after == before, f"legacy dashboard tables changed: before={before} after={after}"
