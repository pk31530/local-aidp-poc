"""Dashboard smoke tests — require the local infrastructure stack running
(./scripts/start.sh; at minimum Postgres reachable at 127.0.0.1:5432).

Runs the actual Streamlit script headlessly (via Streamlit's own AppTest
framework, no browser needed) against the real running stack, and checks it
executes cleanly. This is a real execution of src/dashboard/app.py end to
end, not a mock — Streamlit runs every tab's code on each script run
regardless of which tab is visible, and src/dashboard/data.py opens a real
Postgres connection. Not part of tests/unit and not run in CI for that
reason — see RUNBOOK.md's "Running the tests" section.
"""
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
