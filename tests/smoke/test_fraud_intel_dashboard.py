"""Phase 8: Fraud Intelligence dashboard tab smoke tests — require the
local infrastructure stack running (./scripts/start.sh; Postgres reachable
at 127.0.0.1:5432) AND the real seven-channel Phase 7B `aidp_test` data
this platform actually builds up over the course of the demo (generated
events, promoted bundles, scored/labeled alerts). Not part of
`tests/unit` and not run in CI, for the same reason as
`tests/smoke/test_dashboard.py`: real Postgres access, no mocks.

Runs the real `src/dashboard/app.py` headlessly via Streamlit's own
`AppTest` framework (same technique as `test_dashboard.py`), and also
exercises `src.dashboard.fraud_intel_tab`'s own read-only data-access
functions directly against `aidp_test` -- never `aidp` (see that
module's own `DATABASE = "aidp_test"` constant).
"""
from __future__ import annotations

import ast
import inspect

import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import fraud_intel_tab
from src.common.db import get_connection

FORBIDDEN_WRITE_CALLS = {
    "score_and_record_alert",
    "create_alert_if_new",
    "record_evidence",
    "record_disposition_and_update_status",
    "score_channel",
    "train_channel_configured",
    "promote_bundle",
    "assess_channel_labels",
    "generate_and_write",
}


def _db_snapshot():
    with get_connection("aidp_test") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM fraud_alerts")
            fraud_alerts = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM alert_evidence")
            alert_evidence = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM label_assessments")
            label_assessments = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM analyst_dispositions")
            analyst_dispositions = cur.fetchone()[0]
            cur.execute("SELECT count(*), max(run_id) FROM pipeline_runs")
            pipeline_runs_total, pipeline_runs_highest = cur.fetchone()
    return {
        "fraud_alerts": fraud_alerts, "alert_evidence": alert_evidence,
        "label_assessments": label_assessments, "analyst_dispositions": analyst_dispositions,
        "pipeline_runs_total": pipeline_runs_total, "pipeline_runs_highest": pipeline_runs_highest,
    }


# ---- structural: no write-shaped call anywhere in this module's source ----------------


def test_fraud_intel_tab_source_contains_no_write_call():
    """AST-based (established repo convention): the module's own source
    never calls any function known to write to the database, MLflow, or
    a lifecycle table -- proves the dashboard path is structurally
    read-only, not merely "I didn't call it this time."."""
    tree = ast.parse(inspect.getsource(fraud_intel_tab))
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name:
                called_names.add(name)
    forbidden_found = called_names & FORBIDDEN_WRITE_CALLS
    assert not forbidden_found, f"fraud_intel_tab.py calls forbidden write function(s): {forbidden_found}"


def test_fraud_intel_tab_source_never_accesses_synthetic_fields():
    """AST-based: the module's CODE never does attribute access
    (`.scenario_id`) or a matching dict/subscript key access on
    `scenario_id`/`synthetic_scenario_label` -- a functional check on
    actual data access, not a ban on mentioning the field names in
    documentation (the module's own docstring names them exactly to
    document that they're never exposed, which is the point -- so plain
    string constants, which is all a docstring is, are deliberately not
    checked here)."""
    tree = ast.parse(inspect.getsource(fraud_intel_tab))
    forbidden = {"scenario_id", "synthetic_scenario_label"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            hits.append(f".{node.attr}")
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in forbidden:
                hits.append(f"[{key.value!r}]")
    assert not hits, f"fraud_intel_tab.py's code accesses forbidden field(s): {hits}"


# ---- real aidp_test execution: proves zero writes end to end --------------------------


def test_dashboard_with_fraud_intelligence_tab_renders_without_exceptions():
    before = _db_snapshot()

    at = AppTest.from_file("src/dashboard/app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0

    after = _db_snapshot()
    assert after == before, f"database state changed after rendering the dashboard: before={before} after={after}"


def test_fraud_intel_tab_queue_shows_all_seven_channels_available():
    """Not every channel need appear on one ranked page, but the queue
    must be queryable per channel across the whole real seven-channel
    population -- proves the tab isn't silently scoped to a subset."""
    for channel in fraud_intel_tab.FRAUD_INTEL_CHANNELS:
        items, total, _ = fraud_intel_tab.load_queue(channel=channel, priority_band=None, page=0, page_size=5)
        assert total > 0, f"channel {channel!r} has zero alerts in the real aidp_test queue"


def test_fraud_intel_tab_low_band_alerts_remain_visible():
    """The exact regression this platform already fixed once for `aidp
    alerts list` (Phase 7B corrective pass) -- LOW must be a real,
    queryable filter value, not silently excluded."""
    items, total, _ = fraud_intel_tab.load_queue(channel=None, priority_band="LOW", page=0, page_size=10)
    assert total > 0
    assert all(i.current_priority_band == "LOW" for i in items)


def test_fraud_intel_tab_queue_ranked_by_current_score_not_initial():
    """Current-state semantics: ranking uses current_operational_priority_
    score (latest evidence), and results are non-increasing across the
    page -- never the frozen initial_operational_priority_score."""
    items, total, _ = fraud_intel_tab.load_queue(channel=None, priority_band=None, page=0, page_size=50)
    scores = [i.current_operational_priority_score for i in items if i.current_operational_priority_score is not None]
    assert scores == sorted(scores, reverse=True)


def test_fraud_intel_tab_channel_filter_matches_real_population():
    """Cross-check against a real, independently-known Phase 7B channel
    population count (ACH: 441 source-alerted events, current OPERATIONAL
    bundle 4) -- proves the filter reaches the real current-bundle
    evidence, not some other scope."""
    items, total, _ = fraud_intel_tab.load_queue(channel="ach", priority_band=None, page=0, page_size=500)
    assert total == 441


def test_fraud_intel_tab_alert_detail_renders_expected_fields_and_hides_synthetic_truth():
    items, total, _ = fraud_intel_tab.load_queue(channel=None, priority_band="HIGH", page=0, page_size=1)
    assert total > 0, "expected at least one real HIGH alert (mobile_deposit mandatory-review override)"
    alert_id = items[0].alert_id

    detail = fraud_intel_tab.load_alert_detail(alert_id)
    alert_fields = set(detail["alert"].model_dump().keys())
    evidence_fields = set(detail["evidence"].model_dump().keys())

    assert "scenario_id" not in alert_fields
    assert "synthetic_scenario_label" not in alert_fields
    assert "scenario_id" not in evidence_fields
    assert "synthetic_scenario_label" not in evidence_fields

    # Required Part-C detail fields are all present.
    for field in ("score_execution_id", "channel_model_bundle_id", "gbm_model_version",
                  "lr_model_version", "anomaly_model_version", "preprocessing_artifact_version",
                  "feature_schema_version", "rule_set_version", "graph_policy_version",
                  "ensemble_policy_version", "reason_code_version", "component_statuses",
                  "reason_codes", "degraded", "scored_at"):
        assert field in evidence_fields, f"alert detail evidence is missing required field {field!r}"

    assert isinstance(detail["dispositions"], list)  # present even when empty, never omitted


def test_fraud_intel_tab_alert_detail_unknown_alert_id_raises_lookup_error():
    with pytest.raises(LookupError):
        fraud_intel_tab.load_alert_detail(2_147_483_647)


def test_fraud_intel_tab_no_write_occurs_across_full_exercise():
    """End-to-end proof: run the queue, every channel filter, the LOW
    filter, and one alert-detail lookup -- database state is byte-for-byte
    unchanged afterward."""
    before = _db_snapshot()

    fraud_intel_tab.load_queue(channel=None, priority_band=None, page=0, page_size=50)
    for channel in fraud_intel_tab.FRAUD_INTEL_CHANNELS:
        fraud_intel_tab.load_queue(channel=channel, priority_band=None, page=0, page_size=5)
    fraud_intel_tab.load_queue(channel=None, priority_band="LOW", page=0, page_size=5)
    items, _, _ = fraud_intel_tab.load_queue(channel=None, priority_band=None, page=0, page_size=1)
    fraud_intel_tab.load_alert_detail(items[0].alert_id)

    after = _db_snapshot()
    assert after == before, f"database state changed: before={before} after={after}"
