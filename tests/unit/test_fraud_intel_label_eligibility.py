"""Phase 6: real, versioned label-eligibility policy. No database, Docker,
or network access anywhere in this file.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.control_plane.runs import RunLifecycle, RunRecord
from src.fraud_intel.labels.eligibility import (
    ANALYST_MATURITY_WINDOW_DAYS,
    LABEL_ELIGIBILITY_POLICY_VERSION,
    MATURE_BUT_UNRESOLVED,
    MATURITY_WINDOW_ELAPSED,
    MATURITY_WINDOW_NOT_ELAPSED,
    SYNTHETIC_IMMEDIATE_MATURITY,
    AlertLabelBasis,
    LabelAssessmentInputError,
    _FakeLabelAssessmentStore,
    assess_channel_labels,
    assess_label,
    resolve_label_from_disposition,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeRunStore:
    """Same in-memory contract as tests/unit/test_fraud_intel_scoring_dispatch.py's own fixture."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id, "trigger_source": None, "git_sha": None, "config_snapshot": None,
            "config_hash": None, "dataset_version": None, "model_version": None, "error_type": None,
            "error_message": None, "started_at": datetime.now(timezone.utc), "heartbeat_at": None,
            "completed_at": None, **row,
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


def _lifecycle() -> tuple[RunLifecycle, _FakeRunStore]:
    store = _FakeRunStore()
    return RunLifecycle(store=store), store


def test_policy_version_is_recorded_and_stable():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0)
    assert result["policy_version"] == LABEL_ELIGIBILITY_POLICY_VERSION == "v1"


# ---- synthetic: immediate maturity ------------------------------------------------


def test_synthetic_label_is_immediately_mature():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0)
    assert result["maturity_status"] == "MATURE"
    assert result["eligibility_reason_code"] == SYNTHETIC_IMMEDIATE_MATURITY
    assert result["basis_timestamp"] == T0
    assert result["maturity_due_at"] == T0


def test_synthetic_assessment_requires_source_disposition_id_to_be_none():
    with pytest.raises(LabelAssessmentInputError, match="synthetic"):
        assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=42, resolved_label="RESOLVED_FRAUD", now=T0)


def test_synthetic_resolved_fraud_is_eligible():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0)
    assert result["eligibility_result"] is True


def test_synthetic_resolved_legitimate_is_eligible():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_LEGITIMATE", now=T0)
    assert result["eligibility_result"] is True


# ---- maturity alone is insufficient -- must also be resolved -----------------------


def test_mature_but_unresolved_label_is_ineligible():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="UNRESOLVED", now=T0)
    assert result["maturity_status"] == "MATURE"
    assert result["eligibility_result"] is False
    assert result["eligibility_reason_code"] == MATURE_BUT_UNRESOLVED


def test_mature_but_missing_label_is_ineligible():
    result = assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label=None, now=T0)
    assert result["eligibility_result"] is False
    assert result["eligibility_reason_code"] == MATURE_BUT_UNRESOLVED


# ---- analyst-derived: maturity from basis_timestamp, not event_timestamp -----------


def test_analyst_derived_requires_source_disposition_id():
    with pytest.raises(LabelAssessmentInputError, match="ANALYST_DISPOSITION"):
        assess_label(alert_id=1, label_source="ANALYST_DISPOSITION", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0)


def test_analyst_derived_immature_before_window_elapses():
    disposed_at = T0
    result = assess_label(
        alert_id=1, label_source="ANALYST_DISPOSITION", basis_timestamp=disposed_at, source_disposition_id=99,
        resolved_label="RESOLVED_FRAUD", now=disposed_at + timedelta(days=1),
    )
    assert result["maturity_status"] == "IMMATURE"
    assert result["eligibility_result"] is False
    assert result["eligibility_reason_code"] == MATURITY_WINDOW_NOT_ELAPSED
    assert result["maturity_due_at"] == disposed_at + timedelta(days=ANALYST_MATURITY_WINDOW_DAYS)


def test_analyst_derived_mature_after_window_elapses():
    disposed_at = T0
    result = assess_label(
        alert_id=1, label_source="ANALYST_DISPOSITION", basis_timestamp=disposed_at, source_disposition_id=99,
        resolved_label="RESOLVED_FRAUD", now=disposed_at + timedelta(days=ANALYST_MATURITY_WINDOW_DAYS, hours=1),
    )
    assert result["maturity_status"] == "MATURE"
    assert result["eligibility_result"] is True
    assert result["eligibility_reason_code"] == MATURITY_WINDOW_ELAPSED


def test_analyst_derived_needs_more_info_stays_unresolved_and_ineligible():
    disposed_at = T0
    resolved = resolve_label_from_disposition("NEEDS_MORE_INFO")
    assert resolved == "UNRESOLVED"
    result = assess_label(
        alert_id=1, label_source="ANALYST_DISPOSITION", basis_timestamp=disposed_at, source_disposition_id=99,
        resolved_label=resolved, now=disposed_at + timedelta(days=ANALYST_MATURITY_WINDOW_DAYS, hours=1),
    )
    assert result["eligibility_result"] is False
    assert result["eligibility_reason_code"] == MATURE_BUT_UNRESOLVED


def test_analyst_derived_escalated_stays_unresolved_and_ineligible():
    assert resolve_label_from_disposition("ESCALATED") == "UNRESOLVED"


def test_confirmed_dispositions_resolve_correctly():
    assert resolve_label_from_disposition("CONFIRMED_FRAUD") == "RESOLVED_FRAUD"
    assert resolve_label_from_disposition("CONFIRMED_LEGITIMATE") == "RESOLVED_LEGITIMATE"


def test_maturity_measured_from_basis_timestamp_not_event_timestamp():
    """The whole point of Phase 6 decision 7: an analyst disposition
    recorded long after the underlying event must still measure maturity
    from ITS OWN basis_timestamp (disposed_at), not from some much-earlier
    event_timestamp."""
    event_timestamp = T0 - timedelta(days=365)  # event happened a year ago
    disposed_at = T0  # but the analyst only disposed it just now
    result = assess_label(
        alert_id=1, label_source="ANALYST_DISPOSITION", basis_timestamp=disposed_at, source_disposition_id=99,
        resolved_label="RESOLVED_FRAUD", now=disposed_at + timedelta(days=1),
    )
    # still immature -- despite the event itself being a year old, because
    # maturity counts from disposed_at, not event_timestamp
    assert result["maturity_status"] == "IMMATURE"
    assert result["basis_timestamp"] == disposed_at
    assert result["basis_timestamp"] != event_timestamp


# ---- append-only store / latest-assessment query -------------------------------------


def test_append_assessment_never_updates_a_prior_row():
    store = _FakeLabelAssessmentStore()
    store.append_assessment(**assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="UNRESOLVED", now=T0))
    store.append_assessment(**assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0))
    assert len(store.rows) == 2
    assert store.rows[0].eligibility_result is False  # first row untouched
    assert store.rows[1].eligibility_result is True


def test_get_latest_assessment_returns_most_recent_older_rows_still_queryable():
    store = _FakeLabelAssessmentStore()
    first = store.append_assessment(**assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="UNRESOLVED", now=T0))
    second = store.append_assessment(**assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0))
    latest = store.get_latest_assessment(1)
    assert latest.assessment_id == second.assessment_id
    # the older row is still directly queryable in store.rows
    assert first in store.rows


def test_get_latest_assessment_for_unknown_alert_returns_none():
    store = _FakeLabelAssessmentStore()
    assert store.get_latest_assessment(999) is None


def test_get_latest_assessment_ignores_other_alerts():
    store = _FakeLabelAssessmentStore()
    store.append_assessment(**assess_label(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0))
    store.append_assessment(**assess_label(alert_id=2, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_LEGITIMATE", now=T0))
    latest_for_1 = store.get_latest_assessment(1)
    assert latest_for_1.alert_id == 1
    assert latest_for_1.resolved_label == "RESOLVED_FRAUD"


# ---- assess_channel_labels: explicit label-eligibility command orchestration (Phase 7B Stage 0) ---


def _bases() -> list[AlertLabelBasis]:
    return [
        AlertLabelBasis(alert_id=1, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD"),
        AlertLabelBasis(alert_id=2, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_LEGITIMATE"),
        AlertLabelBasis(alert_id=3, label_source="ANALYST_DISPOSITION", basis_timestamp=T0, source_disposition_id=7, resolved_label="UNRESOLVED"),
    ]


def test_assess_channel_labels_appends_through_its_own_lifecycle_run():
    lifecycle, run_store = _lifecycle()
    store = _FakeLabelAssessmentStore()

    summary = assess_channel_labels(channel="online_banking", lifecycle=lifecycle, load_label_bases=lambda channel: _bases(), store=store, now=T0)

    assert summary["channel"] == "online_banking"
    assert summary["alerts_considered"] == 3
    assert summary["assessments_appended"] == 3
    assert len(store.rows) == 3
    run = run_store.get(summary["run_id"])
    assert run.pipeline_name == "label_eligibility"
    assert run.status == "SUCCESS"


def test_assess_channel_labels_counts_mature_immature_eligible_unresolved():
    lifecycle, _ = _lifecycle()
    store = _FakeLabelAssessmentStore()

    summary = assess_channel_labels(channel="online_banking", lifecycle=lifecycle, load_label_bases=lambda channel: _bases(), store=store, now=T0)

    # alert 1: synthetic, resolved fraud -> MATURE + eligible
    # alert 2: synthetic, resolved legitimate -> MATURE + eligible
    # alert 3: analyst-derived, basis_timestamp == now -> IMMATURE (window has not elapsed) + unresolved (resolved_label carried through as UNRESOLVED regardless of maturity)
    assert summary["mature_count"] == 2
    assert summary["immature_count"] == 1
    assert summary["eligible_count"] == 2
    assert summary["unresolved_count"] == 1


def test_assess_channel_labels_repeated_runs_preserve_history():
    lifecycle, _ = _lifecycle()
    store = _FakeLabelAssessmentStore()

    first = assess_channel_labels(channel="online_banking", lifecycle=lifecycle, load_label_bases=lambda channel: _bases(), store=store, now=T0)
    second = assess_channel_labels(channel="online_banking", lifecycle=lifecycle, load_label_bases=lambda channel: _bases(), store=store, now=T0)

    assert first["run_id"] != second["run_id"]
    # every alert now has TWO assessment rows -- nothing was overwritten
    assert len(store.rows) == 6
    for alert_id in (1, 2, 3):
        matching = [r for r in store.rows if r.alert_id == alert_id]
        assert len(matching) == 2


def test_assess_channel_labels_structural_load_failure_fails_the_run_and_reraises():
    lifecycle, run_store = _lifecycle()
    store = _FakeLabelAssessmentStore()

    def _raising_loader(channel):
        raise RuntimeError("simulated structural failure")

    with pytest.raises(RuntimeError):
        assess_channel_labels(channel="online_banking", lifecycle=lifecycle, load_label_bases=_raising_loader, store=store, now=T0)

    assert len(store.rows) == 0
    (run,) = run_store.rows.values()
    assert run["status"] == "FAILED"


def test_assess_channel_labels_never_trains_or_promotes():
    """AST-based (Phase 7A convention): assess_channel_labels' own source
    never calls anything named train_channel_configured or promote_bundle."""
    import ast
    import inspect

    from src.fraud_intel.labels import eligibility as eligibility_module

    tree = ast.parse(inspect.getsource(eligibility_module.assess_channel_labels))
    forbidden = {"train_channel_configured", "promote_bundle"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            assert name not in forbidden, f"assess_channel_labels must never call {name}"
