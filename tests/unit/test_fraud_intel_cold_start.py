"""Phase 7B Stage 0: candidate-only cold-start evaluation, sourced from a
CANDIDATE bundle's own immutable, already-written training evaluation
report (channel_model_bundles.evaluation_report_ref). No database,
Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.fraud_intel.evaluation.cold_start import (
    MIN_CLASS_COUNT_FOR_GATE,
    ColdStartReportError,
    evaluate_cold_start_promotion_gate,
    load_and_validate_cold_start_report,
)
from src.fraud_intel.models.bundle import ChannelModelBundleRecord

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _report_dict(bundle: ChannelModelBundleRecord, **overrides) -> dict:
    report = dict(
        channel=bundle.channel, training_run_id=bundle.training_run_id, dataset_version=bundle.dataset_version,
        feature_schema_version=bundle.feature_schema_version, gbm_model_version=bundle.gbm_model_version,
        lr_model_version=bundle.lr_model_version, anomaly_model_version=bundle.anomaly_model_version,
        preprocessing_artifact_version=bundle.preprocessing_artifact_version,
        gbm_evaluation={"precision": 0.9, "recall": 0.8, "pr_auc": 0.9, "roc_auc": 0.95, "brier_score": 0.05},
        lr_shadow_evaluation={"precision": 0.7, "recall": 0.6, "pr_auc": 0.7, "roc_auc": 0.8, "brier_score": 0.1},
        eligibility_policy_version="v1", realized_split_fractions={"train": 0.6, "calibration": 0.2, "test": 0.2},
        split_class_counts={
            "train": {"fraud": 10, "legitimate": 10}, "calibration": {"fraud": 10, "legitimate": 10},
            "test": {"fraud": 10, "legitimate": 30},
        },
        gbm_mlflow_run_id="run-gbm-1", lr_mlflow_run_id="run-lr-1", anomaly_mlflow_run_id="run-anomaly-1",
        anomaly_normalization={"method": "none"},
    )
    report.update(overrides)
    return report


def _bundle(evaluation_report_ref=None, **overrides) -> ChannelModelBundleRecord:
    base = dict(
        bundle_id=9, channel="online_banking", bundle_version=4, status="CANDIDATE", created_at=T0,
        gbm_model_version="gbm-9", lr_model_version="lr-9", anomaly_model_version="anomaly-9",
        preprocessing_artifact_version="prep-9", feature_schema_version="fsv-9",
        training_run_id=42, dataset_version="dsv-9", evaluation_report_ref=evaluation_report_ref,
    )
    base.update(overrides)
    return ChannelModelBundleRecord(**base)


def _matching_bundle(**overrides) -> ChannelModelBundleRecord:
    bundle = _bundle(**overrides)
    return bundle.model_copy(update={"evaluation_report_ref": json.dumps(_report_dict(bundle))})


# ---- load_and_validate_cold_start_report ----------------------------------------------


def test_matching_report_loads_and_validates_cleanly():
    bundle = _matching_bundle()
    report = load_and_validate_cold_start_report(bundle)
    assert report.channel == "online_banking"
    assert report.training_run_id == 42
    assert report.dataset_version == "dsv-9"


def test_missing_evaluation_report_ref_raises():
    bundle = _bundle(evaluation_report_ref=None)
    with pytest.raises(ColdStartReportError):
        load_and_validate_cold_start_report(bundle)


def test_malformed_json_raises():
    bundle = _bundle(evaluation_report_ref="not valid json{{{")
    with pytest.raises(ColdStartReportError):
        load_and_validate_cold_start_report(bundle)


def test_report_missing_a_required_field_raises():
    bundle = _bundle()
    incomplete = _report_dict(bundle)
    del incomplete["gbm_evaluation"]
    bundle = bundle.model_copy(update={"evaluation_report_ref": json.dumps(incomplete)})
    with pytest.raises(ColdStartReportError):
        load_and_validate_cold_start_report(bundle)


@pytest.mark.parametrize(
    "field",
    ["channel", "training_run_id", "dataset_version", "feature_schema_version", "gbm_model_version", "lr_model_version", "anomaly_model_version", "preprocessing_artifact_version"],
)
def test_each_cross_checked_field_mismatch_raises(field):
    bundle = _bundle()
    mismatched_report = _report_dict(bundle)
    mismatched_report[field] = "SOMETHING-ELSE" if field != "training_run_id" else 999
    bundle = bundle.model_copy(update={"evaluation_report_ref": json.dumps(mismatched_report)})
    with pytest.raises(ColdStartReportError):
        load_and_validate_cold_start_report(bundle)


def test_missing_required_split_raises():
    bundle = _bundle()
    report = _report_dict(bundle)
    del report["split_class_counts"]["calibration"]
    bundle = bundle.model_copy(update={"evaluation_report_ref": json.dumps(report)})
    with pytest.raises(ColdStartReportError):
        load_and_validate_cold_start_report(bundle)


# ---- evaluate_cold_start_promotion_gate -------------------------------------------------


def test_gate_passes_when_pr_auc_exceeds_prevalence_baseline_and_classes_are_sufficient():
    bundle = _matching_bundle()
    report = load_and_validate_cold_start_report(bundle)
    gate = evaluate_cold_start_promotion_gate(report)
    assert gate["passed"] is True
    assert gate["reasons"] == []
    assert gate["test_fraud_prevalence"] == pytest.approx(10 / 40)
    assert gate["gbm_pr_auc"] == 0.9


def test_gate_fails_when_gbm_pr_auc_does_not_exceed_prevalence_baseline():
    bundle = _bundle()
    report_dict = _report_dict(bundle, gbm_evaluation={"precision": 0.5, "recall": 0.5, "pr_auc": 0.2, "roc_auc": 0.5, "brier_score": 0.2})
    bundle = bundle.model_copy(update={"evaluation_report_ref": json.dumps(report_dict)})
    report = load_and_validate_cold_start_report(bundle)

    gate = evaluate_cold_start_promotion_gate(report)
    assert gate["passed"] is False
    assert any("does not exceed" in r for r in gate["reasons"])


def test_gate_fails_when_a_split_has_too_few_rows_of_a_class():
    bundle = _bundle()
    report_dict = _report_dict(bundle)
    report_dict["split_class_counts"]["train"] = {"fraud": MIN_CLASS_COUNT_FOR_GATE - 1, "legitimate": 10}
    bundle = bundle.model_copy(update={"evaluation_report_ref": json.dumps(report_dict)})
    report = load_and_validate_cold_start_report(bundle)

    gate = evaluate_cold_start_promotion_gate(report)
    assert gate["passed"] is False
    assert any("fraud row(s)" in r for r in gate["reasons"])


def test_gate_never_uses_accuracy_as_a_criterion():
    """AST-based (Phase 7A convention): evaluate_cold_start_promotion_gate's
    own source never reads an 'accuracy' key/attribute anywhere."""
    import ast
    import inspect

    from src.fraud_intel.evaluation import cold_start as cold_start_module

    tree = ast.parse(inspect.getsource(cold_start_module.evaluate_cold_start_promotion_gate))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "accuracy":
            raise AssertionError("evaluate_cold_start_promotion_gate must never key off 'accuracy'")


def test_gate_disclaimer_present_and_load_function_never_promotes():
    """AST-based: load_and_validate_cold_start_report and
    evaluate_cold_start_promotion_gate never call anything named
    promote_bundle -- cold-start evaluation is read-only evidence, never
    a promotion action."""
    import ast
    import inspect

    for fn in (load_and_validate_cold_start_report, evaluate_cold_start_promotion_gate):
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                assert name != "promote_bundle"

    bundle = _matching_bundle()
    report = load_and_validate_cold_start_report(bundle)
    gate = evaluate_cold_start_promotion_gate(report)
    assert "poc_disclaimer" in gate
    assert "not a Citizens Bank" in gate["poc_disclaimer"]
