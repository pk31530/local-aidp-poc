"""Cold-start (pre-first-scoring-run) candidate evaluation (Phase 7A
corrective pass). A channel's first-ever candidate bundle has no live
scored population to evaluate against -- `fraud_alerts`/`alert_evidence`/
`label_assessments` are only ever created by a real scoring run, which
itself requires an OPERATIONAL bundle (src.fraud_intel.scoring.dispatch).
This module reads the bundle's own IMMUTABLE, already-written training
evaluation report (`channel_model_bundles.evaluation_report_ref`,
written once by src.fraud_intel.models.training.train_channel_configured())
instead -- real, held-out test-split evidence, never fabricated, and
never confused with live operational evaluation. No database, MLflow,
Docker, or network access anywhere in this module -- it only parses and
validates a string already in hand.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from src.fraud_intel.models.bundle import ChannelModelBundleRecord

# Phase 7A decision (this corrective pass): POC-scale floors distinct from
# ChannelTrainingRunConfig's own min_train_rows/min_calib_rows/min_test_rows
# (which only require >=1 row of EACH class, not a specific count) --
# these are a stricter, cold-start-gate-specific quality floor.
MIN_CLASS_COUNT_FOR_GATE = 5

REQUIRED_SPLITS = ("train", "calibration", "test")


class ColdStartReportError(ValueError):
    """The candidate bundle's evaluation_report_ref is absent, malformed,
    or does not match the bundle row it is attached to -- cold-start
    evaluation must refuse (and promotion must remain blocked) rather
    than proceed on unverified evidence."""


class TrainingEvaluationReport(BaseModel):
    """The typed, validated content of `channel_model_bundles.
    evaluation_report_ref`, written once at training time and never
    modified afterward."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: str
    training_run_id: int
    dataset_version: str
    # Phase 7B Stage 3 corrective pass: dataset_version above remains the
    # training-derived supervised-population content hash (matches
    # channel_model_bundles.dataset_version -- unchanged, no migration),
    # aliased here under an unambiguous name; source_generation_run_id/
    # source_dataset_version are the DIFFERENT Stage-2 generation identity
    # every row of that population was actually generated under. Neither
    # has a corresponding bundle-row column, so neither is added to
    # _CROSS_CHECK_FIELDS below -- they are traceability-only.
    supervised_population_hash: str
    source_generation_run_id: str
    source_dataset_version: str
    feature_schema_version: str
    gbm_model_version: str
    lr_model_version: str
    anomaly_model_version: str
    preprocessing_artifact_version: str
    gbm_evaluation: dict[str, Any]
    lr_shadow_evaluation: dict[str, Any]
    eligibility_policy_version: str
    realized_split_fractions: dict[str, Any]
    split_class_counts: dict[str, dict[str, int]]
    gbm_mlflow_run_id: str
    lr_mlflow_run_id: str
    anomaly_mlflow_run_id: str
    anomaly_normalization: dict[str, Any]


# The bundle-row fields this report must agree with, exactly (report field
# name -> bundle attribute name).
_CROSS_CHECK_FIELDS: tuple[tuple[str, str], ...] = (
    ("channel", "channel"),
    ("training_run_id", "training_run_id"),
    ("dataset_version", "dataset_version"),
    ("feature_schema_version", "feature_schema_version"),
    ("gbm_model_version", "gbm_model_version"),
    ("lr_model_version", "lr_model_version"),
    ("anomaly_model_version", "anomaly_model_version"),
    ("preprocessing_artifact_version", "preprocessing_artifact_version"),
)


def load_and_validate_cold_start_report(bundle: ChannelModelBundleRecord) -> TrainingEvaluationReport:
    """Parses `bundle.evaluation_report_ref` and cross-checks every field
    in `_CROSS_CHECK_FIELDS` against the bundle row it came from --
    `bundle.bundle_id`/`bundle.bundle_version` themselves need no separate
    check, since the caller already selected this exact bundle row to get
    this exact report. Raises ColdStartReportError on absence, malformed
    JSON/schema, or any mismatch -- never proceeds on unverified
    evidence."""
    if not bundle.evaluation_report_ref:
        raise ColdStartReportError(
            f"bundle {bundle.bundle_id} (channel={bundle.channel!r}, version={bundle.bundle_version}) has no "
            "evaluation_report_ref -- cold-start evaluation requires the training-time held-out report"
        )
    try:
        raw = json.loads(bundle.evaluation_report_ref)
        report = TrainingEvaluationReport.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ColdStartReportError(
            f"bundle {bundle.bundle_id}'s evaluation_report_ref is malformed or missing required fields: {exc}"
        ) from exc

    mismatches: list[str] = []
    for report_field, bundle_field in _CROSS_CHECK_FIELDS:
        report_value = getattr(report, report_field)
        bundle_value = getattr(bundle, bundle_field)
        if report_value != bundle_value:
            mismatches.append(f"{report_field}: report={report_value!r} != bundle.{bundle_field}={bundle_value!r}")
    if mismatches:
        raise ColdStartReportError(
            f"bundle {bundle.bundle_id}'s evaluation_report_ref does not match its own bundle row: {'; '.join(mismatches)}"
        )

    missing_splits = [s for s in REQUIRED_SPLITS if s not in report.split_class_counts]
    if missing_splits:
        raise ColdStartReportError(
            f"bundle {bundle.bundle_id}'s evaluation_report_ref is missing split_class_counts for {missing_splits}"
        )

    return report


def evaluate_cold_start_promotion_gate(report: TrainingEvaluationReport) -> dict[str, Any]:
    """POC demonstration gate on synthetic, held-out training data --
    NOT a Citizens Bank production or regulatory threshold. Accuracy is
    never used as a criterion here."""
    reasons: list[str] = []
    for split_name in REQUIRED_SPLITS:
        counts = report.split_class_counts.get(split_name, {})
        fraud_count = counts.get("fraud", 0)
        legit_count = counts.get("legitimate", 0)
        if fraud_count < MIN_CLASS_COUNT_FOR_GATE:
            reasons.append(f"{split_name} split has only {fraud_count} fraud row(s), fewer than the required {MIN_CLASS_COUNT_FOR_GATE}")
        if legit_count < MIN_CLASS_COUNT_FOR_GATE:
            reasons.append(f"{split_name} split has only {legit_count} legitimate row(s), fewer than the required {MIN_CLASS_COUNT_FOR_GATE}")

    test_counts = report.split_class_counts.get("test", {})
    test_total = test_counts.get("fraud", 0) + test_counts.get("legitimate", 0)
    test_fraud_prevalence: Optional[float] = (test_counts.get("fraud", 0) / test_total) if test_total > 0 else None

    gbm_pr_auc = report.gbm_evaluation.get("pr_auc")
    if test_fraud_prevalence is None or gbm_pr_auc is None:
        reasons.append("test-split fraud prevalence or GBM pr_auc is not computable -- cannot compare against prevalence baseline")
    elif not (gbm_pr_auc > test_fraud_prevalence):
        reasons.append(f"GBM pr_auc ({gbm_pr_auc!r}) does not exceed the test-split fraud prevalence baseline ({test_fraud_prevalence!r})")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "test_fraud_prevalence": test_fraud_prevalence,
        "gbm_pr_auc": gbm_pr_auc,
        "split_class_counts": report.split_class_counts,
        "min_class_count_required": MIN_CLASS_COUNT_FOR_GATE,
        "poc_disclaimer": (
            "POC demonstration gate on synthetic, held-out training data -- not a Citizens Bank "
            "production or regulatory threshold. Accuracy is not used as a criterion."
        ),
    }
