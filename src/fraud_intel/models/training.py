"""Shared, per-channel training framework -- chronological split with a
purge gap, training-window-only preprocessing, calibrated GBM primary, LR
shadow challenger, unsupervised anomaly detector (Phase 5, guide section
14), and candidate channel-bundle registration. Reference channel
(Online/Mobile Banking) only -- the other six channels are Phase 7A.

No real database, MLflow server, or large-scale training is contacted by
this module's own code -- every unit test replaces `mlflow`,
`_train_gbm`/`_train_lr`/`_fit_anomaly`, and the bundle store with fakes,
exactly like tests/unit/test_train.py's existing v1.1 pattern.
`channel_events`/`source_alerts`/`synthetic_labels` are required
parameters, not loaded from a database, because no real Postgres-backed
loader exists until Phase 6/7B applies migration 003.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any, Optional, Sequence

import mlflow
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.common.config import get_settings
from src.common.mlflow_setup import configure_mlflow
from src.common.splits import assign_chronological_split, realized_split_fractions
from src.control_plane.provenance import get_git_sha
from src.control_plane.runs import RunLifecycle
from src.fraud_intel.config import ChannelTrainingRunConfig
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.features.channels.online_banking import (
    ONLINE_BANKING_FEATURE_COLUMNS,
    ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
    compute_online_banking_features,
)
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.models.anomaly import fit_anomaly_model
from src.fraud_intel.models.bundle import ChannelModelBundleStore, create_default_bundle_store
from src.fraud_intel.models.calibration import SplitManifest, fit_calibrator
from src.fraud_intel.models.preprocessing import ChannelPreprocessor

EXPERIMENT_NAME = "fraud-intel-online-banking"

# Phase 4 interim-only label-eligibility policy (guide sections 8, 11, 19).
# The guide's REAL, versioned eligibility policy requires label_assessments
# (section 18), which does not exist until Phase 6's migration 004. This
# policy version string is recorded on every bundle's evaluation_report_ref
# specifically so it is never confused with Phase 6's real policy later.
PHASE4_INTERIM_ELIGIBILITY_POLICY_VERSION = "synthetic_label_eligibility_v1"


class InsufficientTrainingDataError(ValueError):
    """A chronological split partition has too few rows, or fewer than two
    label classes, for GBM/LR/calibration to fit meaningfully. Always
    raised BEFORE any model fitting, MLflow call, or candidate-bundle
    write (guide section 12's Phase 4 requirement)."""


def _phase4_interim_training_eligible(label: SyntheticGroundTruthLabel) -> bool:
    """Phase 4 interim eligibility: true iff the label is a genuine
    synthetic-generator label. This function's only possible input type is
    SyntheticGroundTruthLabel -- it cannot read analyst_disposition,
    outcome_status, maturity_status, or any Phase 6 label_assessments
    field; those simply are not fields on this type. Phase 6 supersedes
    this policy entirely once label_assessments exists for real,
    analyst-derived labels."""
    return label.label_source == "SYNTHETIC_GENERATOR"


def _build_supervised_population(
    *,
    channel_events: Sequence[FraudEvent],
    source_alerts: Sequence[SourceAlertContext],
    synthetic_labels: Sequence[SyntheticGroundTruthLabel],
) -> list[dict[str, Any]]:
    """Pure, in-memory. No database, MLflow, or model-fitting code here --
    only the guide section 11 join: source_alerts -> channel_events ->
    Phase 2 feature vectors -> Phase-4-interim-eligible synthetic labels.

    Returns one dict per supervised training row: {event_id (str),
    event_timestamp, label (bool), **ordered ONLINE_BANKING_FEATURE_COLUMNS}.
    Non-alerted events are still used as history via
    FeatureComputationContext (both historical_events and
    source_alert_history are built from the FULL channel_events/
    source_alerts pools) -- they simply never become a row themselves,
    since the outer loop iterates only event_ids present in source_alerts.
    """
    events_by_id = {event.event_id: event for event in channel_events}
    label_by_event_id = {label.event_id: label for label in synthetic_labels}
    alerted_event_ids = {alert.event_id for alert in source_alerts}

    rows: list[dict[str, Any]] = []
    for event_id in alerted_event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue  # a source alert referencing an unknown event is excluded, never crashes
        if event.channel != "online_banking":
            continue  # reference channel only, Phase 4

        label = label_by_event_id.get(event_id)
        if label is None:
            continue  # no label -> cannot supervise this row
        if not _phase4_interim_training_eligible(label):
            continue

        history = tuple(
            other
            for other in channel_events
            if other.event_id != event.event_id and other.event_timestamp < event.event_timestamp
        )
        prior_alerts = tuple(
            alert
            for alert in source_alerts
            if alert.event_id != event.event_id and alert.source_alert_created_at < event.event_timestamp
        )

        ctx = FeatureComputationContext(
            current_event=event,
            historical_events=history,
            source_alert_history=prior_alerts,
            as_of_time=event.event_timestamp,
        )
        features = compute_online_banking_features(ctx)

        rows.append(
            {
                "event_id": str(event.event_id),
                "event_timestamp": event.event_timestamp,
                "label": bool(label.synthetic_scenario_label),
                **{column: features[column] for column in ONLINE_BANKING_FEATURE_COLUMNS},
            }
        )

    rows.sort(key=lambda row: (row["event_timestamp"], row["event_id"]))
    return rows


def _dataset_version(population_rows: Sequence[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted((row["event_id"], bool(row["label"])) for row in population_rows),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _preprocessing_artifact_version(train_rows: Sequence[dict[str, Any]]) -> str:
    canonical = json.dumps(train_rows, sort_keys=True, separators=(",", ":"), default=str)
    return "pp-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _anomaly_artifact_version(train_rows: Sequence[dict[str, Any]]) -> str:
    canonical = json.dumps(train_rows, sort_keys=True, separators=(",", ":"), default=str)
    return "anom-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _validate_partition(df: pl.DataFrame, name: str, *, min_rows: int) -> None:
    if df.height < min_rows:
        raise InsufficientTrainingDataError(
            f"{name} partition has {df.height} row(s), fewer than the required minimum {min_rows}"
        )
    classes = set(df["label"].to_list())
    if len(classes) < 2:
        raise InsufficientTrainingDataError(
            f"{name} partition contains only class(es) {sorted(classes)} -- both classes are required"
        )


def _rows_as_feature_dicts(part_df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{column: record[column] for column in ONLINE_BANKING_FEATURE_COLUMNS} for record in part_df.to_dicts()]


def _train_gbm(X_train, y_train, config: ChannelTrainingRunConfig):
    import xgboost as xgb

    params: dict[str, Any] = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        eval_metric="aucpr",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    if config.handle_class_imbalance:
        n_pos = sum(1 for y in y_train if y)
        n_neg = len(y_train) - n_pos
        params["scale_pos_weight"] = float(n_neg / max(n_pos, 1))
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def _train_lr(X_train, y_train, config: ChannelTrainingRunConfig):
    params: dict[str, Any] = dict(random_state=config.random_seed, max_iter=1000)
    if config.handle_class_imbalance:
        params["class_weight"] = "balanced"
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model, params


def _fit_anomaly(X_train, config: ChannelTrainingRunConfig, anomaly_artifact_version: str):
    """Thin, monkeypatch-friendly indirection to
    src.fraud_intel.models.anomaly.fit_anomaly_model -- same pattern as
    _train_gbm/_train_lr. Label-agnostic (no y_train parameter at all):
    IsolationForest fit on the SAME source-alerted training-window matrix
    GBM/LR train on (guide section 14, Phase 5 decision 2) -- this detects
    what is unusual WITHIN the alerted population, not across all bank
    transactions."""
    return fit_anomaly_model(X_train, random_seed=config.random_seed, anomaly_artifact_version=anomaly_artifact_version)


def _evaluate(y_true, y_pred, y_proba) -> dict:
    """Same multi-metric shape as src.ml.train._evaluate -- accuracy is
    computed but explicitly marked supplementary, never the headline."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": float(sum(int(p == t) for p, t in zip(y_pred, y_true)) / len(y_true)),  # supplementary only
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _split_manifest_from(split_df: pl.DataFrame, all_ids: Sequence[str]) -> SplitManifest:
    train_ids = frozenset(split_df.filter(pl.col("split") == "train")["event_id"].to_list())
    calibration_ids = frozenset(split_df.filter(pl.col("split") == "calibration")["event_id"].to_list())
    test_ids = frozenset(split_df.filter(pl.col("split") == "test")["event_id"].to_list())
    purge_ids = frozenset(all_ids) - train_ids - calibration_ids - test_ids
    return SplitManifest(train_ids=train_ids, calibration_ids=calibration_ids, purge_ids=purge_ids, test_ids=test_ids)


def train_channel_configured(
    config: ChannelTrainingRunConfig,
    *,
    trigger_source: str = "legacy",
    channel_events: Sequence[FraudEvent],
    source_alerts: Sequence[SourceAlertContext],
    synthetic_labels: Sequence[SyntheticGroundTruthLabel],
    bundle_store: Optional[ChannelModelBundleStore] = None,
    rule_set_version: Optional[str] = None,
    graph_policy_version: Optional[str] = None,
    ensemble_policy_version: Optional[str] = None,
    reason_code_version: Optional[str] = None,
) -> dict:
    """Phase 5 decision 3: rule_set_version/graph_policy_version/
    ensemble_policy_version/reason_code_version are optional, pass-through
    metadata recording which versioned SCORING-time policies this bundle
    was registered against -- training itself never loads or evaluates
    any of them. Omitting them (as every Phase 4-era caller still does)
    reproduces Phase 4's original, deliberately-incomplete bundle exactly;
    passing all four (as Phase 5's own tests do) produces a bundle with
    every REQUIRED_OPERATIONAL_COMPONENTS field populated."""
    lifecycle = RunLifecycle()
    run = lifecycle.begin(
        "train",
        trigger_source=trigger_source,
        git_sha=get_git_sha(),
        config_snapshot=config.redacted_snapshot(),
        config_hash=config.config_hash(),
    )

    dataset_version: Optional[str] = None

    try:
        if config.channel != "online_banking":
            raise ValueError(f"Phase 4 supports only the online_banking reference channel, got {config.channel!r}")

        store = bundle_store if bundle_store is not None else create_default_bundle_store()

        population_rows = _build_supervised_population(
            channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=synthetic_labels
        )
        if not population_rows:
            raise InsufficientTrainingDataError(
                "no source-alerted, eligible online_banking rows available to build a supervised population"
            )

        dataset_version = _dataset_version(population_rows)

        df = pl.DataFrame(
            [
                {
                    "event_id": row["event_id"],
                    "event_timestamp": row["event_timestamp"],
                    "label": row["label"],
                    **{column: row[column] for column in ONLINE_BANKING_FEATURE_COLUMNS},
                }
                for row in population_rows
            ]
        )

        purge_gap = timedelta(seconds=config.purge_gap_seconds)
        split_df = assign_chronological_split(
            df,
            timestamp_col="event_timestamp",
            id_col="event_id",
            train_frac=config.train_frac,
            calib_frac=config.calib_frac,
            test_frac=config.test_frac,
            purge_gap=purge_gap,
        )
        realized_fractions = realized_split_fractions(split_df)

        train_df = split_df.filter(pl.col("split") == "train")
        calib_df = split_df.filter(pl.col("split") == "calibration")
        test_df = split_df.filter(pl.col("split") == "test")

        _validate_partition(train_df, "train", min_rows=config.min_train_rows)
        _validate_partition(calib_df, "calibration", min_rows=config.min_calib_rows)
        _validate_partition(test_df, "test", min_rows=config.min_test_rows)

        manifest = _split_manifest_from(split_df, all_ids=df["event_id"].to_list())

        train_rows = _rows_as_feature_dicts(train_df)
        preprocessor = ChannelPreprocessor.fit(
            train_rows,
            feature_columns=ONLINE_BANKING_FEATURE_COLUMNS,
            feature_schema_version=ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
            preprocessing_artifact_version=_preprocessing_artifact_version(train_rows),
        )

        X_train = preprocessor.transform(train_rows)
        y_train = train_df["label"].to_list()
        X_calib = preprocessor.transform(_rows_as_feature_dicts(calib_df))
        y_calib = calib_df["label"].to_list()
        X_test = preprocessor.transform(_rows_as_feature_dicts(test_df))
        y_test = test_df["label"].to_list()

        gbm_model, gbm_params = _train_gbm(X_train, y_train, config)
        lr_model, lr_params = _train_lr(X_train, y_train, config)

        gbm_calib_raw_proba = [p[1] for p in gbm_model.predict_proba(X_calib)]
        calibrator = fit_calibrator(
            row_ids=calib_df["event_id"].to_list(),
            raw_probabilities=gbm_calib_raw_proba,
            labels=y_calib,
            manifest=manifest,
        )

        gbm_test_raw_proba = [p[1] for p in gbm_model.predict_proba(X_test)]
        gbm_test_calibrated_proba = calibrator.predict(gbm_test_raw_proba)
        gbm_test_pred = [1 if p >= 0.5 else 0 for p in gbm_test_calibrated_proba]
        gbm_eval = _evaluate(y_test, gbm_test_pred, gbm_test_calibrated_proba)

        lr_test_proba = [p[1] for p in lr_model.predict_proba(X_test)]
        lr_test_pred = [1 if p >= 0.5 else 0 for p in lr_test_proba]
        lr_eval = _evaluate(y_test, lr_test_pred, lr_test_proba)  # shadow only -- logged, never scored on

        # Anomaly (guide section 14, Phase 5): unsupervised, X_train only --
        # no y_train argument anywhere in this call.
        anomaly_artifact_version = _anomaly_artifact_version(train_rows)
        anomaly_model, anomaly_normalization = _fit_anomaly(X_train, config, anomaly_artifact_version)

        configure_mlflow()
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"online_banking-gbm-{dataset_version}") as gbm_run:
            mlflow.log_params(gbm_params)
            mlflow.log_param("feature_schema_version", ONLINE_BANKING_FEATURE_SCHEMA_VERSION)
            for key, value in gbm_eval.items():
                if key != "confusion_matrix":
                    mlflow.log_metric(key, value)
            signature = mlflow.models.infer_signature(X_train, gbm_model.predict_proba(X_train))
            gbm_model_info = mlflow.xgboost.log_model(
                gbm_model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"{get_settings().mlflow_model_name}-fraud-intel-online-banking-gbm",
            )
            gbm_mlflow_run_id = gbm_run.info.run_id
            # Phase 6 corrective pass: the preprocessor was previously only
            # ever recorded as a version STRING (preprocessing_artifact_version)
            # -- the actual fitted object was never persisted anywhere
            # reloadable. Logging its deterministic JSON form (never pickle,
            # same ChannelPreprocessor.to_json_dict()/from_json_dict()
            # round-trip already used elsewhere) here, under the gbm run,
            # is what makes real scoring-time reload
            # (src.fraud_intel.scoring.dispatch) possible at all.
            mlflow.log_dict(preprocessor.to_json_dict(), "preprocessor.json")
            gbm_model_version = gbm_model_info.registered_model_version
        # No mlflow.tracking.MlflowClient().set_registered_model_alias(...) call
        # anywhere in this function -- guide section 22's deliberate difference
        # from v1.1's unconditional champion-alias behavior.

        with mlflow.start_run(run_name=f"online_banking-lr-shadow-{dataset_version}") as lr_run:
            mlflow.log_params(lr_params)
            for key, value in lr_eval.items():
                if key != "confusion_matrix":
                    mlflow.log_metric(f"shadow_{key}", value)
            lr_signature = mlflow.models.infer_signature(X_train, lr_model.predict_proba(X_train))
            lr_model_info = mlflow.sklearn.log_model(
                lr_model,
                artifact_path="model",
                signature=lr_signature,
                registered_model_name=f"{get_settings().mlflow_model_name}-fraud-intel-online-banking-lr-shadow",
            )
            lr_mlflow_run_id = lr_run.info.run_id
            lr_model_version = lr_model_info.registered_model_version

        with mlflow.start_run(run_name=f"online_banking-anomaly-{dataset_version}") as anomaly_run:
            mlflow.log_param("anomaly_artifact_version", anomaly_artifact_version)
            mlflow.log_param("contamination", "auto")
            anomaly_model_info = mlflow.sklearn.log_model(
                anomaly_model,
                artifact_path="model",
                registered_model_name=f"{get_settings().mlflow_model_name}-fraud-intel-online-banking-anomaly",
            )
            anomaly_mlflow_run_id = anomaly_run.info.run_id
            anomaly_model_version = anomaly_model_info.registered_model_version
            # Same rationale as preprocessor.json above -- makes real
            # scoring-time reload of the fitted normalization bounds
            # possible (src.fraud_intel.scoring.dispatch).
            mlflow.log_dict(anomaly_normalization.to_json_dict(), "anomaly_normalization.json")

        evaluation_report_ref = json.dumps(
            {
                "gbm": {k: v for k, v in gbm_eval.items() if k != "confusion_matrix"},
                "lr_shadow": {k: v for k, v in lr_eval.items() if k != "confusion_matrix"},
                "eligibility_policy_version": PHASE4_INTERIM_ELIGIBILITY_POLICY_VERSION,
                "realized_split_fractions": realized_fractions,
                "gbm_mlflow_run_id": gbm_mlflow_run_id,
                "lr_mlflow_run_id": lr_mlflow_run_id,
                "anomaly_mlflow_run_id": anomaly_mlflow_run_id,
                "anomaly_normalization": anomaly_normalization.to_json_dict(),
            },
            sort_keys=True,
        )

        bundle = store.register_candidate(
            channel="online_banking",
            gbm_model_version=str(gbm_model_version),
            lr_model_version=str(lr_model_version),
            anomaly_model_version=str(anomaly_model_version),
            preprocessing_artifact_version=preprocessor.preprocessing_artifact_version,
            feature_schema_version=ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
            rule_set_version=rule_set_version,
            graph_policy_version=graph_policy_version,
            ensemble_policy_version=ensemble_policy_version,
            reason_code_version=reason_code_version,
            training_run_id=run.run_id,
            dataset_version=dataset_version,
            evaluation_report_ref=evaluation_report_ref,
        )

    except Exception as exc:
        lifecycle.fail_from_exception(run.run_id, exc, dataset_version=dataset_version)
        raise

    lifecycle.succeed(
        run.run_id,
        dataset_version=dataset_version,
        model_version=str(gbm_model_version),
        artifacts={
            "lr_model_version": str(lr_model_version),
            "anomaly_model_version": str(anomaly_model_version),
            "preprocessing_artifact_version": preprocessor.preprocessing_artifact_version,
            "feature_schema_version": ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
            "bundle_id": bundle.bundle_id,
            "bundle_version": bundle.bundle_version,
            "realized_split_fractions": realized_fractions,
        },
    )

    return {
        "run_id": run.run_id,
        "bundle_id": bundle.bundle_id,
        "bundle_version": bundle.bundle_version,
        "gbm_model_version": str(gbm_model_version),
        "lr_model_version": str(lr_model_version),
        "anomaly_model_version": str(anomaly_model_version),
        "dataset_version": dataset_version,
        "gbm_evaluation": gbm_eval,
        "lr_shadow_evaluation": lr_eval,
        "realized_split_fractions": realized_fractions,
    }
