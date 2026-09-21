"""Shared, per-channel training framework -- chronological split with a
purge gap, training-window-only preprocessing, calibrated GBM primary, LR
shadow challenger, unsupervised anomaly detector (Phase 5, guide section
14), and candidate channel-bundle registration. Phase 7A: routes through
src.fraud_intel.registry.get_channel_adapter(config.channel) for every
channel -- ONE shared training path, not a per-channel copy.

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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.mlflow_setup import configure_mlflow
from src.common.splits import assign_chronological_split, realized_split_fractions
from src.control_plane.provenance import get_git_sha
from src.control_plane.runs import RunLifecycle
from src.fraud_intel.config import ChannelTrainingRunConfig
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.features.history import select_customer_historical_events, select_customer_historical_source_alerts
from src.fraud_intel.models.anomaly import fit_anomaly_model
from src.fraud_intel.models.bundle import ChannelModelBundleStore, create_default_bundle_store
from src.fraud_intel.models.calibration import SplitManifest, fit_calibrator
from src.fraud_intel.models.mlflow_naming import registered_model_name
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.registry import ChannelAdapter, get_channel_adapter

EXPERIMENT_NAME_PREFIX = "fraud-intel"


def _experiment_name(channel: str) -> str:
    return f"{EXPERIMENT_NAME_PREFIX}-{channel}"


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
    adapter: ChannelAdapter,
    channel_events: Sequence[FraudEvent],
    source_alerts: Sequence[SourceAlertContext],
    synthetic_labels: Sequence[SyntheticGroundTruthLabel],
    history_events: Optional[Sequence[FraudEvent]] = None,
    history_source_alerts: Optional[Sequence[SourceAlertContext]] = None,
) -> list[dict[str, Any]]:
    """Pure, in-memory. No database, MLflow, or model-fitting code here --
    only the guide section 11 join: source_alerts -> channel_events ->
    the registered channel adapter's own feature vectors ->
    Phase-4-interim-eligible synthetic labels.

    Returns one dict per supervised training row: {event_id (str),
    event_timestamp, label (bool), **ordered adapter.feature_columns}.

    Phase 7B corrective pass: TARGET selection remains exactly as before
    -- restricted to `channel_events`/`source_alerts` (which callers scope
    to one channel and one generation_run_id, via
    src.fraud_intel.cli_data_access.load_channel_population()) -- guide
    section 15's cross-channel history design never applies to WHICH
    events become supervised rows, only to what history each row's own
    features are computed FROM. That history is now built via the shared
    src.fraud_intel.features.history.select_customer_historical_events()/
    select_customer_historical_source_alerts() -- the same functions
    src.fraud_intel.scoring.dispatch._PostgresScoringDataAccess.
    list_pending() and src.fraud_intel.cli_data_access.
    load_resolved_alert_scoring_contexts() use for real scoring -- over
    `history_events`/`history_source_alerts` (a caller-supplied,
    optionally broader, cross-channel candidate pool for the SAME
    customers). When omitted (the default), `history_events`/
    `history_source_alerts` fall back to `channel_events`/`source_alerts`
    themselves -- the exact pre-fix, channel-scoped-only behavior, kept
    as the default so every existing single-channel test/caller is
    unaffected; the real CLI training path now supplies a real,
    cross-channel pool (src.fraud_intel.cli_data_access.
    load_cross_channel_customer_pool()) explicitly."""
    events_by_id = {event.event_id: event for event in channel_events}
    label_by_event_id = {label.event_id: label for label in synthetic_labels}
    alerted_event_ids = {alert.event_id for alert in source_alerts}

    history_pool = history_events if history_events is not None else channel_events
    alert_history_pool = history_source_alerts if history_source_alerts is not None else source_alerts
    event_customer_ids = {event.event_id: event.customer_id for event in history_pool}

    rows: list[dict[str, Any]] = []
    for event_id in alerted_event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue  # a source alert referencing an unknown event is excluded, never crashes
        if event.channel != adapter.channel:
            continue  # this training run's own channel only

        label = label_by_event_id.get(event_id)
        if label is None:
            continue  # no label -> cannot supervise this row
        if not _phase4_interim_training_eligible(label):
            continue

        history = select_customer_historical_events(
            history_pool, customer_id=event.customer_id, as_of_time=event.event_timestamp, exclude_event_id=event.event_id,
        )
        prior_alerts = select_customer_historical_source_alerts(
            alert_history_pool, event_customer_ids=event_customer_ids, customer_id=event.customer_id,
            as_of_time=event.event_timestamp, exclude_event_id=event.event_id,
        )

        ctx = FeatureComputationContext(
            current_event=event,
            historical_events=history,
            source_alert_history=prior_alerts,
            as_of_time=event.event_timestamp,
        )
        features = adapter.compute_features(ctx)

        rows.append(
            {
                "event_id": str(event.event_id),
                "event_timestamp": event.event_timestamp,
                "label": bool(label.synthetic_scenario_label),
                **{column: features[column] for column in adapter.feature_columns},
            }
        )

    rows.sort(key=lambda row: (row["event_timestamp"], row["event_id"]))
    return rows


def _dataset_version(population_rows: Sequence[dict[str, Any]]) -> str:
    """Phase 7B corrective pass: now hashes each row's FULL computed
    feature vector (every model-input feature), not just its
    (event_id, label) identity pair -- two populations built from
    identical target event IDs but DIFFERENT historical feature context
    (e.g. cross-channel vs. channel-scoped-only history) must never
    silently share the same hash. `event_id` (always unique per row) is
    the sort key, so Python's tuple comparison never needs to fall
    through to comparing the companion dict."""
    normalized = sorted(
        (row["event_id"], {k: v for k, v in row.items() if k not in ("event_id", "event_timestamp")})
        for row in population_rows
    )
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
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


def _rows_as_feature_dicts(part_df: pl.DataFrame, feature_columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{column: record[column] for column in feature_columns} for record in part_df.to_dicts()]


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
    computed but explicitly marked supplementary, never the headline.
    pr_auc/brier_score (Phase 7A corrective pass) are what the cold-start
    promotion gate (src.fraud_intel.evaluation.cold_start) compares
    against the test split's own fraud prevalence -- the same PR-AUC-vs-
    prevalence check the live evaluation path already makes, computed
    here from the SAME held-out test-split predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    single_class = len(set(y_true)) < 2
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if not single_class else float("nan"),
        "pr_auc": average_precision_score(y_true, y_proba) if not single_class else float("nan"),
        "brier_score": brier_score_loss(y_true, y_proba),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": float(sum(int(p == t) for p, t in zip(y_pred, y_true)) / len(y_true)),  # supplementary only
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _class_counts(df: pl.DataFrame) -> dict[str, int]:
    labels = df["label"].to_list()
    return {"fraud": sum(1 for v in labels if v), "legitimate": sum(1 for v in labels if not v)}


def _split_manifest_from(split_df: pl.DataFrame, all_ids: Sequence[str]) -> SplitManifest:
    train_ids = frozenset(split_df.filter(pl.col("split") == "train")["event_id"].to_list())
    calibration_ids = frozenset(split_df.filter(pl.col("split") == "calibration")["event_id"].to_list())
    test_ids = frozenset(split_df.filter(pl.col("split") == "test")["event_id"].to_list())
    purge_ids = frozenset(all_ids) - train_ids - calibration_ids - test_ids
    return SplitManifest(train_ids=train_ids, calibration_ids=calibration_ids, purge_ids=purge_ids, test_ids=test_ids)


class AmbiguousGenerationProvenanceError(ValueError):
    """The source_alerts feeding this training population carry more than
    one distinct (generation_run_id, dataset_version) pair -- training
    must never silently pick one and mix generations together."""


def _source_generation_provenance(source_alerts: Sequence[SourceAlertContext]) -> tuple[str, str]:
    """The Stage-2 generation identity (`genrun-...`/`dsv-...`, minted by
    src.fraud_intel.cli_data_access.generate_and_write) that every row
    contributing to this training population was generated under --
    NOT to be confused with _dataset_version() below, which is a
    completely different, training-DERIVED content hash of the
    supervised population itself. Both values end up in
    evaluation_report_ref under clearly distinct names specifically so
    they are never conflated (source_generation_run_id/
    source_dataset_version here vs. dataset_version/
    supervised_population_hash for the derived hash)."""
    run_ids = {alert.generation_run_id for alert in source_alerts}
    dataset_versions = {alert.dataset_version for alert in source_alerts}
    if len(run_ids) != 1 or len(dataset_versions) != 1:
        raise AmbiguousGenerationProvenanceError(
            f"training population spans more than one Stage-2 generation -- "
            f"generation_run_id(s)={sorted(run_ids)!r}, dataset_version(s)={sorted(dataset_versions)!r}"
        )
    return next(iter(run_ids)), next(iter(dataset_versions))


def train_channel_configured(
    config: ChannelTrainingRunConfig,
    *,
    database: str,
    trigger_source: str = "legacy",
    channel_events: Sequence[FraudEvent],
    source_alerts: Sequence[SourceAlertContext],
    synthetic_labels: Sequence[SyntheticGroundTruthLabel],
    cross_channel_events: Optional[Sequence[FraudEvent]] = None,
    cross_channel_source_alerts: Optional[Sequence[SourceAlertContext]] = None,
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
    every REQUIRED_OPERATIONAL_COMPONENTS field populated.

    Phase 7B Stage 3 corrective pass: `database` is REQUIRED, with no
    default of any kind, specifically so this function can never silently
    fall back to `.env`'s POSTGRES_DB default (production `aidp`) the way
    the bare `RunLifecycle()`/`create_default_bundle_store()` calls used
    to. Every caller -- CLI or test -- must say explicitly which database
    both the `train` pipeline_runs record AND the candidate bundle row
    are written to, and both are guaranteed to be the SAME database
    (there is exactly one `database` value in this function, used for
    both).

    Phase 7B corrective pass: `cross_channel_events`/
    `cross_channel_source_alerts` are optional -- when supplied (the real
    CLI training path always supplies them, via src.fraud_intel.
    cli_data_access.load_cross_channel_customer_pool()), the supervised
    population's per-row HISTORY (never which events become rows) is built
    from this broader, cross-channel pool instead of `channel_events`/
    `source_alerts` alone -- see _build_supervised_population()'s own
    docstring. Omitted, training reproduces the exact pre-fix, channel-
    scoped-only history (every existing test's behavior, unchanged)."""
    if not database:
        raise ValueError("database is required and must not be empty -- train_channel_configured() never assumes a default")

    lifecycle = RunLifecycle(database=database)
    run = lifecycle.begin(
        "train",
        trigger_source=trigger_source,
        git_sha=get_git_sha(),
        config_snapshot=config.redacted_snapshot(),
        config_hash=config.config_hash(),
    )

    dataset_version: Optional[str] = None

    try:
        # Phase 7A: get_channel_adapter() itself is the channel-validity
        # check -- config.channel is a closed 7-value Literal and every
        # value is now registered, so this can only ever raise for a
        # channel that was never registered at all.
        adapter = get_channel_adapter(config.channel)

        store = bundle_store if bundle_store is not None else create_default_bundle_store(database)

        population_rows = _build_supervised_population(
            adapter=adapter, channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=synthetic_labels,
            history_events=cross_channel_events, history_source_alerts=cross_channel_source_alerts,
        )
        if not population_rows:
            raise InsufficientTrainingDataError(
                f"no source-alerted, eligible {config.channel} rows available to build a supervised population"
            )

        population_event_ids = {row["event_id"] for row in population_rows}
        contributing_alerts = [alert for alert in source_alerts if str(alert.event_id) in population_event_ids]
        source_generation_run_id, source_dataset_version = _source_generation_provenance(contributing_alerts)

        dataset_version = _dataset_version(population_rows)

        df = pl.DataFrame(
            [
                {
                    "event_id": row["event_id"],
                    "event_timestamp": row["event_timestamp"],
                    "label": row["label"],
                    **{column: row[column] for column in adapter.feature_columns},
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

        train_rows = _rows_as_feature_dicts(train_df, adapter.feature_columns)
        preprocessor = ChannelPreprocessor.fit(
            train_rows,
            feature_columns=adapter.feature_columns,
            feature_schema_version=adapter.feature_schema_version,
            preprocessing_artifact_version=_preprocessing_artifact_version(train_rows),
        )

        X_train = preprocessor.transform(train_rows)
        y_train = train_df["label"].to_list()
        X_calib = preprocessor.transform(_rows_as_feature_dicts(calib_df, adapter.feature_columns))
        y_calib = calib_df["label"].to_list()
        X_test = preprocessor.transform(_rows_as_feature_dicts(test_df, adapter.feature_columns))
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
        mlflow.set_experiment(_experiment_name(config.channel))

        with mlflow.start_run(run_name=f"{config.channel}-gbm-{dataset_version}") as gbm_run:
            mlflow.log_params(gbm_params)
            mlflow.log_param("feature_schema_version", adapter.feature_schema_version)
            for key, value in gbm_eval.items():
                if key != "confusion_matrix":
                    mlflow.log_metric(key, value)
            signature = mlflow.models.infer_signature(X_train, gbm_model.predict_proba(X_train))
            gbm_model_info = mlflow.xgboost.log_model(
                gbm_model,
                artifact_path="model",
                signature=signature,
                registered_model_name=registered_model_name(config.channel, "gbm"),
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

        with mlflow.start_run(run_name=f"{config.channel}-lr-shadow-{dataset_version}") as lr_run:
            mlflow.log_params(lr_params)
            for key, value in lr_eval.items():
                if key != "confusion_matrix":
                    mlflow.log_metric(f"shadow_{key}", value)
            lr_signature = mlflow.models.infer_signature(X_train, lr_model.predict_proba(X_train))
            lr_model_info = mlflow.sklearn.log_model(
                lr_model,
                artifact_path="model",
                signature=lr_signature,
                registered_model_name=registered_model_name(config.channel, "lr-shadow"),
            )
            lr_mlflow_run_id = lr_run.info.run_id
            lr_model_version = lr_model_info.registered_model_version

        with mlflow.start_run(run_name=f"{config.channel}-anomaly-{dataset_version}") as anomaly_run:
            mlflow.log_param("anomaly_artifact_version", anomaly_artifact_version)
            mlflow.log_param("contamination", "auto")
            anomaly_model_info = mlflow.sklearn.log_model(
                anomaly_model,
                artifact_path="model",
                registered_model_name=registered_model_name(config.channel, "anomaly"),
            )
            anomaly_mlflow_run_id = anomaly_run.info.run_id
            anomaly_model_version = anomaly_model_info.registered_model_version
            # Same rationale as preprocessor.json above -- makes real
            # scoring-time reload of the fitted normalization bounds
            # possible (src.fraud_intel.scoring.dispatch).
            mlflow.log_dict(anomaly_normalization.to_json_dict(), "anomaly_normalization.json")

        # Phase 7A corrective pass: this report is the ONLY evidence source
        # for a channel's cold-start (pre-first-scoring-run) promotion
        # decision (src.fraud_intel.evaluation.cold_start) -- it now
        # carries the bundle-identifying fields (channel/training_run_id/
        # dataset_version/feature_schema_version/component versions) and
        # per-split class counts needed to validate and gate on it,
        # alongside the pre-existing metric summaries. gbm/lr_shadow now
        # include their confusion_matrix too (previously stripped only for
        # the unrelated mlflow.log_metric() call above, which cannot log a
        # nested dict as a scalar metric).
        evaluation_report_ref = json.dumps(
            {
                "channel": config.channel,
                "training_run_id": run.run_id,
                "dataset_version": dataset_version,
                # Phase 7B Stage 3 corrective pass: `dataset_version` above
                # (and channel_model_bundles.dataset_version, unchanged --
                # no migration) is the training-DERIVED content hash of the
                # supervised population, aliased here under an unambiguous
                # name; source_generation_run_id/source_dataset_version are
                # the DIFFERENT, Stage-2 generation identity every row of
                # that population actually came from. Never conflate the
                # two -- see _source_generation_provenance()'s docstring.
                "supervised_population_hash": dataset_version,
                "source_generation_run_id": source_generation_run_id,
                "source_dataset_version": source_dataset_version,
                "feature_schema_version": adapter.feature_schema_version,
                "gbm_model_version": str(gbm_model_version),
                "lr_model_version": str(lr_model_version),
                "anomaly_model_version": str(anomaly_model_version),
                "preprocessing_artifact_version": preprocessor.preprocessing_artifact_version,
                "gbm_evaluation": gbm_eval,
                "lr_shadow_evaluation": lr_eval,
                "eligibility_policy_version": PHASE4_INTERIM_ELIGIBILITY_POLICY_VERSION,
                "realized_split_fractions": realized_fractions,
                "split_class_counts": {
                    "train": _class_counts(train_df), "calibration": _class_counts(calib_df), "test": _class_counts(test_df),
                },
                "gbm_mlflow_run_id": gbm_mlflow_run_id,
                "lr_mlflow_run_id": lr_mlflow_run_id,
                "anomaly_mlflow_run_id": anomaly_mlflow_run_id,
                "anomaly_normalization": anomaly_normalization.to_json_dict(),
            },
            sort_keys=True,
        )

        bundle = store.register_candidate(
            channel=config.channel,
            gbm_model_version=str(gbm_model_version),
            lr_model_version=str(lr_model_version),
            anomaly_model_version=str(anomaly_model_version),
            preprocessing_artifact_version=preprocessor.preprocessing_artifact_version,
            feature_schema_version=adapter.feature_schema_version,
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
            "feature_schema_version": adapter.feature_schema_version,
            "bundle_id": bundle.bundle_id,
            "bundle_version": bundle.bundle_version,
            "realized_split_fractions": realized_fractions,
            "supervised_population_hash": dataset_version,
            "source_generation_run_id": source_generation_run_id,
            "source_dataset_version": source_dataset_version,
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
        "supervised_population_hash": dataset_version,
        "source_generation_run_id": source_generation_run_id,
        "source_dataset_version": source_dataset_version,
        "gbm_evaluation": gbm_eval,
        "lr_shadow_evaluation": lr_eval,
        "realized_split_fractions": realized_fractions,
    }
