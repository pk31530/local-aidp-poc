"""Phase 5: train an XGBoost fraud classifier on the FEATURES parquet,
track it in MLflow, and register the winning run as `fraud-detection-model`.

Uses exactly `src.common.features.MODEL_FEATURE_COLUMNS` as the model input
(fix C2 — the same list Phase 6 serving will use to build its feature
vector), and `is_fraud` as the label only (never as an input feature).

Falls back to scikit-learn's RandomForestClassifier only if XGBoost has a
genuine local compatibility blocker (guide section 41); the fallback is
logged loudly, not silently substituted.

Usage:
    python -m src.ml.train
    python -m src.ml.train --features-path data/output/features/transactions.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.config import PROJECT_ROOT, get_settings
from src.common.db import get_connection
from src.common.features import MODEL_FEATURE_COLUMNS
from src.common.logging import configure_logging, get_logger
from src.common.mlflow_setup import configure_mlflow

DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "output" / "features" / "transactions.parquet"
RANDOM_SEED = get_settings().gen_random_seed

EXPERIMENT_NAME = "fraud-detection"


def _dataset_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pl.read_parquet(path).to_pandas()
    X = df[MODEL_FEATURE_COLUMNS].astype(float)
    y = df["is_fraud"].astype(int)
    return X, y


def _split(X: pd.DataFrame, y: pd.Series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _train_xgboost(X_train, y_train, X_val, y_val):
    import xgboost as xgb

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, params, "xgboost"


def _train_random_forest_fallback(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier

    params = dict(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, params, "random_forest_fallback"


def _evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "accuracy": float((y_pred == y_test).mean()),  # supplementary only, not the headline metric
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(features_path: Path = DEFAULT_FEATURES_PATH) -> dict:
    configure_logging("ml.train")
    log = get_logger(__name__)
    settings = get_settings()

    configure_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset_version = _dataset_version(features_path)
    X, y = _load_dataset(features_path)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)

    log.info(
        "training_data_loaded",
        total_rows=len(X),
        train_rows=len(X_train),
        val_rows=len(X_val),
        test_rows=len(X_test),
        fraud_ratio_overall=round(float(y.mean()), 4),
        dataset_version=dataset_version,
    )

    try:
        model, params, model_type = _train_xgboost(X_train, y_train, X_val, y_val)
    except Exception:
        log.warning("xgboost_unavailable_falling_back_to_random_forest", exc_info=True)
        model, params, model_type = _train_random_forest_fallback(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test)
    log.info("evaluation_complete", model_type=model_type, **{k: v for k, v in metrics.items() if k != "confusion_matrix"})
    log.info("confusion_matrix", **metrics["confusion_matrix"])

    with mlflow.start_run(run_name=f"{model_type}-{dataset_version}") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_param("feature_list", ",".join(MODEL_FEATURE_COLUMNS))
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("random_seed", RANDOM_SEED)

        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("false_positive_rate", metrics["false_positive_rate"])
        mlflow.log_metric("false_negative_rate", metrics["false_negative_rate"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        for k, v in metrics["confusion_matrix"].items():
            mlflow.log_metric(f"confusion_{k}", v)

        cm_path = PROJECT_ROOT / "data" / "models" / "last_confusion_matrix.json"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm_path.write_text(json.dumps(metrics["confusion_matrix"], indent=2))
        mlflow.log_artifact(str(cm_path))

        signature = mlflow.models.infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        log_model_fn = mlflow.xgboost.log_model if model_type == "xgboost" else mlflow.sklearn.log_model
        model_info = log_model_fn(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=settings.mlflow_model_name,
        )

        run_id = run.info.run_id
        registered_version = model_info.registered_model_version

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(settings.mlflow_model_name, "champion", registered_version)
    log.info("model_registered", model_name=settings.mlflow_model_name, version=registered_version, alias="champion", run_id=run_id)

    _record_model_version_in_postgres(
        model_version=str(registered_version),
        model_name=settings.mlflow_model_name,
        mlflow_run_id=run_id,
        metrics=metrics,
    )

    return {
        "run_id": run_id,
        "model_version": registered_version,
        "model_type": model_type,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _record_model_version_in_postgres(model_version: str, model_name: str, mlflow_run_id: str, metrics: dict) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE model_versions SET is_active = false WHERE model_name = %s", (model_name,))
                cur.execute(
                    """
                    INSERT INTO model_versions
                        (model_version, model_name, mlflow_run_id, precision_score, recall_score,
                         f1_score, roc_auc_score, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, true)
                    """,
                    (
                        model_version,
                        model_name,
                        mlflow_run_id,
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        metrics["roc_auc"],
                    ),
                )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud-detection model.")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args.features_path)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
