"""One place that wires up MLflow: the tracking URI, plus the S3-compatible
env vars its client needs to talk to MinIO directly for artifact
upload/download. Call `configure_mlflow()` once at the start of any process
that talks to MLflow (training, and later the API if it loads models
directly) instead of each caller re-deriving these values.
"""
from __future__ import annotations

import os

import mlflow
from mlflow.exceptions import MlflowException

from src.common.config import get_settings


class ModelAliasNotFoundError(Exception):
    """The requested model alias (or the underlying model) does not exist
    in the registry — distinct from a connection, authentication, or
    server failure, which must propagate unchanged."""


# MLflow's SQLAlchemy-backed model registry store (what this repo's Postgres-
# backed MLflow uses) raises exactly this error code — not RESOURCE_DOES_NOT_EXIST
# — when get_model_version_by_alias() can't find the alias; verified against
# mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore.get_model_version_by_alias.
# Checked via the structured `error_code` field, never by matching message text.
_ALIAS_NOT_FOUND_ERROR_CODE = "INVALID_PARAMETER_VALUE"


def configure_mlflow() -> None:
    settings = get_settings()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"http://{settings.minio_endpoint}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def _resolve_model_version(model_name: str, alias: str):
    """Shared lookup: the registered model version behind an alias, its
    MLflow run, and the model_type param used to pick a flavor-specific
    loader. Used by both load_champion_model() (which also loads weights)
    and get_model_alias_info() (metadata only, no weights)."""
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(mv.run_id)
    model_type = run.data.params.get("model_type", "xgboost")
    return client, mv, run, model_type


def load_champion_model(model_name: str):
    """Loads the `champion`-aliased model version with whichever MLflow
    flavor it was actually registered under.

    src/ml/train.py can fall back to a RandomForestClassifier (logged via
    mlflow.sklearn.log_model) if XGBoost training fails; loading every
    champion unconditionally via mlflow.xgboost.load_model would then raise.
    Both serving paths (src/api/main.py, src/ingestion/consumer.py) share
    this loader instead of each hardcoding the XGBoost-only assumption.

    Returns (model, version).
    """
    _, mv, _, model_type = _resolve_model_version(model_name, "champion")
    model_uri = f"models:/{model_name}@champion"
    load_fn = mlflow.xgboost.load_model if model_type == "xgboost" else mlflow.sklearn.load_model
    model = load_fn(model_uri)
    return model, mv.version


def get_model_alias_info(model_name: str, alias: str) -> dict:
    """Metadata only for a registered model alias — never loads model
    weights. Used by the CLI's `model show` command, which must not
    promote, retrain, or otherwise modify the registry.

    Returns only model_name, alias, version, model_type, and the MLflow
    run_id — never parameters, tags, artifact/storage paths, or any
    credential/environment information.

    Raises ModelAliasNotFoundError specifically when the alias/model
    doesn't exist (identified by MLflow's structured error_code, not by
    matching message text). Any other MlflowException — a connection
    failure, an authentication failure, a server error — propagates
    unchanged, since those are operational failures, not a bad CLI input.
    """
    try:
        _, mv, run, model_type = _resolve_model_version(model_name, alias)
    except MlflowException as exc:
        if exc.error_code == _ALIAS_NOT_FOUND_ERROR_CODE:
            raise ModelAliasNotFoundError(f"model {model_name!r} has no alias {alias!r}") from exc
        raise
    return {
        "model_name": model_name,
        "alias": alias,
        "version": mv.version,
        "model_type": model_type,
        "run_id": mv.run_id,
    }
