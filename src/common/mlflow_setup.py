"""One place that wires up MLflow: the tracking URI, plus the S3-compatible
env vars its client needs to talk to MinIO directly for artifact
upload/download. Call `configure_mlflow()` once at the start of any process
that talks to MLflow (training, and later the API if it loads models
directly) instead of each caller re-deriving these values.
"""
from __future__ import annotations

import os

import mlflow

from src.common.config import get_settings


def configure_mlflow() -> None:
    settings = get_settings()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", f"http://{settings.minio_endpoint}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
