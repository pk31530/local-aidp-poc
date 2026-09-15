"""Centralized application configuration.

Secrets/connection info come from environment variables (.env, see
.env.example). Non-secret tunables (thresholds, generation defaults, timezone)
come from the YAML files in config/.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_timezone: str = "Asia/Kolkata"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "aidp"
    postgres_user: str = "aidp"
    postgres_password: str = "aidp_local_only"
    postgres_test_db: str = "aidp_test"

    minio_endpoint: str = "127.0.0.1:9000"
    minio_console_endpoint: str = "127.0.0.1:9001"
    minio_access_key: str = "aidpadmin"
    minio_secret_key: str = "aidpadmin123_local_only"
    minio_secure: bool = False

    redpanda_brokers: str = "127.0.0.1:9092"
    redpanda_topic_transactions: str = "transactions"
    redpanda_topic_fraud_decisions: str = "fraud-decisions"
    redpanda_topic_dlq: str = "dead-letter-transactions"
    redpanda_topic_test: str = "transactions-test"

    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_backend_db: str = "mlflow"
    mlflow_artifact_bucket: str = "mlflow-artifacts"
    mlflow_model_name: str = "fraud-detection-model"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8501

    redpanda_console_port: int = 8080

    gen_customers: int = 10000
    gen_historical_transactions: int = 50000
    gen_fraud_ratio: float = 0.03
    gen_random_seed: int = 42

    @property
    def postgres_dsn(self) -> str:
        return self.postgres_dsn_for(self.postgres_db)

    @property
    def postgres_test_dsn(self) -> str:
        return self.postgres_dsn_for(self.postgres_test_db)

    def postgres_dsn_for(self, database: str) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{database}"
        )


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache
def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_app_settings() -> dict[str, Any]:
    """Non-secret settings from config/settings.yaml (generation defaults,
    velocity windows, streaming/data-lake config, timezone)."""
    return _load_yaml("settings.yaml")


def get_fraud_rules() -> dict[str, Any]:
    """Decision thresholds and reason-code rules from config/fraud_rules.yaml."""
    return _load_yaml("fraud_rules.yaml")
