import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.control_plane.config import (
    BatchRunConfig,
    StreamRunConfig,
    TrainingRunConfig,
    redact_secret_keys,
)

# Imported only here, in tests, to prove the independently-defined defaults
# in src/control_plane/config.py stay aligned with the real entry points —
# config.py itself must not import these modules (would risk a Phase 3
# import cycle once they consume the typed configs).
from src.ingestion.consumer import run as consumer_run
from src.ml.train import DEFAULT_FEATURES_PATH
from src.processing.pipeline import DEFAULT_INPUT

import inspect


# ---- defaults match the real entry points ----------------------------------


def test_batch_default_matches_pipeline_entry_point():
    config = BatchRunConfig()
    assert config.input_path == DEFAULT_INPUT
    assert config.upload_to_minio is True


def test_training_default_matches_train_entry_point():
    config = TrainingRunConfig()
    assert config.features_path == DEFAULT_FEATURES_PATH


def test_stream_defaults_match_consumer_entry_point():
    config = StreamRunConfig()
    params = inspect.signature(consumer_run).parameters
    assert config.group_id == params["group_id"].default
    assert config.raw_bucket == params["raw_bucket"].default
    assert config.duration is None
    assert config.from_beginning is False
    assert config.topic is None
    assert config.dlq_topic is None
    assert config.database is None


# ---- valid overrides ---------------------------------------------------------


def test_batch_valid_overrides():
    config = BatchRunConfig(input_path=Path("data/batch/custom.csv"), upload_to_minio=False)
    assert config.input_path == Path("data/batch/custom.csv")
    assert config.upload_to_minio is False


def test_training_valid_overrides():
    config = TrainingRunConfig(features_path=Path("data/output/features/custom.parquet"))
    assert config.features_path == Path("data/output/features/custom.parquet")


def test_stream_valid_overrides():
    config = StreamRunConfig(
        duration=30.0,
        from_beginning=True,
        topic="transactions-test",
        dlq_topic="dead-letter-transactions-test",
        database="aidp_test",
        group_id="aidp-consumer-test",
        raw_bucket="aidp-raw-test",
    )
    assert config.duration == 30.0
    assert config.from_beginning is True
    assert config.topic == "transactions-test"
    assert config.database == "aidp_test"


# ---- unknown fields rejected --------------------------------------------------


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (BatchRunConfig, {"not_a_real_field": 1}),
        (TrainingRunConfig, {"not_a_real_field": 1}),
        (StreamRunConfig, {"not_a_real_field": 1}),
    ],
)
def test_unknown_field_rejected(model_cls, kwargs):
    with pytest.raises(ValidationError):
        model_cls(**kwargs)


@pytest.mark.parametrize(
    "model_cls, secret_kwargs",
    [
        (BatchRunConfig, {"password": "hunter2"}),
        (TrainingRunConfig, {"api_token": "abc"}),
        (StreamRunConfig, {"postgres_dsn": "postgresql://u:p@h/db"}),
    ],
)
def test_secret_shaped_fields_cannot_enter_the_models(model_cls, secret_kwargs):
    """extra="forbid" means a secret-shaped field is rejected outright, not
    merely redacted after the fact."""
    with pytest.raises(ValidationError):
        model_cls(**secret_kwargs)


# ---- numeric bounds ------------------------------------------------------------


@pytest.mark.parametrize("duration", [0, -1, -0.5])
def test_stream_non_positive_duration_rejected(duration):
    with pytest.raises(ValidationError):
        StreamRunConfig(duration=duration)


@pytest.mark.parametrize("duration", [None, 1, 0.5, 3600])
def test_stream_valid_duration_accepted(duration):
    config = StreamRunConfig(duration=duration)
    assert config.duration == duration


# ---- redaction -----------------------------------------------------------------


def test_redact_secret_keys_redacts_known_secret_shapes():
    data = {
        "password": "x",
        "api_token": "x",
        "auth_secret": "x",
        "postgres_dsn": "x",
        "webhook_url": "x",
        "access_key": "x",
        "group_id": "aidp-consumer",
        "raw_bucket": "aidp-raw",
        "duration": 30.0,
    }
    redacted = redact_secret_keys(data)
    for key in ("password", "api_token", "auth_secret", "postgres_dsn", "webhook_url", "access_key"):
        assert redacted[key] == "***REDACTED***"
    assert redacted["group_id"] == "aidp-consumer"
    assert redacted["raw_bucket"] == "aidp-raw"
    assert redacted["duration"] == 30.0


# ---- deterministic snapshots and hashes ----------------------------------------


def test_redacted_snapshot_is_deterministic():
    config = BatchRunConfig(input_path=Path("data/batch/x.csv"))
    assert config.redacted_snapshot() == config.redacted_snapshot()


def test_config_hash_is_deterministic_and_stable_across_equal_instances():
    a = TrainingRunConfig(features_path=Path("data/output/features/x.parquet"))
    b = TrainingRunConfig(features_path=Path("data/output/features/x.parquet"))
    assert a.config_hash() == b.config_hash()


def test_config_hash_differs_for_different_config():
    a = StreamRunConfig(duration=10)
    b = StreamRunConfig(duration=20)
    assert a.config_hash() != b.config_hash()


# ---- JSON serialisation of Path and None values --------------------------------


def test_batch_snapshot_serialises_path_as_string_and_is_json_dumpable():
    config = BatchRunConfig(input_path=Path("data/batch/custom.csv"))
    snapshot = config.redacted_snapshot()
    assert snapshot["input_path"] == "data/batch/custom.csv"
    assert isinstance(snapshot["input_path"], str)
    json.dumps(snapshot)  # must not raise


def test_stream_snapshot_with_none_fields_is_json_dumpable():
    config = StreamRunConfig()
    snapshot = config.redacted_snapshot()
    assert snapshot["topic"] is None
    json.dumps(snapshot)  # must not raise
