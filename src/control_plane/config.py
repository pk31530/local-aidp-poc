"""Typed, validated run configuration for AiDP v1.2 control-plane workloads.

Each model describes the parameters of a single batch/training/stream
invocation and its safe overrides. Platform-level settings (thresholds,
generation defaults, timezone, connection info) remain sourced from
src/common/config.py and config/*.yaml; these models never duplicate them.

Deliberately independent of src/processing/pipeline.py, src/ml/train.py and
src/ingestion/consumer.py: this module must not import them, so that Phase 3
can have those modules depend on this one without introducing a cycle.
Defaults below are literal values kept in sync with those modules by test
coverage in tests/unit/test_control_plane_config.py, not by import.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.common.config import PROJECT_ROOT

DEFAULT_BATCH_INPUT = PROJECT_ROOT / "data" / "seed" / "historical_transactions.csv"
DEFAULT_TRAINING_FEATURES_PATH = PROJECT_ROOT / "data" / "output" / "features" / "transactions.parquet"

# Keys matching this pattern are redacted from every snapshot. None of the
# fields below are currently secrets, but this is the same rule the Phase 2
# provenance service will apply to config snapshots it persists.
_SECRET_KEY_PATTERN = re.compile(r"password|secret|token|key|credential|dsn|url", re.IGNORECASE)
_REDACTED = "***REDACTED***"


def redact_secret_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `data` with any secret-shaped key's value replaced."""
    return {key: (_REDACTED if _SECRET_KEY_PATTERN.search(key) else value) for key, value in data.items()}


class RunConfig(BaseModel):
    """Shared behaviour for every typed run configuration model."""

    model_config = ConfigDict(extra="forbid")

    def redacted_snapshot(self) -> dict[str, Any]:
        """A deterministic, JSON-serialisable dict with secret-shaped keys redacted."""
        return redact_secret_keys(self.model_dump(mode="json"))

    def config_hash(self) -> str:
        """Stable SHA-256 hex digest of the redacted snapshot's canonical JSON."""
        canonical = json.dumps(self.redacted_snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BatchRunConfig(RunConfig):
    """One invocation of the batch pipeline (src.processing.pipeline.run_pipeline)."""

    input_path: Path = DEFAULT_BATCH_INPUT
    upload_to_minio: bool = True


class TrainingRunConfig(RunConfig):
    """One invocation of model training (src.ml.train.train)."""

    features_path: Path = DEFAULT_TRAINING_FEATURES_PATH


class StreamRunConfig(RunConfig):
    """One invocation of the streaming consumer (src.ingestion.consumer.run)."""

    duration: Optional[float] = Field(default=None, gt=0)
    from_beginning: bool = False
    topic: Optional[str] = None
    dlq_topic: Optional[str] = None
    database: Optional[str] = None
    group_id: str = "aidp-consumer"
    raw_bucket: str = "aidp-raw"
