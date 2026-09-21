"""Typed run configuration for v1.3 per-channel training (guide section
11). Subclasses the same RunConfig base BatchRunConfig/TrainingRunConfig/
StreamRunConfig already use (src.control_plane.config) -- redaction and
config_hash() come free, not reimplemented.
"""
from __future__ import annotations

from pydantic import Field, model_validator

from src.common.config import get_settings
from src.control_plane.config import RunConfig
from src.fraud_intel.events.base import Channel

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_CALIB_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15

DEFAULT_MIN_TRAIN_ROWS = 10
DEFAULT_MIN_CALIB_ROWS = 5
DEFAULT_MIN_TEST_ROWS = 5


class ChannelTrainingRunConfig(RunConfig):
    """One invocation of src.fraud_intel.models.training.train_channel_configured."""

    channel: Channel
    train_frac: float = Field(default=DEFAULT_TRAIN_FRAC, gt=0, lt=1)
    calib_frac: float = Field(default=DEFAULT_CALIB_FRAC, gt=0, lt=1)
    test_frac: float = Field(default=DEFAULT_TEST_FRAC, gt=0, lt=1)
    purge_gap_seconds: float = Field(default=0.0, ge=0)
    handle_class_imbalance: bool = True
    min_train_rows: int = Field(default=DEFAULT_MIN_TRAIN_ROWS, gt=0)
    min_calib_rows: int = Field(default=DEFAULT_MIN_CALIB_ROWS, gt=0)
    min_test_rows: int = Field(default=DEFAULT_MIN_TEST_ROWS, gt=0)
    random_seed: int = Field(default_factory=lambda: get_settings().gen_random_seed)

    @model_validator(mode="after")
    def _fracs_sum_to_one(self) -> "ChannelTrainingRunConfig":
        total = self.train_frac + self.calib_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_frac + calib_frac + test_frac must sum to 1.0, got {total}")
        return self
