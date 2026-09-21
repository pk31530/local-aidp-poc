"""`SourceAlertContext` and `SyntheticGroundTruthLabel` (guide section 6).

Both are genuinely separate objects from `FraudEvent` -- never merged into
`channel_payload`, never read by any scoring or feature-computation code
path (enforced structurally, not just by convention -- see
tests/unit/test_fraud_intel_label_isolation.py).

Canonical field name note: the guide's section 6 table and its
leakage-prevention/Phase-2-leakage-test language both use
`synthetic_scenario_label`; the Phase 1 copy-paste prompt text used
`is_fraud_scenario` for the same field. `synthetic_scenario_label` is used
here, since section 6/8's leakage-prevention tests reference that exact
name repeatedly throughout the guide.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_naive_or_non_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("must be UTC")
    return value


class SourceAlertContext(BaseModel):
    """Mirrors the persisted `source_alerts` table (guide sections 6, 18).
    `event_id` is intentionally not required to be unique across instances
    -- one event can legitimately have more than one source alert."""

    model_config = ConfigDict(extra="forbid")

    source_alert_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_system: str
    event_id: uuid.UUID
    source_alert_created_at: datetime
    source_rule_ids: list[str]
    source_rule_version: str
    source_alert_score: Optional[float] = None
    source_alert_reason_codes: list[str]
    generation_run_id: str
    dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_alert_created_at", "created_at")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _reject_naive_or_non_utc(value)


class SyntheticGroundTruthLabel(BaseModel):
    """Ground truth from the synthetic generator only (guide sections 6, 8).
    Never embedded in `channel_payload`; never read by any feature or
    scoring code path."""

    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID
    scenario_id: Optional[str] = None
    synthetic_scenario_label: bool
    scenario_type: str
    label_source: Literal["SYNTHETIC_GENERATOR"] = "SYNTHETIC_GENERATOR"
    generation_run_id: str
    dataset_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("generated_at")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _reject_naive_or_non_utc(value)
