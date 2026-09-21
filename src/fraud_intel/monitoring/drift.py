"""Alert-volume and score-distribution drift indicators from
alert_evidence history (guide section 25, Phase 7A). Dependency-free
(no scipy) -- a bucketed-histogram total-variation distance for score
drift, and a simple count/rate comparison for volume drift, deliberately
kept this simple since no channel-specific calibration or statistical
significance claim is made anywhere in this POC. No database, MLflow,
Docker, or network access anywhere in this module -- every caller builds
a `DriftWindow` from whatever `alert_evidence` rows it already has (a
real Postgres read is Phase 7B scope, not built here).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_NUM_SCORE_BUCKETS = 10


class DriftWindow(BaseModel):
    """One time-bounded slice of scored alerts -- `scores` are each
    `alert_evidence.operational_priority_score` (or an equivalent
    per-alert score), `window_start`/`window_end` bound the period they
    were scored in. `len(scores)` need not equal any external row count;
    it IS the alert volume for this window, by construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    window_start: datetime
    window_end: datetime
    scores: tuple[float, ...]

    @field_validator("scores")
    @classmethod
    def _scores_in_bounds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for score in value:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"score {score!r} is out of the [0, 1] bound")
        return value

    @model_validator(mode="after")
    def _window_ordered(self) -> "DriftWindow":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be strictly after window_start")
        return self

    def duration_days(self) -> float:
        return (self.window_end - self.window_start).total_seconds() / 86400.0

    def rate_per_day(self) -> float:
        duration = self.duration_days()
        return len(self.scores) / duration if duration > 0 else 0.0


DriftStatus = Literal["ok", "non_computable_empty"]


class AlertVolumeDrift(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_count: int
    current_count: int
    baseline_rate_per_day: float
    current_rate_per_day: float
    relative_change: Optional[float]
    status: DriftStatus
    reason: Optional[str] = None


class ScoreDistributionDrift(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket_edges: tuple[float, ...]
    baseline_bucket_proportions: tuple[float, ...]
    current_bucket_proportions: tuple[float, ...]
    total_variation_distance: Optional[float]  # 0.0 = identical distributions, 1.0 = fully disjoint
    baseline_mean: Optional[float]
    current_mean: Optional[float]
    mean_delta: Optional[float]
    status: DriftStatus
    reason: Optional[str] = None


def compute_alert_volume_drift(baseline: DriftWindow, current: DriftWindow) -> AlertVolumeDrift:
    baseline_rate = baseline.rate_per_day()
    current_rate = current.rate_per_day()
    if baseline_rate == 0:
        relative_change = None
        status: DriftStatus = "non_computable_empty" if len(baseline.scores) == 0 else "ok"
        reason = "baseline window has zero alert volume -- relative_change undefined" if status != "ok" else None
    else:
        relative_change = (current_rate - baseline_rate) / baseline_rate
        status = "ok"
        reason = None
    return AlertVolumeDrift(
        baseline_count=len(baseline.scores), current_count=len(current.scores),
        baseline_rate_per_day=baseline_rate, current_rate_per_day=current_rate,
        relative_change=relative_change, status=status, reason=reason,
    )


def _bucket_proportions(scores: Sequence[float], edges: Sequence[float]) -> tuple[float, ...]:
    counts = [0] * (len(edges) - 1)
    for score in scores:
        for i in range(len(edges) - 1):
            upper_inclusive = i == len(edges) - 2
            if edges[i] <= score < edges[i + 1] or (upper_inclusive and score == edges[-1]):
                counts[i] += 1
                break
    total = len(scores)
    return tuple((c / total) if total > 0 else 0.0 for c in counts)


def compute_score_distribution_drift(
    baseline: DriftWindow, current: DriftWindow, *, num_buckets: int = DEFAULT_NUM_SCORE_BUCKETS
) -> ScoreDistributionDrift:
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
    edges = tuple(i / num_buckets for i in range(num_buckets + 1))

    if not baseline.scores or not current.scores:
        empty = tuple(0.0 for _ in range(num_buckets))
        return ScoreDistributionDrift(
            bucket_edges=edges, baseline_bucket_proportions=empty, current_bucket_proportions=empty,
            total_variation_distance=None, baseline_mean=None, current_mean=None, mean_delta=None,
            status="non_computable_empty", reason="at least one window has no scored alerts",
        )

    baseline_props = _bucket_proportions(baseline.scores, edges)
    current_props = _bucket_proportions(current.scores, edges)
    # Total variation distance between two discrete distributions: half the
    # L1 distance between their probability vectors -- 0.0 identical, 1.0
    # fully disjoint. Dependency-free (no scipy) by construction.
    tvd = sum(abs(b - c) for b, c in zip(baseline_props, current_props)) / 2.0

    baseline_mean = sum(baseline.scores) / len(baseline.scores)
    current_mean = sum(current.scores) / len(current.scores)

    return ScoreDistributionDrift(
        bucket_edges=edges, baseline_bucket_proportions=baseline_props, current_bucket_proportions=current_props,
        total_variation_distance=tvd, baseline_mean=baseline_mean, current_mean=current_mean,
        mean_delta=current_mean - baseline_mean, status="ok",
    )
