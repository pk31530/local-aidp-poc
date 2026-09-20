"""Versioned ensemble policy (guide section 16). `compute_operational_
priority_score` has no `lr_probability` parameter anywhere in its
signature -- a code-level fact, not a convention (guide section 13),
verified by an explicit structural test.
"""
from __future__ import annotations

import functools
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.config import PROJECT_ROOT

ENSEMBLE_POLICY_CONFIG_DIR = PROJECT_ROOT / "config" / "fraud_intel"

PriorityBand = Literal["LOW", "MEDIUM", "HIGH"]
_BAND_SEVERITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class EnsemblePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str
    policy_version: str
    weight_rule: float = Field(ge=0)
    weight_gbm: float = Field(ge=0)
    weight_anomaly: float = Field(ge=0)
    weight_graph: float = Field(ge=0)
    # rule_score_contribution (RuleEvaluationResult.score_contribution) is
    # an unbounded sum of fired SCORE_CONTRIBUTING rules -- rule_score_cap
    # (Phase 5 decision 7) bounds it to [0, cap] before it is normalized
    # into the weighted sum, exactly like every other component being in
    # [0, 1].
    rule_score_cap: float = Field(gt=0)
    high_threshold: float = Field(ge=0, le=1)
    medium_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "EnsemblePolicy":
        if self.medium_threshold > self.high_threshold:
            raise ValueError("medium_threshold must be <= high_threshold")
        return self


def compute_operational_priority_score(
    *,
    rule_score_contribution: float,
    calibrated_gbm_probability: float,
    anomaly_score: float,
    graph_risk_score: float,
    policy: EnsemblePolicy,
) -> float:
    """Guide section 16's exact formula, with the rule_score_cap boundary
    (Phase 5 decision 7) applied first: capped_rule_score =
    min(max(rule_score_contribution, 0), rule_score_cap), then normalized
    to [0, 1] by dividing by the cap before weighting. The final weighted
    sum is clipped to [0, 1] as a last safety net regardless of whether
    the configured weights sum to exactly 1.0."""
    capped_rule_score = min(max(rule_score_contribution, 0.0), policy.rule_score_cap)
    normalized_rule = capped_rule_score / policy.rule_score_cap

    score = (
        policy.weight_rule * normalized_rule
        + policy.weight_gbm * calibrated_gbm_probability
        + policy.weight_anomaly * anomaly_score
        + policy.weight_graph * graph_risk_score
    )
    return float(min(max(score, 0.0), 1.0))


def max_priority_band(a: PriorityBand, b: PriorityBand) -> PriorityBand:
    return a if _BAND_SEVERITY[a] >= _BAND_SEVERITY[b] else b


def band_priority(
    score: float,
    policy: EnsemblePolicy,
    *,
    mandatory_review_hit: bool,
    provider_status: str,
    minimum_priority_band: Optional[PriorityBand],
) -> PriorityBand:
    """HIGH if mandatory_review_hit. Otherwise, if provider_status is
    UNAVAILABLE/ERROR, the band is floored at minimum_priority_band (never
    lower -- max_priority_band, not an unconditional override). Otherwise,
    thresholds from policy. LOW is a real, retained band -- never dropped
    (guide section 18). This function covers ONLY the rule provider's own
    floor, exactly per guide section 16's original signature; the
    orchestrator applies additional GBM/anomaly/graph floors on top (Phase
    5 decision 6) via this same max_priority_band() helper."""
    if mandatory_review_hit:
        return "HIGH"
    if score >= policy.high_threshold:
        band: PriorityBand = "HIGH"
    elif score >= policy.medium_threshold:
        band = "MEDIUM"
    else:
        band = "LOW"

    if provider_status in ("UNAVAILABLE", "ERROR") and minimum_priority_band is not None:
        band = max_priority_band(band, minimum_priority_band)
    return band


@functools.lru_cache
def load_ensemble_policy(channel: str) -> EnsemblePolicy:
    path = ENSEMBLE_POLICY_CONFIG_DIR / f"ensemble_policy_{channel}.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EnsemblePolicy.model_validate(raw)
