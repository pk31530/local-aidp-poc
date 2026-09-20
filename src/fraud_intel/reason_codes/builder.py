"""Explainable, versioned reason-code contract (guide section 17).

Reason codes are explanatory evidence, not causal findings, adverse-action
reasons, or regulatory conclusions. A reason code states which signal
contributed to a score (e.g. "GBM feature X was high," "shared-device ring
detected") -- it is not a determination that the signal *caused* fraud,
not an adverse-action reason under any regulation, and not a compliance or
legal conclusion of any kind (guide sections 17, 23). This module never
references scenario_id, synthetic-label fields, or analyst_disposition --
verified structurally by tests/unit/test_fraud_intel_reason_codes.py.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from src.control_plane.provenance import truncate_text
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.rules.provider import RuleEvaluationResult

REASON_CODE_VERSION = "v1"
MAX_TEXT_LENGTH = 200
MAX_GBM_CONTRIBUTION_CODES = 5

ANOMALY_FLAG_THRESHOLD_DEFAULT = 0.7
GRAPH_SHARED_DEVICE_FLAG_THRESHOLD = 2
GRAPH_FAN_IN_FLAG_THRESHOLD = 3
GRAPH_NEAR_FRAUD_LINKED_MAX_HOPS = 2

Layer = Literal["rule", "gbm", "anomaly", "graph", "orchestrator"]
# "orchestrator" (Phase 6): reserved for a catastrophic-scoring-failure
# reason code (SCORING_UNAVAILABLE) -- never emitted by build_reason_codes()
# itself, only constructed directly by src.fraud_intel.alerts.queue when
# score_source_alert() raises before returning any ScoredAlert at all.
Severity = Literal["informational", "contributing", "mandatory"]

_RULE_SEVERITY_BY_CATEGORY = {
    "MANDATORY_REVIEW": "mandatory",
    "SCORE_CONTRIBUTING": "contributing",
    "INFORMATIONAL": "informational",
}


class ReasonCode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    text: str
    layer: Layer
    severity: Severity


def _safe_text(text: str) -> str:
    return truncate_text(text, MAX_TEXT_LENGTH) or ""


def _upstream_reason_codes(source_alert: SourceAlertContext) -> list[ReasonCode]:
    return [
        ReasonCode(
            code=f"UPSTREAM_{code}",
            text=_safe_text(f"Upstream (simulated) source alert reason: {code}."),
            layer="rule",
            severity="informational",
        )
        for code in source_alert.source_alert_reason_codes
    ]


def _rule_reason_codes(rule_result: RuleEvaluationResult) -> list[ReasonCode]:
    codes = []
    for rule_id, reason_code in zip(rule_result.fired_rule_ids, rule_result.reason_codes):
        category = rule_result.rule_categories.get(rule_id, "INFORMATIONAL")
        codes.append(
            ReasonCode(
                code=reason_code,
                text=_safe_text(f"Rule {rule_id} fired ({category})."),
                layer="rule",
                severity=_RULE_SEVERITY_BY_CATEGORY[category],
            )
        )
    if rule_result.provider_status != "OK":
        codes.append(
            ReasonCode(
                code="RULE_PROVIDER_UNAVAILABLE",
                text=_safe_text("The rule provider was unavailable for this evaluation."),
                layer="rule",
                severity="informational",
            )
        )
    return codes


def _gbm_reason_codes(feature_contributions: Sequence[tuple[str, float]]) -> list[ReasonCode]:
    """`feature_contributions`: (feature_name, contribution_value) pairs
    for the scored row -- computing that array (XGBoost's
    Booster.predict(pred_contribs=True), guide section 17) is the
    orchestrator's job; this function only formats the already-computed
    top contributors into reason codes."""
    ranked = sorted(feature_contributions, key=lambda pair: (-abs(pair[1]), pair[0]))[:MAX_GBM_CONTRIBUTION_CODES]
    codes = []
    for feature_name, contribution in ranked:
        direction = "increased" if contribution > 0 else "decreased"
        codes.append(
            ReasonCode(
                code=f"GBM_FEATURE_{feature_name.upper()}",
                text=_safe_text(f"{feature_name} {direction} the fraud score (contribution {contribution:+.4f})."),
                layer="gbm",
                severity="contributing",
            )
        )
    return codes


def _anomaly_reason_codes(anomaly_score: float, *, threshold: float = ANOMALY_FLAG_THRESHOLD_DEFAULT) -> list[ReasonCode]:
    if anomaly_score < threshold:
        return []
    return [
        ReasonCode(
            code="ANOMALY_SCORE_ELEVATED",
            text=_safe_text(f"Anomaly score {anomaly_score:.2f} exceeded the configured threshold {threshold:.2f}."),
            layer="anomaly",
            severity="contributing",
        )
    ]


def _graph_reason_codes(*, shared_device_count: int, fan_in_count: int, shortest_path: int | None) -> list[ReasonCode]:
    codes = []
    if shared_device_count >= GRAPH_SHARED_DEVICE_FLAG_THRESHOLD:
        codes.append(
            ReasonCode(
                code="SHARED_DEVICE_RING",
                text=_safe_text(f"Device linked to {shared_device_count} distinct customers."),
                layer="graph",
                severity="contributing",
            )
        )
    if fan_in_count >= GRAPH_FAN_IN_FLAG_THRESHOLD:
        codes.append(
            ReasonCode(
                code="RAPID_RECIPIENT_FAN_IN",
                text=_safe_text(f"Recipient linked to {fan_in_count} distinct senders."),
                layer="graph",
                severity="contributing",
            )
        )
    if shortest_path is not None and shortest_path <= GRAPH_NEAR_FRAUD_LINKED_MAX_HOPS:
        codes.append(
            ReasonCode(
                code="NEAR_FRAUD_LINKED_ENTITY",
                text=_safe_text(f"{shortest_path} hop(s) from a resolved-fraud-linked entity."),
                layer="graph",
                severity="contributing",
            )
        )
    return codes


def build_reason_codes(
    *,
    rule_result: RuleEvaluationResult,
    source_alert: SourceAlertContext,
    feature_contributions: Sequence[tuple[str, float]],
    anomaly_score: float,
    graph_shared_device_count: int,
    graph_fan_in_count: int,
    graph_shortest_path: int | None,
    component_statuses: Mapping[str, Mapping[str, Any]],
) -> list[ReasonCode]:
    """Deterministic layer order (rule -> gbm -> anomaly -> graph), each
    layer's own deterministic sub-order, then a first-occurrence-wins
    dedup by code. A degraded component (per component_statuses) emits its
    own *_UNAVAILABLE informational code instead of that layer's normal
    codes."""
    codes: list[ReasonCode] = []
    codes.extend(_upstream_reason_codes(source_alert))
    codes.extend(_rule_reason_codes(rule_result))

    if component_statuses.get("gbm", {}).get("status") == "OK":
        codes.extend(_gbm_reason_codes(feature_contributions))
    else:
        codes.append(
            ReasonCode(code="GBM_UNAVAILABLE", text="Primary model scoring was unavailable for this alert.", layer="gbm", severity="informational")
        )

    if component_statuses.get("anomaly", {}).get("status") == "OK":
        codes.extend(_anomaly_reason_codes(anomaly_score))
    else:
        codes.append(
            ReasonCode(code="ANOMALY_UNAVAILABLE", text="Anomaly scoring was unavailable for this alert.", layer="anomaly", severity="informational")
        )

    if component_statuses.get("graph", {}).get("status") == "OK":
        codes.extend(
            _graph_reason_codes(
                shared_device_count=graph_shared_device_count,
                fan_in_count=graph_fan_in_count,
                shortest_path=graph_shortest_path,
            )
        )
    else:
        codes.append(
            ReasonCode(code="GRAPH_UNAVAILABLE", text="Graph scoring was unavailable for this alert.", layer="graph", severity="informational")
        )

    seen: set[str] = set()
    deduped: list[ReasonCode] = []
    for code in codes:
        if code.code in seen:
            continue
        seen.add(code.code)
        deduped.append(code)
    return deduped
