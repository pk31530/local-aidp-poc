"""The scoring orchestrator (guide section 4/25 Phase 5) -- THE single
integration point sequencing context validation -> feature adapter -> rule
enrichment -> GBM -> anomaly -> graph -> ensemble policy -> priority band
-> reason codes -> an evidence-ready result object. No other code path may
duplicate this sequencing logic.

`score_source_alert()` is pure: no database, MLflow, or file I/O; no
alert/evidence persistence (Phase 6 builds that). The existing
`SourceAlertContext` is passed through unchanged and read-only -- this
function never creates or mutates one.

Component-failure policy (Phase 5 decision 6), applied on top of
`band_priority()`'s own rule-provider floor:
- Rule failure: RuleProvider's own result already floors at MEDIUM
  (guide section 10) -- band_priority() handles this directly.
- GBM/primary-model failure: never invent a probability. Record
  `calibrated_gbm_probability=None`, mark `degraded=True`, and force the
  band to at least HIGH -- a missing primary signal must never be allowed
  to look like a confident low-risk score.
- Anomaly or graph failure: the remaining valid components still produce
  a score (their own weights are simply not overridden -- no
  renormalization of the other weights), mark `degraded=True`, force the
  band to at least MEDIUM.
- Multiple failures: floors are applied independently and each can only
  raise the band (`max_priority_band`), so the combined result is always
  the single highest required floor.
- The existing source alert is never dropped: this function always
  returns a `ScoredAlert`, even when every non-rule component fails.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from src.fraud_intel.ensemble.policy import (
    EnsemblePolicy,
    band_priority,
    compute_operational_priority_score,
    max_priority_band,
)
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import (
    EntityKey,
    GraphPolicy,
    ResolvedFraudEntityEvidence,
    beneficiary_fan_in_count,
    build_entity_graph,
    compute_graph_risk_score,
    graph_counterparty_key,
    shared_device_across_distinct_customers_count,
    shortest_path_to_fraud_linked_entity,
    validate_resolved_fraud_evidence,
)
from src.fraud_intel.models.anomaly import AnomalyNormalization, score_anomaly
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.reason_codes.builder import REASON_CODE_VERSION, ReasonCode, build_reason_codes
from src.fraud_intel.registry import get_channel_adapter
from src.fraud_intel.rules.provider import RuleEvaluationResult, RuleProvider

GBM_SCORING_FAILED = "GBM_SCORING_FAILED"
ANOMALY_SCORING_FAILED = "ANOMALY_SCORING_FAILED"
GRAPH_SCORING_FAILED = "GRAPH_SCORING_FAILED"

GBM_FAILURE_MINIMUM_BAND = "HIGH"
DEGRADED_COMPONENT_MINIMUM_BAND = "MEDIUM"


@dataclass(frozen=True)
class LoadedChannelBundle:
    """Runtime container holding the ACTUAL loaded model/artifact objects
    -- distinct from `ChannelModelBundleRecord` (Phase 4), which only
    holds version *strings*. Loading these objects from MLflow using those
    version strings is explicitly not the orchestrator's job (a separate,
    not-yet-built concern, analogous to v1.1's `load_champion_model()`).

    Phase 6 decision 3: this bundle also carries its own PINNED provenance
    -- bundle_id and every component/policy version string the bundle was
    registered with. `alert_evidence`'s provenance columns
    (src.fraud_intel.alerts.queue) are populated from THESE fields, not
    from the live ensemble_policy/graph_policy objects passed to
    score_source_alert() -- the bundle's own recorded versions represent
    "what this bundle was validated/trained against," which may diverge
    from "what policy is live right now" if a policy YAML changes without
    a retrain. This also means every one of these fields is available even
    when scoring fails catastrophically (they come from this argument, not
    from any scoring OUTPUT), which is exactly why a catastrophic-failure
    evidence row can still have a fully populated bundle/policy provenance
    trail.
    """

    channel: str
    bundle_id: int
    bundle_version: int
    gbm_model: Any
    lr_model: Any
    anomaly_model: Any
    anomaly_normalization: AnomalyNormalization
    preprocessor: ChannelPreprocessor
    gbm_model_version: str
    lr_model_version: str
    anomaly_model_version: str
    preprocessing_artifact_version: str
    feature_schema_version: str
    rule_set_version: str
    graph_policy_version: str
    ensemble_policy_version: str
    reason_code_version: str


class ComponentStatus(BaseModel):
    """Typed technical/audit provenance (Phase 5 decision 6) -- distinct
    from the analyst-facing reason codes. Never carries raw exception
    text; `error_code` is always one of a small set of stable constants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str  # "OK" | "ERROR"
    error_code: Optional[str] = None


class ScoredAlert(BaseModel):
    """The orchestrator's complete, typed, evidence-ready result. No
    persistence anywhere -- verified structurally (no psycopg2 import, no
    SQL, in this module)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: uuid.UUID
    source_alert_id: uuid.UUID
    score_execution_id: uuid.UUID

    rule_result: RuleEvaluationResult
    calibrated_gbm_probability: Optional[float]
    lr_probability: Optional[float]  # shadow only -- logged, never used for score/band
    anomaly_score: Optional[float]
    graph_risk_score: Optional[float]

    operational_priority_score: float
    priority_band: str

    reason_codes: list[ReasonCode]
    degraded: bool
    component_statuses: dict[str, ComponentStatus]

    channel_model_bundle_version: int
    feature_schema_version: str
    ensemble_policy_version: str
    graph_policy_version: str
    reason_code_version: str
    rule_set_version: str

    config_hash: Optional[str]
    git_sha: Optional[str]
    event_time: datetime
    scored_at: datetime


def _gbm_feature_contributions(gbm_model: Any, X: Sequence[Sequence[float]], feature_names: Sequence[str]) -> list[tuple[str, float]]:
    """XGBoost's native Booster.predict(pred_contribs=True) -- no new
    dependency (guide section 17). Best-effort: a model without a real
    XGBoost booster (e.g. a fake in a unit test) simply yields no GBM
    feature-contribution reason codes, which is a graceful degradation of
    explanation detail only, not a scoring failure."""
    try:
        import xgboost as xgb

        booster = gbm_model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(feature_names))
        contributions = booster.predict(dmatrix, pred_contribs=True)
        row = contributions[0][:-1]  # last column is the bias term
        return list(zip(feature_names, [float(v) for v in row]))
    except Exception:
        return []


def score_source_alert(
    *,
    event: FraudEvent,
    source_alert: SourceAlertContext,
    context: FeatureComputationContext,
    bundle: LoadedChannelBundle,
    rule_provider: RuleProvider,
    ensemble_policy: EnsemblePolicy,
    graph_policy: GraphPolicy,
    resolved_fraud_evidence: Sequence[ResolvedFraudEntityEvidence],
    score_execution_id: uuid.UUID,
    config_hash: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> ScoredAlert:
    # Phase 7A: get_channel_adapter() itself is the channel-validity check
    # -- event.channel is a closed 7-value Literal (FraudEvent's own
    # Channel type) and every value is now registered, so this can only
    # ever raise for a channel that was never registered at all, not for a
    # legitimate channel this function merely used to special-case.
    adapter = get_channel_adapter(event.channel)
    if context.current_event.event_id != event.event_id:
        raise ValueError("context.current_event must be the same event passed as `event`")
    if source_alert.event_id != event.event_id:
        raise ValueError("source_alert.event_id must match event.event_id")

    # Guide section 15 / Phase 5 decision 5: raises GraphLeakageError before
    # anything else runs if any evidence item is not strictly prior to
    # this event -- the current event's own label can never leak in.
    fraud_linked_entities = validate_resolved_fraud_evidence(
        resolved_fraud_evidence, current_event_timestamp=event.event_timestamp
    )

    features = adapter.compute_features(context)
    rule_result = rule_provider.evaluate(event=event, source_alert=source_alert, features=features)
    mandatory_review_hit = any(
        rule_result.rule_categories.get(rule_id) == "MANDATORY_REVIEW" for rule_id in rule_result.fired_rule_ids
    )

    component_statuses: dict[str, ComponentStatus] = {
        "rules": ComponentStatus(
            status="OK" if rule_result.provider_status == "OK" else "ERROR",
            error_code=rule_result.provider_error_code,
        )
    }
    degraded = rule_result.provider_status != "OK"

    feature_row = {column: features[column] for column in adapter.feature_columns}

    gbm_probability: Optional[float] = None
    feature_contributions: list[tuple[str, float]] = []
    try:
        X = bundle.preprocessor.transform([feature_row])
        gbm_probability = float(bundle.gbm_model.predict_proba(X)[0][1])
        component_statuses["gbm"] = ComponentStatus(status="OK")
        feature_contributions = _gbm_feature_contributions(bundle.gbm_model, X, adapter.feature_columns)
    except Exception:
        component_statuses["gbm"] = ComponentStatus(status="ERROR", error_code=GBM_SCORING_FAILED)
        degraded = True
        X = None

    # LR shadow -- best-effort, never gates anything, never raises out of
    # this function (a shadow-only failure is not itself a degraded-score
    # condition).
    lr_probability: Optional[float] = None
    try:
        X_lr = X if X is not None else bundle.preprocessor.transform([feature_row])
        lr_probability = float(bundle.lr_model.predict_proba(X_lr)[0][1])
    except Exception:
        lr_probability = None

    anomaly_score: Optional[float] = None
    try:
        X_anomaly = X if X is not None else bundle.preprocessor.transform([feature_row])
        anomaly_score = score_anomaly(bundle.anomaly_model, bundle.anomaly_normalization, X_anomaly)[0]
        component_statuses["anomaly"] = ComponentStatus(status="OK")
    except Exception:
        component_statuses["anomaly"] = ComponentStatus(status="ERROR", error_code=ANOMALY_SCORING_FAILED)
        degraded = True

    graph_risk_score: Optional[float] = None
    shared_device_count = 0
    fan_in_count = 0
    shortest_path: Optional[int] = None
    try:
        graph = build_entity_graph(
            historical_events=context.historical_events, current_event_timestamp=event.event_timestamp,
            policy=graph_policy, entity_extractor=adapter.extract_entities,
        )
        customer_key: EntityKey = ("customer", event.customer_id)
        device_key: Optional[EntityKey] = ("device", event.device_id) if event.device_id else None
        # Phase 7A: generalizes what was online_banking's own hardcoded
        # `isinstance(event.channel_payload, OnlineBankingPayload) and
        # event.channel_payload.target_account` check -- every channel's
        # own extract_entities() already knows which (if any) entity plays
        # the counterparty role (recipient/beneficiary/card/atm/
        # check_payee); ACH deliberately has none (Phase 7A decision 3).
        recipient_key: Optional[EntityKey] = graph_counterparty_key(adapter.extract_entities(event))

        shared_device_count = shared_device_across_distinct_customers_count(graph, device_key) if device_key else 0
        fan_in_count = beneficiary_fan_in_count(graph, recipient_key) if recipient_key else 0
        shortest_path = shortest_path_to_fraud_linked_entity(graph, customer_key, fraud_linked_entities)
        graph_risk_score = compute_graph_risk_score(
            graph=graph,
            customer_key=customer_key,
            device_key=device_key,
            recipient_key=recipient_key,
            fraud_linked_entities=fraud_linked_entities,
            policy=graph_policy,
        )
        component_statuses["graph"] = ComponentStatus(status="OK")
    except Exception:
        component_statuses["graph"] = ComponentStatus(status="ERROR", error_code=GRAPH_SCORING_FAILED)
        degraded = True

    # Degraded components contribute 0 (neutral) -- their own weight is
    # never redistributed to the remaining components (no renormalization,
    # per Phase 5 decision 6).
    operational_priority_score = compute_operational_priority_score(
        rule_score_contribution=rule_result.score_contribution,
        calibrated_gbm_probability=gbm_probability if gbm_probability is not None else 0.0,
        anomaly_score=anomaly_score if anomaly_score is not None else 0.0,
        graph_risk_score=graph_risk_score if graph_risk_score is not None else 0.0,
        policy=ensemble_policy,
    )

    band = band_priority(
        operational_priority_score,
        ensemble_policy,
        mandatory_review_hit=mandatory_review_hit,
        provider_status=rule_result.provider_status,
        minimum_priority_band=rule_result.minimum_priority_band,
    )
    if component_statuses["gbm"].status == "ERROR":
        band = max_priority_band(band, GBM_FAILURE_MINIMUM_BAND)
    if component_statuses["anomaly"].status == "ERROR" or component_statuses["graph"].status == "ERROR":
        band = max_priority_band(band, DEGRADED_COMPONENT_MINIMUM_BAND)

    reason_codes = build_reason_codes(
        rule_result=rule_result,
        source_alert=source_alert,
        feature_contributions=feature_contributions,
        anomaly_score=anomaly_score if anomaly_score is not None else 0.0,
        graph_shared_device_count=shared_device_count,
        graph_fan_in_count=fan_in_count,
        graph_shortest_path=shortest_path,
        component_statuses={name: {"status": cs.status, "error_code": cs.error_code} for name, cs in component_statuses.items()},
    )

    return ScoredAlert(
        event_id=event.event_id,
        source_alert_id=source_alert.source_alert_id,
        score_execution_id=score_execution_id,
        rule_result=rule_result,
        calibrated_gbm_probability=gbm_probability,
        lr_probability=lr_probability,
        anomaly_score=anomaly_score,
        graph_risk_score=graph_risk_score,
        operational_priority_score=operational_priority_score,
        priority_band=band,
        reason_codes=reason_codes,
        degraded=degraded,
        component_statuses=component_statuses,
        channel_model_bundle_version=bundle.bundle_version,
        feature_schema_version=bundle.feature_schema_version,
        ensemble_policy_version=ensemble_policy.policy_version,
        graph_policy_version=graph_policy.graph_policy_version,
        reason_code_version=REASON_CODE_VERSION,
        rule_set_version=rule_result.rule_set_version,
        config_hash=config_hash,
        git_sha=git_sha,
        event_time=event.event_timestamp,
        scored_at=datetime.now(timezone.utc),
    )
