"""Shadow-candidate evaluation (Phase 7A corrective pass): in-memory
candidate scoring plus a pure, typed comparison between an OPERATIONAL
bundle's real scores and a CANDIDATE bundle's shadow scores over the
same source-alert population.

`score_candidate_shadow()` is the ONLY function here that does real work
beyond comparison -- it calls the existing PURE
src.fraud_intel.scoring.orchestrator.score_source_alert() (no database,
MLflow, or persistence inside that function either) once per alert, in
memory, and NEVER src.fraud_intel.alerts.queue.score_and_record_alert().
This module never contacts a database or MLflow itself, and never
imports src.fraud_intel.alerts.queue or src.fraud_intel.models.promotion
-- structurally verified by tests/unit/test_fraud_intel_shadow_candidate.py.
A candidate's scores are therefore held in memory only, as
`CandidateShadowScore` objects; nothing here can, even in principle,
create a fraud_alerts/alert_evidence row, influence the analyst queue or
a disposition, or trigger a promotion -- there is no code path here that
writes anything, and no function that calls
src.fraud_intel.models.promotion.promote_bundle. A per-alert scoring
failure is captured as a `CandidateShadowScoringError` and skipped --
never allowed to raise out of the batch or touch that alert's real,
already-persisted operational evidence.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, brier_score_loss

from src.control_plane.provenance import redact_credentials, truncate_text
from src.fraud_intel.ensemble.policy import EnsemblePolicy
from src.fraud_intel.evaluation.capacity import CountCapacity, FractionCapacity, resolve_capacity_count
from src.fraud_intel.evaluation.cross_channel import AlertOutcome, MetricValue, PriorityBand
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy, ResolvedFraudEntityEvidence
from src.fraud_intel.rules.provider import RuleProvider
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle, score_source_alert

DEFAULT_MIN_ROWS_FOR_SCORE_DIFF_STDEV = 2
MAX_SHADOW_ERROR_MESSAGE_LENGTH = 500


@dataclass(frozen=True)
class CandidateScoringInput:
    """Everything needed to re-score one already-resolved alert against a
    CANDIDATE bundle via the pure score_source_alert() path -- the exact
    same shape as src.fraud_intel.scoring.dispatch.PendingScoringItem,
    defined separately here (not imported from dispatch.py) so this
    module never pulls in dispatch.py's own import of
    src.fraud_intel.alerts.queue, even transitively."""

    source_alert_id: uuid.UUID
    event: FraudEvent
    source_alert: SourceAlertContext
    context: FeatureComputationContext
    resolved_fraud_evidence: tuple[ResolvedFraudEntityEvidence, ...]


class CandidateShadowScore(BaseModel):
    """One CANDIDATE bundle's shadow score for a source alert that also
    has a real OPERATIONAL `AlertOutcome` row. Never written to
    fraud_alerts/alert_evidence -- this type exists purely as this
    module's input, not a persisted record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_alert_id: uuid.UUID
    channel: str
    event_timestamp: datetime
    candidate_priority_score: float = Field(ge=0, le=1)
    candidate_priority_band: PriorityBand


class CandidateShadowScoringError(BaseModel):
    """A safe, typed record of one alert's candidate re-scoring failure --
    never the raw exception, never propagated out of score_candidate_shadow()
    to abort the whole batch, and never written anywhere: the alert's
    real, already-persisted OPERATIONAL alert/evidence is completely
    untouched by this failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_alert_id: uuid.UUID
    error_type: str
    message: str


def score_candidate_shadow(
    scoring_inputs: Sequence[CandidateScoringInput],
    *,
    bundle: LoadedChannelBundle,
    rule_provider: RuleProvider,
    ensemble_policy: EnsemblePolicy,
    graph_policy: GraphPolicy,
    config_hash: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> tuple[list[CandidateShadowScore], list[CandidateShadowScoringError]]:
    """Scores every input through the PURE
    src.fraud_intel.scoring.orchestrator.score_source_alert() path (never
    src.fraud_intel.alerts.queue.score_and_record_alert()) against the
    CANDIDATE `bundle`, entirely in memory -- no fraud_alerts/alert_evidence
    row is ever created, no disposition is ever touched, and nothing here
    can promote the candidate. A fresh score_execution_id is minted per
    alert purely for score_source_alert()'s own audit-trail field; it is
    never persisted by this function. One alert's scoring failure is
    captured as a CandidateShadowScoringError and the batch continues --
    the same "one bad alert must not block the rest" principle
    src.fraud_intel.scoring.dispatch.score_channel() already applies to
    real operational scoring."""
    scores: list[CandidateShadowScore] = []
    errors: list[CandidateShadowScoringError] = []
    for item in scoring_inputs:
        try:
            scored = score_source_alert(
                event=item.event,
                source_alert=item.source_alert,
                context=item.context,
                bundle=bundle,
                rule_provider=rule_provider,
                ensemble_policy=ensemble_policy,
                graph_policy=graph_policy,
                resolved_fraud_evidence=item.resolved_fraud_evidence,
                score_execution_id=uuid.uuid4(),
                config_hash=config_hash,
                git_sha=git_sha,
            )
            scores.append(
                CandidateShadowScore(
                    source_alert_id=item.source_alert_id, channel=item.event.channel, event_timestamp=item.event.event_timestamp,
                    candidate_priority_score=scored.operational_priority_score, candidate_priority_band=scored.priority_band,
                )
            )
        except Exception as exc:
            safe_message = truncate_text(redact_credentials(str(exc)), MAX_SHADOW_ERROR_MESSAGE_LENGTH) or ""
            errors.append(
                CandidateShadowScoringError(source_alert_id=item.source_alert_id, error_type=type(exc).__name__, message=safe_message)
            )
    return scores, errors


class ShadowCandidateComparisonResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: str
    operational_bundle_id: int
    operational_bundle_version: int
    candidate_bundle_id: int
    candidate_bundle_version: int

    matched_source_alert_count: int
    operational_only_source_alert_count: int
    candidate_only_source_alert_count: int

    capacity_mode: str
    capacity_value: float
    recall_target: float

    precision_at_capacity_delta: MetricValue
    recall_at_capacity_delta: MetricValue
    workload_reduction_at_recall_target_delta: MetricValue
    pr_auc_delta: MetricValue
    brier_score_delta: MetricValue

    priority_band_migration_counts: dict[str, dict[str, int]]
    score_difference_mean: MetricValue
    score_difference_stdev: MetricValue
    ranking_disagreement_at_capacity: MetricValue

    fraud_captured_operational: int
    fraud_captured_candidate: int
    fraud_gained_by_candidate: int
    fraud_lost_by_candidate: int

    # Deliberately no field, method, or note here suggests a promotion
    # decision -- this result is comparison data only. Promotion remains
    # exclusively src.fraud_intel.models.promotion.promote_bundle(), a
    # manually-triggered, separately-approved action this module never
    # calls and does not import.


def _rank_ids(rows: Sequence[tuple], score_index: int, ts_index: int, id_index: int) -> list:
    """Deterministic ranking, same rule as cross_channel._rank(): score
    descending, event_timestamp ascending, source_alert_id ascending."""
    return sorted(rows, key=lambda r: (-r[score_index], r[ts_index], str(r[id_index])))


def _minimum_alerts_for_recall_target(ranked_fraud_flags: Sequence[bool], total_fraud: int, recall_target: float) -> Optional[int]:
    cumulative = 0
    for i, is_fraud in enumerate(ranked_fraud_flags, start=1):
        if is_fraud:
            cumulative += 1
        if cumulative / total_fraud >= recall_target:
            return i
    return None


def _scalar_metrics(
    *,
    paired: Sequence[tuple],  # (source_alert_id, event_timestamp, score, is_fraud)
    total: int,
    total_fraud: int,
    single_class: bool,
    capacity,
    recall_target: float,
) -> dict:
    """precision@capacity / recall@capacity / workload_reduction_at_recall_target
    / pr_auc / brier for ONE score column over the matched population --
    shared logic for both the operational and the candidate side, so the
    delta computation compares apples to apples."""
    if total == 0:
        empty = MetricValue.non_computable("non_computable_empty", "no matched rows")
        return {"precision_at_capacity": empty, "recall_at_capacity": empty, "workload_reduction": empty, "pr_auc": empty, "brier": empty, "reviewed_ids": set(), "fraud_captured": 0}
    if single_class:
        sc = MetricValue.non_computable("non_computable_single_class", "matched population has only one resolved-label class")
        # precision@capacity/recall@capacity/workload_reduction may still
        # be partially computable depending on which class is missing --
        # kept conservatively non-computable here since the delta itself
        # (this function's caller) needs both sides on equal footing.
        ranked = _rank_ids([(r[0], r[1], r[2], r[3]) for r in paired], 2, 1, 0)
        alerts_reviewed = resolve_capacity_count(capacity, total)
        reviewed_ids = {r[0] for r in ranked[:alerts_reviewed]}
        fraud_captured = sum(1 for r in ranked[:alerts_reviewed] if r[3])
        return {"precision_at_capacity": sc, "recall_at_capacity": sc, "workload_reduction": sc, "pr_auc": sc, "brier": sc, "reviewed_ids": reviewed_ids, "fraud_captured": fraud_captured}

    ranked = _rank_ids([(r[0], r[1], r[2], r[3]) for r in paired], 2, 1, 0)
    alerts_reviewed = resolve_capacity_count(capacity, total)
    reviewed = ranked[:alerts_reviewed]
    reviewed_ids = {r[0] for r in reviewed}
    fraud_captured = sum(1 for r in reviewed if r[3])

    precision_at_capacity = MetricValue.ok(fraud_captured / alerts_reviewed) if alerts_reviewed > 0 else MetricValue.non_computable(
        "non_computable_empty", "capacity resolves to zero alerts reviewed"
    )
    recall_at_capacity = MetricValue.ok(fraud_captured / total_fraud) if alerts_reviewed > 0 else MetricValue.non_computable(
        "non_computable_empty", "capacity resolves to zero alerts reviewed"
    )

    fraud_flags_ranked = [r[3] for r in ranked]
    min_alerts = _minimum_alerts_for_recall_target(fraud_flags_ranked, total_fraud, recall_target)
    if min_alerts is None:
        workload_reduction = MetricValue.non_computable(
            "non_computable_unreachable_target", f"recall target {recall_target} not reached even reviewing all {total} alerts"
        )
    else:
        workload_reduction = MetricValue.ok(1 - (min_alerts / total))

    y_true = [1 if r[3] else 0 for r in paired]
    y_score = [r[2] for r in paired]
    pr_auc = MetricValue.ok(float(average_precision_score(y_true, y_score)))
    brier = MetricValue.ok(float(brier_score_loss(y_true, y_score)))

    return {
        "precision_at_capacity": precision_at_capacity, "recall_at_capacity": recall_at_capacity,
        "workload_reduction": workload_reduction, "pr_auc": pr_auc, "brier": brier,
        "reviewed_ids": reviewed_ids, "fraud_captured": fraud_captured,
    }


def _delta(candidate: MetricValue, operational: MetricValue) -> MetricValue:
    if candidate.status != "ok" or operational.status != "ok":
        # Prefer surfacing the candidate side's reason when both are
        # non-computable for different reasons; otherwise whichever is
        # non-computable.
        non_ok = candidate if candidate.status != "ok" else operational
        return MetricValue.non_computable(non_ok.status, non_ok.reason or "metric not computable on one or both sides")
    return MetricValue.ok(candidate.value - operational.value)


def compare_operational_vs_candidate(
    channel: str,
    operational_outcomes: Sequence[AlertOutcome],
    candidate_scores: Sequence[CandidateShadowScore],
    *,
    operational_bundle_id: int,
    operational_bundle_version: int,
    candidate_bundle_id: int,
    candidate_bundle_version: int,
    capacity,
    recall_target: float,
) -> ShadowCandidateComparisonResult:
    if not (0 < recall_target <= 1):
        raise ValueError(f"recall_target must be in (0, 1], got {recall_target}")
    for outcome in operational_outcomes:
        if outcome.channel != channel:
            raise ValueError(f"operational outcome for channel {outcome.channel!r} does not match requested channel {channel!r}")
    for score in candidate_scores:
        if score.channel != channel:
            raise ValueError(f"candidate score for channel {score.channel!r} does not match requested channel {channel!r}")

    operational_by_id = {o.source_alert_id: o for o in operational_outcomes}
    candidate_by_id = {c.source_alert_id: c for c in candidate_scores}
    matched_ids = set(operational_by_id) & set(candidate_by_id)
    operational_only = set(operational_by_id) - set(candidate_by_id)
    candidate_only = set(candidate_by_id) - set(operational_by_id)

    total = len(matched_ids)
    matched_pairs = [(sid, operational_by_id[sid], candidate_by_id[sid]) for sid in matched_ids]
    total_fraud = sum(1 for _, o, _ in matched_pairs if o.resolved_label == "RESOLVED_FRAUD")
    single_class = total > 0 and (total_fraud == 0 or total_fraud == total)

    if total == 0:
        empty = MetricValue.non_computable("non_computable_empty", "operational and candidate populations share no source_alert_id")
        return ShadowCandidateComparisonResult(
            channel=channel, operational_bundle_id=operational_bundle_id, operational_bundle_version=operational_bundle_version,
            candidate_bundle_id=candidate_bundle_id, candidate_bundle_version=candidate_bundle_version,
            matched_source_alert_count=0, operational_only_source_alert_count=len(operational_only),
            candidate_only_source_alert_count=len(candidate_only),
            capacity_mode=capacity.mode, capacity_value=float(capacity.value), recall_target=recall_target,
            precision_at_capacity_delta=empty, recall_at_capacity_delta=empty, workload_reduction_at_recall_target_delta=empty,
            pr_auc_delta=empty, brier_score_delta=empty, priority_band_migration_counts={},
            score_difference_mean=empty, score_difference_stdev=empty, ranking_disagreement_at_capacity=empty,
            fraud_captured_operational=0, fraud_captured_candidate=0, fraud_gained_by_candidate=0, fraud_lost_by_candidate=0,
        )

    operational_paired = [(sid, o.event_timestamp, o.operational_priority_score, o.resolved_label == "RESOLVED_FRAUD") for sid, o, c in matched_pairs]
    candidate_paired = [(sid, c.event_timestamp, c.candidate_priority_score, o.resolved_label == "RESOLVED_FRAUD") for sid, o, c in matched_pairs]

    operational_metrics = _scalar_metrics(paired=operational_paired, total=total, total_fraud=total_fraud, single_class=single_class, capacity=capacity, recall_target=recall_target)
    candidate_metrics = _scalar_metrics(paired=candidate_paired, total=total, total_fraud=total_fraud, single_class=single_class, capacity=capacity, recall_target=recall_target)

    precision_at_capacity_delta = _delta(candidate_metrics["precision_at_capacity"], operational_metrics["precision_at_capacity"])
    recall_at_capacity_delta = _delta(candidate_metrics["recall_at_capacity"], operational_metrics["recall_at_capacity"])
    workload_reduction_delta = _delta(candidate_metrics["workload_reduction"], operational_metrics["workload_reduction"])
    pr_auc_delta = _delta(candidate_metrics["pr_auc"], operational_metrics["pr_auc"])
    brier_score_delta = _delta(candidate_metrics["brier"], operational_metrics["brier"])

    # ---- priority-band migration counts (always computable, even single-class) ----
    migration: dict[str, dict[str, int]] = {}
    for _, o, c in matched_pairs:
        migration.setdefault(o.priority_band, {}).setdefault(c.candidate_priority_band, 0)
        migration[o.priority_band][c.candidate_priority_band] += 1

    # ---- score-difference distribution ----
    diffs = [c.candidate_priority_score - o.operational_priority_score for _, o, c in matched_pairs]
    score_difference_mean = MetricValue.ok(sum(diffs) / len(diffs))
    if len(diffs) >= DEFAULT_MIN_ROWS_FOR_SCORE_DIFF_STDEV:
        mean = sum(diffs) / len(diffs)
        variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        score_difference_stdev = MetricValue.ok(variance**0.5)
    else:
        score_difference_stdev = MetricValue.non_computable("non_computable_empty", "fewer than 2 matched rows -- stdev undefined")

    # ---- ranking disagreement at capacity: 1 - Jaccard overlap of the two "reviewed" sets ----
    operational_reviewed = operational_metrics["reviewed_ids"]
    candidate_reviewed = candidate_metrics["reviewed_ids"]
    if not operational_reviewed and not candidate_reviewed:
        ranking_disagreement = MetricValue.non_computable("non_computable_empty", "capacity resolves to zero alerts reviewed")
    else:
        union = operational_reviewed | candidate_reviewed
        intersection = operational_reviewed & candidate_reviewed
        jaccard = len(intersection) / len(union) if union else 1.0
        ranking_disagreement = MetricValue.ok(1 - jaccard)

    fraud_captured_operational = operational_metrics["fraud_captured"]
    fraud_captured_candidate = candidate_metrics["fraud_captured"]
    operational_fraud_ids = {sid for sid, o, _ in matched_pairs if o.resolved_label == "RESOLVED_FRAUD" and sid in operational_reviewed}
    candidate_fraud_ids = {sid for sid, o, _ in matched_pairs if o.resolved_label == "RESOLVED_FRAUD" and sid in candidate_reviewed}
    fraud_gained_by_candidate = len(candidate_fraud_ids - operational_fraud_ids)
    fraud_lost_by_candidate = len(operational_fraud_ids - candidate_fraud_ids)

    return ShadowCandidateComparisonResult(
        channel=channel, operational_bundle_id=operational_bundle_id, operational_bundle_version=operational_bundle_version,
        candidate_bundle_id=candidate_bundle_id, candidate_bundle_version=candidate_bundle_version,
        matched_source_alert_count=total, operational_only_source_alert_count=len(operational_only),
        candidate_only_source_alert_count=len(candidate_only),
        capacity_mode=capacity.mode, capacity_value=float(capacity.value), recall_target=recall_target,
        precision_at_capacity_delta=precision_at_capacity_delta, recall_at_capacity_delta=recall_at_capacity_delta,
        workload_reduction_at_recall_target_delta=workload_reduction_delta, pr_auc_delta=pr_auc_delta, brier_score_delta=brier_score_delta,
        priority_band_migration_counts=migration, score_difference_mean=score_difference_mean, score_difference_stdev=score_difference_stdev,
        ranking_disagreement_at_capacity=ranking_disagreement,
        fraud_captured_operational=fraud_captured_operational, fraud_captured_candidate=fraud_captured_candidate,
        fraud_gained_by_candidate=fraud_gained_by_candidate, fraud_lost_by_candidate=fraud_lost_by_candidate,
    )
