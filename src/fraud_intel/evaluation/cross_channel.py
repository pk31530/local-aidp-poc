"""Cross-channel evaluation (Phase 7A, guide section 21). Operates ONLY
on the source-alerted, RESOLVED population (`AlertOutcome` rows) -- never
the full synthetic event population, and never an unresolved/pending
alert. No database, MLflow, Docker, or network access anywhere in this
module; every unit test supplies fixture `AlertOutcome` rows directly.

Accuracy is deliberately never the headline metric: `ChannelEvaluationResult`
computes it (as `accuracy`, alongside every other metric) but nothing in
this module's own output ordering, naming, or any summary privileges it
over precision/recall/PR-AUC.

Every metric is a `MetricValue` -- `status="ok"` with a real `value`, or
an explicit `status="non_computable_empty"`/`"non_computable_single_class"`/
`"non_computable_unreachable_target"` with `value=None` and a human-
readable `reason` -- never a bare 0.0/1.0 standing in for "could not be
computed."
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from src.fraud_intel.evaluation.capacity import CountCapacity, FractionCapacity, resolve_capacity_count

PriorityBand = Literal["LOW", "MEDIUM", "HIGH"]
ResolvedLabel = Literal["RESOLVED_FRAUD", "RESOLVED_LEGITIMATE"]

MetricStatus = Literal["ok", "non_computable_empty", "non_computable_single_class", "non_computable_unreachable_target"]


class AlertOutcome(BaseModel):
    """One source-alerted, RESOLVED row -- the only kind of row this
    module ever reads. `baseline_priority_score` is the SAME
    compute_operational_priority_score() formula, called with
    weight_gbm=weight_anomaly=weight_graph=0 on the same EnsemblePolicy
    (guide section 21's "same mechanism, not a separate code path") --
    callers build both scores themselves; this module never re-derives
    either one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_alert_id: uuid.UUID
    channel: str
    event_timestamp: datetime
    operational_priority_score: float = Field(ge=0, le=1)
    baseline_priority_score: float = Field(ge=0, le=1)
    priority_band: PriorityBand
    resolved_label: ResolvedLabel


class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: Optional[float]
    status: MetricStatus
    reason: Optional[str] = None

    @classmethod
    def ok(cls, value: float) -> "MetricValue":
        return cls(value=value, status="ok")

    @classmethod
    def non_computable(cls, status: MetricStatus, reason: str) -> "MetricValue":
        return cls(value=None, status=status, reason=reason)


class RulesOnlyBaselineResult(BaseModel):
    """Guide section 21's "rules-only baseline" -- the SAME mechanism as
    the model score (src.fraud_intel.ensemble.policy.compute_operational_priority_score()
    called with weight_gbm=weight_anomaly=weight_graph=0 on the real
    EnsemblePolicy), never a separately-implemented formula. Callers
    compute `baseline_priority_score` themselves and supply it on every
    AlertOutcome; this module only compares the two score columns.
    Ranked continuous-score metrics only (PR-AUC/ROC-AUC/Brier/
    precision@capacity/recall@capacity) -- the baseline has no discrete
    priority band of its own to derive a threshold-based confusion matrix
    from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_auc: MetricValue
    roc_auc: MetricValue
    brier_score: MetricValue
    precision_at_capacity: MetricValue
    recall_at_capacity: MetricValue


class ChannelEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: str
    total_source_alerts: int
    total_resolved_fraud: int
    total_resolved_legitimate: int

    precision: MetricValue
    recall: MetricValue
    pr_auc: MetricValue
    roc_auc: MetricValue  # secondary, per guide section 21 -- never the headline
    brier_score: MetricValue
    confusion_matrix_at_operational_threshold: dict[str, int]
    per_band_counts: dict[str, dict[str, int]]

    capacity_mode: str
    capacity_value: float
    alerts_reviewed: int
    fraud_captured: int
    precision_at_capacity: MetricValue
    recall_at_capacity: MetricValue

    recall_target: float
    minimum_alerts_required_for_recall_target: Optional[int]
    workload_reduction_at_recall_target: MetricValue

    false_positive_reduction: MetricValue

    rules_only_baseline: RulesOnlyBaselineResult

    # Guide section 21 also names a "clearly-separate, secondary shadow-
    # candidate summary" (comparing the OPERATIONAL bundle against a not-
    # yet-promoted CANDIDATE bundle's shadow scores). No shadow-candidate
    # scoring mechanism exists anywhere in this codebase yet -- inventing
    # one here would mean fabricating comparison data this module cannot
    # actually source. Deliberately omitted and flagged, not silently
    # faked; a real implementation needs a shadow-scoring data source
    # first (Phase 7B+ scope).

    accuracy: MetricValue  # computed but explicitly supplementary -- never headlined


class ChannelSummary(BaseModel):
    """Macro (unweighted mean across channels with a computable value) and
    volume-weighted (weighted by total_source_alerts) summaries for one
    metric name -- channels with a non-"ok" status for that metric are
    excluded from both, never treated as 0.0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    macro: Optional[float]
    volume_weighted: Optional[float]
    channels_included: list[str]
    channels_excluded_non_computable: list[str]


class CrossChannelEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    per_channel: dict[str, ChannelEvaluationResult]
    summaries: dict[str, ChannelSummary]


def _rank(outcomes: Sequence[AlertOutcome]) -> list[AlertOutcome]:
    """Deterministic ranking (Phase 7A decision 6): operational_priority_score
    descending, event_timestamp ascending, source_alert_id ascending."""
    return sorted(
        outcomes,
        key=lambda o: (-o.operational_priority_score, o.event_timestamp, str(o.source_alert_id)),
    )


def _y_true_and_scores(outcomes: Sequence[AlertOutcome]) -> tuple[list[int], list[float]]:
    y_true = [1 if o.resolved_label == "RESOLVED_FRAUD" else 0 for o in outcomes]
    y_score = [o.operational_priority_score for o in outcomes]
    return y_true, y_score


def evaluate_channel(
    channel: str,
    outcomes: Sequence[AlertOutcome],
    *,
    capacity: Union[CountCapacity, FractionCapacity],
    recall_target: float,
) -> ChannelEvaluationResult:
    if not (0 < recall_target <= 1):
        raise ValueError(f"recall_target must be in (0, 1], got {recall_target}")

    total = len(outcomes)
    total_fraud = sum(1 for o in outcomes if o.resolved_label == "RESOLVED_FRAUD")
    total_legit = total - total_fraud
    single_class = total > 0 and (total_fraud == 0 or total_legit == 0)

    ranked = _rank(outcomes)
    alerts_reviewed = resolve_capacity_count(capacity, total)
    reviewed = ranked[:alerts_reviewed]
    fraud_captured = sum(1 for o in reviewed if o.resolved_label == "RESOLVED_FRAUD")

    guard: Optional[MetricValue] = None
    if total == 0:
        guard = MetricValue.non_computable("non_computable_empty", f"channel {channel!r} has no resolved source-alerted rows")
    elif single_class:
        guard = MetricValue.non_computable(
            "non_computable_single_class", f"channel {channel!r} has only one resolved-label class present"
        )

    # ---- confusion matrix / precision / recall / accuracy at the operational threshold ----
    predicted_positive = [o.priority_band in ("MEDIUM", "HIGH") for o in outcomes]
    y_true = [o.resolved_label == "RESOLVED_FRAUD" for o in outcomes]
    if guard is not None:
        precision = recall = accuracy = guard
        cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    else:
        tp = sum(1 for p, t in zip(predicted_positive, y_true) if p and t)
        fp = sum(1 for p, t in zip(predicted_positive, y_true) if p and not t)
        fn = sum(1 for p, t in zip(predicted_positive, y_true) if not p and t)
        tn = sum(1 for p, t in zip(predicted_positive, y_true) if not p and not t)
        cm = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        precision = MetricValue.ok(tp / (tp + fp)) if (tp + fp) > 0 else MetricValue.non_computable(
            "non_computable_single_class", "no rows flagged MEDIUM/HIGH -- precision undefined"
        )
        recall = MetricValue.ok(tp / (tp + fn)) if (tp + fn) > 0 else MetricValue.non_computable(
            "non_computable_single_class", "no resolved-fraud rows -- recall undefined"
        )
        accuracy = MetricValue.ok((tp + tn) / total)

    # ---- PR-AUC / ROC-AUC / Brier (need real score variance across both classes) ----
    if guard is not None:
        pr_auc = roc_auc = brier = guard
    else:
        y_true_int, y_score = _y_true_and_scores(outcomes)
        pr_auc = MetricValue.ok(float(average_precision_score(y_true_int, y_score)))
        roc_auc = MetricValue.ok(float(roc_auc_score(y_true_int, y_score)))
        brier = MetricValue.ok(float(brier_score_loss(y_true_int, y_score)))

    # ---- per-priority-band alert/fraud counts (always computable, even single-class) ----
    per_band_counts: dict[str, dict[str, int]] = {
        band: {"alerts": 0, "fraud": 0} for band in ("LOW", "MEDIUM", "HIGH")
    }
    for o in outcomes:
        per_band_counts[o.priority_band]["alerts"] += 1
        if o.resolved_label == "RESOLVED_FRAUD":
            per_band_counts[o.priority_band]["fraud"] += 1

    # ---- precision@capacity / recall@capacity ----
    if total == 0:
        precision_at_capacity = recall_at_capacity = MetricValue.non_computable(
            "non_computable_empty", f"channel {channel!r} has no resolved source-alerted rows"
        )
    elif total_fraud == 0:
        precision_at_capacity = MetricValue.ok(0.0) if alerts_reviewed > 0 else MetricValue.non_computable(
            "non_computable_empty", "capacity resolves to zero alerts reviewed"
        )
        recall_at_capacity = MetricValue.non_computable(
            "non_computable_single_class", f"channel {channel!r} has no resolved-fraud rows -- recall@capacity undefined"
        )
    elif alerts_reviewed == 0:
        precision_at_capacity = recall_at_capacity = MetricValue.non_computable(
            "non_computable_empty", "capacity resolves to zero alerts reviewed"
        )
    else:
        precision_at_capacity = MetricValue.ok(fraud_captured / alerts_reviewed)
        recall_at_capacity = MetricValue.ok(fraud_captured / total_fraud)

    # ---- minimum review volume for the stated recall target / workload reduction ----
    if total_fraud == 0:
        minimum_alerts_required = None
        workload_reduction = MetricValue.non_computable(
            "non_computable_single_class", f"channel {channel!r} has no resolved-fraud rows -- recall target is unreachable"
        )
    else:
        # Smallest prefix length N (in ranked order) whose cumulative fraud
        # count / total_fraud >= recall_target.
        cumulative_fraud = 0
        minimum_alerts_required = None
        for i, o in enumerate(ranked, start=1):
            if o.resolved_label == "RESOLVED_FRAUD":
                cumulative_fraud += 1
            if cumulative_fraud / total_fraud >= recall_target:
                minimum_alerts_required = i
                break
        if minimum_alerts_required is None:
            # Reviewing every alert still yields recall == 1.0 whenever
            # total_fraud > 0, so this branch is unreachable in practice --
            # kept as an explicit, honest non-computable path rather than a
            # silent fallback to "review everything."
            workload_reduction = MetricValue.non_computable(
                "non_computable_unreachable_target", f"recall target {recall_target} not reached even reviewing all {total} alerts"
            )
        else:
            workload_reduction = MetricValue.ok(1 - (minimum_alerts_required / total))

    # ---- false-positive reduction vs. the "review every source alert" baseline ----
    if total_legit == 0:
        false_positive_reduction = MetricValue.non_computable(
            "non_computable_single_class", f"channel {channel!r} has no resolved-legitimate rows -- false_positive_reduction undefined"
        )
    else:
        baseline_false_positives = total_legit  # the baseline reviews every source alert
        model_false_positives = sum(1 for p, t in zip(predicted_positive, y_true) if p and not t)
        false_positive_reduction = MetricValue.ok(1 - (model_false_positives / baseline_false_positives))

    # ---- rules-only baseline (guide section 21): same score-based metrics,
    # computed off baseline_priority_score instead of operational_priority_score ----
    if guard is not None:
        rules_only_baseline = RulesOnlyBaselineResult(
            pr_auc=guard, roc_auc=guard, brier_score=guard, precision_at_capacity=guard, recall_at_capacity=guard,
        )
    else:
        y_true_int, _ = _y_true_and_scores(outcomes)
        baseline_scores = [o.baseline_priority_score for o in outcomes]
        baseline_pr_auc = MetricValue.ok(float(average_precision_score(y_true_int, baseline_scores)))
        baseline_roc_auc = MetricValue.ok(float(roc_auc_score(y_true_int, baseline_scores)))
        baseline_brier = MetricValue.ok(float(brier_score_loss(y_true_int, baseline_scores)))

        baseline_ranked = sorted(outcomes, key=lambda o: (-o.baseline_priority_score, o.event_timestamp, str(o.source_alert_id)))
        baseline_reviewed = baseline_ranked[:alerts_reviewed]
        baseline_fraud_captured = sum(1 for o in baseline_reviewed if o.resolved_label == "RESOLVED_FRAUD")
        if alerts_reviewed == 0:
            baseline_precision_at_capacity = baseline_recall_at_capacity = MetricValue.non_computable(
                "non_computable_empty", "capacity resolves to zero alerts reviewed"
            )
        elif total_fraud == 0:
            baseline_precision_at_capacity = MetricValue.ok(0.0)
            baseline_recall_at_capacity = MetricValue.non_computable(
                "non_computable_single_class", f"channel {channel!r} has no resolved-fraud rows -- recall@capacity undefined"
            )
        else:
            baseline_precision_at_capacity = MetricValue.ok(baseline_fraud_captured / alerts_reviewed)
            baseline_recall_at_capacity = MetricValue.ok(baseline_fraud_captured / total_fraud)

        rules_only_baseline = RulesOnlyBaselineResult(
            pr_auc=baseline_pr_auc, roc_auc=baseline_roc_auc, brier_score=baseline_brier,
            precision_at_capacity=baseline_precision_at_capacity, recall_at_capacity=baseline_recall_at_capacity,
        )

    return ChannelEvaluationResult(
        channel=channel,
        total_source_alerts=total,
        total_resolved_fraud=total_fraud,
        total_resolved_legitimate=total_legit,
        precision=precision,
        recall=recall,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier_score=brier,
        confusion_matrix_at_operational_threshold=cm,
        per_band_counts=per_band_counts,
        capacity_mode=capacity.mode,
        capacity_value=float(capacity.value),
        alerts_reviewed=alerts_reviewed,
        fraud_captured=fraud_captured,
        precision_at_capacity=precision_at_capacity,
        recall_at_capacity=recall_at_capacity,
        recall_target=recall_target,
        minimum_alerts_required_for_recall_target=minimum_alerts_required,
        workload_reduction_at_recall_target=workload_reduction,
        false_positive_reduction=false_positive_reduction,
        rules_only_baseline=rules_only_baseline,
        accuracy=accuracy,
    )


def _summarize(per_channel: dict[str, ChannelEvaluationResult], metric_name: str) -> ChannelSummary:
    included: list[str] = []
    excluded: list[str] = []
    weighted_numerator = 0.0
    weighted_denominator = 0
    values: list[float] = []
    for channel, result in per_channel.items():
        metric: MetricValue = getattr(result, metric_name)
        if metric.status != "ok" or metric.value is None:
            excluded.append(channel)
            continue
        included.append(channel)
        values.append(metric.value)
        weighted_numerator += metric.value * result.total_source_alerts
        weighted_denominator += result.total_source_alerts

    macro = (sum(values) / len(values)) if values else None
    volume_weighted = (weighted_numerator / weighted_denominator) if weighted_denominator > 0 else None
    return ChannelSummary(
        metric=metric_name, macro=macro, volume_weighted=volume_weighted,
        channels_included=sorted(included), channels_excluded_non_computable=sorted(excluded),
    )


def evaluate_cross_channel(
    outcomes_by_channel: dict[str, Sequence[AlertOutcome]],
    *,
    capacity: Union[CountCapacity, FractionCapacity],
    recall_target: float,
) -> CrossChannelEvaluationReport:
    """Per-channel results plus macro and volume-weighted summaries.
    `capacity`/`recall_target` are applied identically to every channel --
    there is no implicit, per-channel-varying default (Phase 7A decision
    6). Accuracy is included in `summaries` like every other metric --
    nothing here treats it as the headline."""
    per_channel = {
        channel: evaluate_channel(channel, outcomes, capacity=capacity, recall_target=recall_target)
        for channel, outcomes in outcomes_by_channel.items()
    }
    summaries = {
        metric_name: _summarize(per_channel, metric_name)
        for metric_name in (
            "precision", "recall", "pr_auc", "roc_auc", "brier_score", "precision_at_capacity", "recall_at_capacity",
            "workload_reduction_at_recall_target", "false_positive_reduction", "accuracy",
        )
    }
    return CrossChannelEvaluationReport(per_channel=per_channel, summaries=summaries)
