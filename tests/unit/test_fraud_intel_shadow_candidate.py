"""Phase 7A corrective pass: shadow-candidate evaluation
(src.fraud_intel.evaluation.shadow_candidate). No database, MLflow,
Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.evaluation.capacity import CountCapacity
from src.fraud_intel.evaluation.cross_channel import AlertOutcome
from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScore, compare_operational_vs_candidate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _op(*, source_alert_id, minute=0, score=0.5, band="MEDIUM", label="RESOLVED_FRAUD", channel="online_banking"):
    return AlertOutcome(
        source_alert_id=source_alert_id, channel=channel, event_timestamp=T0 + timedelta(minutes=minute),
        operational_priority_score=score, baseline_priority_score=score, priority_band=band, resolved_label=label,
    )


def _cand(*, source_alert_id, minute=0, score=0.5, band="MEDIUM", channel="online_banking"):
    return CandidateShadowScore(
        source_alert_id=source_alert_id, channel=channel, event_timestamp=T0 + timedelta(minutes=minute),
        candidate_priority_score=score, candidate_priority_band=band,
    )


def _compare(operational, candidate, *, capacity=None, recall_target=0.8, channel="online_banking"):
    return compare_operational_vs_candidate(
        channel, operational, candidate, operational_bundle_id=1, operational_bundle_version=1,
        candidate_bundle_id=2, candidate_bundle_version=1, capacity=capacity or CountCapacity(value=2),
        recall_target=recall_target,
    )


# ---- structural: candidate scoring never persists, never imports operational-write code ----


def test_module_never_imports_alert_persistence_or_promotion_code():
    import ast
    import inspect

    from src.fraud_intel.evaluation import shadow_candidate as module

    tree = ast.parse(inspect.getsource(module))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
    forbidden = {"src.fraud_intel.alerts.queue", "src.fraud_intel.models.promotion", "src.common.db", "psycopg2"}
    assert forbidden.isdisjoint(imported_modules), imported_modules & forbidden


def test_module_has_no_database_or_promotion_shaped_code():
    """AST-based (Name/Attribute/Call nodes only) -- ignores the module's
    own docstring, which legitimately explains that promote_bundle is
    never called, so that explanation cannot itself trip this check (the
    same false-positive class already fixed elsewhere in this codebase,
    e.g. tests/unit/test_fraud_intel_migration_004.py)."""
    import ast
    import inspect

    from src.fraud_intel.evaluation import shadow_candidate as module

    tree = ast.parse(inspect.getsource(module))
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    assert "promote_bundle" not in identifiers
    assert "cursor" not in identifiers


def test_result_has_no_promotion_decision_field():
    """Structural: ShadowCandidateComparisonResult carries comparison data
    only -- no field that could be read as an automatic recommendation."""
    from src.fraud_intel.evaluation.shadow_candidate import ShadowCandidateComparisonResult

    field_names = set(ShadowCandidateComparisonResult.model_fields)
    for forbidden_substring in ("promote", "recommend", "approve"):
        assert not any(forbidden_substring in name.lower() for name in field_names)


def test_candidate_scores_are_a_separate_frozen_type_never_written_back():
    """CandidateShadowScore is immutable (frozen) and structurally
    distinct from AlertOutcome/FraudAlertRecord/AlertEvidenceRecord --
    nothing in this module can accidentally treat a candidate score as an
    operational one."""
    from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScore

    assert CandidateShadowScore.model_config.get("frozen") is True
    score = _cand(source_alert_id=uuid.uuid4())
    with pytest.raises(Exception):
        score.candidate_priority_score = 0.9  # frozen -- must raise


def test_candidate_only_alert_is_excluded_from_the_matched_population_entirely():
    """Proves a candidate-only source alert (one the operational bundle
    never scored) cannot smuggle a fraud/legit judgment into any metric --
    it is counted, never compared."""
    op_id = uuid.uuid4()
    candidate_only_id = uuid.uuid4()
    result = _compare(
        [_op(source_alert_id=op_id, score=0.9, band="HIGH")],
        [_cand(source_alert_id=op_id, score=0.9, band="HIGH"), _cand(source_alert_id=candidate_only_id, score=0.99, band="HIGH")],
    )
    assert result.matched_source_alert_count == 1
    assert result.candidate_only_source_alert_count == 1
    assert candidate_only_id not in result.priority_band_migration_counts.get("HIGH", {})


# ---- operational/candidate population matching -------------------------------------


def test_join_is_strictly_by_source_alert_id():
    op_id = uuid.uuid4()
    other_id = uuid.uuid4()
    result = _compare([_op(source_alert_id=op_id)], [_cand(source_alert_id=other_id)])
    assert result.matched_source_alert_count == 0
    assert result.operational_only_source_alert_count == 1
    assert result.candidate_only_source_alert_count == 1


def test_channel_mismatch_raises():
    op_id = uuid.uuid4()
    with pytest.raises(ValueError, match="channel"):
        _compare([_op(source_alert_id=op_id, channel="wire")], [_cand(source_alert_id=op_id)], channel="wire")
    with pytest.raises(ValueError, match="channel"):
        compare_operational_vs_candidate(
            "atm", [_op(source_alert_id=op_id, channel="atm")], [_cand(source_alert_id=op_id, channel="wire")],
            operational_bundle_id=1, operational_bundle_version=1, candidate_bundle_id=2, candidate_bundle_version=1,
            capacity=CountCapacity(value=1), recall_target=0.5,
        )


# ---- metric deltas and priority-band migrations -------------------------------------


def test_identical_scores_produce_zero_deltas_and_diagonal_migration():
    ids = [uuid.uuid4() for _ in range(4)]
    labels = ["RESOLVED_FRAUD", "RESOLVED_FRAUD", "RESOLVED_LEGITIMATE", "RESOLVED_LEGITIMATE"]
    scores = [0.9, 0.7, 0.3, 0.1]
    bands = ["HIGH", "MEDIUM", "LOW", "LOW"]
    operational = [_op(source_alert_id=ids[i], minute=i, score=scores[i], band=bands[i], label=labels[i]) for i in range(4)]
    candidate = [_cand(source_alert_id=ids[i], minute=i, score=scores[i], band=bands[i]) for i in range(4)]

    result = _compare(operational, candidate, capacity=CountCapacity(value=2))
    assert result.precision_at_capacity_delta.value == pytest.approx(0.0)
    assert result.recall_at_capacity_delta.value == pytest.approx(0.0)
    assert result.pr_auc_delta.value == pytest.approx(0.0)
    assert result.brier_score_delta.value == pytest.approx(0.0)
    assert result.score_difference_mean.value == pytest.approx(0.0)
    assert result.score_difference_stdev.value == pytest.approx(0.0)
    assert result.ranking_disagreement_at_capacity.value == pytest.approx(0.0)
    # every row migrates to the SAME band -- diagonal only
    for from_band, to_counts in result.priority_band_migration_counts.items():
        assert set(to_counts) == {from_band}


def test_fraud_gained_and_lost_are_computed_from_capacity_limited_reviewed_sets():
    ids = [uuid.uuid4() for _ in range(4)]
    # operational ranks: id0(0.9,fraud), id1(0.8,fraud), id2(0.3,legit), id3(0.1,legit)
    # candidate ranks:   id0(0.95,fraud), id2(0.6,legit), id1(0.2,fraud), id3(0.1,legit)
    operational = [
        _op(source_alert_id=ids[0], minute=0, score=0.9, band="HIGH", label="RESOLVED_FRAUD"),
        _op(source_alert_id=ids[1], minute=1, score=0.8, band="HIGH", label="RESOLVED_FRAUD"),
        _op(source_alert_id=ids[2], minute=2, score=0.3, band="LOW", label="RESOLVED_LEGITIMATE"),
        _op(source_alert_id=ids[3], minute=3, score=0.1, band="LOW", label="RESOLVED_LEGITIMATE"),
    ]
    candidate = [
        _cand(source_alert_id=ids[0], minute=0, score=0.95, band="HIGH"),
        _cand(source_alert_id=ids[1], minute=1, score=0.2, band="LOW"),
        _cand(source_alert_id=ids[2], minute=2, score=0.6, band="MEDIUM"),
        _cand(source_alert_id=ids[3], minute=3, score=0.1, band="LOW"),
    ]
    result = _compare(operational, candidate, capacity=CountCapacity(value=2))
    # operational top-2: id0(fraud), id1(fraud) -> fraud_captured=2
    assert result.fraud_captured_operational == 2
    # candidate top-2: id0(fraud), id2(legit) -> fraud_captured=1
    assert result.fraud_captured_candidate == 1
    assert result.fraud_lost_by_candidate == 1  # id1 dropped out
    assert result.fraud_gained_by_candidate == 0


# ---- empty and single-class populations ---------------------------------------------


def test_empty_populations_are_non_computable_not_zero():
    result = _compare([], [])
    assert result.matched_source_alert_count == 0
    assert result.pr_auc_delta.status == "non_computable_empty"
    assert result.precision_at_capacity_delta.status == "non_computable_empty"


def test_single_class_matched_population_marks_classification_deltas_non_computable():
    ids = [uuid.uuid4(), uuid.uuid4()]
    operational = [_op(source_alert_id=i, score=0.9, band="HIGH", label="RESOLVED_FRAUD") for i in ids]
    candidate = [_cand(source_alert_id=i, score=0.5, band="MEDIUM") for i in ids]
    result = _compare(operational, candidate, capacity=CountCapacity(value=1))
    assert result.pr_auc_delta.status == "non_computable_single_class"
    assert result.precision_at_capacity_delta.status == "non_computable_single_class"
    assert result.recall_at_capacity_delta.status == "non_computable_single_class"
    # migration counts and score-difference stats remain computable regardless
    assert result.priority_band_migration_counts
    assert result.score_difference_mean.status == "ok"


def test_recall_target_out_of_range_raises():
    op_id = uuid.uuid4()
    with pytest.raises(ValueError):
        _compare([_op(source_alert_id=op_id)], [_cand(source_alert_id=op_id)], recall_target=0.0)
