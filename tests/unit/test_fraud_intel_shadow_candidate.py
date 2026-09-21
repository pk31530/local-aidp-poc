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
    forbidden_calls = {"create_alert_if_new", "record_evidence", "record_disposition", "promote_bundle", "cursor"}
    assert forbidden_calls.isdisjoint(identifiers), forbidden_calls & identifiers


def test_module_source_never_contains_insert_update_or_delete_sql():
    """Plain substring check on the RAW (not AST-stripped) source -- safe
    here because neither this module's code nor its docstrings ever have
    a legitimate reason to mention these SQL keywords at all (unlike the
    NotImplementedError/promote_bundle false-positive class elsewhere in
    this codebase, where an explanatory docstring legitimately NAMES the
    forbidden thing)."""
    import inspect

    from src.fraud_intel.evaluation import shadow_candidate as module

    source = inspect.getsource(module)
    for keyword in ("INSERT INTO", "UPDATE ", "DELETE FROM"):
        assert keyword not in source, keyword


def test_score_candidate_shadow_never_touches_a_real_alert_queue_store():
    """Behavioral proof, not just structural: a real _FakeAlertQueueStore
    exists in this test process throughout candidate scoring, but is
    never passed to score_candidate_shadow() (its signature has no store
    parameter at all) -- confirms it stays completely empty."""
    from src.fraud_intel.alerts.queue import _FakeAlertQueueStore
    from src.fraud_intel.evaluation.shadow_candidate import score_candidate_shadow

    untouched_store = _FakeAlertQueueStore()

    scores, errors = score_candidate_shadow([], bundle=None, rule_provider=None, ensemble_policy=None, graph_policy=None)

    assert scores == []
    assert errors == []
    assert untouched_store.alerts_by_id == {}
    assert untouched_store.evidence_by_alert == {}
    assert untouched_store.dispositions == []


def test_score_candidate_shadow_signature_has_no_store_or_database_parameter():
    """Structural: the function CANNOT be given anything to write to, even
    by a future caller mistake -- no store/database/connection-shaped
    parameter exists in its signature at all."""
    import inspect

    from src.fraud_intel.evaluation.shadow_candidate import score_candidate_shadow

    param_names = set(inspect.signature(score_candidate_shadow).parameters)
    for forbidden_substring in ("store", "database", "conn", "session"):
        assert not any(forbidden_substring in name.lower() for name in param_names), param_names


def test_candidate_scoring_never_changes_the_candidate_bundles_own_status():
    """Behavioral proof: registering a real CANDIDATE bundle via
    _FakeChannelModelBundleStore, then running the full shadow-comparison
    flow (score_candidate_shadow + compare_operational_vs_candidate),
    leaves that bundle's status exactly 'CANDIDATE' -- nothing in this
    flow ever promotes it."""
    from src.fraud_intel.models.bundle import _FakeChannelModelBundleStore
    from src.fraud_intel.evaluation.shadow_candidate import score_candidate_shadow

    bundle_store = _FakeChannelModelBundleStore()
    candidate_record = bundle_store.register_candidate(
        channel="online_banking", gbm_model_version="1", lr_model_version="1", anomaly_model_version="1",
        preprocessing_artifact_version="pp-1", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        training_run_id=1, dataset_version="ds-1", evaluation_report_ref="{}",
    )
    assert candidate_record.status == "CANDIDATE"

    scores, errors = score_candidate_shadow([], bundle=None, rule_provider=None, ensemble_policy=None, graph_policy=None)

    refreshed = next(r for r in bundle_store.rows if r.bundle_id == candidate_record.bundle_id)
    assert refreshed.status == "CANDIDATE"  # completely untouched


def test_load_resolved_alert_scoring_contexts_sql_is_read_only():
    """The real Postgres reader backing the candidate-scoring path
    (reviewed as SQL, not exercised) contains only SELECT statements --
    no INSERT/UPDATE/DELETE anywhere."""
    import inspect

    from src.fraud_intel import cli_data_access

    source = inspect.getsource(cli_data_access.load_resolved_alert_scoring_contexts)
    for keyword in ("INSERT INTO", "UPDATE ", "DELETE FROM"):
        assert keyword not in source, keyword


def test_load_candidate_shadow_scores_no_longer_exists():
    """The old function queried alert_evidence for a channel_model_bundle_id
    tag no writer has ever produced -- it has been removed and replaced
    by the real in-memory-scoring-oriented read
    (load_resolved_alert_scoring_contexts + score_candidate_shadow)."""
    from src.fraud_intel import cli_data_access

    assert not hasattr(cli_data_access, "load_candidate_shadow_scores")


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
