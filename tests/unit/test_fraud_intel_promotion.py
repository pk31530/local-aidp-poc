"""Phase 6 (bundle-promotion verification, deterministic full-channel row
locking, atomic retire-and-promote) plus Phase 7B Stage 5's corrective
pass (canonical MLflow naming used by the verifier, mandatory freshly-
recomputed cold-start promotion-gate enforcement for a channel's
first-ever promotion, and a real `model_promotion` RunLifecycle audit
trail). No real MLflow/PostgreSQL contact anywhere in this file -- every
test uses `_FakeBundlePromotionStore`, a fake `ModelVersionVerifier`, and
a fake RunLifecycle run store.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.control_plane.runs import RunLifecycle as RealRunLifecycle
from src.control_plane.runs import RunRecord
from src.fraud_intel.models import promotion as promotion_module
from src.fraud_intel.models.bundle import IncompleteBundleError, _FakeChannelModelBundleStore
from src.fraud_intel.models.promotion import (
    BundlePromotionRaceError,
    BundleVerificationFailedError,
    ColdStartPromotionGateFailedError,
    _FakeBundlePromotionStore,
    promote_bundle,
    verify_bundle_components,
)

DATABASE = "aidp_test"


def _passing_cold_start_report(**overrides) -> str:
    base = dict(
        channel="online_banking", training_run_id=42, dataset_version="ds-abc123",
        supervised_population_hash="ds-abc123", source_generation_run_id="genrun-test",
        source_dataset_version="dsv-test", feature_schema_version="v1",
        gbm_model_version="1", lr_model_version="1", anomaly_model_version="1",
        preprocessing_artifact_version="pp-abc123",
        gbm_evaluation={"precision": 1.0, "recall": 1.0, "pr_auc": 0.9, "roc_auc": 1.0, "brier_score": 0.01},
        lr_shadow_evaluation={"precision": 1.0, "recall": 1.0, "pr_auc": 0.8, "roc_auc": 1.0, "brier_score": 0.02},
        eligibility_policy_version="v1",
        realized_split_fractions={"train": 0.7, "calibration": 0.15, "test": 0.15},
        split_class_counts={
            "train": {"fraud": 10, "legitimate": 10},
            "calibration": {"fraud": 10, "legitimate": 10},
            "test": {"fraud": 10, "legitimate": 10},
        },
        gbm_mlflow_run_id="run-gbm-1", lr_mlflow_run_id="run-lr-1", anomaly_mlflow_run_id="run-anomaly-1",
        anomaly_normalization={"method": "none"},
    )
    base.update(overrides)
    return json.dumps(base)


def _complete_fields(**overrides) -> dict:
    base = dict(
        channel="online_banking", gbm_model_version="1", lr_model_version="1", anomaly_model_version="1",
        preprocessing_artifact_version="pp-abc123", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        training_run_id=42, dataset_version="ds-abc123", evaluation_report_ref=_passing_cold_start_report(),
    )
    base.update(overrides)
    return base


class _AlwaysTrueVerifier:
    def verify(self, model_name, version, *, expected_run_id=None):
        return True


class _AlwaysFalseVerifier:
    def verify(self, model_name, version, *, expected_run_id=None):
        return False


class _RaisingVerifier:
    def verify(self, model_name, version, *, expected_run_id=None):
        raise ConnectionError("simulated MLflow contact failure")


class _RecordingVerifier:
    """Records every (model_name, version, expected_run_id) it was asked
    to verify -- lets a test prove the canonical MLflow name (never the
    old, buggy f"{channel}-{label}" construction) and the expected run id
    were both passed through."""

    def __init__(self):
        self.calls: list[tuple] = []

    def verify(self, model_name, version, *, expected_run_id=None):
        self.calls.append((model_name, version, expected_run_id))
        return True


# ---- shared fake RunLifecycle plumbing -------------------------------------------------


class _FakeRunStore:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id, "trigger_source": None, "git_sha": None, "config_snapshot": None,
            "config_hash": None, "dataset_version": None, "model_version": None, "error_type": None,
            "error_message": None, "started_at": datetime.now(timezone.utc), "heartbeat_at": None,
            "completed_at": None, **row,
        }
        self.rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self.rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self.rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        return [RunRecord(**r) for r in list(self.rows.values())[:limit]]


@pytest.fixture(autouse=True)
def run_store(monkeypatch):
    store = _FakeRunStore()
    monkeypatch.setattr(promotion_module, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store))
    return store


# ---- verify_bundle_components: every referenced component -------------------------


def test_verification_passes_for_a_fully_matching_candidate():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    problems = verify_bundle_components(bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert problems == []


def test_verification_reports_missing_gbm_model_version():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(gbm_model_version=None))
    problems = verify_bundle_components(bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("gbm_model_version is missing" in p for p in problems)


def test_verification_reports_unverifiable_lr_model_version():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    problems = verify_bundle_components(bundle, model_version_verifier=_AlwaysFalseVerifier())
    assert any("lr-shadow" in p and "could not be verified" in p for p in problems)


def test_verification_reports_verifier_failure_without_raising():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    problems = verify_bundle_components(bundle, model_version_verifier=_RaisingVerifier())
    assert any("verification raised" in p for p in problems)


def test_verification_checks_rule_set_version_against_the_real_loaded_yaml():
    store = _FakeChannelModelBundleStore()
    stale_bundle = store.register_candidate(**_complete_fields(rule_set_version="v999-does-not-exist"))
    problems = verify_bundle_components(stale_bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("rule_set_version" in p for p in problems)


def test_verification_checks_graph_policy_version_against_the_real_loaded_yaml():
    store = _FakeChannelModelBundleStore()
    stale_bundle = store.register_candidate(**_complete_fields(graph_policy_version="v999-does-not-exist"))
    problems = verify_bundle_components(stale_bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("graph_policy_version" in p for p in problems)


def test_verification_checks_ensemble_policy_version_against_the_real_loaded_yaml():
    store = _FakeChannelModelBundleStore()
    stale_bundle = store.register_candidate(**_complete_fields(ensemble_policy_version="v999-does-not-exist"))
    problems = verify_bundle_components(stale_bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("ensemble_policy_version" in p for p in problems)


def test_verification_checks_reason_code_version_against_the_current_constant():
    store = _FakeChannelModelBundleStore()
    stale_bundle = store.register_candidate(**_complete_fields(reason_code_version="v999-does-not-exist"))
    problems = verify_bundle_components(stale_bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("reason_code_version" in p for p in problems)


def test_verification_checks_preprocessing_and_feature_schema_presence():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(preprocessing_artifact_version=None, feature_schema_version=None))
    problems = verify_bundle_components(bundle, model_version_verifier=_AlwaysTrueVerifier())
    assert any("preprocessing_artifact_version" in p for p in problems)
    assert any("feature_schema_version" in p for p in problems)


# ---- Phase 7B Stage 5: canonical MLflow naming used by the verifier -------------------


def test_verifier_is_queried_with_the_canonical_model_names_and_expected_run_ids():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    verifier = _RecordingVerifier()
    problems = verify_bundle_components(bundle, model_version_verifier=verifier)
    assert problems == []
    assert set(verifier.calls) == {
        ("fraud-detection-model-fraud-intel-online-banking-gbm", "1", "run-gbm-1"),
        ("fraud-detection-model-fraud-intel-online-banking-lr-shadow", "1", "run-lr-1"),
        ("fraud-detection-model-fraud-intel-online-banking-anomaly", "1", "run-anomaly-1"),
    }


def test_verification_fails_when_registered_version_points_at_the_wrong_run_id():
    """The exact bug this corrective pass fixes proven the other way: even
    with the correct model name and version, a mismatched run_id must
    fail verification -- not merely 'the version exists'."""

    class _WrongRunVerifier:
        def verify(self, model_name, version, *, expected_run_id=None):
            # Simulates a registered version whose actual run_id differs
            # from what evaluation_report_ref expects.
            return False

    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    problems = verify_bundle_components(bundle, model_version_verifier=_WrongRunVerifier())
    assert len(problems) == 3
    for p in problems:
        assert "expected_run_id=" in p


# ---- promote_bundle: eligibility + verification gate ------------------------------


def test_promotion_refused_for_an_incomplete_bundle():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields(anomaly_model_version=None))
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(IncompleteBundleError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
        )


def test_promotion_refused_when_verification_fails():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_AlwaysFalseVerifier(), store=promotion_store,
        )
    # refused BEFORE any lock/transaction -- the bundle's status is untouched
    assert bundle_store.rows[0].status == "CANDIDATE"


def test_promotion_refused_when_verifier_itself_cannot_run():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_RaisingVerifier(), store=promotion_store,
        )


# ---- Phase 7B Stage 5: mandatory manual approval + database ---------------------------


def test_promoted_by_remains_required_even_though_the_gate_passes():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(ValueError, match="promoted_by"):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="", database=DATABASE,
            model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
        )
    assert bundle_store.rows[0].status == "CANDIDATE"


def test_database_is_required_and_rejected_when_empty():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(ValueError, match="database"):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database="",
            model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
        )


def test_run_lifecycle_is_constructed_with_the_same_database_passed_to_promote_bundle(monkeypatch, run_store):
    captured = {}

    class _SpyLifecycle:
        def __init__(self, database=None):
            captured["database"] = database
            self._real = RealRunLifecycle(store=run_store)

        def begin(self, *a, **k):
            return self._real.begin(*a, **k)

        def succeed(self, *a, **k):
            return self._real.succeed(*a, **k)

        def fail_from_exception(self, *a, **k):
            return self._real.fail_from_exception(*a, **k)

    monkeypatch.setattr(promotion_module, "RunLifecycle", _SpyLifecycle)

    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    promote_bundle(
        channel="online_banking", bundle_version=1, promoted_by="analyst1", database="aidp_test",
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    assert captured["database"] == "aidp_test"


# ---- Phase 7B Stage 5: cold-start promotion-gate enforcement --------------------------


def test_cold_start_gate_is_recomputed_and_first_promotion_succeeds_when_it_passes(run_store):
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    promoted = promote_bundle(
        channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    assert promoted.status == "OPERATIONAL"

    (run,) = run_store.rows.values()
    assert run["status"] == "SUCCESS"
    assert run["pipeline_name"] == "model_promotion"
    assert run["trigger_source"] == "cli"
    assert run["completed_at"] is not None
    artifacts = run["artifacts"]
    assert artifacts["evaluation_mode"] == "candidate_training_holdout"
    assert artifacts["promotion_gate_passed"] is True
    assert artifacts["promotion_gate_reasons"] == []
    assert artifacts["source_generation_run_id"] == "genrun-test"
    assert artifacts["source_dataset_version"] == "dsv-test"
    assert artifacts["supervised_population_hash"] == "ds-abc123"
    assert artifacts["test_fraud_count"] == 10
    assert artifacts["test_legitimate_count"] == 10
    assert artifacts["gbm_pr_auc"] == 0.9
    assert artifacts["channel"] == "online_banking"
    assert artifacts["bundle_id"] == promoted.bundle_id
    assert artifacts["bundle_version"] == 1
    assert artifacts["promoted_by"] == "analyst1"
    assert artifacts["previous_status"] == "CANDIDATE"
    assert artifacts["resulting_status"] == "OPERATIONAL"
    assert artifacts["gbm_model_version"] == "1"
    assert artifacts["lr_model_version"] == "1"
    assert artifacts["anomaly_model_version"] == "1"
    assert artifacts["preprocessing_artifact_version"] == "pp-abc123"
    assert artifacts["feature_schema_version"] == "v1"
    assert artifacts["rule_set_version"] == "v1"
    assert artifacts["graph_policy_version"] == "v1"
    assert artifacts["ensemble_policy_version"] == "v1"
    assert artifacts["reason_code_version"] == "v1"


def test_a_failed_cold_start_gate_prevents_promote_locked_and_leaves_bundle_candidate(run_store):
    """Too few test-split rows of each class -- the pilot's real gate
    floor -- must block promotion before promote_locked() is ever
    called."""
    failing_report = _passing_cold_start_report(
        split_class_counts={
            "train": {"fraud": 10, "legitimate": 10},
            "calibration": {"fraud": 10, "legitimate": 10},
            "test": {"fraud": 1, "legitimate": 1},
        }
    )
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields(evaluation_report_ref=failing_report))
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    with pytest.raises(ColdStartPromotionGateFailedError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
        )

    # promote_locked() was never reached -- the bundle is still CANDIDATE
    assert bundle_store.rows[0].status == "CANDIDATE"
    (run,) = run_store.rows.values()
    assert run["status"] == "FAILED"
    assert run["error_type"] == "ColdStartPromotionGateFailedError"


def test_cold_start_gate_is_skipped_when_an_operational_bundle_already_exists(run_store):
    """A channel's SECOND promotion (an OPERATIONAL bundle already exists)
    is not subject to the cold-start gate -- the candidate's report can
    even be absent/malformed and promotion still proceeds, because
    get_operational_bundle() short-circuits the cold-start branch
    entirely."""
    bundle_store = _FakeChannelModelBundleStore()
    first = bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    promote_bundle(
        channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )

    second = bundle_store.register_candidate(**_complete_fields(evaluation_report_ref="not valid json{{{"))
    promoted_second = promote_bundle(
        channel="online_banking", bundle_version=2, promoted_by="analyst2", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    assert promoted_second.status == "OPERATIONAL"
    refreshed_first = next(r for r in bundle_store.rows if r.bundle_id == first.bundle_id)
    assert refreshed_first.status == "RETIRED"


def test_failed_verification_records_failed_run_without_swallowing_the_original_error(run_store):
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_AlwaysFalseVerifier(), store=promotion_store,
        )

    (run,) = run_store.rows.values()
    assert run["status"] == "FAILED"
    assert run["error_type"] == "BundleVerificationFailedError"


# ---- atomic retire + promote --------------------------------------------------------


def test_successful_promotion_retires_old_and_promotes_new_atomically():
    bundle_store = _FakeChannelModelBundleStore()
    first = bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    promote_bundle(
        channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    assert bundle_store.rows[0].status == "OPERATIONAL"
    assert bundle_store.rows[0].promoted_by == "analyst1"
    assert bundle_store.rows[0].promoted_at is not None

    second = bundle_store.register_candidate(**_complete_fields())
    promote_bundle(
        channel="online_banking", bundle_version=2, promoted_by="analyst2", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    refreshed_first = next(r for r in bundle_store.rows if r.bundle_id == first.bundle_id)
    refreshed_second = next(r for r in bundle_store.rows if r.bundle_id == second.bundle_id)
    assert refreshed_first.status == "RETIRED"  # atomically demoted
    assert refreshed_second.status == "OPERATIONAL"
    # exactly one OPERATIONAL bundle for the channel at any time
    operational = [r for r in bundle_store.rows if r.status == "OPERATIONAL"]
    assert len(operational) == 1


def test_promotion_with_no_prior_operational_bundle_succeeds():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    result = promote_bundle(
        channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    assert result.status == "OPERATIONAL"


# ---- concurrency / race protection ---------------------------------------------------


def test_promotion_aborts_if_candidate_changed_between_verification_and_lock():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    # Simulate a race: something else retires the candidate bundle after
    # verification completed but before the locking transaction runs.
    bundle_store.rows[0] = bundle_store.rows[0].model_copy(update={"status": "RETIRED"})

    with pytest.raises(BundlePromotionRaceError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database=DATABASE,
            model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
        )


def test_promotion_aborts_if_candidate_disappears_between_verification_and_lock():
    """Exercises promote_locked()'s own re-confirmation directly (the
    candidate existed for get_bundle()/verification, then vanished before
    the locking step specifically -- promote_bundle() itself has no seam
    to inject a mutation strictly between its two internal calls, so this
    isolates the race-check mechanism at the store level instead)."""
    bundle_store = _FakeChannelModelBundleStore()
    candidate = bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    bundle_store.rows = []  # candidate vanished entirely, after verification already captured `candidate`

    with pytest.raises(BundlePromotionRaceError):
        promotion_store.promote_locked(
            channel="online_banking", candidate_bundle_id=candidate.bundle_id, promoted_by="analyst1", expected_candidate=candidate,
        )


def test_second_of_two_racing_promotions_is_rejected():
    """Two candidates for the same channel; simulate two 'concurrent'
    promote_bundle calls (sequential here, but each independently verifies
    against the state at ITS OWN verification time) -- the second call's
    re-confirmation at lock time must still be consistent with whichever
    promotion actually committed first."""
    bundle_store = _FakeChannelModelBundleStore()
    candidate_a = bundle_store.register_candidate(**_complete_fields())
    candidate_b = bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    promote_bundle(
        channel="online_banking", bundle_version=candidate_a.bundle_version, promoted_by="analyst1", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    # candidate_b is still CANDIDATE and can still be promoted (retiring A) --
    # promotion never silently produces two OPERATIONAL rows.
    promote_bundle(
        channel="online_banking", bundle_version=candidate_b.bundle_version, promoted_by="analyst2", database=DATABASE,
        model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store,
    )
    operational = [r for r in bundle_store.rows if r.status == "OPERATIONAL"]
    assert len(operational) == 1
    assert operational[0].bundle_id == candidate_b.bundle_id


def test_promotion_never_triggered_automatically_by_training():
    """Structural: src.fraud_intel.models.training has no reference to
    promote_bundle anywhere."""
    import ast
    import inspect

    from src.fraud_intel.models import training as training_module

    tree = ast.parse(inspect.getsource(training_module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert "promote_bundle" not in names


def test_promotion_never_triggered_automatically_by_evaluate():
    """Structural twin of the training check above, for the CLI's
    evaluate handlers -- neither the cold-start nor the live evaluation
    path may ever call promote_bundle/promote_locked."""
    import ast
    import inspect

    from src.cli import __main__ as cli_main

    for fn in (cli_main._handle_fraud_intel_evaluate, cli_main._evaluate_candidate_cold_start, cli_main._evaluate_live):
        tree = ast.parse(inspect.getsource(fn))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        assert "promote_bundle" not in names
        assert "promote_locked" not in names
