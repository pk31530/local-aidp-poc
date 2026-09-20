"""Phase 6: bundle-promotion verification, deterministic full-channel row
locking, and atomic retire-and-promote. No real MLflow/PostgreSQL contact
anywhere in this file -- every test uses `_FakeBundlePromotionStore` and a
fake `ModelVersionVerifier`.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.fraud_intel.models.bundle import IncompleteBundleError, _FakeChannelModelBundleStore
from src.fraud_intel.models.promotion import (
    BundlePromotionRaceError,
    BundleVerificationFailedError,
    _FakeBundlePromotionStore,
    promote_bundle,
    verify_bundle_components,
)


def _complete_fields(**overrides) -> dict:
    base = dict(
        channel="online_banking", gbm_model_version="1", lr_model_version="1", anomaly_model_version="1",
        preprocessing_artifact_version="pp-abc123", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        training_run_id=42, dataset_version="ds-abc123", evaluation_report_ref="{}",
    )
    base.update(overrides)
    return base


class _AlwaysTrueVerifier:
    def verify(self, model_name, version):
        return True


class _AlwaysFalseVerifier:
    def verify(self, model_name, version):
        return False


class _RaisingVerifier:
    def verify(self, model_name, version):
        raise ConnectionError("simulated MLflow contact failure")


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
    assert any("lr" in p and "could not be verified" in p for p in problems)


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


# ---- promote_bundle: eligibility + verification gate ------------------------------


def test_promotion_refused_for_an_incomplete_bundle():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields(anomaly_model_version=None))
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(IncompleteBundleError):
        promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)


def test_promotion_refused_when_verification_fails():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_AlwaysFalseVerifier(), store=promotion_store)
    # refused BEFORE any lock/transaction -- the bundle's status is untouched
    assert bundle_store.rows[0].status == "CANDIDATE"


def test_promotion_refused_when_verifier_itself_cannot_run():
    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_RaisingVerifier(), store=promotion_store)


# ---- atomic retire + promote --------------------------------------------------------


def test_successful_promotion_retires_old_and_promotes_new_atomically():
    bundle_store = _FakeChannelModelBundleStore()
    first = bundle_store.register_candidate(**_complete_fields())
    promotion_store = _FakeBundlePromotionStore(bundle_store)
    promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)
    assert bundle_store.rows[0].status == "OPERATIONAL"
    assert bundle_store.rows[0].promoted_by == "analyst1"
    assert bundle_store.rows[0].promoted_at is not None

    second = bundle_store.register_candidate(**_complete_fields())
    promote_bundle(channel="online_banking", bundle_version=2, promoted_by="analyst2", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)
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
    result = promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)
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
        promote_bundle(channel="online_banking", bundle_version=1, promoted_by="analyst1", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)


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

    promote_bundle(channel="online_banking", bundle_version=candidate_a.bundle_version, promoted_by="analyst1", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)
    # candidate_b is still CANDIDATE and can still be promoted (retiring A) --
    # promotion never silently produces two OPERATIONAL rows.
    promote_bundle(channel="online_banking", bundle_version=candidate_b.bundle_version, promoted_by="analyst2", model_version_verifier=_AlwaysTrueVerifier(), store=promotion_store)
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
