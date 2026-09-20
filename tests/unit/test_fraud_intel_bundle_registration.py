"""Phase 4: ChannelModelBundleStore -- injectable pattern, candidate-bundle
shape, and the incomplete-bundle (missing anomaly component) promotion
safeguard. No database, Docker, or network access anywhere in this file
-- every test uses _FakeChannelModelBundleStore.
"""
from __future__ import annotations

import pytest

from src.fraud_intel.models.bundle import (
    REQUIRED_OPERATIONAL_COMPONENTS,
    IncompleteBundleError,
    _FakeChannelModelBundleStore,
    validate_promotion_eligible,
)


def _complete_fields(**overrides) -> dict:
    base = dict(
        channel="online_banking",
        gbm_model_version="1",
        lr_model_version="1",
        anomaly_model_version="1",
        preprocessing_artifact_version="pp-abc123",
        feature_schema_version="v1",
        training_run_id=42,
        dataset_version="ds-abc123",
        evaluation_report_ref="{}",
    )
    base.update(overrides)
    return base


def test_register_candidate_starts_at_version_one():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    assert bundle.bundle_version == 1
    assert bundle.status == "CANDIDATE"
    assert bundle.bundle_id == 1


def test_register_candidate_increments_version_per_channel():
    store = _FakeChannelModelBundleStore()
    first = store.register_candidate(**_complete_fields())
    second = store.register_candidate(**_complete_fields())
    assert first.bundle_version == 1
    assert second.bundle_version == 2
    assert first.bundle_id != second.bundle_id


def test_register_candidate_versions_are_independent_per_channel():
    store = _FakeChannelModelBundleStore()
    ob_1 = store.register_candidate(**_complete_fields(channel="online_banking"))
    ach_1 = store.register_candidate(**_complete_fields(channel="ach"))
    ob_2 = store.register_candidate(**_complete_fields(channel="online_banking"))
    assert ob_1.bundle_version == 1
    assert ach_1.bundle_version == 1
    assert ob_2.bundle_version == 2


def test_bundle_versions_are_immutable_once_registered():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    with pytest.raises(Exception):
        bundle.bundle_version = 99  # frozen-like pydantic model rejects mutation attempts differently per config


# ---- Phase 4's incomplete (anomaly-less) candidate bundle -------------------------


def test_phase4_candidate_bundle_has_anomaly_model_version_null():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    assert bundle.anomaly_model_version is None
    assert bundle.status == "CANDIDATE"


def test_incomplete_bundle_is_not_promotion_eligible():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    assert bundle.is_promotion_eligible() is False
    with pytest.raises(IncompleteBundleError, match="anomaly_model_version"):
        validate_promotion_eligible(bundle)


def test_complete_bundle_is_promotion_eligible():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    assert bundle.is_promotion_eligible() is True
    validate_promotion_eligible(bundle)  # must not raise


@pytest.mark.parametrize("missing_field", REQUIRED_OPERATIONAL_COMPONENTS)
def test_promotion_rejects_a_bundle_missing_any_single_required_component(missing_field):
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(**{missing_field: None}))
    with pytest.raises(IncompleteBundleError, match=missing_field):
        validate_promotion_eligible(bundle)


def test_phase5_registers_a_new_bundle_version_never_mutates_phase4s_row():
    """Phase 4's candidate bundle is never updated in place -- a later
    "Phase 5" registration (simulated here as a second register_candidate
    call with a complete component set) must produce a NEW row, leaving
    the first bundle's own fields exactly as they were."""
    store = _FakeChannelModelBundleStore()
    phase4_bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    phase4_bundle_snapshot = phase4_bundle.model_copy(deep=True)

    phase5_bundle = store.register_candidate(**_complete_fields(anomaly_model_version="1"))

    assert phase5_bundle.bundle_id != phase4_bundle.bundle_id
    assert phase5_bundle.bundle_version == phase4_bundle.bundle_version + 1
    assert phase4_bundle == phase4_bundle_snapshot  # untouched
    assert phase4_bundle.anomaly_model_version is None  # still incomplete, still CANDIDATE
    assert phase4_bundle.status == "CANDIDATE"
    assert len(store.rows) == 2
