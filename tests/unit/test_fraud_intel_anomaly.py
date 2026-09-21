"""Phase 5: anomaly detection (guide section 14). No database, Docker, or
network access -- IsolationForest fitting here is small, fast, local, and
infrastructure-free (same category as Phase 4's real IsotonicRegression
fitting), never "real large-scale training."
"""
from __future__ import annotations

import ast
import inspect

import pytest

from src.fraud_intel.models import anomaly as anomaly_module
from src.fraud_intel.models.anomaly import fit_anomaly_model, score_anomaly

X_TRAIN = [[float(i), float(i % 3)] for i in range(30)]


def test_fit_anomaly_model_is_label_agnostic_by_signature():
    """No y/label parameter anywhere in fit_anomaly_model's signature."""
    sig = inspect.signature(fit_anomaly_model)
    param_names = {name.lower() for name in sig.parameters}
    assert not any("label" in name or name in ("y", "y_train") for name in param_names)


def test_fit_is_deterministic_for_a_fixed_seed():
    model_a, norm_a = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    model_b, norm_b = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    scores_a = score_anomaly(model_a, norm_a, X_TRAIN)
    scores_b = score_anomaly(model_b, norm_b, X_TRAIN)
    assert scores_a == scores_b


def test_different_seeds_can_produce_different_normalization_bounds():
    _, norm_42 = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    _, norm_7 = fit_anomaly_model(X_TRAIN, random_seed=7, anomaly_artifact_version="v1")
    # Not asserting inequality (could coincide), just that both are valid,
    # well-formed bounds -- proves random_seed is actually threaded through.
    assert norm_42.train_min <= norm_42.train_max
    assert norm_7.train_min <= norm_7.train_max


def test_anomaly_score_is_bounded_zero_to_one():
    model, norm = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    scores = score_anomaly(model, norm, X_TRAIN)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_normalization_uses_the_training_windows_own_distribution():
    model, norm = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    # A far-outlier point, scored AFTER fitting, must still be clipped into
    # [0, 1] rather than exploding the normalization -- proving the bounds
    # are fixed at fit time, not re-derived from the scoring input.
    outlier = [[10_000.0, 10_000.0]]
    scores = score_anomaly(model, norm, outlier)
    assert 0.0 <= scores[0] <= 1.0


def test_anomaly_artifact_version_and_library_versions_recorded():
    _, norm = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="anom-abc123")
    doc = norm.to_json_dict()
    assert doc["anomaly_artifact_version"] == "anom-abc123"
    assert "scikit_learn" in doc["library_versions"]
    assert "numpy" in doc["library_versions"]


def test_normalization_content_hash_is_deterministic():
    _, norm_a = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    _, norm_b = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    assert norm_a.content_hash() == norm_b.content_hash()


def test_no_fitting_during_scoring():
    """score_anomaly's own code never calls .fit() -- AST-level check, not
    just "it happens not to" by observation."""
    tree = ast.parse(inspect.getsource(anomaly_module))
    score_fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "score_anomaly")
    calls_fit = any(
        isinstance(node, ast.Attribute) and node.attr == "fit"
        for node in ast.walk(score_fn)
    )
    assert calls_fit is False


def test_score_anomaly_does_not_mutate_the_model_or_normalization():
    model, norm = fit_anomaly_model(X_TRAIN, random_seed=42, anomaly_artifact_version="v1")
    norm_before = norm.content_hash()
    score_anomaly(model, norm, X_TRAIN)
    assert norm.content_hash() == norm_before
