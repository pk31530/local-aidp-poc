"""Integration: feature pipeline -> model.

Loads the real registered `fraud-detection-model@champion`, runs
`src.common.features.compute_features()` (the same function used by both
batch and real-time paths) for a suspicious vs. a normal transaction, and
confirms the model's predictions are directionally correct end to end —
not just that compute_features() returns the right dict shape in
isolation.
"""
from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import pandas as pd
import pytest

from src.common.features import (
    MODEL_FEATURE_COLUMNS,
    CustomerProfileSnapshot,
    RecentEvent,
    RiskLookups,
    compute_features,
)
from src.common.mlflow_setup import configure_mlflow

PROFILE = CustomerProfileSnapshot(
    avg_transaction_amount=8000.0,
    stddev_transaction_amount=1500.0,
    known_devices=frozenset({"DEV100001"}),
    known_countries=frozenset({"India"}),
    home_country="India",
)


@pytest.fixture(scope="module")
def loaded_model():
    configure_mlflow()
    return mlflow.xgboost.load_model("models:/fraud-detection-model@champion")


def _predict(model, features: dict, amount: float) -> float:
    X = pd.DataFrame([{**features, "amount": amount}])[MODEL_FEATURE_COLUMNS].astype(float)
    return float(model.predict_proba(X)[:, 1][0])


def test_suspicious_transaction_scores_higher_than_normal(loaded_model):
    # Matches the guide's worked demo scenario almost exactly.
    suspicious_features = compute_features(
        amount=82000.0,
        merchant="Electronics",
        country="Singapore",
        device_id="DEV998",
        transaction_timestamp=datetime(2026, 6, 1, 20, 45, tzinfo=timezone.utc),  # 2:15am IST
        profile=PROFILE,
        recent_events=[
            RecentEvent("transaction_failed", datetime(2026, 6, 1, 20, 30, tzinfo=timezone.utc)),
            RecentEvent("transaction_failed", datetime(2026, 6, 1, 20, 35, tzinfo=timezone.utc)),
            RecentEvent("transaction_failed", datetime(2026, 6, 1, 20, 40, tzinfo=timezone.utc)),
        ],
        risk_lookups=RiskLookups.empty(),
    )
    normal_features = compute_features(
        amount=4500.0,
        merchant="Grocery",
        country="India",
        device_id="DEV100001",
        transaction_timestamp=datetime(2026, 6, 1, 13, 45, tzinfo=timezone.utc),  # 7:15pm IST
        profile=PROFILE,
        recent_events=[],
        risk_lookups=RiskLookups.empty(),
    )

    suspicious_score = _predict(loaded_model, suspicious_features, 82000.0)
    normal_score = _predict(loaded_model, normal_features, 4500.0)

    assert suspicious_score > normal_score
    assert suspicious_score > 0.5
    assert normal_score < 0.5
