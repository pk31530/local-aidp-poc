"""Unsupervised anomaly detection, per channel (guide section 14).

`IsolationForest` fit on the SAME source-alerted training population as
GBM/LR (Phase 4's chronological split) -- never calibration/test rows,
never the full bank-wide event population. This means the model learns
"unusual within the population of events that already received a source
alert," not "unusual across all bank transactions" -- a narrower,
deliberately scoped notion of anomaly for this POC (Phase 5 decision 2).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class AnomalyNormalization:
    """Deterministic min-max bounds fit on the TRAINING window's own raw
    decision_function distribution -- computed once, at training time,
    never re-derived at scoring time from a different population."""

    train_min: float
    train_max: float
    anomaly_artifact_version: str
    library_versions: Mapping[str, str]

    def normalize(self, raw_decision_value: float) -> float:
        span = self.train_max - self.train_min
        if span <= 0:
            return 0.0
        value = (self.train_max - raw_decision_value) / span
        return float(min(max(value, 0.0), 1.0))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "train_min": self.train_min,
            "train_max": self.train_max,
            "anomaly_artifact_version": self.anomaly_artifact_version,
            "library_versions": dict(self.library_versions),
        }

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fit_anomaly_model(
    X_train: Sequence[Sequence[float]], *, random_seed: int, anomaly_artifact_version: str
) -> tuple[IsolationForest, AnomalyNormalization]:
    """Label-agnostic: no `y` argument anywhere in this function. `X_train`
    must already be the training-window-only, source-alerted-only
    preprocessed matrix (the same one GBM/LR train on) -- never
    calibration or test rows."""
    model = IsolationForest(random_state=random_seed, contamination="auto", n_jobs=-1)
    model.fit(X_train)
    raw_scores = model.decision_function(X_train)
    normalization = AnomalyNormalization(
        train_min=float(np.min(raw_scores)),
        train_max=float(np.max(raw_scores)),
        anomaly_artifact_version=anomaly_artifact_version,
        library_versions={"scikit_learn": sklearn.__version__, "numpy": np.__version__},
    )
    return model, normalization


def score_anomaly(
    model: IsolationForest, normalization: AnomalyNormalization, X: Sequence[Sequence[float]]
) -> list[float]:
    """Pure inference -- never calls .fit(). Returns one bounded [0, 1]
    anomaly_score per row, higher meaning more anomalous."""
    raw_scores = model.decision_function(X)
    return [normalization.normalize(float(value)) for value in raw_scores]
