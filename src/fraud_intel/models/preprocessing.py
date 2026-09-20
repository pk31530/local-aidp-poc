"""Training-window-only preprocessing artifact (guide sections 9, 12, 22).

Serialized as deterministic, canonical-JSON -- never pickle/joblib, so
there is no arbitrary-code-execution surface on load and the artifact is
fully human-inspectable, matching src.common.features.RiskLookups'
existing JSON-artifact precedent.

`ChannelPreprocessor` is an immutable, frozen value: `.fit(...)` is a
classmethod constructor, not an in-place mutation, so there is no
"forgot to call fit first" state to guard against -- an instance either
exists (and is fit) or does not.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn

UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN__"
UNKNOWN_CATEGORY_INDEX = 0


class FeatureSchemaMismatchError(ValueError):
    """The feature names/order supplied at fit or transform time do not
    exactly match the preprocessor's own ordered feature_columns. Missing,
    unexpected, AND reordered features all raise this -- never silently
    reordered, padded, truncated, or otherwise adapted (guide section 9's
    FeatureSchemaMismatchError contract)."""


def _validate_feature_names(rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> None:
    expected = list(feature_columns)
    for row in rows:
        actual = list(row.keys())
        if actual != expected:
            missing = [c for c in expected if c not in row]
            unexpected = [c for c in row if c not in expected]
            raise FeatureSchemaMismatchError(
                f"feature row does not match the expected ordered schema -- "
                f"missing={missing}, unexpected={unexpected}, "
                f"expected_order={expected}, actual_order={actual}"
            )


@dataclass(frozen=True)
class ChannelPreprocessor:
    feature_columns: tuple[str, ...]
    feature_schema_version: str
    preprocessing_artifact_version: str
    numeric_imputation_values: Mapping[str, float]
    means: Mapping[str, float]
    scales: Mapping[str, float]
    categorical_encodings: Mapping[str, Mapping[str, int]]
    unknown_category_policy: str
    library_versions: Mapping[str, str]

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_columns: Sequence[str],
        feature_schema_version: str,
        preprocessing_artifact_version: str,
    ) -> "ChannelPreprocessor":
        """Fits imputation values, mean/scale (numeric columns), and a
        vocabulary + reserved-unknown-index encoding (any column whose
        training-window values are all strings) -- on these `rows` only.
        The reference channel's current feature output (guide section 9)
        has no categorical columns, so categorical_encodings is empty in
        practice for online_banking today; the mechanism itself is real
        and exercised directly by tests/unit/test_fraud_intel_training.py
        against a standalone fixture, not left as an untested promise."""
        _validate_feature_names(rows, feature_columns)

        numeric_imputation_values: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        categorical_encodings: dict[str, dict[str, int]] = {}

        for column in feature_columns:
            values = [row[column] for row in rows]
            if values and all(isinstance(v, str) for v in values):
                vocabulary = sorted({v for v in values})
                encoding = {UNKNOWN_CATEGORY_TOKEN: UNKNOWN_CATEGORY_INDEX}
                for index, token in enumerate(vocabulary, start=1):
                    encoding[token] = index
                categorical_encodings[column] = encoding
            else:
                numeric_values = [float(v) for v in values if v is not None]
                median = float(statistics.median(numeric_values)) if numeric_values else 0.0
                numeric_imputation_values[column] = median
                filled = [float(v) if v is not None else median for v in values]
                mean = float(sum(filled) / len(filled)) if filled else 0.0
                variance = float(sum((v - mean) ** 2 for v in filled) / len(filled)) if filled else 0.0
                scale = variance**0.5
                means[column] = mean
                scales[column] = scale if scale > 0 else 1.0  # never divide by zero

        return cls(
            feature_columns=tuple(feature_columns),
            feature_schema_version=feature_schema_version,
            preprocessing_artifact_version=preprocessing_artifact_version,
            numeric_imputation_values=numeric_imputation_values,
            means=means,
            scales=scales,
            categorical_encodings=categorical_encodings,
            unknown_category_policy=(
                f"unseen categorical values map to the reserved {UNKNOWN_CATEGORY_TOKEN!r} "
                f"index ({UNKNOWN_CATEGORY_INDEX}); never raise, never silently invent a new category"
            ),
            library_versions={"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        )

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        _validate_feature_names(rows, self.feature_columns)
        transformed: list[list[float]] = []
        for row in rows:
            vector: list[float] = []
            for column in self.feature_columns:
                value = row[column]
                if column in self.categorical_encodings:
                    vector.append(float(self.categorical_encodings[column].get(value, UNKNOWN_CATEGORY_INDEX)))
                else:
                    numeric = float(value) if value is not None else self.numeric_imputation_values[column]
                    vector.append((numeric - self.means[column]) / self.scales[column])
            transformed.append(vector)
        return transformed

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "feature_schema_version": self.feature_schema_version,
            "preprocessing_artifact_version": self.preprocessing_artifact_version,
            "numeric_imputation_values": dict(self.numeric_imputation_values),
            "means": dict(self.means),
            "scales": dict(self.scales),
            "categorical_encodings": {k: dict(v) for k, v in self.categorical_encodings.items()},
            "unknown_category_policy": self.unknown_category_policy,
            "library_versions": dict(self.library_versions),
        }

    def to_canonical_json(self) -> str:
        """Deterministic, sorted-key serialization -- the same
        canonicalization technique RunConfig.config_hash() already uses."""
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "ChannelPreprocessor":
        return cls(
            feature_columns=tuple(data["feature_columns"]),
            feature_schema_version=data["feature_schema_version"],
            preprocessing_artifact_version=data["preprocessing_artifact_version"],
            numeric_imputation_values=dict(data["numeric_imputation_values"]),
            means=dict(data["means"]),
            scales=dict(data["scales"]),
            categorical_encodings={k: dict(v) for k, v in data["categorical_encodings"].items()},
            unknown_category_policy=data["unknown_category_policy"],
            library_versions=dict(data["library_versions"]),
        )
