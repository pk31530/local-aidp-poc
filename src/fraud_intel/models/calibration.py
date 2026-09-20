"""GBM probability calibration on the calibration window only (guide
sections 12, 13). `SplitManifest` records exactly which row IDs belong to
each partition of one chronological split -- including the rows the purge
gap removed entirely -- and backs a RUNTIME (not just documentation) guard
in `fit_calibrator()` against calibration ever touching train, purge, or
test rows.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

DEFAULT_CALIBRATION_METHOD = "isotonic"


class CalibrationLeakageError(ValueError):
    """A row identified as train, purge, or test data was supplied to
    calibration fitting -- raised before any fitting happens."""


@dataclass(frozen=True)
class SplitManifest:
    """Immutable. Built once, immediately after assign_chronological_split
    (src.common.splits), from that call's own output -- never
    reconstructed or guessed at inside fit_calibrator() itself."""

    train_ids: frozenset[str]
    calibration_ids: frozenset[str]
    purge_ids: frozenset[str]
    test_ids: frozenset[str]

    def __post_init__(self) -> None:
        groups = {
            "train": self.train_ids,
            "calibration": self.calibration_ids,
            "purge": self.purge_ids,
            "test": self.test_ids,
        }
        for (name_a, ids_a), (name_b, ids_b) in itertools.combinations(groups.items(), 2):
            overlap = ids_a & ids_b
            if overlap:
                raise ValueError(f"SplitManifest partitions {name_a!r} and {name_b!r} overlap: {sorted(overlap)}")


def fit_calibrator(
    *,
    row_ids: Sequence[str],
    raw_probabilities: Sequence[float],
    labels: Sequence[int],
    manifest: SplitManifest,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> IsotonicRegression:
    """Guide section 12: "calibration must never touch the test window --
    add a runtime assertion, not just a docstring, that raises if it's
    attempted." The three assertions below are that runtime guard, not
    merely true-by-absence -- a test constructs a manifest and deliberately
    passes a row_id from manifest.test_ids to confirm this raises."""
    if len(row_ids) != len(raw_probabilities) or len(row_ids) != len(labels):
        raise ValueError("row_ids, raw_probabilities, and labels must be the same length")

    supplied = set(row_ids)
    outside_calibration = supplied - manifest.calibration_ids
    if outside_calibration:
        raise CalibrationLeakageError(
            f"fit_calibrator received row id(s) outside the calibration partition: {sorted(outside_calibration)}"
        )
    forbidden = manifest.train_ids | manifest.purge_ids | manifest.test_ids
    leaked = supplied & forbidden
    if leaked:
        raise CalibrationLeakageError(f"fit_calibrator received train/purge/test row id(s): {sorted(leaked)}")

    if method != "isotonic":
        raise ValueError(f"unsupported calibration method {method!r}; only 'isotonic' is implemented in Phase 4")

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(raw_probabilities, dtype=float), np.asarray(labels, dtype=float))
    return calibrator
