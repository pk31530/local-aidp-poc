"""Shared train/val/test split assignment (Finding 1 fix).

Merchant/country risk lookups (src/processing/risk_lookups.py) must be fit
on the train partition only, and src/ml/train.py must evaluate on exactly
that same partition — otherwise smoothed fraud rates for val/test rows leak
into a feature the model then trains on. `assign_split` is the single source
of truth for how rows are partitioned; its result is persisted as a "split"
column on the FEATURES output so training never re-derives its own,
independent split.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def assign_split(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Stratified 70/15/15 train/val/test split on `is_fraud`.

    Returns `df` with a new "split" column ("train" | "val" | "test") added;
    row order and all existing columns are preserved.
    """
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("split"))

    indices = list(range(df.height))
    labels = df["is_fraud"].to_list()

    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_SIZE + TEST_SIZE), stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), stratify=temp_labels, random_state=seed
    )

    split_col = [""] * df.height
    for i in train_idx:
        split_col[i] = "train"
    for i in val_idx:
        split_col[i] = "val"
    for i in test_idx:
        split_col[i] = "test"

    return df.with_columns(pl.Series("split", split_col))


# ----------------------------------------------------------------------
# v1.3 addition (guide section 12) -- purely additive, assign_split() above
# is unchanged and still used by v1.1/v1.2. Chronological, not stratified:
# rows are cut by TIME order, with a configurable purge gap around each
# boundary, for the reference channel's supervised training population
# (src.fraud_intel.models.training).
# ----------------------------------------------------------------------


def _group_start_indices(sorted_timestamps: list) -> list[int]:
    """Row indices, in a pre-sorted-by-timestamp list, where a new
    distinct timestamp value begins. Always includes 0 and len(...) (a
    virtual boundary just past the last row), so every real boundary lies
    strictly between two entries of this list."""
    starts = [0]
    for i in range(1, len(sorted_timestamps)):
        if sorted_timestamps[i] != sorted_timestamps[i - 1]:
            starts.append(i)
    starts.append(len(sorted_timestamps))
    return starts


def _nearest_group_boundary(candidates: list[int], raw_index: int) -> int:
    """The candidate group-start index nearest raw_index; ties broken
    toward the earlier boundary, so the choice is fully deterministic for
    a fixed dataset."""
    return min(candidates, key=lambda start: (abs(start - raw_index), start))


def realized_split_fractions(df: pl.DataFrame, split_col: str = "split") -> dict[str, float]:
    """The ACTUAL row-count fraction each partition ended up with, after
    timestamp-group-boundary snapping and the purge gap -- guide section
    12 requires this be recorded, since it will not exactly equal the
    requested train_frac/calib_frac/test_frac."""
    total = df.height
    if total == 0:
        return {"train": 0.0, "calibration": 0.0, "test": 0.0}
    return {name: df.filter(pl.col(split_col) == name).height / total for name in ("train", "calibration", "test")}


def assign_chronological_split(
    df: pl.DataFrame,
    timestamp_col: str,
    id_col: str,
    train_frac: float,
    calib_frac: float,
    test_frac: float,
    purge_gap: timedelta = timedelta(0),
) -> pl.DataFrame:
    """Sorts by (timestamp_col, id_col) for a fully deterministic order,
    then places the train/calibration and calibration/test boundaries at
    the nearest actual TIMESTAMP-GROUP boundary to the requested row-count
    fraction cut -- rows sharing the same timestamp are never split across
    two partitions. A purge_gap-wide time buffer is then dropped from both
    sides of each boundary (rows in the buffer belong to no partition and
    are removed from the returned frame entirely). Adds a "split" column
    ("train"|"calibration"|"test").

    Generic validation only (label-agnostic -- this function has no label
    column parameter and cannot check class balance; that is
    src.fraud_intel.models.training's job): raises ValueError if the frame
    is empty, if fewer than two distinct timestamp groups exist (no room
    for two real boundaries), or if any partition is empty after the
    purge gap is applied.
    """
    if df.height == 0:
        raise ValueError("assign_chronological_split requires at least one row")
    total_frac = train_frac + calib_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"train_frac + calib_frac + test_frac must sum to 1.0, got {total_frac}")
    if min(train_frac, calib_frac, test_frac) < 0:
        raise ValueError("train_frac, calib_frac, and test_frac must each be non-negative")

    sorted_df = df.sort([timestamp_col, id_col])
    timestamps = sorted_df[timestamp_col].to_list()
    n = len(timestamps)

    group_starts = _group_start_indices(timestamps)  # includes 0 and n
    inner_group_starts = group_starts[1:-1]
    if not inner_group_starts:
        raise ValueError(
            "assign_chronological_split requires at least two distinct timestamp "
            "groups to place a train/calibration boundary without splitting a group"
        )

    raw_boundary_1 = round(n * train_frac)
    raw_boundary_2 = round(n * (train_frac + calib_frac))

    boundary_1 = _nearest_group_boundary(inner_group_starts, raw_boundary_1)
    candidates_2 = [s for s in inner_group_starts if s > boundary_1]
    if not candidates_2:
        raise ValueError(
            "assign_chronological_split requires at least three distinct timestamp "
            "groups to place both the train/calibration and calibration/test "
            "boundaries without splitting a group"
        )
    boundary_2 = _nearest_group_boundary(candidates_2, raw_boundary_2)

    boundary_1_time = timestamps[boundary_1]
    boundary_2_time = timestamps[boundary_2]
    purge_gap_seconds = purge_gap.total_seconds()

    split_labels: list[str | None] = []
    for i, ts in enumerate(timestamps):
        if i < boundary_1:
            distance = (boundary_1_time - ts).total_seconds()
            split_labels.append(None if (purge_gap_seconds > 0 and distance <= purge_gap_seconds) else "train")
        elif i < boundary_2:
            distance_to_b1 = (ts - boundary_1_time).total_seconds()
            distance_to_b2 = (boundary_2_time - ts).total_seconds()
            too_close = purge_gap_seconds > 0 and (distance_to_b1 < purge_gap_seconds or distance_to_b2 <= purge_gap_seconds)
            split_labels.append(None if too_close else "calibration")
        else:
            distance = (ts - boundary_2_time).total_seconds()
            split_labels.append(None if (purge_gap_seconds > 0 and distance < purge_gap_seconds) else "test")

    result = sorted_df.with_columns(pl.Series("split", split_labels)).filter(pl.col("split").is_not_null())

    if result.height == 0:
        raise ValueError("purge_gap removed every row -- reduce purge_gap or increase dataset size")
    for name in ("train", "calibration", "test"):
        if result.filter(pl.col("split") == name).height == 0:
            raise ValueError(f"purge_gap and/or split fractions leave the {name!r} partition empty")

    return result
