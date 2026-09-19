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
