"""CLEAN stage: duplicate removal, timestamp normalization, null handling,
and a business-rule check Pydantic's type system can't express (allowed
payment methods). Everything rejected here is reported, not dropped
silently (guide section 22 / Phase 4 acceptance: "rejected records
reported").
"""
from __future__ import annotations

import polars as pl

from src.common.logging import get_logger

log = get_logger(__name__)

ALLOWED_PAYMENT_METHODS = {"CARD", "UPI", "NETBANKING", "WALLET"}

REQUIRED_FIELDS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "merchant",
    "country",
    "device_id",
    "payment_method",
]


def clean_transactions(df: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    if df.height == 0:
        return df, []

    before = df.height
    rejected: list[dict] = []

    # timestamp normalization: canonical UTC (RAW already does this, but
    # CLEAN re-asserts it as its own explicit responsibility).
    df = df.with_columns(pl.col("transaction_timestamp").dt.convert_time_zone("UTC"))

    # null handling
    null_mask = pl.any_horizontal([pl.col(c).is_null() for c in REQUIRED_FIELDS])
    for row in df.filter(null_mask).to_dicts():
        rejected.append({**row, "rejection_reason": "null_in_required_field"})
    df = df.filter(~null_mask)

    # business-rule rejection: unrecognized payment method
    bad_payment_mask = ~pl.col("payment_method").is_in(list(ALLOWED_PAYMENT_METHODS))
    for row in df.filter(bad_payment_mask).to_dicts():
        rejected.append({**row, "rejection_reason": f"invalid_payment_method:{row['payment_method']}"})
    df = df.filter(~bad_payment_mask)

    # duplicate removal by transaction_id (keep first occurrence)
    df = df.with_row_index("_ridx")
    deduped = df.unique(subset=["transaction_id"], keep="first", maintain_order=True)
    kept_idx = deduped["_ridx"]
    dup_df = df.filter(~pl.col("_ridx").is_in(kept_idx))
    for row in dup_df.drop("_ridx").to_dicts():
        rejected.append({**row, "rejection_reason": "duplicate_transaction_id"})
    df = deduped.drop("_ridx")

    log.info("clean_stage_complete", input=before, output=df.height, rejected=len(rejected))
    return df, rejected
