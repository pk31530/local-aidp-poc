"""Projects the two documented column sets (guide section 8) out of the one
enriched DataFrame that src/processing/enrich.py produces."""
from __future__ import annotations

import polars as pl

CURATED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "merchant",
    "country",
    "device_id",
    "payment_method",
    "is_fraud",
    "customer_average_amount",
    "is_new_device",
    "is_new_country",
    "transaction_count_1h",
    "transaction_count_24h",
    "failed_attempts_1h",
]

FEATURES_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "amount_vs_customer_average",
    "amount_zscore",
    "new_device_flag",
    "new_country_flag",
    "night_transaction_flag",
    "transactions_last_10m",
    "transactions_last_1h",
    "transactions_last_24h",
    "failed_attempts_last_1h",
    "merchant_risk_score",
    "country_risk_score",
    "is_fraud",
]


def to_curated(enriched_df: pl.DataFrame) -> pl.DataFrame:
    return enriched_df.select(CURATED_COLUMNS)


def to_features(enriched_df: pl.DataFrame) -> pl.DataFrame:
    return enriched_df.select(FEATURES_COLUMNS)
