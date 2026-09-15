"""Computes merchant/country risk-score lookups from observed historical
fraud rates — data-derived, not manually assigned per-merchant or
per-country assumptions. Only possible on labeled (historical/training)
data; the resulting lookup artifact is then reused unchanged by real-time
serving via src.common.features.RiskLookups.load() (fix C2).
"""
from __future__ import annotations

import polars as pl

from src.common.features import RiskLookups

_ALPHA = 5.0  # Laplace-style smoothing strength (shrinks low-volume groups toward the overall rate)
_MIN_SCORE = 0.02
_MAX_SCORE = 0.95


def _smoothed_rate_by_group(df: pl.DataFrame, group_col: str, overall_rate: float) -> dict:
    agg = df.group_by(group_col).agg(
        [
            pl.col("is_fraud").sum().alias("fraud_count"),
            pl.col("is_fraud").count().alias("total_count"),
        ]
    )
    out = {}
    for row in agg.to_dicts():
        smoothed = (row["fraud_count"] + _ALPHA * overall_rate) / (row["total_count"] + _ALPHA)
        out[row[group_col]] = round(min(max(smoothed, _MIN_SCORE), _MAX_SCORE), 4)
    return out


def compute_risk_lookups(labeled_df: pl.DataFrame) -> RiskLookups:
    overall_rate = float(labeled_df["is_fraud"].mean()) if labeled_df.height else 0.03
    merchant_risk = _smoothed_rate_by_group(labeled_df, "merchant", overall_rate)
    country_risk = _smoothed_rate_by_group(labeled_df, "country", overall_rate)
    return RiskLookups(
        merchant_risk=merchant_risk,
        country_risk=country_risk,
        default_merchant_risk=round(overall_rate, 4),
        default_country_risk=round(overall_rate, 4),
    )
