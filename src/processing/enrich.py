"""Batch adapter onto the shared feature module (fix C2).

Builds `CustomerProfileSnapshot` (from the generator's customers.parquet
baseline) and `RecentEvent` history (from this customer's own prior
transactions + the historical failed-attempts file) for every row, then
calls the exact same `compute_features()` that the real-time adapter in
src/common/feature_store.py calls. One computation pass produces every
CURATED- and FEATURES-layer column; src/processing/views.py then projects
the two column subsets from it.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.common.features import (
    DEFAULT_PROFILE,
    CustomerProfileSnapshot,
    RecentEvent,
    RiskLookups,
    compute_features,
)
from src.common.logging import get_logger

log = get_logger(__name__)


def _profile_lookup(customers_df: pl.DataFrame) -> dict:
    lookup = {}
    for row in customers_df.to_dicts():
        lookup[row["customer_id"]] = CustomerProfileSnapshot(
            avg_transaction_amount=float(row["avg_transaction_amount"]),
            stddev_transaction_amount=float(row["stddev_transaction_amount"]),
            known_devices=frozenset(row["known_devices"]),
            known_countries=frozenset(row["known_countries"]),
            home_country=row["home_country"],
        )
    return lookup


def enrich_transactions(
    clean_df: pl.DataFrame,
    customers_df: pl.DataFrame,
    failed_attempts_df: pl.DataFrame,
    risk_lookups: RiskLookups,
) -> pl.DataFrame:
    """Returns one row per input transaction with every original field plus
    every compute_features() key merged in."""
    if clean_df.height == 0:
        return clean_df

    profile_lookup = _profile_lookup(customers_df)

    fa_by_customer: dict[str, list[datetime]] = {}
    if failed_attempts_df.height > 0:
        for row in failed_attempts_df.to_dicts():
            ts = row["occurred_at"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            fa_by_customer.setdefault(row["customer_id"], []).append(ts)
    for customer_id in fa_by_customer:
        fa_by_customer[customer_id].sort()

    sorted_df = clean_df.sort("transaction_timestamp")
    out_rows = []

    for customer_id, group in sorted_df.group_by("customer_id", maintain_order=True):
        customer_id = customer_id[0] if isinstance(customer_id, tuple) else customer_id
        profile = profile_lookup.get(customer_id, DEFAULT_PROFILE)
        failed_ts = fa_by_customer.get(customer_id, [])

        tx_history: list[datetime] = []

        for row in group.sort("transaction_timestamp").to_dicts():
            ts = row["transaction_timestamp"]

            recent_events = [RecentEvent("transaction_success", t) for t in tx_history]
            recent_events += [RecentEvent("transaction_failed", t) for t in failed_ts if t < ts]

            features = compute_features(
                amount=row["amount"],
                merchant=row["merchant"],
                country=row["country"],
                device_id=row["device_id"],
                transaction_timestamp=ts,
                profile=profile,
                recent_events=recent_events,
                risk_lookups=risk_lookups,
            )

            out_rows.append({**row, **features})
            tx_history.append(ts)

    enriched_df = pl.DataFrame(out_rows).sort("transaction_timestamp")
    log.info("enrich_stage_complete", rows=enriched_df.height)
    return enriched_df
