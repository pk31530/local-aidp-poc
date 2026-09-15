"""Historical transaction + fraud-pattern generator (Phase 3).

Produces:
  - the historical transactions table (the Phase 4 batch pipeline's input
    file), and
  - a supplementary "failed attempts" event stream, used by Phase 4 to give
    the failed_attempts / velocity features real historical signal instead
    of always-zero.

`is_fraud` is the training ground-truth label. It is generated here but is
never treated as a model input feature (see src/common/features.py, fix C2).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from src.generator.customers import COUNTRIES, DEVICE_PREFIX

NEW_DEVICE_PREFIX = "DEVNEW"

MERCHANTS = [
    ("Grocery", False),
    ("Dining", False),
    ("Fuel", False),
    ("Utilities", False),
    ("Pharmacy", False),
    ("Fashion", False),
    ("Entertainment", False),
    ("Electronics", True),
    ("Travel", True),
    ("Jewellery", True),
]
MERCHANT_NAMES = [m for m, _ in MERCHANTS]
MERCHANT_WEIGHTS = [0.18, 0.16, 0.14, 0.10, 0.10, 0.10, 0.08, 0.06, 0.05, 0.03]
RISKY_MERCHANTS = [m for m, risky in MERCHANTS if risky]

PAYMENT_METHODS = ["CARD", "UPI", "NETBANKING", "WALLET"]
PAYMENT_WEIGHTS = [0.40, 0.35, 0.15, 0.10]

FRAUD_PATTERNS = [
    "HIGH_AMOUNT",
    "NEW_DEVICE",
    "NEW_COUNTRY",
    "NIGHT_TIME",
    "RISKY_MERCHANT",
    "RAPID_VELOCITY",
    "FAILED_ATTEMPTS",
]


def _split_counts(total: int, n_customers: int, rng: np.random.Generator) -> np.ndarray:
    """Distribute `total` transactions across customers with a moderate skew
    (some customers more active than others), summing to exactly `total`."""
    weights = rng.gamma(shape=2.0, scale=1.0, size=n_customers) + 0.05
    weights = weights / weights.sum()
    counts = np.floor(weights * total).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        idx = rng.choice(n_customers, size=remainder, replace=True)
        for i in idx:
            counts[i] += 1
    return counts


def _random_daytime_timestamp(rng: np.random.Generator, day: datetime) -> datetime:
    hour = int(np.clip(rng.normal(loc=14, scale=4), 6, 22))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return day.replace(hour=hour, minute=minute, second=second, microsecond=0)


def _random_night_timestamp(rng: np.random.Generator, day: datetime) -> datetime:
    # Night window: 23:00-05:00 local (config/fraud_rules.yaml night_window).
    hour = int(rng.choice([23, 0, 1, 2, 3, 4]))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return day.replace(hour=hour, minute=minute, second=second, microsecond=0)


def generate_history(
    customers: pl.DataFrame,
    total_transactions: int,
    fraud_ratio: float,
    seed: int,
    reference_date: datetime,
    lookback_days: int = 90,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic for a given (customers, total_transactions, fraud_ratio,
    seed, reference_date, lookback_days). Returns (transactions_df,
    failed_attempts_df)."""
    rng = np.random.default_rng(seed + 1)  # independent stream from customer generation

    customer_records = customers.to_dicts()
    n_customers = len(customer_records)
    counts = _split_counts(total_transactions, n_customers, rng)

    rows: list[dict[str, Any]] = []
    tx_counter = 0
    for cust, count in zip(customer_records, counts):
        known_devices = list(cust["known_devices"])
        known_countries = list(cust["known_countries"])
        home_country = cust["home_country"]
        avg_amount = cust["avg_transaction_amount"]
        stddev_amount = max(cust["stddev_transaction_amount"], 1.0)

        for _ in range(int(count)):
            tx_counter += 1
            transaction_id = f"TXH{seed}{tx_counter:07d}"
            day_offset = int(rng.integers(0, lookback_days))
            day = reference_date - timedelta(days=day_offset)
            ts = _random_daytime_timestamp(rng, day)

            amount = round(float(np.clip(rng.normal(avg_amount, stddev_amount), 10, None)), 2)
            country = home_country if rng.random() > 0.05 else str(rng.choice(known_countries))
            device_id = str(rng.choice(known_devices))
            merchant = str(rng.choice(MERCHANT_NAMES, p=MERCHANT_WEIGHTS))
            payment_method = str(rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS))

            rows.append(
                {
                    "transaction_id": transaction_id,
                    "customer_id": cust["customer_id"],
                    "transaction_timestamp": ts.isoformat(),
                    "amount": amount,
                    "merchant": merchant,
                    "country": country,
                    "device_id": device_id,
                    "payment_method": payment_method,
                    "is_fraud": False,
                }
            )

    n_fraud = round(total_transactions * fraud_ratio)
    fraud_indices = rng.choice(len(rows), size=n_fraud, replace=False)

    failed_attempts_rows: list[dict[str, Any]] = []

    rows_by_customer: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        rows_by_customer.setdefault(r["customer_id"], []).append(idx)

    customer_by_id = {c["customer_id"]: c for c in customer_records}

    for idx in fraud_indices:
        idx = int(idx)
        row = rows[idx]
        row["is_fraud"] = True
        cust = customer_by_id[row["customer_id"]]
        n_patterns = int(rng.integers(2, 4))  # combine 2-3 patterns, like a real case
        patterns = rng.choice(FRAUD_PATTERNS, size=n_patterns, replace=False)

        base_ts = datetime.fromisoformat(row["transaction_timestamp"])

        for pattern in patterns:
            if pattern == "HIGH_AMOUNT":
                multiplier = rng.uniform(5, 15)
                row["amount"] = round(float(cust["avg_transaction_amount"] * multiplier), 2)
            elif pattern == "NEW_DEVICE":
                row["device_id"] = f"{NEW_DEVICE_PREFIX}{int(rng.integers(100000, 999999))}"
            elif pattern == "NEW_COUNTRY":
                others = [c for c in COUNTRIES if c not in cust["known_countries"]]
                row["country"] = str(rng.choice(others))
            elif pattern == "NIGHT_TIME":
                night_ts = _random_night_timestamp(rng, base_ts.replace(hour=0, minute=0, second=0))
                row["transaction_timestamp"] = night_ts.isoformat()
                base_ts = night_ts
            elif pattern == "RISKY_MERCHANT":
                row["merchant"] = str(rng.choice(RISKY_MERCHANTS))
            elif pattern == "RAPID_VELOCITY":
                siblings = [i for i in rows_by_customer[row["customer_id"]] if i != idx]
                if siblings:
                    burst_size = min(len(siblings), int(rng.integers(2, 4)))
                    burst = rng.choice(siblings, size=burst_size, replace=False)
                    for b_idx in np.atleast_1d(burst):
                        offset_minutes = int(rng.integers(1, 8))
                        rows[int(b_idx)]["transaction_timestamp"] = (
                            base_ts + timedelta(minutes=offset_minutes)
                        ).isoformat()
            elif pattern == "FAILED_ATTEMPTS":
                n_fail = int(rng.integers(1, 4))
                for _ in range(n_fail):
                    minutes_before = int(rng.integers(1, 30))
                    failed_attempts_rows.append(
                        {
                            "customer_id": row["customer_id"],
                            "transaction_id": row["transaction_id"],
                            "event_type": "transaction_failed",
                            "amount": row["amount"],
                            "country": row["country"],
                            "device_id": row["device_id"],
                            "occurred_at": (base_ts - timedelta(minutes=minutes_before)).isoformat(),
                        }
                    )

    transactions_df = pl.DataFrame(rows)
    if failed_attempts_rows:
        failed_attempts_df = pl.DataFrame(failed_attempts_rows)
    else:
        failed_attempts_df = pl.DataFrame(
            schema={
                "customer_id": pl.Utf8,
                "transaction_id": pl.Utf8,
                "event_type": pl.Utf8,
                "amount": pl.Float64,
                "country": pl.Utf8,
                "device_id": pl.Utf8,
                "occurred_at": pl.Utf8,
            }
        )
    return transactions_df, failed_attempts_df
