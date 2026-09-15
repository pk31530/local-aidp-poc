"""Synthetic customer generator (Phase 3).

Produces one row per customer carrying both dimension fields (for the
`customers` table) and profile-baseline fields (for the `customer_profiles`
online feature store, fix C1) — the same row seeds both tables directly, no
extra derivation step.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from faker import Faker

COUNTRIES = ["India", "USA", "UK", "Singapore", "UAE", "Germany", "Australia", "Canada", "Nigeria"]
COUNTRY_WEIGHTS = [0.55, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04]

DEVICE_PREFIX = "DEV"


def _make_device_id(rng: np.random.Generator) -> str:
    return f"{DEVICE_PREFIX}{int(rng.integers(100000, 999999))}"


def generate_customers(n: int, seed: int, reference_date: datetime) -> pl.DataFrame:
    """Deterministic for a given (n, seed, reference_date)."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    rows = []
    for i in range(n):
        customer_id = f"C{1000 + i}"
        home_country = str(rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS))

        known_countries = [home_country]
        if rng.random() < 0.15:
            other = str(rng.choice([c for c in COUNTRIES if c != home_country]))
            known_countries.append(other)

        num_devices = int(rng.integers(1, 3))  # 1-2 known devices
        known_devices = [_make_device_id(rng) for _ in range(num_devices)]

        avg_amount = float(np.round(rng.lognormal(mean=8.2, sigma=0.55), 2))
        stddev_amount = float(np.round(avg_amount * rng.uniform(0.15, 0.35), 2))

        signup_days_ago = int(rng.integers(30, 730))
        signup_date = (reference_date - timedelta(days=signup_days_ago)).date()

        rows.append(
            {
                "customer_id": customer_id,
                "full_name": fake.name(),
                "home_country": home_country,
                "known_countries": known_countries,
                "known_devices": known_devices,
                "avg_transaction_amount": avg_amount,
                "stddev_transaction_amount": stddev_amount,
                "signup_date": signup_date.isoformat(),
            }
        )

    return pl.DataFrame(rows)


def demo_customer_row(customer_id: str, reference_date: datetime) -> dict:
    """A fixed, hand-specified customer used by the guide's worked demo
    scenario (fix C3): typical spend ~8,000, home country India, one known
    device — so `./scripts/run_stream.sh` / the API demo can reference this
    exact customer_id and get the documented result.
    """
    return {
        "customer_id": customer_id,
        "full_name": "Demo Customer",
        "home_country": "India",
        "known_countries": ["India"],
        "known_devices": ["DEV100001"],
        "avg_transaction_amount": 8000.0,
        "stddev_transaction_amount": 1500.0,
        "signup_date": (reference_date - timedelta(days=400)).date().isoformat(),
    }
