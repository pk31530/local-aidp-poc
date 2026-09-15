"""The one shared feature-transformation module (fix C2).

`compute_features()` is a pure function: given a transaction's fields, the
customer's profile snapshot, and their recent event history, it returns
every CURATED/FEATURES-layer computed field. It has no idea whether its
inputs came from a Polars DataFrame (batch pipeline, src/processing) or a
live Postgres query (real-time serving, src/api and src/ingestion) — both
call sites build the same two input shapes (`CustomerProfileSnapshot`,
`Sequence[RecentEvent]`) via their own adapter and then call this same
function, so feature logic is defined exactly once.

Unknown/new customers get `DEFAULT_PROFILE` (fix C3) instead of raising —
every ratio/flag has a defined, non-crashing behavior for a customer with no
profile and no history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from src.common.config import get_app_settings, get_fraud_rules
from src.common.timeutil import to_app_tz


@dataclass(frozen=True)
class CustomerProfileSnapshot:
    avg_transaction_amount: float = 0.0
    stddev_transaction_amount: float = 0.0
    known_devices: frozenset = field(default_factory=frozenset)
    known_countries: frozenset = field(default_factory=frozenset)
    home_country: str | None = None


# fix C3: default-profile fallback for a customer_id with no profile row and
# no history at all. Every feature below has a defined value against this.
DEFAULT_PROFILE = CustomerProfileSnapshot()


@dataclass(frozen=True)
class RecentEvent:
    event_type: str  # "transaction_success" | "transaction_failed"
    occurred_at: datetime
    device_id: str | None = None
    country: str | None = None
    amount: float | None = None


@dataclass(frozen=True)
class RiskLookups:
    merchant_risk: dict
    country_risk: dict
    default_merchant_risk: float = 0.2
    default_country_risk: float = 0.2

    def merchant_score(self, merchant: str) -> float:
        return float(self.merchant_risk.get(merchant, self.default_merchant_risk))

    def country_score(self, country: str) -> float:
        return float(self.country_risk.get(country, self.default_country_risk))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "merchant_risk": self.merchant_risk,
                    "country_risk": self.country_risk,
                    "default_merchant_risk": self.default_merchant_risk,
                    "default_country_risk": self.default_country_risk,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "RiskLookups":
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def empty(cls) -> "RiskLookups":
        """Neutral fallback (fix C3) when no lookup artifact exists yet —
        every merchant/country scores at the default, nothing crashes."""
        return cls(merchant_risk={}, country_risk={})


_WINDOWS = get_app_settings()["velocity_windows"]
_NIGHT = get_fraud_rules()["night_window"]

WINDOW_10M = timedelta(minutes=_WINDOWS["transactions_last_10m_minutes"])
WINDOW_1H = timedelta(minutes=_WINDOWS["transactions_last_1h_minutes"])
WINDOW_24H = timedelta(minutes=_WINDOWS["transactions_last_24h_minutes"])
WINDOW_FAILED_1H = timedelta(minutes=_WINDOWS["failed_attempts_1h_minutes"])
NIGHT_START_HOUR = _NIGHT["start_hour"]
NIGHT_END_HOUR = _NIGHT["end_hour"]


def _is_night_hour(hour: int) -> bool:
    if NIGHT_START_HOUR > NIGHT_END_HOUR:
        # wraps midnight, e.g. 23 -> 5
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR
    return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR


def compute_features(
    *,
    amount: float,
    merchant: str,
    country: str,
    device_id: str,
    transaction_timestamp: datetime,
    profile: CustomerProfileSnapshot,
    recent_events: Sequence[RecentEvent],
    risk_lookups: RiskLookups,
) -> dict:
    """Pure, side-effect-free. `recent_events` must contain only events for
    this customer strictly before `transaction_timestamp` (callers/adapters
    are responsible for that filter); this function derives the 10m/1h/24h
    windows from it.
    """
    avg = profile.avg_transaction_amount
    stddev = profile.stddev_transaction_amount

    amount_vs_customer_average = (amount / avg) if avg > 0 else 1.0
    amount_zscore = ((amount - avg) / stddev) if stddev > 0 else 0.0

    is_new_device = device_id not in profile.known_devices
    is_new_country = country not in profile.known_countries

    local_ts = to_app_tz(transaction_timestamp)
    night_transaction_flag = _is_night_hour(local_ts.hour)

    tx_events = [e for e in recent_events if e.event_type == "transaction_success"]
    failed_events = [e for e in recent_events if e.event_type == "transaction_failed"]

    transactions_last_10m = sum(1 for e in tx_events if transaction_timestamp - e.occurred_at <= WINDOW_10M)
    transactions_last_1h = sum(1 for e in tx_events if transaction_timestamp - e.occurred_at <= WINDOW_1H)
    transactions_last_24h = sum(1 for e in tx_events if transaction_timestamp - e.occurred_at <= WINDOW_24H)
    failed_attempts_last_1h = sum(
        1 for e in failed_events if transaction_timestamp - e.occurred_at <= WINDOW_FAILED_1H
    )

    return {
        # CURATED-layer names
        "customer_average_amount": avg,
        "is_new_device": is_new_device,
        "is_new_country": is_new_country,
        "transaction_count_1h": transactions_last_1h,
        "transaction_count_24h": transactions_last_24h,
        "failed_attempts_1h": failed_attempts_last_1h,
        # FEATURES-layer names (ML-ready)
        "amount_vs_customer_average": amount_vs_customer_average,
        "amount_zscore": amount_zscore,
        "new_device_flag": is_new_device,
        "new_country_flag": is_new_country,
        "night_transaction_flag": night_transaction_flag,
        "transactions_last_10m": transactions_last_10m,
        "transactions_last_1h": transactions_last_1h,
        "transactions_last_24h": transactions_last_24h,
        "failed_attempts_last_1h": failed_attempts_last_1h,
        "merchant_risk_score": risk_lookups.merchant_score(merchant),
        "country_risk_score": risk_lookups.country_score(country),
    }


# ----------------------------------------------------------------------
# Model input feature order. XGBoost/scikit-learn need a fixed column order
# at both train and inference time; this is the single source of truth for
# it (fix C2 — batch training and real-time serving both import this list
# instead of hand-writing the column order twice).
# is_fraud is deliberately excluded: it is the label, never a model input.
# ----------------------------------------------------------------------
MODEL_FEATURE_COLUMNS = [
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
]
