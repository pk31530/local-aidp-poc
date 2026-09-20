"""Shared, channel-agnostic feature core (guide section 9). Pure functions
only -- no database, Docker, or network access anywhere in this module.

`FeatureComputationContext` is the single, typed, immutable, validated
input every feature function accepts -- never a raw dict or loose scalar
parameters. Its own construction enforces the full as-of-time contract
(guide section 12) FAIL-FAST: a boundary or future-dated
`historical_events`/`source_alert_history` entry raises a `ValidationError`
immediately, it is never silently filtered out.

Label-leakage boundary: every function in this module accepts only
`FraudEvent`/`SourceAlertContext` objects and plain primitives. Nothing
here ever imports or references `SyntheticGroundTruthLabel`,
`analyst_disposition`, `outcome_status`, or `training_eligible` --
verified structurally by
tests/unit/test_fraud_intel_feature_leakage.py, not just by convention.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.common.features import NIGHT_END_HOUR, NIGHT_START_HOUR, WINDOW_1H, WINDOW_10M, WINDOW_24H
from src.common.timeutil import to_app_tz
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext

FEATURE_SCHEMA_VERSION = "v1"

SHARED_FEATURE_COLUMNS: list[str] = [
    "amount_vs_entity_average",
    "amount_zscore",
    "is_new_device",
    "is_new_ip",
    "night_transaction_flag",
    "events_last_10m",
    "events_last_1h",
    "events_last_24h",
    "prior_alert_count",
    "customer_observed_tenure_days",
    "account_observed_tenure_days",
    "counterparty_risk_score",
]


def _reject_naive_or_non_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("must be UTC")
    return value


def _is_night_hour(hour: int) -> bool:
    """Same rule as src.common.features._is_night_hour, reimplemented here
    (not imported -- that name is private to its module) against the same
    public NIGHT_START_HOUR/NIGHT_END_HOUR constants (config/fraud_rules.yaml's
    night_window), so v1.3 uses the identical, pinned-timezone definition of
    "night" as v1.1/v1.2 (fix M2) without reaching into another module's
    private API."""
    if NIGHT_START_HOUR > NIGHT_END_HOUR:
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR
    return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR


class FeatureComputationContext(BaseModel):
    """The single typed input to every v1.3 feature function.

    Validation, enforced at construction (fails fast, never filters
    silently):
    - `as_of_time` must be UTC-aware.
    - `as_of_time` must equal `current_event.event_timestamp` -- v1.3
      scores an event as of its own event time, never an arbitrary later
      "now" (guide section 12).
    - No `historical_events` entry may share `current_event.event_id`.
    - Every `historical_events` entry must have
      `event_timestamp < as_of_time` -- a boundary (`==`) or future
      (`>`) entry raises.
    - Every `source_alert_history` entry must have
      `source_alert_created_at < as_of_time` -- same fail-fast rule. This
      also guarantees the current event's own just-fired source alert (if
      any) can never be counted as a prior alert: its
      `source_alert_created_at` cannot be strictly before the current
      event's own `event_timestamp`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    current_event: FraudEvent
    historical_events: tuple[FraudEvent, ...]
    source_alert_history: tuple[SourceAlertContext, ...]
    as_of_time: datetime

    @field_validator("as_of_time")
    @classmethod
    def _validate_as_of_utc(cls, value: datetime) -> datetime:
        return _reject_naive_or_non_utc(value)

    @model_validator(mode="after")
    def _validate_as_of_time_contract(self) -> "FeatureComputationContext":
        if self.as_of_time != self.current_event.event_timestamp:
            raise ValueError(
                "as_of_time must equal current_event.event_timestamp for event-time "
                f"fraud scoring; got as_of_time={self.as_of_time!r}, "
                f"current_event.event_timestamp={self.current_event.event_timestamp!r}"
            )

        for record in self.historical_events:
            if record.event_id == self.current_event.event_id:
                raise ValueError(
                    f"historical_events must not include the current event (event_id={record.event_id})"
                )
            if record.event_timestamp >= self.as_of_time:
                raise ValueError(
                    f"historical_events entry {record.event_id} has event_timestamp "
                    f"{record.event_timestamp!r} >= as_of_time {self.as_of_time!r} -- "
                    "future or boundary records are not permitted; the caller must not "
                    "pre-filter loosely and rely on this constructor to catch the mistake"
                )

        for alert in self.source_alert_history:
            if alert.source_alert_created_at >= self.as_of_time:
                raise ValueError(
                    f"source_alert_history entry {alert.source_alert_id} has "
                    f"source_alert_created_at {alert.source_alert_created_at!r} >= "
                    f"as_of_time {self.as_of_time!r} -- future or boundary source alerts "
                    "are not permitted (this also guarantees the current event's own "
                    "source alert, if any, is never counted as a prior alert)"
                )

        return self

    def same_customer_history(self) -> tuple[FraudEvent, ...]:
        return tuple(r for r in self.historical_events if r.customer_id == self.current_event.customer_id)

    def same_account_history(self) -> tuple[FraudEvent, ...]:
        return tuple(r for r in self.historical_events if r.account_id == self.current_event.account_id)


class CounterpartyRiskLookups(BaseModel):
    """A generic, channel-agnostic "counterparty risk" lookup (guide
    section 9's "counterparty risk score"). Deliberately NOT a reuse of
    src.common.features.RiskLookups -- that class's fields are named for
    v1.1's specific merchant/country concepts; v1.3 needs one generic keyed
    lookup that a channel adapter can point at whichever field it
    considers "the counterparty" (beneficiary account, target account,
    merchant id, recipient handle, ...).

    The real fitted values are a training-window-only artifact (guide
    section 12) that does not exist until Phase 4 -- this class defines
    only the read contract, with a neutral default for every unseen key
    and for Phase 2's fixture-only tests.
    """

    model_config = ConfigDict(frozen=True)

    risk_by_key: Mapping[str, float]
    default_risk: float = 0.2

    def score(self, key: Optional[str]) -> float:
        if key is None:
            return float(self.default_risk)
        return float(self.risk_by_key.get(key, self.default_risk))

    @classmethod
    def empty(cls) -> "CounterpartyRiskLookups":
        return cls(risk_by_key={}, default_risk=0.2)


def _mean_and_population_stddev(values: Sequence[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return float(mean), 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    return float(mean), float(variance**0.5)


def compute_shared_features(
    ctx: FeatureComputationContext,
    *,
    counterparty_key: Optional[str] = None,
    counterparty_risk: CounterpartyRiskLookups = CounterpartyRiskLookups.empty(),
) -> dict[str, float | int | bool]:
    """Every value is derived from `ctx` alone plus the one genuinely
    external, well-typed artifact (`counterparty_risk` -- a training-time
    fit, not something derivable from a single scoring call's own
    history). No loose scalar parameters, no raw dicts.

    Velocity, amount-average/zscore, device/ip novelty, and tenure are all
    scoped to `ctx.same_customer_history()` (or, for the account-tenure
    feature, `ctx.same_account_history()`) -- never the raw, possibly
    broader `ctx.historical_events` -- so a caller that ever passes a
    wider history set (e.g. for a future shared-device lookup) cannot
    silently pollute these entity-scoped aggregates.
    """
    event = ctx.current_event
    same_customer = ctx.same_customer_history()
    same_account = ctx.same_account_history()

    amounts = [r.amount_minor_units for r in same_customer]
    mean_amount, stddev_amount = _mean_and_population_stddev(amounts)
    amount_vs_entity_average = (event.amount_minor_units / mean_amount) if mean_amount > 0 else 1.0
    amount_zscore = ((event.amount_minor_units - mean_amount) / stddev_amount) if stddev_amount > 0 else 0.0

    known_devices = {r.device_id for r in same_customer if r.device_id}
    is_new_device = (event.device_id not in known_devices) if event.device_id else False

    known_ips = {r.ip_address for r in same_customer if r.ip_address}
    is_new_ip = (event.ip_address not in known_ips) if event.ip_address else False

    local_ts = to_app_tz(event.event_timestamp)
    night_transaction_flag = _is_night_hour(local_ts.hour)

    events_last_10m = sum(1 for r in same_customer if ctx.as_of_time - r.event_timestamp <= WINDOW_10M)
    events_last_1h = sum(1 for r in same_customer if ctx.as_of_time - r.event_timestamp <= WINDOW_1H)
    events_last_24h = sum(1 for r in same_customer if ctx.as_of_time - r.event_timestamp <= WINDOW_24H)

    # FeatureComputationContext's own validation already guarantees every
    # entry in source_alert_history is strictly prior to as_of_time, so no
    # further time filtering is needed here.
    prior_alert_count = len(ctx.source_alert_history)

    customer_observed_tenure_days = (
        (ctx.as_of_time - min(r.event_timestamp for r in same_customer)).days if same_customer else 0
    )
    account_observed_tenure_days = (
        (ctx.as_of_time - min(r.event_timestamp for r in same_account)).days if same_account else 0
    )

    counterparty_risk_score = counterparty_risk.score(counterparty_key)

    return {
        "amount_vs_entity_average": float(amount_vs_entity_average),
        "amount_zscore": float(amount_zscore),
        "is_new_device": bool(is_new_device),
        "is_new_ip": bool(is_new_ip),
        "night_transaction_flag": bool(night_transaction_flag),
        "events_last_10m": int(events_last_10m),
        "events_last_1h": int(events_last_1h),
        "events_last_24h": int(events_last_24h),
        "prior_alert_count": int(prior_alert_count),
        "customer_observed_tenure_days": int(customer_observed_tenure_days),
        "account_observed_tenure_days": int(account_observed_tenure_days),
        "counterparty_risk_score": float(counterparty_risk_score),
    }


def ordered_feature_vector(
    features: Mapping[str, float | int | bool], columns: Sequence[str]
) -> list[float | int | bool]:
    """Projects a feature dict into the exact, fixed column order a model
    bundle expects (guide section 9's "ordered feature-schema version and
    feature-name list") -- the single mechanism that guarantees
    deterministic ordering regardless of how the dict itself was built.
    Raises if any expected column is absent -- never silently pads."""
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"feature output is missing required columns: {missing}")
    return [features[c] for c in columns]
