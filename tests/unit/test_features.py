from datetime import datetime, timedelta, timezone

import pytest

from src.common.features import (
    DEFAULT_PROFILE,
    MODEL_FEATURE_COLUMNS,
    CustomerProfileSnapshot,
    RecentEvent,
    RiskLookups,
    compute_features,
)
from src.common.timeutil import APP_TZ

PROFILE = CustomerProfileSnapshot(
    avg_transaction_amount=1000.0,
    stddev_transaction_amount=200.0,
    known_devices=frozenset({"DEV1"}),
    known_countries=frozenset({"India"}),
    home_country="India",
)

NEUTRAL_RISK = RiskLookups.empty()


def _ts(hour: int, minute: int = 0) -> datetime:
    """A timestamp at the given local (Asia/Kolkata) hour, expressed in UTC
    to exercise the timezone-conversion path, not just a naive local clock."""
    local = datetime(2026, 6, 15, hour, minute, tzinfo=APP_TZ)
    return local.astimezone(timezone.utc)


def test_amount_ratio_and_zscore_normal_profile():
    f = compute_features(
        amount=2000.0,
        merchant="Grocery",
        country="India",
        device_id="DEV1",
        transaction_timestamp=_ts(14),
        profile=PROFILE,
        recent_events=[],
        risk_lookups=NEUTRAL_RISK,
    )
    assert f["amount_vs_customer_average"] == pytest.approx(2.0)
    assert f["amount_zscore"] == pytest.approx((2000.0 - 1000.0) / 200.0)


def test_default_profile_fallback_does_not_crash_or_divide_by_zero():
    f = compute_features(
        amount=500.0,
        merchant="Grocery",
        country="India",
        device_id="DEVX",
        transaction_timestamp=_ts(14),
        profile=DEFAULT_PROFILE,
        recent_events=[],
        risk_lookups=NEUTRAL_RISK,
    )
    assert f["amount_vs_customer_average"] == 1.0  # neutral, no history
    assert f["amount_zscore"] == 0.0
    assert f["is_new_device"] is True
    assert f["is_new_country"] is True


def test_new_device_and_country_flags():
    known = compute_features(
        amount=100.0, merchant="Grocery", country="India", device_id="DEV1",
        transaction_timestamp=_ts(14), profile=PROFILE, recent_events=[], risk_lookups=NEUTRAL_RISK,
    )
    unknown = compute_features(
        amount=100.0, merchant="Grocery", country="Singapore", device_id="DEVNEW1",
        transaction_timestamp=_ts(14), profile=PROFILE, recent_events=[], risk_lookups=NEUTRAL_RISK,
    )
    assert known["is_new_device"] is False and known["is_new_country"] is False
    assert unknown["is_new_device"] is True and unknown["is_new_country"] is True


@pytest.mark.parametrize(
    "hour,expected",
    [
        (2, True),    # 2am IST -> night
        (23, True),   # 11pm IST -> night (wraps midnight)
        (4, True),    # 4am IST -> night (end_hour=5 exclusive upper bound)
        (5, False),   # 5am IST -> day starts
        (14, False),  # 2pm IST -> day
        (22, False),  # 10pm IST -> day (before night starts at 23)
    ],
)
def test_night_transaction_flag_pinned_to_app_timezone(hour, expected):
    """Fix M2: explicitly tests night_transaction_flag against the pinned
    Asia/Kolkata timezone, using a UTC-expressed input timestamp so the
    conversion path is actually exercised."""
    f = compute_features(
        amount=100.0, merchant="Grocery", country="India", device_id="DEV1",
        transaction_timestamp=_ts(hour), profile=PROFILE, recent_events=[], risk_lookups=NEUTRAL_RISK,
    )
    assert f["night_transaction_flag"] is expected


def test_velocity_windows_count_only_transaction_success_events_in_range():
    now = _ts(14)
    events = [
        RecentEvent("transaction_success", now - timedelta(minutes=5)),   # in 10m, 1h, 24h
        RecentEvent("transaction_success", now - timedelta(minutes=40)),  # in 1h, 24h only
        RecentEvent("transaction_success", now - timedelta(hours=20)),    # in 24h only
        RecentEvent("transaction_success", now - timedelta(hours=30)),    # outside all windows
        RecentEvent("transaction_failed", now - timedelta(minutes=5)),    # not a "transaction", excluded
    ]
    f = compute_features(
        amount=100.0, merchant="Grocery", country="India", device_id="DEV1",
        transaction_timestamp=now, profile=PROFILE, recent_events=events, risk_lookups=NEUTRAL_RISK,
    )
    assert f["transactions_last_10m"] == 1
    assert f["transactions_last_1h"] == 2
    assert f["transactions_last_24h"] == 3


def test_failed_attempts_last_1h_counts_only_failed_events_in_window():
    now = _ts(14)
    events = [
        RecentEvent("transaction_failed", now - timedelta(minutes=10)),
        RecentEvent("transaction_failed", now - timedelta(minutes=50)),
        RecentEvent("transaction_failed", now - timedelta(hours=3)),  # outside 1h
        RecentEvent("transaction_success", now - timedelta(minutes=1)),  # wrong type
    ]
    f = compute_features(
        amount=100.0, merchant="Grocery", country="India", device_id="DEV1",
        transaction_timestamp=now, profile=PROFILE, recent_events=events, risk_lookups=NEUTRAL_RISK,
    )
    assert f["failed_attempts_last_1h"] == 2
    assert f["failed_attempts_1h"] == 2  # CURATED-layer alias


def test_risk_lookup_default_fallback_for_unseen_key():
    lookups = RiskLookups(merchant_risk={"Electronics": 0.8}, country_risk={"Singapore": 0.5})
    assert lookups.merchant_score("Electronics") == 0.8
    assert lookups.merchant_score("SomeUnseenMerchant") == lookups.default_merchant_risk
    assert lookups.country_score("SomeUnseenCountry") == lookups.default_country_risk


def test_risk_lookup_save_and_load_roundtrip(tmp_path):
    lookups = RiskLookups(merchant_risk={"Electronics": 0.8}, country_risk={"Singapore": 0.5})
    path = tmp_path / "risk_lookups.json"
    lookups.save(path)
    loaded = RiskLookups.load(path)
    assert loaded.merchant_score("Electronics") == 0.8
    assert loaded.country_score("Singapore") == 0.5


def test_model_feature_columns_excludes_label_and_matches_output_keys():
    f = compute_features(
        amount=100.0, merchant="Grocery", country="India", device_id="DEV1",
        transaction_timestamp=_ts(14), profile=PROFILE, recent_events=[], risk_lookups=NEUTRAL_RISK,
    )
    assert "is_fraud" not in MODEL_FEATURE_COLUMNS
    for col in MODEL_FEATURE_COLUMNS:
        assert col == "amount" or col in f, f"{col} missing from compute_features() output"
