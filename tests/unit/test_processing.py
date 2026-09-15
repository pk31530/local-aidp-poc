from datetime import datetime, timezone

import polars as pl
import pytest

from src.processing.clean import clean_transactions
from src.processing.enrich import enrich_transactions
from src.processing.risk_lookups import compute_risk_lookups
from src.processing.views import to_curated, to_features


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _base_row(**overrides) -> dict:
    row = {
        "transaction_id": "TX1",
        "customer_id": "C1",
        "transaction_timestamp": _ts("2026-06-01T10:00:00"),
        "amount": 100.0,
        "merchant": "Grocery",
        "country": "India",
        "device_id": "DEV1",
        "payment_method": "CARD",
        "schema_version": 1,
        "is_fraud": False,
    }
    row.update(overrides)
    return row


def test_clean_removes_duplicates_nulls_and_bad_payment_method():
    df = pl.DataFrame(
        [
            _base_row(transaction_id="TX1"),
            _base_row(transaction_id="TX1"),  # duplicate
            _base_row(transaction_id="TX2", merchant=None),  # null required field
            _base_row(transaction_id="TX3", payment_method="BITCOIN"),  # not allowed
            _base_row(transaction_id="TX4"),  # valid
        ]
    )
    clean_df, rejected = clean_transactions(df)

    assert clean_df.height == 2  # TX1 (first occurrence) + TX4
    assert sorted(clean_df["transaction_id"].to_list()) == ["TX1", "TX4"]
    reasons = {r["transaction_id"]: r["rejection_reason"] for r in rejected}
    assert reasons["TX1"] == "duplicate_transaction_id"
    assert reasons["TX2"] == "null_in_required_field"
    assert "invalid_payment_method" in reasons["TX3"]


def test_clean_empty_input_is_a_noop():
    df = pl.DataFrame(
        schema={
            "transaction_id": pl.Utf8, "customer_id": pl.Utf8,
            "transaction_timestamp": pl.Datetime(time_zone="UTC"), "amount": pl.Float64,
            "merchant": pl.Utf8, "country": pl.Utf8, "device_id": pl.Utf8,
            "payment_method": pl.Utf8, "schema_version": pl.Int64, "is_fraud": pl.Boolean,
        }
    )
    clean_df, rejected = clean_transactions(df)
    assert clean_df.height == 0
    assert rejected == []


def test_risk_lookups_high_fraud_merchant_scores_higher():
    df = pl.DataFrame(
        [
            _base_row(transaction_id="TX1", merchant="Electronics", country="India", is_fraud=True),
            _base_row(transaction_id="TX2", merchant="Electronics", country="India", is_fraud=True),
            _base_row(transaction_id="TX3", merchant="Electronics", country="India", is_fraud=False),
            _base_row(transaction_id="TX4", merchant="Grocery", country="India", is_fraud=False),
            _base_row(transaction_id="TX5", merchant="Grocery", country="India", is_fraud=False),
        ]
    )
    lookups = compute_risk_lookups(df)
    assert lookups.merchant_score("Electronics") > lookups.merchant_score("Grocery")
    # unseen merchant falls back to the overall (neutral) rate, fix C3
    assert lookups.merchant_score("NeverSeenBefore") == lookups.default_merchant_risk


def test_enrich_then_project_curated_and_features():
    customers_df = pl.DataFrame(
        [
            {
                "customer_id": "C1",
                "full_name": "Test Customer",
                "home_country": "India",
                "known_countries": ["India"],
                "known_devices": ["DEV1"],
                "avg_transaction_amount": 100.0,
                "stddev_transaction_amount": 20.0,
                "signup_date": "2025-01-01",
            }
        ]
    )
    clean_df = pl.DataFrame(
        [
            _base_row(transaction_id="TX1", transaction_timestamp=_ts("2026-06-01T10:00:00")),
            _base_row(transaction_id="TX2", transaction_timestamp=_ts("2026-06-01T10:05:00"), amount=5000.0, is_fraud=True),
        ]
    )
    failed_attempts_df = pl.DataFrame(
        [{"customer_id": "C1", "transaction_id": "TX2", "event_type": "transaction_failed",
          "amount": 5000.0, "country": "India", "device_id": "DEV1", "occurred_at": "2026-06-01T09:55:00+00:00"}]
    )

    from src.common.features import RiskLookups

    enriched = enrich_transactions(clean_df, customers_df, failed_attempts_df, RiskLookups.empty())
    assert enriched.height == 2

    tx2 = enriched.filter(pl.col("transaction_id") == "TX2").to_dicts()[0]
    assert tx2["transactions_last_1h"] == 1  # TX1 is 5 minutes earlier
    assert tx2["failed_attempts_last_1h"] == 1
    assert tx2["amount_vs_customer_average"] == pytest.approx(50.0)

    curated = to_curated(enriched)
    features = to_features(enriched)
    assert "customer_average_amount" in curated.columns
    assert "amount_vs_customer_average" in features.columns
    assert "amount_vs_customer_average" not in curated.columns
