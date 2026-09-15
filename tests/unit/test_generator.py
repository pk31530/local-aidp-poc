from datetime import datetime

from src.common.schemas import Transaction
from src.common.timeutil import APP_TZ
from src.generator.customers import generate_customers
from src.generator.history import generate_history

REF_DATE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=APP_TZ)


def test_generate_customers_is_repeatable():
    a = generate_customers(50, seed=42, reference_date=REF_DATE)
    b = generate_customers(50, seed=42, reference_date=REF_DATE)
    assert a.equals(b)


def test_generate_customers_respects_size():
    df = generate_customers(37, seed=1, reference_date=REF_DATE)
    assert df.height == 37
    assert df["customer_id"].n_unique() == 37


def test_generate_history_is_repeatable():
    customers = generate_customers(30, seed=7, reference_date=REF_DATE)
    tx_a, fa_a = generate_history(customers, total_transactions=200, fraud_ratio=0.03, seed=7, reference_date=REF_DATE)
    tx_b, fa_b = generate_history(customers, total_transactions=200, fraud_ratio=0.03, seed=7, reference_date=REF_DATE)
    assert tx_a.equals(tx_b)
    assert fa_a.equals(fa_b)


def test_generate_history_respects_size_and_fraud_ratio():
    customers = generate_customers(200, seed=11, reference_date=REF_DATE)
    total = 2000
    fraud_ratio = 0.03
    tx, _ = generate_history(customers, total_transactions=total, fraud_ratio=fraud_ratio, seed=11, reference_date=REF_DATE)

    assert tx.height == total
    fraud_count = int(tx["is_fraud"].sum())
    assert fraud_count == round(total * fraud_ratio)


def test_generated_transactions_match_shared_schema():
    customers = generate_customers(50, seed=3, reference_date=REF_DATE)
    tx, _ = generate_history(customers, total_transactions=300, fraud_ratio=0.03, seed=3, reference_date=REF_DATE)

    for row in tx.to_dicts():
        Transaction(
            transaction_id=row["transaction_id"],
            customer_id=row["customer_id"],
            transaction_timestamp=row["transaction_timestamp"],
            amount=row["amount"],
            merchant=row["merchant"],
            country=row["country"],
            device_id=row["device_id"],
            payment_method=row["payment_method"],
            is_fraud=row["is_fraud"],
        )


def test_different_seeds_produce_different_data():
    a = generate_customers(20, seed=1, reference_date=REF_DATE)
    b = generate_customers(20, seed=2, reference_date=REF_DATE)
    assert not a.equals(b)
