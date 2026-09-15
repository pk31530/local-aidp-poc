import csv
import json

import pytest

from src.processing.raw import validate_batch_file

VALID_ROW = {
    "transaction_id": "TX1",
    "customer_id": "C1",
    "transaction_timestamp": "2026-06-01T10:00:00+05:30",
    "amount": "100.50",
    "merchant": "Grocery",
    "country": "India",
    "device_id": "DEV1",
    "payment_method": "CARD",
    "is_fraud": "false",
}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_valid_csv_rows_pass_validation(tmp_path):
    path = tmp_path / "batch.csv"
    _write_csv(path, [VALID_ROW])
    df, rejected = validate_batch_file(path)
    assert df.height == 1
    assert rejected == []
    assert df["amount"][0] == pytest.approx(100.50)


def test_missing_transaction_id_is_rejected_not_crashed(tmp_path):
    bad_row = dict(VALID_ROW)
    del bad_row["transaction_id"]
    path = tmp_path / "batch.csv"
    _write_csv(path, [bad_row])
    df, rejected = validate_batch_file(path)
    assert df.height == 0
    assert len(rejected) == 1
    assert "rejection_reason" in rejected[0]


@pytest.mark.parametrize("amount_value", ["-50.00", "0", "not_a_number", "50000000"])
def test_out_of_bounds_or_non_numeric_amount_is_rejected(tmp_path, amount_value):
    bad_row = dict(VALID_ROW, amount=amount_value)
    path = tmp_path / "batch.csv"
    _write_csv(path, [bad_row])
    df, rejected = validate_batch_file(path)
    assert df.height == 0
    assert len(rejected) == 1


def test_invalid_timestamp_is_rejected(tmp_path):
    bad_row = dict(VALID_ROW, transaction_timestamp="not-a-timestamp")
    path = tmp_path / "batch.csv"
    _write_csv(path, [bad_row])
    df, rejected = validate_batch_file(path)
    assert df.height == 0
    assert len(rejected) == 1


def test_one_bad_row_does_not_block_other_valid_rows(tmp_path):
    good = dict(VALID_ROW, transaction_id="TX_GOOD")
    bad = dict(VALID_ROW, transaction_id="TX_BAD", amount="not_a_number")
    path = tmp_path / "batch.csv"
    _write_csv(path, [good, bad])
    df, rejected = validate_batch_file(path)
    assert df.height == 1
    assert df["transaction_id"][0] == "TX_GOOD"
    assert len(rejected) == 1
    assert rejected[0]["transaction_id"] == "TX_BAD"


def test_json_input_supported(tmp_path):
    path = tmp_path / "batch.json"
    payload = [dict(VALID_ROW, is_fraud=False)]
    path.write_text(json.dumps(payload))
    df, rejected = validate_batch_file(path)
    assert df.height == 1
    assert rejected == []


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "batch.txt"
    path.write_text("nonsense")
    with pytest.raises(ValueError):
        validate_batch_file(path)
