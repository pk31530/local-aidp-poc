import csv

import pytest

from src.processing import pipeline as pipeline_module

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


def test_run_pipeline_records_failed_status_and_reraises_on_stage_exception(tmp_path, monkeypatch):
    # Redirect all local writes into tmp_path so this test never touches the
    # real data/output or data/models directories.
    monkeypatch.setattr(pipeline_module, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline_module, "MODELS_DIR", tmp_path / "models")

    input_path = tmp_path / "batch.csv"
    _write_csv(input_path, [VALID_ROW])

    def _boom(*args, **kwargs):
        raise RuntimeError("clean stage exploded")

    monkeypatch.setattr(pipeline_module, "clean_transactions", _boom)

    calls = []
    monkeypatch.setattr(
        pipeline_module,
        "record_pipeline_run",
        lambda status, records_processed, records_rejected, started_at: calls.append(
            (status, records_processed, records_rejected)
        ),
    )

    with pytest.raises(RuntimeError, match="clean stage exploded"):
        pipeline_module.run_pipeline(input_path, upload_to_minio=False)

    assert len(calls) == 1
    status, records_processed, records_rejected = calls[0]
    assert status == "FAILED"
    assert records_processed == 0
    assert records_rejected == 0
