"""RAW stage: validate the batch input file, persist valid records unchanged.

Batch flow (guide section 19): File -> Validation -> RAW. Only schema/bounds
-valid rows (via the shared src.common.schemas.Transaction, fix M4) proceed;
everything else is rejected and reported, never silently dropped and never
crashes the pipeline (guide section 22).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import polars as pl
from pydantic import ValidationError

from src.common.logging import get_logger
from src.common.schemas import Transaction

log = get_logger(__name__)

EMPTY_SCHEMA = {
    "transaction_id": pl.Utf8,
    "customer_id": pl.Utf8,
    "transaction_timestamp": pl.Datetime(time_zone="UTC"),
    "amount": pl.Float64,
    "merchant": pl.Utf8,
    "country": pl.Utf8,
    "device_id": pl.Utf8,
    "payment_method": pl.Utf8,
    "schema_version": pl.Int64,
    "is_fraud": pl.Boolean,
}


def _read_input_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("transactions", [])
        yield from data
    else:
        raise ValueError(f"Unsupported batch file type: {path.suffix} (supported: .csv, .json)")


def _coerce_is_fraud(value) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def validate_batch_file(path: Path) -> tuple[pl.DataFrame, list[dict]]:
    """Returns (valid_transactions_df, rejected_rows). `rejected_rows` carry
    the original row plus a `rejection_reason` string — reported, not
    discarded silently."""
    valid_rows: list[dict] = []
    rejected_rows: list[dict] = []

    for raw_row in _read_input_rows(path):
        try:
            tx = Transaction(
                transaction_id=raw_row["transaction_id"],
                customer_id=raw_row["customer_id"],
                transaction_timestamp=raw_row["transaction_timestamp"],
                amount=raw_row["amount"],
                merchant=raw_row["merchant"],
                country=raw_row["country"],
                device_id=raw_row["device_id"],
                payment_method=raw_row["payment_method"],
                is_fraud=_coerce_is_fraud(raw_row.get("is_fraud")),
            )
        except (ValidationError, KeyError, ValueError) as exc:
            rejected_rows.append({**raw_row, "rejection_reason": str(exc)[:300]})
            continue
        valid_rows.append(tx.model_dump())

    if valid_rows:
        df = pl.DataFrame(valid_rows)
        # transaction_timestamp comes back tz-aware from pydantic (whatever
        # offset the source data used); normalize to UTC, one canonical tz
        # for every downstream stage.
        df = df.with_columns(pl.col("transaction_timestamp").dt.convert_time_zone("UTC"))
    else:
        df = pl.DataFrame(schema=EMPTY_SCHEMA)

    log.info("raw_validation_complete", valid=df.height, rejected=len(rejected_rows))
    return df, rejected_rows
