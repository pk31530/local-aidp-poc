"""Batch pipeline orchestrator (Phase 4): File -> RAW -> CLEAN -> CURATED ->
FEATURES.

Usage:
    python -m src.processing.pipeline
    python -m src.processing.pipeline --input data/batch/some_file.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

from src.common.config import PROJECT_ROOT, get_settings
from src.common.db import get_connection
from src.common.logging import configure_logging, get_logger
from src.common.splits import assign_split
from src.processing.clean import clean_transactions
from src.processing.enrich import enrich_transactions
from src.processing.raw import validate_batch_file
from src.processing.risk_lookups import compute_risk_lookups
from src.processing.views import to_curated, to_features

SEED_DIR = PROJECT_ROOT / "data" / "seed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

DEFAULT_INPUT = SEED_DIR / "historical_transactions.csv"
CUSTOMERS_PATH = SEED_DIR / "customers.parquet"
FAILED_ATTEMPTS_PATH = SEED_DIR / "historical_failed_attempts.csv"

BUCKETS = {
    "raw": "aidp-raw",
    "clean": "aidp-clean",
    "curated": "aidp-curated",
    "features": "aidp-features",
}


def _write_rejected(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_and_upload(df: pl.DataFrame, layer: str, upload: bool) -> Path:
    layer_dir = OUTPUT_DIR / layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    local_path = layer_dir / "transactions.parquet"
    df.write_parquet(local_path)
    if upload:
        from src.common.storage import upload_file

        upload_file(local_path, BUCKETS[layer], "transactions/transactions.parquet")
    return local_path


def record_pipeline_run(status: str, records_processed: int, records_rejected: int, started_at: float) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (pipeline_name, status, records_processed, records_rejected, started_at, completed_at)
                    VALUES ('batch', %s, %s, %s, to_timestamp(%s), now())
                    """,
                    (status, records_processed, records_rejected, started_at),
                )
    finally:
        conn.close()


def run_pipeline(input_path: Path, upload_to_minio: bool = True) -> dict:
    configure_logging("processing.pipeline")
    log = get_logger(__name__)
    settings = get_settings()
    started_at = time.time()

    log.info("pipeline_started", input=str(input_path))

    records_processed = 0
    records_rejected = 0

    try:
        # ---- RAW ----
        raw_df, rejected_raw = validate_batch_file(input_path)
        _write_and_upload(raw_df, "raw", upload_to_minio)
        _write_rejected(rejected_raw, OUTPUT_DIR / "rejected" / "raw_rejected.csv")
        records_rejected = len(rejected_raw)

        # ---- CLEAN ----
        clean_df, rejected_clean = clean_transactions(raw_df)
        _write_and_upload(clean_df, "clean", upload_to_minio)
        _write_rejected(rejected_clean, OUTPUT_DIR / "rejected" / "clean_rejected.csv")
        records_rejected += len(rejected_clean)

        # ---- split assignment (must happen before risk lookups so they are
        # fit on the train partition only, eliminating target leakage) ----
        clean_df = assign_split(clean_df, seed=settings.gen_random_seed)

        # ---- risk lookups (derived from the train partition only) ----
        risk_lookups = compute_risk_lookups(clean_df.filter(pl.col("split") == "train"))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        risk_lookups.save(MODELS_DIR / "risk_lookups.json")
        if upload_to_minio:
            from src.common.storage import upload_file

            upload_file(MODELS_DIR / "risk_lookups.json", "aidp-model-output", "risk_lookups.json")

        # ---- CURATED + FEATURES (one enrichment pass, two projections) ----
        customers_df = pl.read_parquet(CUSTOMERS_PATH)
        failed_attempts_df = (
            pl.read_csv(FAILED_ATTEMPTS_PATH) if FAILED_ATTEMPTS_PATH.exists() else pl.DataFrame()
        )
        enriched_df = enrich_transactions(clean_df, customers_df, failed_attempts_df, risk_lookups)

        curated_df = to_curated(enriched_df)
        _write_and_upload(curated_df, "curated", upload_to_minio)

        features_df = to_features(enriched_df)
        _write_and_upload(features_df, "features", upload_to_minio)

        records_processed = curated_df.height
        summary = {
            "input_rows": raw_df.height + len(rejected_raw),
            "raw_valid": raw_df.height,
            "raw_rejected": len(rejected_raw),
            "clean_valid": clean_df.height,
            "clean_rejected": len(rejected_clean),
            "curated_rows": curated_df.height,
            "features_rows": features_df.height,
            "total_rejected": records_rejected,
        }
        log.info("pipeline_complete", **summary)
    except Exception:
        log.error(
            "pipeline_failed",
            records_processed=records_processed,
            records_rejected=records_rejected,
            exc_info=True,
        )
        record_pipeline_run(
            "FAILED", records_processed=records_processed, records_rejected=records_rejected, started_at=started_at
        )
        raise

    record_pipeline_run(
        "SUCCESS", records_processed=records_processed, records_rejected=records_rejected, started_at=started_at
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the batch RAW->CLEAN->CURATED->FEATURES pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading outputs to MinIO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.input, upload_to_minio=not args.no_upload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
