"""Batch pipeline orchestrator (Phase 4): File -> RAW -> CLEAN -> CURATED ->
FEATURES.

Usage:
    python -m src.processing.pipeline
    python -m src.processing.pipeline --input data/batch/some_file.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import polars as pl

from src.common.config import PROJECT_ROOT, get_settings
from src.common.logging import configure_logging, get_logger
from src.common.splits import assign_split
from src.control_plane.config import BatchRunConfig
from src.control_plane.provenance import get_git_sha
from src.control_plane.runs import RunLifecycle
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


def run_pipeline(input_path: Path, upload_to_minio: bool = True) -> dict:
    """Legacy entry point — signature and defaults unchanged. Builds a typed
    config and delegates all workload/lifecycle logic to
    run_pipeline_configured(); trigger_source="legacy" identifies calls made
    this way rather than through the future CLI (trigger_source="cli")."""
    config = BatchRunConfig(input_path=input_path, upload_to_minio=upload_to_minio)
    return run_pipeline_configured(config, trigger_source="legacy")


def run_pipeline_configured(config: BatchRunConfig, *, trigger_source: str = "legacy") -> dict:
    configure_logging("processing.pipeline")
    log = get_logger(__name__)
    settings = get_settings()

    log.info("pipeline_started", input=str(config.input_path))

    lifecycle = RunLifecycle()
    run = lifecycle.begin(
        "batch",
        trigger_source=trigger_source,
        git_sha=get_git_sha(),
        config_snapshot=config.redacted_snapshot(),
        config_hash=config.config_hash(),
    )

    records_processed = 0
    records_rejected = 0
    dataset_version = None

    try:
        # Reading/hashing the input file happens inside the protected block —
        # a missing/unreadable/corrupted file must still record FAILED, not
        # crash before any run record exists.
        dataset_version = hashlib.sha256(config.input_path.read_bytes()).hexdigest()[:16]

        # ---- RAW ----
        raw_df, rejected_raw = validate_batch_file(config.input_path)
        raw_path = _write_and_upload(raw_df, "raw", config.upload_to_minio)
        _write_rejected(rejected_raw, OUTPUT_DIR / "rejected" / "raw_rejected.csv")
        records_rejected = len(rejected_raw)

        # ---- CLEAN ----
        clean_df, rejected_clean = clean_transactions(raw_df)
        clean_path = _write_and_upload(clean_df, "clean", config.upload_to_minio)
        _write_rejected(rejected_clean, OUTPUT_DIR / "rejected" / "clean_rejected.csv")
        records_rejected += len(rejected_clean)

        # ---- split assignment (must happen before risk lookups so they are
        # fit on the train partition only, eliminating target leakage) ----
        clean_df = assign_split(clean_df, seed=settings.gen_random_seed)

        # ---- risk lookups (derived from the train partition only) ----
        risk_lookups = compute_risk_lookups(clean_df.filter(pl.col("split") == "train"))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        risk_lookups_path = MODELS_DIR / "risk_lookups.json"
        risk_lookups.save(risk_lookups_path)
        if config.upload_to_minio:
            from src.common.storage import upload_file

            upload_file(risk_lookups_path, "aidp-model-output", "risk_lookups.json")

        # ---- CURATED + FEATURES (one enrichment pass, two projections) ----
        customers_df = pl.read_parquet(CUSTOMERS_PATH)
        failed_attempts_df = (
            pl.read_csv(FAILED_ATTEMPTS_PATH) if FAILED_ATTEMPTS_PATH.exists() else pl.DataFrame()
        )
        enriched_df = enrich_transactions(clean_df, customers_df, failed_attempts_df, risk_lookups)

        curated_df = to_curated(enriched_df)
        curated_path = _write_and_upload(curated_df, "curated", config.upload_to_minio)

        features_df = to_features(enriched_df)
        features_path = _write_and_upload(features_df, "features", config.upload_to_minio)

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
    except Exception as exc:
        log.error(
            "pipeline_failed",
            records_processed=records_processed,
            records_rejected=records_rejected,
            exc_info=True,
        )
        lifecycle.fail_from_exception(
            run.run_id,
            exc,
            records_processed=records_processed,
            records_rejected=records_rejected,
            dataset_version=dataset_version,
        )
        raise

    lifecycle.succeed(
        run.run_id,
        records_processed=records_processed,
        records_rejected=records_rejected,
        dataset_version=dataset_version,
        artifacts={
            "raw_path": str(raw_path),
            "clean_path": str(clean_path),
            "curated_path": str(curated_path),
            "features_path": str(features_path),
            "risk_lookups_path": str(risk_lookups_path),
        },
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
