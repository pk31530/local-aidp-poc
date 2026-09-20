"""Phase 7: Redpanda consumer.

Generator -> Redpanda -> Consumer -> feature engineering -> model ->
decision -> PostgreSQL -> MinIO.

Two distinct failure paths (fix H3):
  - malformed messages (bad JSON, schema/bounds violation) go straight to
    the dead-letter-transactions topic — retrying them would never help.
  - transient infra failures (Postgres/MinIO briefly unavailable) are
    retried with backoff (src/common/retry.py); only routed to the DLQ if
    retries are exhausted, so the message and its failure reason aren't
    silently lost.

Usage:
    python -m src.ingestion.consumer
    python -m src.ingestion.consumer --duration 30
    python -m src.ingestion.consumer --from-beginning
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from confluent_kafka import Consumer, KafkaException, Producer
from pydantic import ValidationError

from src.common.config import PROJECT_ROOT, get_settings
from src.common.db import get_connection
from src.common.features import RiskLookups
from src.common.logging import bind_transaction_id, clear_transaction_id, configure_logging, get_logger
from src.common.mlflow_setup import configure_mlflow, load_champion_model
from src.common.retry import transient_retry
from src.common.schemas import Transaction
from src.common.scoring import score_and_persist
from src.common.storage import upload_file
from src.control_plane.config import StreamRunConfig
from src.control_plane.provenance import get_git_sha
from src.control_plane.runs import RunLifecycle

RISK_LOOKUPS_PATH = PROJECT_ROOT / "data" / "models" / "risk_lookups.json"
FLUSH_BATCH_SIZE = 10
HEARTBEAT_EVERY_N_MESSAGES = 10


@transient_retry()
def _score_with_retry(model, model_version, risk_lookups, database=None, **kwargs):
    # A fresh connection per attempt (not one shared/reused across retries):
    # if Postgres was briefly down, a prior attempt's connection is dead and
    # retrying on it would keep failing even after Postgres recovers.
    conn = get_connection(database)
    try:
        return score_and_persist(conn, model, model_version, risk_lookups, **kwargs)
    finally:
        conn.close()


class DeadLetterPublisher:
    def __init__(self, producer: Producer, topic: str, log):
        self._producer = producer
        self._topic = topic
        self._log = log

    def send(self, raw_value: bytes, reason: str) -> None:
        envelope = {
            "original_message": raw_value.decode("utf-8", errors="replace"),
            "rejection_reason": reason,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._producer.produce(self._topic, value=json.dumps(envelope).encode("utf-8"))
        self._producer.poll(0)
        self._log.warning("sent_to_dlq", reason=reason)


class RawEventBuffer:
    """Micro-batches successfully processed events to Parquet in the
    aidp-raw MinIO bucket, mirroring the batch pipeline's RAW layer for
    stream-sourced data."""

    def __init__(self, bucket: str, log):
        self._bucket = bucket
        self._log = log
        self._rows: list[dict] = []

    def add(self, event: dict) -> None:
        self._rows.append(event)
        if len(self._rows) >= FLUSH_BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        df = pl.DataFrame(self._rows)
        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / "batch.parquet"
            df.write_parquet(local_path)
            object_name = f"streaming/{datetime.now(timezone.utc):%Y%m%d}/{uuid.uuid4().hex}.parquet"
            upload_file(local_path, self._bucket, object_name)
        self._log.info("raw_batch_flushed", rows=len(self._rows), bucket=self._bucket)
        self._rows = []


def _load_model_or_raise():
    configure_mlflow()
    settings = get_settings()
    model, version = load_champion_model(settings.mlflow_model_name)
    return model, str(version)


def run(
    duration: float | None,
    from_beginning: bool,
    topic: str | None = None,
    dlq_topic: str | None = None,
    database: str | None = None,
    group_id: str = "aidp-consumer",
    raw_bucket: str = "aidp-raw",
) -> dict:
    """Legacy entry point — signature and defaults unchanged. Builds a typed
    config and delegates all workload/lifecycle logic to run_configured();
    trigger_source="legacy" identifies calls made this way rather than
    through the future CLI (trigger_source="cli")."""
    config = StreamRunConfig(
        duration=duration,
        from_beginning=from_beginning,
        topic=topic,
        dlq_topic=dlq_topic,
        database=database,
        group_id=group_id,
        raw_bucket=raw_bucket,
    )
    return run_configured(config, trigger_source="legacy")


def run_configured(config: StreamRunConfig, *, trigger_source: str = "legacy") -> dict:
    """`config.topic`/`config.dlq_topic`/`config.database` default to the
    production topic/DLQ-topic/database; tests pass the isolated
    `transactions-test` topic and `aidp_test` database instead (fix H4), with
    a distinct `group_id` so a test run doesn't share committed offsets with
    the production consumer group."""
    configure_logging("ingestion.consumer")
    log = get_logger(__name__)
    settings = get_settings()
    topic = config.topic or settings.redpanda_topic_transactions
    dlq_topic = config.dlq_topic or settings.redpanda_topic_dlq

    lifecycle = RunLifecycle(database=config.database)
    # fix H3/pipeline_runs-status: the run is recorded before model loading
    # so a model-load failure (or any other setup/loop failure) is still
    # traceable to a run, and gets corrected to FAILED below instead of
    # leaving zero trace or a falsely-SUCCESS row.
    run = lifecycle.begin(
        "stream",
        trigger_source=trigger_source,
        git_sha=get_git_sha(),
        config_snapshot=config.redacted_snapshot(),
        config_hash=config.config_hash(),
    )

    processed = 0
    rejected = 0
    model_version = None
    consumer = None
    dlq_producer = None
    raw_buffer = None

    try:
        model, model_version = _load_model_or_raise()
        risk_lookups = RiskLookups.load(RISK_LOOKUPS_PATH) if RISK_LOOKUPS_PATH.exists() else RiskLookups.empty()

        consumer = Consumer(
            {
                "bootstrap.servers": settings.redpanda_brokers,
                "group.id": config.group_id,
                "auto.offset.reset": "earliest" if config.from_beginning else "latest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([topic])

        dlq_producer = Producer({"bootstrap.servers": settings.redpanda_brokers})
        dlq = DeadLetterPublisher(dlq_producer, dlq_topic, log)
        raw_buffer = RawEventBuffer(config.raw_bucket, log)

        start = time.monotonic()

        log.info(
            "consumer_started", model_version=model_version, from_beginning=config.from_beginning, run_id=run.run_id
        )

        try:
            while config.duration is None or (time.monotonic() - start) < config.duration:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())

                raw_value = msg.value()

                try:
                    payload = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    dlq.send(raw_value, f"invalid_json: {exc}")
                    rejected += 1
                    consumer.commit(msg)
                    continue

                try:
                    tx = Transaction(**payload)
                except ValidationError as exc:
                    dlq.send(raw_value, f"schema_validation_failed: {exc}")
                    rejected += 1
                    consumer.commit(msg)
                    continue

                bind_transaction_id(tx.transaction_id)
                try:
                    try:
                        result = _score_with_retry(
                            model,
                            model_version,
                            risk_lookups,
                            database=config.database,
                            transaction_id=tx.transaction_id,
                            customer_id=tx.customer_id,
                            amount=tx.amount,
                            merchant=tx.merchant,
                            country=tx.country,
                            device_id=tx.device_id,
                            payment_method=tx.payment_method,
                            transaction_timestamp=tx.transaction_timestamp,
                            source="stream",
                        )
                    except Exception as exc:
                        # fix H3: retries (inside _score_with_retry) are already
                        # exhausted by the time we get here — this is a genuine
                        # processing failure, not a malformed message.
                        log.error("processing_failed_after_retries", error=str(exc), exc_info=True)
                        dlq.send(raw_value, f"processing_failed_after_retries: {exc}")
                        rejected += 1
                        consumer.commit(msg)
                        continue

                    raw_buffer.add(payload)
                    processed += 1
                    consumer.commit(msg)

                    log.info(
                        "transaction_scored",
                        customer_id=tx.customer_id,
                        fraud_probability=round(result.fraud_probability, 4),
                        decision=result.decision,
                    )

                    # Best-effort progress signal only: a failure here is
                    # logged and ignored, never allowed to affect message
                    # processing, commits, or DLQ routing.
                    if processed % HEARTBEAT_EVERY_N_MESSAGES == 0:
                        try:
                            lifecycle.heartbeat(run.run_id)
                        except Exception:
                            log.warning("heartbeat_failed", run_id=run.run_id, exc_info=True)
                finally:
                    clear_transaction_id()
        except KeyboardInterrupt:
            # Operator-initiated stop (Ctrl+C): treated the same as a
            # duration expiring — a graceful, expected way to end a local
            # demo consumer, not a failure. Falls through to the same
            # SUCCESS recording below as a normal loop exit, so the run
            # never remains stuck in RUNNING.
            log.info("consumer_interrupted_by_user")
    except Exception as exc:
        # A genuine crash (Kafka error, model-load failure, or anything else
        # unhandled) — record it as FAILED before re-raising so pipeline_runs
        # never misrepresents a crash as SUCCESS or drops it entirely.
        log.error("consumer_run_failed", exc_info=True)
        lifecycle.fail_from_exception(
            run.run_id, exc, records_processed=processed, records_rejected=rejected, model_version=model_version
        )
        raise
    finally:
        if raw_buffer is not None:
            raw_buffer.flush()
        if consumer is not None:
            consumer.close()
        if dlq_producer is not None:
            dlq_producer.flush(10)

    lifecycle.succeed(
        run.run_id,
        records_processed=processed,
        records_rejected=rejected,
        model_version=model_version,
        artifacts={"raw_bucket": config.raw_bucket},
    )

    summary = {"processed": processed, "rejected": rejected, "run_id": run.run_id}
    log.info("consumer_stopped", **summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume and score live transactions from Redpanda.")
    parser.add_argument("--duration", type=float, default=None, help="seconds to run; omit to run until Ctrl+C")
    parser.add_argument("--from-beginning", action="store_true", help="replay the topic from the earliest offset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.duration, args.from_beginning)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
