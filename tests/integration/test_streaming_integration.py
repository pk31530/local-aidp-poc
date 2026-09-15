"""Integration: producer -> Redpanda -> consumer -> PostgreSQL -> MinIO.

Uses the isolated `transactions-test` topic and `aidp_test` database (fix
H4) end to end, via the real production code paths
(src.ingestion.consumer.run, confluent_kafka Producer) — not reimplemented
test-only logic.

Each test uses a brand-new consumer group (`auto.offset.reset=latest`) and
starts the consumer *before* producing, in a background thread, so it only
sees messages produced by that test — not the accumulated history of
`transactions-test` from earlier test runs (which `from_beginning=True`
with a fresh group would otherwise replay in full every time).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

from src.common.config import get_settings
from src.common.db import get_connection
from src.common.storage import get_minio_client
from src.ingestion import consumer as consumer_module

TEST_DB = "aidp_test"


def _unique_group_id() -> str:
    return f"integration-test-{uuid.uuid4().hex[:8]}"


def _run_consumer_in_background(topic: str, duration: float = 10.0):
    """Starts consumer.run() in a background thread with a fresh consumer
    group (auto.offset.reset=latest), waits for it to be actively
    subscribed, and returns (thread, result_holder)."""
    result: dict = {}

    def target():
        result["summary"] = consumer_module.run(
            duration=duration,
            from_beginning=False,
            topic=topic,
            database=TEST_DB,
            group_id=_unique_group_id(),
        )

    thread = threading.Thread(target=target)
    thread.start()
    time.sleep(2.5)  # let the consumer group join and get partition assignment
    return thread, result


def test_producer_to_consumer_to_postgres_and_minio(seeded_customer):
    settings = get_settings()
    test_topic = settings.redpanda_topic_test
    transaction_id = f"TXIT{uuid.uuid4().hex[:10].upper()}"

    thread, result = _run_consumer_in_background(test_topic)

    before = datetime.now(timezone.utc)
    event = {
        "transaction_id": transaction_id,
        "customer_id": seeded_customer,
        "transaction_timestamp": before.isoformat(),
        "amount": 555.55,
        "merchant": "Grocery",
        "country": "India",
        "device_id": "DEVTEST1",
        "payment_method": "CARD",
        "schema_version": 1,
        "is_fraud": False,
    }
    producer = Producer({"bootstrap.servers": settings.redpanda_brokers})
    producer.produce(test_topic, value=json.dumps(event).encode("utf-8"))
    producer.flush(10)

    thread.join(timeout=20)
    summary = result["summary"]

    assert summary["processed"] == 1
    assert summary["rejected"] == 0

    # consumer -> PostgreSQL
    conn = get_connection(TEST_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT amount, source FROM transactions WHERE transaction_id = %s", (transaction_id,))
            row = cur.fetchone()
            assert row is not None
            assert float(row[0]) == 555.55
            assert row[1] == "stream"

            cur.execute("SELECT decision FROM fraud_decisions WHERE transaction_id = %s", (transaction_id,))
            decision_row = cur.fetchone()
            assert decision_row is not None
            assert decision_row[0] in ("APPROVE", "MONITOR", "REVIEW", "BLOCK")

            cur.execute(
                "SELECT count(*) FROM recent_events WHERE customer_id = %s AND transaction_id = %s",
                (seeded_customer, transaction_id),
            )
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()

    # consumer -> MinIO: confirm a raw batch object actually landed (not
    # just that run() returned without raising).
    client = get_minio_client()
    date_prefix = f"streaming/{before:%Y%m%d}/"
    objects = list(client.list_objects("aidp-raw", prefix=date_prefix, recursive=True))
    recent_objects = [o for o in objects if o.last_modified >= before.replace(microsecond=0)]
    assert len(recent_objects) >= 1


def test_malformed_message_goes_to_dlq_not_postgres(seeded_customer):
    settings = get_settings()
    test_topic = settings.redpanda_topic_test

    thread, result = _run_consumer_in_background(test_topic)

    malformed = {
        "customer_id": seeded_customer,
        "amount": -100,  # invalid, and transaction_id is missing entirely
    }
    producer = Producer({"bootstrap.servers": settings.redpanda_brokers})
    producer.produce(test_topic, value=json.dumps(malformed).encode("utf-8"))
    producer.flush(10)

    thread.join(timeout=20)
    summary = result["summary"]

    assert summary["processed"] == 0
    assert summary["rejected"] == 1

    conn = get_connection(TEST_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM transactions")
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()
