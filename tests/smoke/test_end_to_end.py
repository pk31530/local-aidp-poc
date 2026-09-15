"""Smoke test (guide section 23): proves the full path in one run —

generate -> publish -> consume -> feature calculation -> model score ->
decision -> store -> retrieve through API

— against the isolated `transactions-test` topic and `aidp_test` database
(fix H4), using the real production code at every step (no mocks): the
Redpanda producer, `src.ingestion.consumer.run()`, and the FastAPI app via
TestClient for the final retrieval.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone

import psycopg2.extras
import pytest
from confluent_kafka import Producer
from fastapi.testclient import TestClient

from src.api.main import app, get_conn
from src.common.config import get_settings
from src.common.db import get_connection
from src.ingestion import consumer as consumer_module

TEST_DB = "aidp_test"


def _override_get_conn():
    conn = get_connection(TEST_DB)
    try:
        yield conn
    finally:
        conn.close()


app.dependency_overrides[get_conn] = _override_get_conn


@pytest.fixture(autouse=True)
def clean_test_db():
    conn = get_connection(TEST_DB)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "TRUNCATE fraud_decisions, fraud_scores, transactions, recent_events, "
                    "customer_profiles, customers, pipeline_runs RESTART IDENTITY CASCADE"
                )
    finally:
        conn.close()
    yield


@pytest.fixture
def seeded_customer() -> str:
    customer_id = "SMOKECUST1"
    conn = get_connection(TEST_DB)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO customers (customer_id, full_name, home_country) VALUES (%s, %s, %s)",
                    (customer_id, "Smoke Test Customer", "India"),
                )
                cur.execute(
                    """
                    INSERT INTO customer_profiles
                        (customer_id, avg_transaction_amount, stddev_transaction_amount,
                         known_devices, known_countries, home_country)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        customer_id, 8000.0, 1500.0,
                        psycopg2.extras.Json(["DEV100001"]),
                        psycopg2.extras.Json(["India"]),
                        "India",
                    ),
                )
    finally:
        conn.close()
    return customer_id


def test_full_path_generate_to_retrieve(seeded_customer):
    settings = get_settings()
    test_topic = settings.redpanda_topic_test
    transaction_id = f"TXSMOKE{uuid.uuid4().hex[:8].upper()}"

    # ---- generate + publish ----
    event = {
        "transaction_id": transaction_id,
        "customer_id": seeded_customer,
        "transaction_timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": 82000.0,
        "merchant": "Electronics",
        "country": "Singapore",
        "device_id": "DEVNEW999",
        "payment_method": "CARD",
        "schema_version": 1,
        "is_fraud": True,
    }

    result: dict = {}

    def consume():
        result["summary"] = consumer_module.run(
            duration=10,
            from_beginning=False,
            topic=test_topic,
            database=TEST_DB,
            group_id=f"smoke-test-{uuid.uuid4().hex[:8]}",
        )

    # ---- consume (started first so a fresh, latest-offset group doesn't miss the message) ----
    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(2.5)

    producer = Producer({"bootstrap.servers": settings.redpanda_brokers})
    producer.produce(test_topic, value=json.dumps(event).encode("utf-8"))
    producer.flush(10)

    thread.join(timeout=20)
    summary = result["summary"]

    # ---- consume -> feature calculation -> model score -> decision -> store ----
    assert summary["processed"] == 1, "consumer did not process the published message"
    assert summary["rejected"] == 0

    # ---- retrieve through API ----
    with TestClient(app) as client:
        tx_response = client.get(f"/transactions/{transaction_id}")
        assert tx_response.status_code == 200
        tx_body = tx_response.json()
        assert tx_body["customer_id"] == seeded_customer
        assert tx_body["amount"] == 82000.0

        decision_response = client.get(f"/decisions/{transaction_id}")
        assert decision_response.status_code == 200
        decision_body = decision_response.json()

        # A ₹82,000 transaction on a new device in a new country, against a
        # customer whose average is ₹8,000, must not come back as a quiet
        # APPROVE — the whole point of the platform is to catch this.
        assert decision_body["decision"] in ("MONITOR", "REVIEW", "BLOCK")
        assert decision_body["fraud_probability"] > 0.4
        assert "NEW_DEVICE" in decision_body["reason_codes"]
        assert "NEW_COUNTRY" in decision_body["reason_codes"]
