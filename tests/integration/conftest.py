"""Fixtures shared by the integration suite.

Every integration test runs against the isolated `aidp_test` database (fix
H4) — never the demo database other phases' live verification has been
populating. Each test gets a clean slate via the autouse fixture below.
"""
from __future__ import annotations

import psycopg2.extras
import pytest

from src.common.db import get_connection

TEST_DB = "aidp_test"


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
    """Inserts one known customer + profile into aidp_test and returns its id."""
    customer_id = "TESTCUST1"
    conn = get_connection(TEST_DB)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO customers (customer_id, full_name, home_country) VALUES (%s, %s, %s)",
                    (customer_id, "Integration Test Customer", "India"),
                )
                cur.execute(
                    """
                    INSERT INTO customer_profiles
                        (customer_id, avg_transaction_amount, stddev_transaction_amount,
                         known_devices, known_countries, home_country)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        customer_id, 1000.0, 200.0,
                        psycopg2.extras.Json(["DEVTEST1"]),
                        psycopg2.extras.Json(["India"]),
                        "India",
                    ),
                )
    finally:
        conn.close()
    return customer_id
