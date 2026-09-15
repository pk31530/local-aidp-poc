"""Integration: model -> FastAPI.

Uses FastAPI's TestClient (which triggers the app's real lifespan —
loading the actual registered model) with the `get_conn` dependency
overridden to the isolated `aidp_test` database (fix H4), so a real
`POST /score` request exercises the real endpoint code against isolated
data instead of the demo database other phases' testing has populated.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app, get_conn
from src.common.db import get_connection

TEST_DB = "aidp_test"


def _override_get_conn():
    conn = get_connection(TEST_DB)
    try:
        yield conn
    finally:
        conn.close()


app.dependency_overrides[get_conn] = _override_get_conn


def test_score_endpoint_persists_to_isolated_test_db(seeded_customer):
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.post(
            "/score",
            json={
                "customer_id": seeded_customer,
                "amount": 500.0,
                "merchant": "Grocery",
                "country": "India",
                "device_id": "DEVTEST1",
                "payment_method": "CARD",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] in ("APPROVE", "MONITOR", "REVIEW", "BLOCK")
        assert 0.0 <= body["fraud_probability"] <= 1.0

        transaction_id = body["transaction_id"]

        tx_response = client.get(f"/transactions/{transaction_id}")
        assert tx_response.status_code == 200
        assert tx_response.json()["customer_id"] == seeded_customer

        decision_response = client.get(f"/decisions/{transaction_id}")
        assert decision_response.status_code == 200
        assert decision_response.json()["decision"] == body["decision"]

    conn = get_connection(TEST_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM transactions WHERE transaction_id = %s", (transaction_id,))
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def test_score_endpoint_rejects_invalid_amount():
    with TestClient(app) as client:
        response = client.post(
            "/score",
            json={
                "customer_id": "ANYONE",
                "amount": -50.0,
                "merchant": "Grocery",
                "country": "India",
                "device_id": "DEV1",
                "payment_method": "CARD",
            },
        )
        assert response.status_code == 422  # fix M4
