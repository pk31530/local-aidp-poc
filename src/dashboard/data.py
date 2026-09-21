"""Query/health functions backing the Streamlit dashboard (Phase 8).

Reads directly from the same Postgres tables the API/consumer write to —
no separate dashboard-only data path, so what's shown is exactly what the
platform has actually persisted.
"""
from __future__ import annotations

import json

import httpx
import pandas as pd
import psycopg2.extras

from src.common.config import PROJECT_ROOT, get_settings
from src.common.db import get_connection

CONFUSION_MATRIX_PATH = PROJECT_ROOT / "data" / "models" / "last_confusion_matrix.json"

# Phase 8 corrective pass: the dashboard (every tab, including the legacy
# v1.1 ones below) must never resolve a bare get_connection() call to
# `settings.postgres_db`'s default ("aidp") -- it explicitly targets the
# existing `postgres_test_db` setting (aidp_test by default, overridable
# via .env's POSTGRES_TEST_DB), the SAME setting `tests/integration`/
# `tests/smoke` already use, rather than introducing a new one. See
# RUNBOOK.md's "v1.3 fraud intelligence" section.
def _dashboard_database() -> str:
    return get_settings().postgres_test_db


def _query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection(_dashboard_database())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    finally:
        conn.close()


def get_executive_metrics() -> dict:
    df = _query_df(
        """
        SELECT t.amount, fd.decision, fd.risk_level
        FROM transactions t
        JOIN fraud_decisions fd ON fd.transaction_id = t.transaction_id
        """
    )
    if df.empty:
        return {
            "transactions_processed": 0, "transaction_value": 0.0, "fraud_alerts": 0,
            "fraud_rate": 0.0, "blocked_transactions": 0, "potential_fraud_value": 0.0,
        }

    df["amount"] = df["amount"].astype(float)
    high_risk = df[df["decision"].isin(["REVIEW", "BLOCK"])]
    blocked = df[df["decision"] == "BLOCK"]

    return {
        "transactions_processed": len(df),
        "transaction_value": float(df["amount"].sum()),
        "fraud_alerts": len(high_risk),
        "fraud_rate": round(len(high_risk) / len(df), 4) if len(df) else 0.0,
        "blocked_transactions": len(blocked),
        "potential_fraud_value": float(high_risk["amount"].sum()),
    }


def get_active_model() -> dict | None:
    df = _query_df(
        """
        SELECT model_version, model_name, precision_score, recall_score, f1_score, roc_auc_score, registered_at
        FROM model_versions
        WHERE is_active = true
        ORDER BY registered_at DESC LIMIT 1
        """
    )
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    for k in ("precision_score", "recall_score", "f1_score", "roc_auc_score"):
        row[k] = float(row[k]) if row[k] is not None else None
    return row


def get_confusion_matrix() -> dict | None:
    if not CONFUSION_MATRIX_PATH.exists():
        return None
    return json.loads(CONFUSION_MATRIX_PATH.read_text())


def get_live_transactions(limit: int = 100) -> pd.DataFrame:
    return _query_df(
        """
        SELECT t.transaction_timestamp AS time, t.transaction_id, t.customer_id AS customer,
               t.amount, t.country, fd.fraud_probability AS fraud_score, fd.decision, fd.risk_level
        FROM transactions t
        JOIN fraud_decisions fd ON fd.transaction_id = t.transaction_id
        ORDER BY t.created_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_fraud_analysis() -> dict:
    df = _query_df(
        """
        SELECT t.transaction_id, t.amount, t.country, t.merchant, t.transaction_timestamp,
               fd.fraud_probability, fd.decision, fd.risk_level
        FROM transactions t
        JOIN fraud_decisions fd ON fd.transaction_id = t.transaction_id
        """
    )
    if df.empty:
        return {"data": df}

    df["amount"] = df["amount"].astype(float)
    df["fraud_probability"] = df["fraud_probability"].astype(float)
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    return {"data": df}


def get_platform_health() -> dict:
    settings = get_settings()
    results: dict[str, bool] = {}

    try:
        conn = get_connection(settings.postgres_test_db)
        conn.close()
        results["PostgreSQL"] = True
    except Exception:
        results["PostgreSQL"] = False

    try:
        r = httpx.get(f"http://{settings.minio_endpoint}/minio/health/live", timeout=3)
        results["MinIO"] = r.status_code == 200
    except Exception:
        results["MinIO"] = False

    try:
        r = httpx.get("http://127.0.0.1:9644/v1/status/ready", timeout=3)
        results["Redpanda"] = r.status_code == 200
    except Exception:
        results["Redpanda"] = False

    try:
        r = httpx.get(f"{settings.mlflow_tracking_uri}/health", timeout=3)
        results["MLflow"] = r.status_code == 200
    except Exception:
        results["MLflow"] = False

    try:
        r = httpx.get(f"http://{settings.api_host}:{settings.api_port}/health", timeout=3)
        results["FastAPI"] = r.status_code == 200
    except Exception:
        results["FastAPI"] = False

    return results
