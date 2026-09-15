"""Shared "score one transaction and persist the result" pipeline.

Used identically by the real-time API (src/api/main.py) and the streaming
consumer (src/ingestion/consumer.py) — extends the fix C2 "no duplicated
logic" principle from feature computation to the whole scoring path, so the
two serving paths can't silently drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import psycopg2.extras

from src.common.feature_store import fetch_customer_profile, fetch_recent_events, record_event
from src.common.features import MODEL_FEATURE_COLUMNS, RiskLookups, compute_features
from src.decisioning.engine import classify, reason_codes


@dataclass(frozen=True)
class ScoreResult:
    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    reason_codes: list


def score_and_persist(
    conn,
    model,
    model_version: str,
    risk_lookups: RiskLookups,
    *,
    transaction_id: str,
    customer_id: str,
    amount: float,
    merchant: str,
    country: str,
    device_id: str,
    payment_method: str,
    transaction_timestamp: datetime,
    source: str,
) -> ScoreResult:
    # fix C1: real, live queries against the online feature/profile store.
    profile = fetch_customer_profile(conn, customer_id)
    recent_events = fetch_recent_events(conn, customer_id, transaction_timestamp)

    # fix C2: the exact same feature function the batch pipeline uses.
    features = compute_features(
        amount=amount,
        merchant=merchant,
        country=country,
        device_id=device_id,
        transaction_timestamp=transaction_timestamp,
        profile=profile,
        recent_events=recent_events,
        risk_lookups=risk_lookups,
    )

    X = pd.DataFrame([{**features, "amount": amount}])[MODEL_FEATURE_COLUMNS].astype(float)
    fraud_probability = float(model.predict_proba(X)[:, 1][0])

    risk_level, decision = classify(fraud_probability)
    codes = reason_codes(features)

    # ON CONFLICT DO NOTHING: safe to retry this whole call (fix H3) without
    # double-inserting if a prior attempt partially got through.
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions
                    (transaction_id, customer_id, amount, merchant, country, device_id, payment_method, transaction_timestamp, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (transaction_id, customer_id, amount, merchant, country, device_id, payment_method, transaction_timestamp, source),
            )
            cur.execute(
                """
                INSERT INTO fraud_scores (transaction_id, fraud_probability, model_version)
                VALUES (%s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (transaction_id, fraud_probability, model_version),
            )
            cur.execute(
                """
                INSERT INTO fraud_decisions
                    (transaction_id, customer_id, fraud_probability, risk_level, decision, reason_codes, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (transaction_id, customer_id, fraud_probability, risk_level, decision, psycopg2.extras.Json(codes), model_version),
            )
        # fix C1: the online store is live — the *next* transaction for this
        # customer (whether via API or stream) sees this event.
        record_event(
            conn,
            customer_id=customer_id,
            transaction_id=transaction_id,
            event_type="transaction_success",
            amount=amount,
            country=country,
            device_id=device_id,
            occurred_at=transaction_timestamp,
        )

    return ScoreResult(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        decision=decision,
        reason_codes=codes,
    )
