"""Real-time adapter onto the online feature/profile store (fix C1).

Builds the same `CustomerProfileSnapshot` / `RecentEvent` shapes that
src/processing's batch adapter builds from a Polars DataFrame — but here by
querying live Postgres (`customer_profiles`, `recent_events`) at scoring
time. Both adapters feed the same `compute_features()` in
src/common/features.py, so real-time and batch features are computed by
identical logic (fix C2); only how the inputs are fetched differs.

Used by the FastAPI /score endpoint (Phase 6) and the streaming consumer
(Phase 7).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

from src.common.features import DEFAULT_PROFILE, WINDOW_24H, CustomerProfileSnapshot, RecentEvent

# Slightly wider than the largest feature window (24h) so events right at
# the boundary are never missed due to clock skew between rows.
_LOOKBACK = WINDOW_24H + timedelta(minutes=5)


def fetch_customer_profile(conn, customer_id: str) -> CustomerProfileSnapshot:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT avg_transaction_amount, stddev_transaction_amount,
                   known_devices, known_countries, home_country
            FROM customer_profiles
            WHERE customer_id = %s
            """,
            (customer_id,),
        )
        row = cur.fetchone()

    if row is None:
        # fix C3: unknown/new customer -> safe default, never a crash.
        return DEFAULT_PROFILE

    return CustomerProfileSnapshot(
        avg_transaction_amount=float(row["avg_transaction_amount"] or 0.0),
        stddev_transaction_amount=float(row["stddev_transaction_amount"] or 0.0),
        known_devices=frozenset(row["known_devices"] or []),
        known_countries=frozenset(row["known_countries"] or []),
        home_country=row["home_country"],
    )


def fetch_recent_events(conn, customer_id: str, as_of: datetime) -> list[RecentEvent]:
    """Real, live query against recent_events (fix C1) — not stubbed."""
    window_start = as_of - _LOOKBACK
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT event_type, occurred_at, device_id, country, amount
            FROM recent_events
            WHERE customer_id = %s
              AND occurred_at >= %s
              AND occurred_at < %s
            ORDER BY occurred_at
            """,
            (customer_id, window_start, as_of),
        )
        rows = cur.fetchall()

    return [
        RecentEvent(
            event_type=r["event_type"],
            occurred_at=r["occurred_at"],
            device_id=r["device_id"],
            country=r["country"],
            amount=float(r["amount"]) if r["amount"] is not None else None,
        )
        for r in rows
    ]


def record_event(
    conn,
    *,
    customer_id: str,
    transaction_id: str | None,
    event_type: str,
    amount: float | None,
    country: str | None,
    device_id: str | None,
    occurred_at: datetime,
) -> None:
    """Appends one event to the online store, so the *next* transaction for
    this customer sees it (fix C1: the store is live, not a snapshot)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recent_events
                (customer_id, transaction_id, event_type, amount, country, device_id, occurred_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (customer_id, transaction_id, event_type, amount, country, device_id, occurred_at),
        )
