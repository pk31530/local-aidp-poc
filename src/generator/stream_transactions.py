"""Phase 7: Redpanda producer — publishes live synthetic transaction events
to the `transactions` topic at a configurable rate.

Usage:
    python -m src.generator.stream_transactions --rate 10
    python -m src.generator.stream_transactions --rate 10 --duration 30
    python -m src.generator.stream_transactions --rate 1 --inject-malformed-rate 0.1
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import psycopg2.extras
from confluent_kafka import Producer

from src.common.config import get_settings
from src.common.db import get_connection
from src.common.logging import configure_logging, get_logger

MERCHANTS = [
    "Grocery", "Dining", "Fuel", "Utilities", "Pharmacy", "Fashion",
    "Entertainment", "Electronics", "Travel", "Jewellery",
]
MERCHANT_WEIGHTS = [0.18, 0.16, 0.14, 0.10, 0.10, 0.10, 0.08, 0.06, 0.05, 0.03]
RISKY_MERCHANTS = ["Electronics", "Travel", "Jewellery"]

PAYMENT_METHODS = ["CARD", "UPI", "NETBANKING", "WALLET"]
PAYMENT_WEIGHTS = [0.40, 0.35, 0.15, 0.10]

OTHER_COUNTRIES = ["USA", "UK", "Singapore", "UAE", "Germany", "Australia", "Canada", "Nigeria", "India"]

FRAUD_PATTERNS = ["HIGH_AMOUNT", "NEW_DEVICE", "NEW_COUNTRY", "NIGHT_TIME", "RISKY_MERCHANT"]


def _load_customer_pool(limit: int = 2000, database: str | None = None) -> list[dict]:
    """Pulls a pool of real, already-profiled customers from Postgres (the
    same customer_profiles Phase 3 seeded and Phase 6's API reads live) so
    streamed traffic looks like activity from real accounts. `database`
    lets tests point this at the isolated aidp_test database (fix H4)
    instead of the demo database."""
    conn = get_connection(database)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT customer_id, avg_transaction_amount, stddev_transaction_amount,
                       known_devices, known_countries, home_country
                FROM customer_profiles
                ORDER BY random()
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def _build_event(customer: dict, rng: np.random.Generator, inject_fraud: bool, inject_malformed: bool) -> dict:
    known_devices = list(customer["known_devices"]) or [f"DEV{int(rng.integers(100000, 999999))}"]
    known_countries = list(customer["known_countries"]) or [customer["home_country"] or "India"]
    avg = float(customer["avg_transaction_amount"] or 1000.0)
    stddev = max(float(customer["stddev_transaction_amount"] or 200.0), 1.0)

    amount = round(float(np.clip(rng.normal(avg, stddev), 10, None)), 2)
    device_id = str(rng.choice(known_devices))
    country = str(rng.choice(known_countries))
    merchant = str(rng.choice(MERCHANTS, p=MERCHANT_WEIGHTS))
    payment_method = str(rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS))
    now = datetime.now(timezone.utc)

    if inject_fraud:
        pattern = str(rng.choice(FRAUD_PATTERNS))
        if pattern == "HIGH_AMOUNT":
            amount = round(avg * float(rng.uniform(5, 15)), 2)
        elif pattern == "NEW_DEVICE":
            device_id = f"DEVNEW{int(rng.integers(100000, 999999))}"
        elif pattern == "NEW_COUNTRY":
            others = [c for c in OTHER_COUNTRIES if c not in known_countries]
            country = str(rng.choice(others)) if others else "Singapore"
        elif pattern == "RISKY_MERCHANT":
            merchant = str(rng.choice(RISKY_MERCHANTS))
        # NIGHT_TIME is left to real clock time — a demo run spanning the
        # night window will show it naturally; we don't fake the timestamp
        # for live-streamed events.

    event = {
        "transaction_id": f"TXS{uuid.uuid4().hex[:12].upper()}",
        "customer_id": customer["customer_id"],
        "transaction_timestamp": now.isoformat(),
        "amount": amount,
        "merchant": merchant,
        "country": country,
        "device_id": device_id,
        "payment_method": payment_method,
        "schema_version": 1,  # fix M3
        "is_fraud": inject_fraud,  # ground-truth label for demo narration only; never sent to the model as a feature
    }

    if inject_malformed:
        # Deliberately break the message so the consumer's DLQ path (fix
        # M4/error handling) can be exercised for real, not just unit-tested.
        broken = dict(event)
        choice = rng.integers(0, 3)
        if choice == 0:
            del broken["transaction_id"]
        elif choice == 1:
            broken["amount"] = -999
        else:
            broken["transaction_timestamp"] = "not-a-timestamp"
        return broken

    return event


def _delivery_callback(log):
    def _cb(err, msg):
        if err is not None:
            log.error("delivery_failed", error=str(err))
    return _cb


def run(
    rate: float,
    duration: float | None,
    fraud_ratio: float,
    inject_malformed_rate: float,
    seed: int,
    topic: str | None = None,
    database: str | None = None,
) -> dict:
    """`topic`/`database` default to the production topic/database; tests
    pass the isolated `transactions-test` topic and `aidp_test` database
    instead (fix H4)."""
    configure_logging("generator.stream")
    log = get_logger(__name__)
    settings = get_settings()
    topic = topic or settings.redpanda_topic_transactions

    customers = _load_customer_pool(database=database)
    if not customers:
        raise RuntimeError("No customers found in customer_profiles — run scripts/seed_data.sh first.")

    rng = np.random.default_rng(seed)
    producer = Producer({"bootstrap.servers": settings.redpanda_brokers})

    log.info("stream_started", rate=rate, duration=duration, fraud_ratio=fraud_ratio, customer_pool_size=len(customers), topic=topic)

    interval = 1.0 / rate
    sent = 0
    fraud_sent = 0
    malformed_sent = 0
    start = time.monotonic()

    try:
        while duration is None or (time.monotonic() - start) < duration:
            customer = customers[int(rng.integers(0, len(customers)))]
            inject_fraud = rng.random() < fraud_ratio
            inject_malformed = rng.random() < inject_malformed_rate

            event = _build_event(customer, rng, inject_fraud, inject_malformed)

            producer.produce(
                topic,
                key=str(event.get("customer_id", "unknown")).encode("utf-8"),
                value=json.dumps(event).encode("utf-8"),
                callback=_delivery_callback(log),
            )
            producer.poll(0)

            sent += 1
            fraud_sent += int(inject_fraud)
            malformed_sent += int(inject_malformed)

            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("stream_interrupted_by_user")
    finally:
        producer.flush(10)

    summary = {"sent": sent, "fraud_sent": fraud_sent, "malformed_sent": malformed_sent}
    log.info("stream_complete", **summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream synthetic transactions to Redpanda.")
    parser.add_argument("--rate", type=float, default=1.0, help="events per second (e.g. 1, 10, 100)")
    parser.add_argument("--duration", type=float, default=None, help="seconds to run; omit to run until Ctrl+C")
    parser.add_argument("--fraud-ratio", type=float, default=get_settings().gen_fraud_ratio)
    parser.add_argument("--inject-malformed-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=get_settings().gen_random_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.rate, args.duration, args.fraud_ratio, args.inject_malformed_rate, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
