"""Phase 3 seeding CLI.

Generates synthetic customers + historical transactions + a failed-attempts
event stream, writes them under data/seed/, and loads the customer dimension
+ online feature-store baseline directly into Postgres (`customers` and
`customer_profiles`, fix C1/C3) so no downstream phase can hit a cold-start
customer lookup.

Usage:
    python -m src.generator.seed
    python -m src.generator.seed --customers 2000 --transactions 8000 --fraud-ratio 0.03 --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime

import polars as pl
import psycopg2
import psycopg2.extras

from src.common.config import PROJECT_ROOT, get_settings
from src.common.logging import configure_logging, get_logger
from src.common.timeutil import APP_TZ
from src.generator.customers import demo_customer_row, generate_customers
from src.generator.history import generate_history

SEED_DIR = PROJECT_ROOT / "data" / "seed"

# Guaranteed demo customer ids referenced by the guide's worked scenario
# (fix C3) — always present regardless of --customers, so the documented
# demo (C101: typical spend ~8,000, then a suspicious 82,000 transaction)
# is reproducible on any run.
GUARANTEED_DEMO_CUSTOMER_IDS = ["C101"]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate synthetic AiDP POC data.")
    parser.add_argument("--customers", type=int, default=settings.gen_customers)
    parser.add_argument("--transactions", type=int, default=settings.gen_historical_transactions)
    parser.add_argument("--fraud-ratio", type=float, default=settings.gen_fraud_ratio)
    parser.add_argument("--seed", type=int, default=settings.gen_random_seed)
    parser.add_argument("--skip-db-load", action="store_true", help="Only write files, do not load Postgres.")
    return parser.parse_args()


def load_customer_profiles(customers_df, settings) -> None:
    """Upsert `customers` + `customer_profiles` rows (fix C1/C3). Idempotent."""
    log = get_logger(__name__)
    records = customers_df.to_dicts()

    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                customer_rows = [
                    (r["customer_id"], r["full_name"], r["home_country"], r["signup_date"])
                    for r in records
                ]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO customers (customer_id, full_name, home_country, signup_date)
                    VALUES %s
                    ON CONFLICT (customer_id) DO UPDATE SET
                        full_name = EXCLUDED.full_name,
                        home_country = EXCLUDED.home_country,
                        signup_date = EXCLUDED.signup_date
                    """,
                    customer_rows,
                )

                profile_rows = [
                    (
                        r["customer_id"],
                        r["avg_transaction_amount"],
                        r["stddev_transaction_amount"],
                        psycopg2.extras.Json(r["known_devices"]),
                        psycopg2.extras.Json(r["known_countries"]),
                        r["home_country"],
                    )
                    for r in records
                ]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO customer_profiles
                        (customer_id, avg_transaction_amount, stddev_transaction_amount,
                         known_devices, known_countries, home_country)
                    VALUES %s
                    ON CONFLICT (customer_id) DO UPDATE SET
                        avg_transaction_amount = EXCLUDED.avg_transaction_amount,
                        stddev_transaction_amount = EXCLUDED.stddev_transaction_amount,
                        known_devices = EXCLUDED.known_devices,
                        known_countries = EXCLUDED.known_countries,
                        home_country = EXCLUDED.home_country,
                        updated_at = now()
                    """,
                    profile_rows,
                )
        log.info("customer_profiles_loaded", count=len(records))
    finally:
        conn.close()


def main() -> None:
    configure_logging("generator.seed")
    log = get_logger(__name__)
    args = parse_args()
    settings = get_settings()

    reference_date = datetime.now(tz=APP_TZ)

    log.info(
        "generation_started",
        customers=args.customers,
        transactions=args.transactions,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )

    customers_df = generate_customers(args.customers, args.seed, reference_date)

    # Guarantee the demo customer id(s) exist even if not in the normal
    # C{1000+i} range (fix C3).
    demo_rows = [demo_customer_row(cid, reference_date) for cid in GUARANTEED_DEMO_CUSTOMER_IDS]
    customers_df = pl.concat([customers_df, pl.DataFrame(demo_rows)], how="vertical")

    transactions_df, failed_attempts_df = generate_history(
        customers_df, args.transactions, args.fraud_ratio, args.seed, reference_date
    )

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    customers_path = SEED_DIR / "customers.parquet"
    transactions_path = SEED_DIR / "historical_transactions.csv"
    failed_attempts_path = SEED_DIR / "historical_failed_attempts.csv"

    customers_df.write_parquet(customers_path)
    transactions_df.write_csv(transactions_path)
    failed_attempts_df.write_csv(failed_attempts_path)

    fraud_count = int(transactions_df["is_fraud"].sum())
    log.info(
        "generation_complete",
        customers_written=customers_df.height,
        transactions_written=transactions_df.height,
        fraud_count=fraud_count,
        fraud_ratio_actual=round(fraud_count / transactions_df.height, 4),
        failed_attempts_written=failed_attempts_df.height,
        customers_path=str(customers_path),
        transactions_path=str(transactions_path),
        failed_attempts_path=str(failed_attempts_path),
    )

    if not args.skip_db_load:
        load_customer_profiles(customers_df, settings)


if __name__ == "__main__":
    main()
