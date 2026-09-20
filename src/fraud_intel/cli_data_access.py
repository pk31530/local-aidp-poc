"""Minimal, real Postgres data access backing the v1.3 CLI's `generate`/
`train`/`score` commands (Phase 6). Reviewed as SQL, not exercised by any
unit test -- every CLI test monkeypatches these functions directly, same
precedent as every other "_Postgres*Store" in this codebase. Reference
channel (online_banking) only for population loading (train/score);
`generate` supports all seven channels, since Phase 1's generators already
exist for all of them.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import psycopg2.extras

from src.common.db import get_connection
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.generator.ach import generate_ach_events
from src.fraud_intel.generator.atm import generate_atm_events
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.generator.debit_card import generate_debit_card_events
from src.fraud_intel.generator.mobile_deposit import generate_mobile_deposit_events
from src.fraud_intel.generator.online_banking import generate_online_banking_events
from src.fraud_intel.generator.p2p import generate_p2p_events
from src.fraud_intel.generator.wire import generate_wire_events

_GENERATORS = {
    "ach": generate_ach_events,
    "wire": generate_wire_events,
    "mobile_deposit": generate_mobile_deposit_events,
    "online_banking": generate_online_banking_events,
    "atm": generate_atm_events,
    "debit_card": generate_debit_card_events,
    "p2p": generate_p2p_events,
}


def generate_and_write(*, channel: str, count: int, seed: int, database: str, reference_date: date) -> dict[str, Any]:
    if channel not in _GENERATORS:
        raise ValueError(f"unknown channel {channel!r}")
    generation_run_id = f"genrun-{uuid.uuid4()}"
    dataset_version = f"dsv-{uuid.uuid4().hex[:16]}"
    customers = generate_customers(n=max(count // 5, 1), seed=seed, reference_date=reference_date)
    results = _GENERATORS[channel](
        seed=seed, n=count, reference_date=reference_date, customers=customers,
        generation_run_id=generation_run_id, dataset_version=dataset_version,
    )
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor() as cur:
                for event, source_alert, label in results:
                    cur.execute(
                        "INSERT INTO channel_events (event_id, channel, customer_id, account_id, event_timestamp, "
                        "amount_minor_units, direction, device_id, ip_address, channel_payload, scenario_id, "
                        "schema_version, generation_run_id, dataset_version) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                        (
                            str(event.event_id), event.channel, event.customer_id, event.account_id,
                            event.event_timestamp, event.amount_minor_units, event.direction, event.device_id,
                            event.ip_address, psycopg2.extras.Json(event.channel_payload.model_dump(mode="json")),
                            event.scenario_id, event.schema_version, generation_run_id, dataset_version,
                        ),
                    )
                    if source_alert is not None:
                        cur.execute(
                            "INSERT INTO source_alerts (source_alert_id, source_system, event_id, "
                            "source_alert_created_at, source_rule_ids, source_rule_version, source_alert_score, "
                            "source_alert_reason_codes, generation_run_id, dataset_version) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT (source_system, source_alert_id) DO NOTHING",
                            (
                                str(source_alert.source_alert_id), source_alert.source_system, str(event.event_id),
                                source_alert.source_alert_created_at, psycopg2.extras.Json(source_alert.source_rule_ids),
                                source_alert.source_rule_version, source_alert.source_alert_score,
                                psycopg2.extras.Json(source_alert.source_alert_reason_codes),
                                generation_run_id, dataset_version,
                            ),
                        )
                    cur.execute(
                        "INSERT INTO synthetic_event_labels (event_id, scenario_id, synthetic_scenario_label, "
                        "scenario_type, generation_run_id, dataset_version) VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (event_id) DO NOTHING",
                        (
                            str(event.event_id), label.scenario_id, label.synthetic_scenario_label,
                            label.scenario_type, generation_run_id, dataset_version,
                        ),
                    )
    finally:
        conn.close()
    return {"channel": channel, "count": len(results), "generation_run_id": generation_run_id, "dataset_version": dataset_version}


def load_online_banking_population(
    database: str,
) -> tuple[list[FraudEvent], list[SourceAlertContext], list[SyntheticGroundTruthLabel]]:
    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM channel_events WHERE channel = 'online_banking'")
                events = [
                    FraudEvent(
                        event_id=row["event_id"], channel=row["channel"], customer_id=row["customer_id"],
                        account_id=row["account_id"], event_timestamp=row["event_timestamp"],
                        amount_minor_units=row["amount_minor_units"], direction=row["direction"],
                        device_id=row["device_id"], ip_address=row["ip_address"], scenario_id=row["scenario_id"],
                        schema_version=row["schema_version"], channel_payload=OnlineBankingPayload(**row["channel_payload"]),
                    )
                    for row in cur.fetchall()
                ]
                event_ids = [str(e.event_id) for e in events]
                if not event_ids:
                    return [], [], []

                cur.execute("SELECT * FROM source_alerts WHERE event_id = ANY(%s)", (event_ids,))
                source_alerts = [
                    SourceAlertContext(
                        source_alert_id=row["source_alert_id"], source_system=row["source_system"],
                        event_id=row["event_id"], source_alert_created_at=row["source_alert_created_at"],
                        source_rule_ids=row["source_rule_ids"], source_rule_version=row["source_rule_version"],
                        source_alert_score=row["source_alert_score"],
                        source_alert_reason_codes=row["source_alert_reason_codes"],
                        generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                        created_at=row["created_at"],
                    )
                    for row in cur.fetchall()
                ]

                cur.execute("SELECT * FROM synthetic_event_labels WHERE event_id = ANY(%s)", (event_ids,))
                labels = [
                    SyntheticGroundTruthLabel(
                        event_id=row["event_id"], scenario_id=row["scenario_id"],
                        synthetic_scenario_label=row["synthetic_scenario_label"], scenario_type=row["scenario_type"],
                        generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                        generated_at=row["generated_at"],
                    )
                    for row in cur.fetchall()
                ]
    finally:
        conn.close()
    return events, source_alerts, labels
