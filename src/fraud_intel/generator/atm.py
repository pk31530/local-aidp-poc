"""ATM channel synthetic generator (guide section 7). Deterministic for a
fixed (seed, n, reference_date). Emits (FraudEvent, SourceAlertContext |
None, SyntheticGroundTruthLabel) tuples -- the label is always produced
separately, never merged into the event.

Typology: skimming-enabled rapid cash-out (fraud scenario) vs. a
legitimate but unusually large single withdrawal (hard false positive).
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator._shared import build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "atm"
FRAUD_SCENARIO_TYPE = "ATM_SKIMMING_CASH_OUT"
HARD_FP_SCENARIO_TYPE = "ATM_LEGITIMATE_LARGE_WITHDRAWAL"

GEO_BUCKETS = ["urban", "suburban", "rural", "international"]


def generate_atm_events(
    *,
    seed: int,
    n: int,
    reference_date: date,
    customers: Sequence[SyntheticCustomer],
    generation_run_id: str,
    dataset_version: str,
) -> list[tuple[FraudEvent, Optional[SourceAlertContext], SyntheticGroundTruthLabel]]:
    rng = channel_rng(seed, CHANNEL)
    draws = draw_scenarios(
        rng,
        n,
        reference_date=reference_date,
        customers=customers,
        channel=CHANNEL,
        fraud_scenario_type=FRAUD_SCENARIO_TYPE,
        hard_fp_scenario_type=HARD_FP_SCENARIO_TYPE,
    )

    results = []
    for draw in draws:
        event_id = uuid.UUID(bytes=rng.bytes(16))
        base_amount = int(rng.integers(2_000, 20_000))
        amount_minor_units = base_amount * int(rng.integers(3, 10)) if draw.kind != "NORMAL" else base_amount

        payload = ATMPayload(
            atm_id=f"ATM{int(rng.integers(1000, 9999))}",
            atm_geo_bucket=str(rng.choice(GEO_BUCKETS)),
            transaction_type="withdrawal",
            card_present_flag=True,
        )
        event = FraudEvent(
            event_id=event_id,
            channel=CHANNEL,
            customer_id=draw.customer_id,
            account_id=draw.account_id,
            event_timestamp=draw.event_timestamp,
            amount_minor_units=amount_minor_units,
            direction="debit",
            scenario_id=draw.scenario_id,
            channel_payload=payload,
        )
        source_alert = build_source_alert_context(
            draw, rng=rng, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version
        )
        label = build_label(draw, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version)
        results.append((event, source_alert, label))
    return results
