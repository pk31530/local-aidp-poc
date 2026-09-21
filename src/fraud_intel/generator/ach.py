"""ACH channel synthetic generator (guide section 7). Deterministic for a
fixed (seed, n, reference_date). Emits (FraudEvent, SourceAlertContext |
None, SyntheticGroundTruthLabel) tuples -- the label is always produced
separately, never merged into the event.

Typology: account-takeover batch fraud (fraud scenario) vs. a legitimate
but unusually large same-day batch (hard false positive) -- both may fire
the simulated upstream rule engine, per guide section 7.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator._shared import FRAUD_SCENARIO, HARD_FALSE_POSITIVE, build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "ach"
FRAUD_SCENARIO_TYPE = "ACH_ACCOUNT_TAKEOVER_BATCH"
HARD_FP_SCENARIO_TYPE = "ACH_LEGITIMATE_LARGE_BATCH"

SEC_CODES = ["PPD", "CCD", "WEB", "TEL"]


def generate_ach_events(
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
        base_amount = int(rng.integers(5_000, 50_000))
        if draw.kind == FRAUD_SCENARIO:
            amount_minor_units = base_amount * int(rng.integers(5, 15))
        elif draw.kind == HARD_FALSE_POSITIVE:
            amount_minor_units = base_amount * int(rng.integers(4, 10))
        else:
            amount_minor_units = base_amount

        payload = ACHPayload(
            sec_code=str(rng.choice(SEC_CODES)),
            originating_routing_number=f"{int(rng.integers(100000000, 999999999))}",
            receiving_routing_number=f"{int(rng.integers(100000000, 999999999))}",
            batch_id=f"BATCH{int(rng.integers(100000, 999999))}",
            effective_entry_date=draw.event_timestamp.date(),
            company_id=f"COMP{int(rng.integers(1000, 9999))}",
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
