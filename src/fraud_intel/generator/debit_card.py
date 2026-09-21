"""Debit Card channel synthetic generator (guide section 7). Deterministic
for a fixed (seed, n, reference_date). Emits (FraudEvent,
SourceAlertContext | None, SyntheticGroundTruthLabel) tuples -- the label
is always produced separately, never merged into the event.

Typology: card-not-present testing (small, rapid, fraud scenario) vs. a
legitimate cross-border card-present purchase (hard false positive).
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator._shared import FRAUD_SCENARIO, HARD_FALSE_POSITIVE, build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "debit_card"
FRAUD_SCENARIO_TYPE = "DEBIT_CARD_NOT_PRESENT_TESTING"
HARD_FP_SCENARIO_TYPE = "DEBIT_CARD_LEGITIMATE_CROSS_BORDER"

POS_ENTRY_MODES = ["chip", "swipe", "cnp", "contactless"]


def generate_debit_card_events(
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
        is_fraud = draw.kind == FRAUD_SCENARIO
        is_hard_fp = draw.kind == HARD_FALSE_POSITIVE
        amount_minor_units = int(rng.integers(50, 2_000)) if is_fraud else int(rng.integers(1_000, 30_000))

        payload = DebitCardPayload(
            merchant_id=f"MER{int(rng.integers(10000, 99999))}",
            mcc_code=str(int(rng.integers(1000, 9999))),
            pos_entry_mode=str(rng.choice(POS_ENTRY_MODES)),
            card_present_flag=not is_fraud,
            cross_border_flag=is_hard_fp,
            # Deterministic synthetic token, never a real PAN (Phase 7A
            # additive correction) -- draws from a smaller range for a
            # fraud scenario, simulating the same compromised card token
            # being used across multiple card-not-present testing events.
            card_token=f"CARDTOK{int(rng.integers(100, 999)) if is_fraud else int(rng.integers(100000, 999999))}",
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
