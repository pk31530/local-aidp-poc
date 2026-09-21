"""Mobile Check Deposit channel synthetic generator (guide section 7).
Deterministic for a fixed (seed, n, reference_date). Emits (FraudEvent,
SourceAlertContext | None, SyntheticGroundTruthLabel) tuples -- the label
is always produced separately, never merged into the event.

Typology: duplicate/altered check deposit (fraud scenario) vs. rapid but
legitimate resubmission of a low-quality image (hard false positive).
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator._shared import FRAUD_SCENARIO, build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "mobile_deposit"
FRAUD_SCENARIO_TYPE = "MOBILE_DEPOSIT_DUPLICATE_ALTERED_CHECK"
HARD_FP_SCENARIO_TYPE = "MOBILE_DEPOSIT_RAPID_LEGITIMATE_RESUBMISSION"


def generate_mobile_deposit_events(
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
        amount_minor_units = int(rng.integers(1_000, 30_000))

        payload = MobileDepositPayload(
            duplicate_image_hash_flag=is_fraud,
            car_lar_mismatch_flag=is_fraud and bool(rng.integers(0, 2)),
            signature_verification_flag=not is_fraud,
            endorsement_present_flag=not is_fraud,
            micr_consistency_flag=not is_fraud,
            image_quality_score=float(rng.uniform(0.3, 0.6)) if is_fraud else float(rng.uniform(0.8, 1.0)),
            # Deterministic synthetic token, never a real payee name/account
            # (Phase 7A additive correction) -- draws from a smaller range
            # for a fraud scenario, simulating the same duplicated/altered
            # check being redeposited to the same payee across events.
            check_payee_token=f"PAYEETOK{int(rng.integers(100, 999)) if is_fraud else int(rng.integers(100000, 999999))}",
        )
        event = FraudEvent(
            event_id=event_id,
            channel=CHANNEL,
            customer_id=draw.customer_id,
            account_id=draw.account_id,
            event_timestamp=draw.event_timestamp,
            amount_minor_units=amount_minor_units,
            direction="credit",
            scenario_id=draw.scenario_id,
            channel_payload=payload,
        )
        source_alert = build_source_alert_context(
            draw, rng=rng, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version
        )
        label = build_label(draw, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version)
        results.append((event, source_alert, label))
    return results
