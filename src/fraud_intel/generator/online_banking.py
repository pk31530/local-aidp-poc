"""Online/Mobile Banking channel synthetic generator (guide section 7).
Deterministic for a fixed (seed, n, reference_date). Emits (FraudEvent,
SourceAlertContext | None, SyntheticGroundTruthLabel) tuples -- the label
is always produced separately, never merged into the event.

Typology: credential-stuffing login followed by a transfer (fraud
scenario) vs. a legitimate new-device transfer (hard false positive). This
is also the Phase 2 reference channel.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator._shared import FRAUD_SCENARIO, build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "online_banking"
FRAUD_SCENARIO_TYPE = "ONLINE_BANKING_CREDENTIAL_STUFFING_TRANSFER"
HARD_FP_SCENARIO_TYPE = "ONLINE_BANKING_LEGITIMATE_NEW_DEVICE_TRANSFER"

LOGIN_METHODS = ["password", "biometric", "sso"]


def generate_online_banking_events(
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
        base_amount = int(rng.integers(5_000, 40_000))
        amount_minor_units = base_amount * int(rng.integers(3, 12)) if draw.kind != "NORMAL" else base_amount

        payload = OnlineBankingPayload(
            session_id=f"SESS{int(rng.integers(100000, 999999))}",
            login_method=str(rng.choice(LOGIN_METHODS)),
            mfa_used_flag=not is_fraud,
            transaction_type="transfer",
            target_account=f"TGT{int(rng.integers(100000, 999999))}",
        )
        event = FraudEvent(
            event_id=event_id,
            channel=CHANNEL,
            customer_id=draw.customer_id,
            account_id=draw.account_id,
            event_timestamp=draw.event_timestamp,
            amount_minor_units=amount_minor_units,
            direction="debit",
            device_id=f"DEV{int(rng.integers(100000, 999999))}",
            scenario_id=draw.scenario_id,
            channel_payload=payload,
        )
        source_alert = build_source_alert_context(
            draw, rng=rng, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version
        )
        label = build_label(draw, event_id=event_id, generation_run_id=generation_run_id, dataset_version=dataset_version)
        results.append((event, source_alert, label))
    return results
