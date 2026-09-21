"""Shared, deterministic helpers used by all seven per-channel generators
(guide section 7). Each channel's `generate_<channel>_events()` seeds its
own `np.random.default_rng(seed)` and calls these pure functions to decide
per-event scenario category, timestamp, and source-alert/label shape --
payload construction stays entirely channel-specific. No database, Docker,
or network access anywhere in this module.

Determinism note: every timestamp field below (`source_alert_created_at`,
`created_at`, `generated_at`) is set from `draw.event_timestamp`, never
from a model's wall-clock `default_factory` -- using the real-time default
would make two generator calls with the same seed disagree, breaking the
"deterministic for a fixed seed" requirement.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np

from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator.customers import SyntheticCustomer

NORMAL = "NORMAL"
HARD_FALSE_POSITIVE = "HARD_FALSE_POSITIVE"
FRAUD_SCENARIO = "FRAUD_SCENARIO"

# "including scenarios below 5%" (guide section 7).
DEFAULT_FRAUD_PREVALENCE = 0.03
DEFAULT_HARD_FP_PREVALENCE = 0.05
LOOKBACK_DAYS = 30

# A hard false positive fires a source alert almost as often as real fraud
# -- that is the point (guide section 7): legitimate-but-unusual behavior
# that also trips the upstream rule engine, so the triage layer is
# meaningfully tested, not just the upstream rule.
SOURCE_ALERT_FIRE_PROBABILITY = {
    NORMAL: 0.02,
    HARD_FALSE_POSITIVE: 0.90,
    FRAUD_SCENARIO: 0.85,
}

RULE_SET_VERSION = "v1"
SOURCE_SYSTEM = "LocalYamlRuleProvider (simulated upstream)"

# Fixed (never Python's randomized hash()) per-channel salt so each
# channel's np.random.default_rng([seed, salt]) stream diverges from its
# very first draw. Without this, two channel generators seeded with the
# identical bare `seed` can draw byte-identical UUIDs at analogous
# positions in their independent-but-parallel-structured RNG streams,
# producing cross-channel source_alert_id/event_id collisions -- caught by
# tests/unit/test_fraud_intel_generators.py's combined-batch uniqueness
# test. The salt values themselves are arbitrary but must stay fixed
# forever once assigned, since changing one changes every downstream
# generated id for that channel.
CHANNEL_SALTS = {
    "ach": 0,
    "wire": 1,
    "mobile_deposit": 2,
    "online_banking": 3,
    "atm": 4,
    "debit_card": 5,
    "p2p": 6,
}


def channel_rng(seed: int, channel: str) -> np.random.Generator:
    return np.random.default_rng([seed, CHANNEL_SALTS[channel]])


@dataclass(frozen=True)
class ScenarioDraw:
    kind: str
    scenario_id: Optional[str]
    scenario_type: str
    fires_source_alert: bool
    event_timestamp: datetime
    customer_id: str
    account_id: str


def draw_scenarios(
    rng: np.random.Generator,
    n: int,
    *,
    reference_date: date,
    customers: Sequence[SyntheticCustomer],
    channel: str,
    fraud_scenario_type: str,
    hard_fp_scenario_type: str,
    fraud_prevalence: float = DEFAULT_FRAUD_PREVALENCE,
    hard_fp_prevalence: float = DEFAULT_HARD_FP_PREVALENCE,
) -> list[ScenarioDraw]:
    if not customers:
        raise ValueError("customers must be non-empty")

    reference_dt = datetime(reference_date.year, reference_date.month, reference_date.day, tzinfo=timezone.utc)
    kind_draws = rng.random(n)
    customer_idx = rng.integers(0, len(customers), size=n)
    timestamp_offsets = rng.uniform(0, LOOKBACK_DAYS * 86400, size=n)
    fire_draws = rng.random(n)

    draws: list[ScenarioDraw] = []
    for i in range(n):
        d = float(kind_draws[i])
        if d < fraud_prevalence:
            kind = FRAUD_SCENARIO
            scenario_type = fraud_scenario_type
            scenario_id = f"v1.3-{channel}-fraud-{i:06d}"
        elif d < fraud_prevalence + hard_fp_prevalence:
            kind = HARD_FALSE_POSITIVE
            scenario_type = hard_fp_scenario_type
            scenario_id = f"v1.3-{channel}-hardfp-{i:06d}"
        else:
            kind = NORMAL
            scenario_type = "NORMAL"
            scenario_id = None

        fires = float(fire_draws[i]) < SOURCE_ALERT_FIRE_PROBABILITY[kind]
        customer = customers[int(customer_idx[i])]
        event_timestamp = reference_dt - timedelta(seconds=float(timestamp_offsets[i]))

        draws.append(
            ScenarioDraw(
                kind=kind,
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                fires_source_alert=fires,
                event_timestamp=event_timestamp,
                customer_id=customer.customer_id,
                account_id=customer.account_id,
            )
        )
    return draws


def build_source_alert_context(
    draw: ScenarioDraw,
    *,
    rng: np.random.Generator,
    event_id: uuid.UUID,
    generation_run_id: str,
    dataset_version: str,
) -> Optional[SourceAlertContext]:
    if not draw.fires_source_alert:
        return None
    return SourceAlertContext(
        source_alert_id=uuid.UUID(bytes=rng.bytes(16)),
        source_system=SOURCE_SYSTEM,
        event_id=event_id,
        source_alert_created_at=draw.event_timestamp,
        source_rule_ids=[f"{draw.scenario_type}_RULE"],
        source_rule_version=RULE_SET_VERSION,
        source_alert_score=None,
        source_alert_reason_codes=[f"{draw.scenario_type}_REASON"],
        generation_run_id=generation_run_id,
        dataset_version=dataset_version,
        created_at=draw.event_timestamp,
    )


def build_label(
    draw: ScenarioDraw,
    *,
    event_id: uuid.UUID,
    generation_run_id: str,
    dataset_version: str,
) -> SyntheticGroundTruthLabel:
    return SyntheticGroundTruthLabel(
        event_id=event_id,
        scenario_id=draw.scenario_id,
        synthetic_scenario_label=(draw.kind == FRAUD_SCENARIO),
        scenario_type=draw.scenario_type,
        generation_run_id=generation_run_id,
        dataset_version=dataset_version,
        generated_at=draw.event_timestamp,
    )
