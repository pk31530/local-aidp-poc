"""Debit Card channel synthetic generator (guide section 7). Deterministic
for a fixed (seed, n, reference_date). Emits (FraudEvent,
SourceAlertContext | None, SyntheticGroundTruthLabel) tuples -- the label
is always produced separately, never merged into the event.

Typology: card-not-present testing (small, rapid, fraud scenario) vs. a
legitimate cross-border card-present purchase (hard false positive).

Synthetic-label-leakage correction (channel generator version v2). The
original implementation assigned four payload fields DETERMINISTICALLY
from `is_fraud`:

  * ``card_present_flag = not is_fraud`` -- which the feature adapter
    turns into ``card_not_present_flag`` (a real model feature), making
    that feature an EXACT COPY of the label. Verified over the approved
    deterministic population: 117/117 fraud rows True, 339/339
    legitimate rows False. Any model trained on it learns one boolean
    and reports perfect held-out metrics that say nothing about
    detection quality.
  * ``cross_border_flag = is_hard_fp`` -- so ``cross_border_flag=True``
    implied LEGITIMATE with certainty (0 fraud / 268 legitimate). It
    feeds the ``DEBIT_CARD_CROSS_BORDER_NEW_DEVICE`` rule and the
    ``cross_border_new_device_combo_flag`` feature.
  * ``card_token`` -- drawn from a 3-digit range for fraud and a 6-digit
    range otherwise, so the token's very LENGTH encoded the label. The
    token is a graph entity (``extract_entities``), so this leaked into
    the graph component of the operational score.
  * ``amount_minor_units`` -- fraud drawn from [50, 2000) and legitimate
    from [1000, 30000), leaving a pure-fraud region below 1000 that
    covered 51% of fraud rows with certainty. This propagates into the
    ``amount_vs_entity_average`` and ``amount_zscore`` model features.

Every one of those is now drawn PROBABILISTICALLY, with distributions
that OVERLAP between fraud and legitimate activity. The typology is
preserved -- card-testing fraud is still predominantly card-not-present
and predominantly small-amount, and the hard-false-positive scenario is
still cross-border -- but no payload field, and therefore no feature
derived from one, equals or inverts the label. Determinism for a fixed
(seed, n, reference_date) is unchanged.

Because the generated events genuinely changed, this channel's
generation identity must change too: see
``src.fraud_intel.cli_data_access.CHANNEL_GENERATOR_VERSION``.
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
from src.fraud_intel.generator._shared import FRAUD_SCENARIO, HARD_FALSE_POSITIVE, NORMAL, build_label, build_source_alert_context, channel_rng, draw_scenarios
from src.fraud_intel.generator.customers import SyntheticCustomer

CHANNEL = "debit_card"
FRAUD_SCENARIO_TYPE = "DEBIT_CARD_NOT_PRESENT_TESTING"
HARD_FP_SCENARIO_TYPE = "DEBIT_CARD_LEGITIMATE_CROSS_BORDER"

POS_ENTRY_MODES = ["chip", "swipe", "cnp", "contactless"]

# ---- leakage-safe payload distributions (channel generator version v2) --------------
#
# Every probability below is strictly between 0 and 1 for EVERY scenario
# kind. That is the structural property that matters: a field whose
# probability is 0 or 1 for any kind becomes a one-sided certainty
# ("value=True => LEGITIMATE") even when it is not a full label copy.
# The one deliberate exception is HARD_FALSE_POSITIVE's cross-border
# probability, which stays 1.0 because being cross-border IS that
# typology's definition -- it is safe only because fraud and normal
# activity are now cross-border too, so cross_border_flag=True no longer
# implies a legitimate row.

# Card-testing fraud is predominantly card-not-present, but card-present
# fraud (a cloned/stolen card used at a terminal) is real, and plenty of
# legitimate activity is card-not-present (e-commerce). Both values occur
# in both populations.
CARD_NOT_PRESENT_PROBABILITY = {
    FRAUD_SCENARIO: 0.70,
    HARD_FALSE_POSITIVE: 0.40,
    NORMAL: 0.25,
}

# Cross-border is the hard-false-positive typology's defining trait (1.0,
# unchanged), but fraud and ordinary activity now cross borders too, so
# the flag no longer identifies a legitimate row.
CROSS_BORDER_PROBABILITY = {
    FRAUD_SCENARIO: 0.30,
    HARD_FALSE_POSITIVE: 1.0,
    NORMAL: 0.08,
}

# Amounts: ONE shared support for every scenario kind, so no threshold can
# identify the label with certainty in either direction. Only the MIXTURE
# WEIGHT differs. Every event is either a "small-value" draw (exponent
# concentrates it near the floor) or an ordinary uniform draw; card
# testing is predominantly the former, which preserves the "small, rapid
# probe" character of the typology.
#
# The legitimate weights are deliberately non-trivial rather than 0: real
# card portfolios are full of small legitimate purchases (transit, coffee,
# subscriptions). Without them the floor of the amount range would be a
# pure-fraud band -- a one-sided certainty rather than a full label copy,
# but still a proxy. With them, both populations reach both ends of the
# range.
AMOUNT_MIN_MINOR_UNITS = 50
AMOUNT_MAX_MINOR_UNITS = 30_000
SMALL_AMOUNT_SKEW_EXPONENT = 6.0
SMALL_AMOUNT_PROBABILITY = {
    FRAUD_SCENARIO: 0.80,
    HARD_FALSE_POSITIVE: 0.25,
    NORMAL: 0.30,
}

# Card tokens: every token has the identical shape (CARDTOK + 6 digits)
# drawn from the identical value space, so neither the token's length nor
# its magnitude encodes the label. The card-testing typology -- the same
# compromised card reappearing across many events -- is preserved by
# giving fraud a much SMALLER pool of tokens, drawn from that same space.
CARD_TOKEN_SPACE = 1_000_000
COMPROMISED_CARD_POOL_SIZE = 250


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

    # The pool of "already compromised" cards, drawn once from the SAME
    # token space every other event draws from. Fraud events reuse these,
    # which is what makes a card token repeat across a testing run -- but
    # the tokens themselves are indistinguishable from any other card's.
    compromised_card_tokens = rng.integers(0, CARD_TOKEN_SPACE, size=COMPROMISED_CARD_POOL_SIZE)

    results = []
    for draw in draws:
        event_id = uuid.UUID(bytes=rng.bytes(16))
        is_fraud = draw.kind == FRAUD_SCENARIO

        # Shared support, kind-specific mixture weight -- never a
        # kind-specific range.
        is_small_amount = float(rng.random()) < SMALL_AMOUNT_PROBABILITY[draw.kind]
        amount_unit = float(rng.random()) ** (SMALL_AMOUNT_SKEW_EXPONENT if is_small_amount else 1.0)
        amount_minor_units = int(
            AMOUNT_MIN_MINOR_UNITS + (AMOUNT_MAX_MINOR_UNITS - AMOUNT_MIN_MINOR_UNITS) * amount_unit
        )

        card_not_present = float(rng.random()) < CARD_NOT_PRESENT_PROBABILITY[draw.kind]
        cross_border = float(rng.random()) < CROSS_BORDER_PROBABILITY[draw.kind]

        if is_fraud:
            token_index = int(compromised_card_tokens[int(rng.integers(0, COMPROMISED_CARD_POOL_SIZE))])
        else:
            token_index = int(rng.integers(0, CARD_TOKEN_SPACE))

        payload = DebitCardPayload(
            merchant_id=f"MER{int(rng.integers(10000, 99999))}",
            mcc_code=str(int(rng.integers(1000, 9999))),
            pos_entry_mode=str(rng.choice(POS_ENTRY_MODES)),
            card_present_flag=not card_not_present,
            cross_border_flag=cross_border,
            # Deterministic synthetic token, never a real PAN (Phase 7A
            # additive correction). Fixed width and one shared value space
            # for every scenario kind -- see COMPROMISED_CARD_POOL_SIZE.
            card_token=f"CARDTOK{token_index:06d}",
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
