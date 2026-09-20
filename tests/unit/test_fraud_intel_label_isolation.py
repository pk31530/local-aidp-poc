"""Phase 1 leakage-prevention tests (guide section 6): proves, structurally,
that no label/scenario/disposition field can ever appear inside a
FraudEvent or any channel payload -- by construction, not by convention.
Re-asserted in Phase 2/4/5 against the real feature-computation code; this
file only covers the event/payload/label CONTRACTS themselves, which is
all that exists as of Phase 1.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import ChannelPayload, FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.generator.ach import generate_ach_events
from src.fraud_intel.generator.customers import generate_customers

FORBIDDEN_FIELD_NAMES = {
    "synthetic_scenario_label",
    "analyst_disposition",
    "outcome_status",
    "training_eligible",
}

ALL_CHANNEL_PAYLOAD_MODELS = [
    ACHPayload,
    WirePayload,
    MobileDepositPayload,
    OnlineBankingPayload,
    ATMPayload,
    DebitCardPayload,
    P2PPayload,
]


def test_fraud_event_never_declares_a_forbidden_label_field():
    field_names = set(FraudEvent.model_fields.keys())
    assert field_names.isdisjoint(FORBIDDEN_FIELD_NAMES)


def test_no_channel_payload_declares_a_forbidden_label_field():
    for model in ALL_CHANNEL_PAYLOAD_MODELS:
        field_names = set(model.model_fields.keys())
        assert field_names.isdisjoint(FORBIDDEN_FIELD_NAMES), f"{model.__name__} leaks a label field"


def test_scenario_id_is_the_only_synthetic_only_field_on_fraud_event():
    """scenario_id is explicitly permitted (guide section 6) -- it is the
    ONE synthetic-only field allowed directly on FraudEvent, and it is
    optional. No other scenario/label-shaped field may exist alongside it."""
    field_names = set(FraudEvent.model_fields.keys())
    assert "scenario_id" in field_names
    assert FraudEvent.model_fields["scenario_id"].is_required() is False
    suspicious = {name for name in field_names if "label" in name or "disposition" in name or "outcome" in name}
    assert suspicious == set()


def test_synthetic_ground_truth_label_and_source_alert_context_are_distinct_models_from_fraud_event():
    assert SyntheticGroundTruthLabel is not FraudEvent
    assert SourceAlertContext is not FraudEvent
    assert not issubclass(SyntheticGroundTruthLabel, FraudEvent)
    assert not issubclass(SourceAlertContext, FraudEvent)
    assert not issubclass(FraudEvent, SyntheticGroundTruthLabel)
    assert not issubclass(FraudEvent, SourceAlertContext)


def test_fraud_event_serialization_never_includes_label_fields_even_via_channel_payload():
    """Round-trips a real event through JSON and confirms no forbidden
    field appears anywhere in the serialized structure, including nested
    inside channel_payload."""
    payload = ACHPayload(
        sec_code="PPD",
        originating_routing_number="123456789",
        receiving_routing_number="987654321",
        batch_id="BATCH1",
        effective_entry_date=date(2026, 1, 1),
        company_id="COMP1",
    )
    event = FraudEvent(
        channel="ach",
        customer_id="FIC1000",
        account_id="FIA100000",
        event_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        amount_minor_units=5000,
        direction="debit",
        scenario_id="v1.3-ach-fraud-000001",
        channel_payload=payload,
    )
    dumped = event.model_dump(mode="json")
    assert set(dumped.keys()).isdisjoint(FORBIDDEN_FIELD_NAMES)
    assert set(dumped["channel_payload"].keys()).isdisjoint(FORBIDDEN_FIELD_NAMES)


def test_generated_tuples_keep_label_genuinely_separate_from_the_event():
    """A fraud-scenario event from a real generator run must never carry
    the label's own fields, even though the label says it is fraud."""
    customers = generate_customers(n=20, seed=7, reference_date=date(2026, 1, 1))
    results = generate_ach_events(
        seed=7,
        n=500,
        reference_date=date(2026, 1, 1),
        customers=customers,
        generation_run_id="genrun-iso-1",
        dataset_version="dsv-iso-1",
    )
    fraud_pairs = [(event, label) for event, _, label in results if label.synthetic_scenario_label]
    assert fraud_pairs, "expected at least one fraud-scenario event in this fixture batch"

    for event, label in fraud_pairs:
        event_dump = event.model_dump(mode="json")
        assert "synthetic_scenario_label" not in event_dump
        assert label.event_id == event.event_id
        # The only synthetic trace permitted directly on the event is the
        # optional scenario_id -- and it must match the label's own, never
        # diverge into a second, independent value.
        assert event.scenario_id == label.scenario_id


def test_channel_payload_annotation_covers_exactly_the_seven_channel_models():
    from typing import get_args

    union_members = get_args(get_args(ChannelPayload)[0])
    assert {m.model_fields["channel"].default for m in union_members} == {
        "ach",
        "wire",
        "mobile_deposit",
        "online_banking",
        "atm",
        "debit_card",
        "p2p",
    }
