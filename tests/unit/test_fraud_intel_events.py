"""Phase 1: schema validation for FraudEvent, the discriminated channel
payload union, SourceAlertContext, and SyntheticGroundTruthLabel. No
database, Docker, or network access -- models/fixtures only.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.events.wire import WirePayload

UTC_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
IST = timezone(timedelta(hours=5, minutes=30))


def _ach_payload() -> ACHPayload:
    return ACHPayload(
        sec_code="PPD",
        originating_routing_number="123456789",
        receiving_routing_number="987654321",
        batch_id="BATCH1",
        effective_entry_date=date(2026, 1, 1),
        company_id="COMP1",
    )


def _base_event_kwargs() -> dict:
    return dict(
        channel="ach",
        customer_id="FIC1000",
        account_id="FIA100000",
        event_timestamp=UTC_NOW,
        amount_minor_units=5000,
        direction="debit",
        channel_payload=_ach_payload(),
    )


# ---- FraudEvent base contract ------------------------------------------------


def test_fraud_event_accepts_valid_input():
    event = FraudEvent(**_base_event_kwargs())
    assert isinstance(event.event_id, uuid.UUID)
    assert event.channel == "ach"


def test_fraud_event_event_id_defaults_to_uuid():
    event = FraudEvent(**_base_event_kwargs())
    assert isinstance(event.event_id, uuid.UUID)


def test_fraud_event_rejects_naive_timestamp():
    kwargs = _base_event_kwargs()
    kwargs["event_timestamp"] = datetime(2026, 1, 1, 12, 0, 0)  # naive
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


def test_fraud_event_rejects_non_utc_timestamp():
    kwargs = _base_event_kwargs()
    kwargs["event_timestamp"] = datetime(2026, 1, 1, 12, 0, 0, tzinfo=IST)
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


def test_fraud_event_rejects_float_amount():
    kwargs = _base_event_kwargs()
    kwargs["amount_minor_units"] = 5000.0  # float, not int -- must be rejected
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


def test_fraud_event_rejects_zero_or_negative_amount():
    kwargs = _base_event_kwargs()
    kwargs["amount_minor_units"] = 0
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


def test_fraud_event_rejects_unsupported_schema_version():
    kwargs = _base_event_kwargs()
    kwargs["schema_version"] = 999
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


def test_fraud_event_accepts_supported_schema_version():
    kwargs = _base_event_kwargs()
    kwargs["schema_version"] = 1
    event = FraudEvent(**kwargs)
    assert event.schema_version == 1


def test_fraud_event_rejects_mismatched_channel_and_payload():
    """FraudEvent.channel must equal channel_payload.channel -- Pydantic's
    discriminated union alone does not enforce this (guide section 6)."""
    kwargs = _base_event_kwargs()
    kwargs["channel"] = "wire"  # payload is still ACHPayload
    with pytest.raises(ValidationError, match="does not match"):
        FraudEvent(**kwargs)


def test_fraud_event_scenario_id_optional_and_not_required():
    kwargs = _base_event_kwargs()
    assert "scenario_id" not in kwargs
    event = FraudEvent(**kwargs)
    assert event.scenario_id is None


def test_fraud_event_rejects_extra_fields():
    kwargs = _base_event_kwargs()
    kwargs["unexpected_field"] = "should not be allowed"
    with pytest.raises(ValidationError):
        FraudEvent(**kwargs)


# ---- Discriminated channel payload union --------------------------------------


CHANNEL_PAYLOADS = {
    "ach": _ach_payload(),
    "wire": WirePayload(
        wire_type="international",
        beneficiary_bank_id="BENBANK1",
        beneficiary_account="BENACCT1",
        purpose_code="GDS",
    ),
    "mobile_deposit": MobileDepositPayload(
        duplicate_image_hash_flag=False,
        car_lar_mismatch_flag=False,
        signature_verification_flag=True,
        endorsement_present_flag=True,
        micr_consistency_flag=True,
        image_quality_score=0.9,
    ),
    "online_banking": OnlineBankingPayload(
        session_id="SESS1",
        login_method="password",
        mfa_used_flag=True,
        transaction_type="transfer",
        target_account="TGT1",
    ),
    "atm": ATMPayload(atm_id="ATM1", atm_geo_bucket="urban", transaction_type="withdrawal", card_present_flag=True),
    "debit_card": DebitCardPayload(
        merchant_id="MER1", mcc_code="5411", pos_entry_mode="chip", card_present_flag=True, cross_border_flag=False
    ),
    "p2p": P2PPayload(recipient_handle="@r1", network="Zelle-like", memo_present_flag=True, recipient_is_new_flag=False),
}


@pytest.mark.parametrize("channel,payload", CHANNEL_PAYLOADS.items())
def test_discriminated_union_routes_each_channel_payload(channel, payload):
    kwargs = _base_event_kwargs()
    kwargs["channel"] = channel
    kwargs["channel_payload"] = payload
    event = FraudEvent(**kwargs)
    assert event.channel_payload.channel == channel
    assert type(event.channel_payload) is type(payload)


def test_mobile_deposit_image_quality_score_bounded():
    with pytest.raises(ValidationError):
        MobileDepositPayload(
            duplicate_image_hash_flag=False,
            car_lar_mismatch_flag=False,
            signature_verification_flag=True,
            endorsement_present_flag=True,
            micr_consistency_flag=True,
            image_quality_score=1.5,  # out of [0, 1]
        )


def test_wire_type_restricted_to_allowlisted_values():
    with pytest.raises(ValidationError):
        WirePayload(
            wire_type="interplanetary",  # not domestic/international
            beneficiary_bank_id="BENBANK1",
            beneficiary_account="BENACCT1",
            purpose_code="GDS",
        )


# ---- SourceAlertContext --------------------------------------------------------


def _source_alert_kwargs(event_id: uuid.UUID) -> dict:
    return dict(
        source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id,
        source_alert_created_at=UTC_NOW,
        source_rule_ids=["RULE1"],
        source_rule_version="v1",
        source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        created_at=UTC_NOW,
    )


def test_source_alert_context_accepts_valid_input():
    event_id = uuid.uuid4()
    alert = SourceAlertContext(**_source_alert_kwargs(event_id))
    assert isinstance(alert.source_alert_id, uuid.UUID)
    assert alert.event_id == event_id


def test_source_alert_context_rejects_naive_created_at():
    kwargs = _source_alert_kwargs(uuid.uuid4())
    kwargs["source_alert_created_at"] = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(ValidationError):
        SourceAlertContext(**kwargs)


def test_source_alert_context_event_id_is_not_declared_unique_by_the_model():
    """event_id is intentionally not a uniqueness constraint at the model
    level -- one event can have more than one source alert (guide section
    6). Two independently-constructed contexts for the same event_id must
    both validate and must carry distinct source_alert_id values."""
    event_id = uuid.uuid4()
    first = SourceAlertContext(**_source_alert_kwargs(event_id))
    second = SourceAlertContext(**_source_alert_kwargs(event_id))
    assert first.event_id == second.event_id == event_id
    assert first.source_alert_id != second.source_alert_id


def test_source_alert_context_ids_default_to_distinct_uuids():
    ids = {SourceAlertContext(**_source_alert_kwargs(uuid.uuid4())).source_alert_id for _ in range(50)}
    assert len(ids) == 50


# ---- SyntheticGroundTruthLabel --------------------------------------------------


def test_synthetic_ground_truth_label_is_a_separate_object_from_fraud_event():
    event = FraudEvent(**_base_event_kwargs())
    label = SyntheticGroundTruthLabel(
        event_id=event.event_id,
        scenario_id="v1.3-ach-fraud-000001",
        synthetic_scenario_label=True,
        scenario_type="ACH_ACCOUNT_TAKEOVER_BATCH",
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        generated_at=UTC_NOW,
    )
    assert type(label) is not type(event)
    assert not hasattr(event, "synthetic_scenario_label")
    # Independently serializable -- proves it is genuinely a separate object,
    # not a view into FraudEvent.
    dumped_event = event.model_dump(mode="json")
    dumped_label = label.model_dump(mode="json")
    assert "synthetic_scenario_label" not in dumped_event
    assert "synthetic_scenario_label" in dumped_label
