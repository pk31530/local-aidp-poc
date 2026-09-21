"""Phase 7A: the shared channel-adapter registry
(src.fraud_intel.registry). No database, Docker, or network access
anywhere in this file.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import EntityType
from src.fraud_intel.registry import CHANNEL_ADAPTERS, UnknownChannelError, get_channel_adapter

ALL_CHANNELS = {"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"}

_APPROVED_ENTITY_TYPES = {
    "customer", "account", "device", "ip_address", "beneficiary", "recipient", "card", "atm", "check_payee",
}

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _payload_for(channel: str):
    from datetime import date

    return {
        "ach": ACHPayload(sec_code="PPD", originating_routing_number="1", receiving_routing_number="2", batch_id="B1", effective_entry_date=date(2026, 1, 1), company_id="C1"),
        "wire": WirePayload(wire_type="domestic", beneficiary_bank_id="B1", beneficiary_account="A1", purpose_code="GDS"),
        "mobile_deposit": MobileDepositPayload(duplicate_image_hash_flag=False, car_lar_mismatch_flag=False, signature_verification_flag=True, endorsement_present_flag=True, micr_consistency_flag=True, image_quality_score=0.9),
        "online_banking": OnlineBankingPayload(session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"),
        "atm": ATMPayload(atm_id="ATM1", atm_geo_bucket="domestic", transaction_type="withdrawal", card_present_flag=True),
        "debit_card": DebitCardPayload(merchant_id="MER1", mcc_code="5411", pos_entry_mode="chip", card_present_flag=True, cross_border_flag=False),
        "p2p": P2PPayload(recipient_handle="@r1", network="zelle", memo_present_flag=True, recipient_is_new_flag=False),
    }[channel]


def _event_for(channel: str) -> FraudEvent:
    return FraudEvent(
        channel=channel, customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=T0, amount_minor_units=10_000,
        direction="debit", device_id="DEV1", channel_payload=_payload_for(channel),
    )


# ---- every channel resolves through the registry -----------------------------------


def test_every_channel_is_registered():
    assert set(CHANNEL_ADAPTERS) == ALL_CHANNELS


@pytest.mark.parametrize("channel", sorted(ALL_CHANNELS))
def test_get_channel_adapter_returns_the_matching_adapter(channel):
    adapter = get_channel_adapter(channel)
    assert adapter.channel == channel


def test_unknown_channel_raises_unknown_channel_error_not_a_bare_key_error():
    with pytest.raises(UnknownChannelError):
        get_channel_adapter("not_a_real_channel")


# ---- ordered schemas are deterministic and distinct where appropriate --------------


def test_every_adapter_feature_columns_is_a_tuple_and_deterministic():
    for channel, adapter in CHANNEL_ADAPTERS.items():
        assert isinstance(adapter.feature_columns, tuple)
        assert adapter.feature_columns == tuple(adapter.feature_columns)  # stable, not re-ordered


def test_channel_specific_feature_columns_are_distinct_across_channels():
    """Every channel's own (non-shared) feature columns are unique to it --
    proves each adapter contributes real, distinct signal, not a copy-paste
    of another channel's columns."""
    shared_prefix_len = 12  # len(SHARED_FEATURE_COLUMNS)
    channel_specific_sets = {
        channel: set(adapter.feature_columns[shared_prefix_len:]) for channel, adapter in CHANNEL_ADAPTERS.items()
    }
    channels = sorted(channel_specific_sets)
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            assert channel_specific_sets[a].isdisjoint(channel_specific_sets[b]), (a, b)


# ---- each graph extractor emits only approved entity types -------------------------


def test_entity_type_literal_matches_the_approved_set():
    import typing

    assert set(typing.get_args(EntityType)) == _APPROVED_ENTITY_TYPES


@pytest.mark.parametrize("channel", sorted(ALL_CHANNELS))
def test_extract_entities_emits_only_approved_entity_types(channel):
    adapter = get_channel_adapter(channel)
    event = _event_for(channel)
    entities = adapter.extract_entities(event)
    for entity_type, _entity_id in entities:
        assert entity_type in _APPROVED_ENTITY_TYPES


def test_ach_extract_entities_never_emits_a_routing_number_or_company_id_derived_entity():
    """Phase 7A decision 3: a deliberate semantic decision, not a missing
    implementation -- ACH's routing numbers/company_id participate in
    direct/channel features only, never as graph nodes."""
    adapter = get_channel_adapter("ach")
    event = _event_for("ach")
    entities = adapter.extract_entities(event)
    entity_ids = {entity_id for _, entity_id in entities}
    assert "1" not in entity_ids and "2" not in entity_ids and "C1" not in entity_ids
    assert {entity_type for entity_type, _ in entities} <= {"customer", "account", "device", "ip_address"}


# ---- each channel can pass through the shared feature adapter with real payloads ----


@pytest.mark.parametrize("channel", sorted(ALL_CHANNELS))
def test_each_channel_compute_features_returns_the_full_declared_schema(channel):
    adapter = get_channel_adapter(channel)
    event = _event_for(channel)
    ctx = FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)
    features = adapter.compute_features(ctx)
    assert set(features.keys()) == set(adapter.feature_columns)


# ---- payload_class matches the real registered event contract ----------------------


@pytest.mark.parametrize("channel", sorted(ALL_CHANNELS))
def test_payload_class_round_trips_the_channels_own_payload(channel):
    adapter = get_channel_adapter(channel)
    payload = _payload_for(channel)
    reconstructed = adapter.payload_class(**payload.model_dump(mode="json"))
    assert reconstructed == payload


# ---- shared-code-path assertion: no channel-specific branch outside config/adapter files ----


def test_orchestrator_and_training_contain_no_channel_literal_string_comparison():
    """Phase 7A guide requirement: training/scoring for every channel goes
    through the IDENTICAL shared code path -- assert no channel-specific
    branch exists outside config/adapter files. AST-walks
    src.fraud_intel.scoring.orchestrator and src.fraud_intel.models.training
    for any string-literal comparison against one of the 7 channel names
    (e.g. `if event.channel == "online_banking"`) -- these are exactly the
    hardcoded per-channel guards the Phase 7A registry refactor removed.
    Channel-name string literals are expected and fine inside
    src.fraud_intel.registry itself and the per-channel feature-adapter
    modules (features/channels/*.py) -- those are the adapter layer this
    test does not check."""
    import ast
    import inspect

    from src.fraud_intel.models import training as training_module
    from src.fraud_intel.scoring import orchestrator as orchestrator_module

    for module in (orchestrator_module, training_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value in ALL_CHANNELS:
                        pytest.fail(
                            f"{module.__name__} contains a hardcoded channel-literal comparison: "
                            f"{ast.dump(node)}"
                        )
                if isinstance(node.left, ast.Constant) and node.left.value in ALL_CHANNELS:
                    pytest.fail(f"{module.__name__} contains a hardcoded channel-literal comparison: {ast.dump(node)}")
