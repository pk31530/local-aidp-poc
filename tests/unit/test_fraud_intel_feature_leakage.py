"""Phase 2 leakage-prevention tests (guide section 6/9): proves,
structurally, that no label/scenario/disposition/outcome/training-
eligibility field or object can ever reach a feature function's input or
output. Re-asserted here against the real feature-computation code (the
guide's section 6 leakage tests covered only the event/label contracts
themselves, in Phase 1).
"""
from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.features import core as core_module
from src.fraud_intel.features.channels import ach as ach_module
from src.fraud_intel.features.channels import atm as atm_module
from src.fraud_intel.features.channels import debit_card as debit_card_module
from src.fraud_intel.features.channels import mobile_deposit as mobile_deposit_module
from src.fraud_intel.features.channels import online_banking as online_banking_module
from src.fraud_intel.features.channels import p2p as p2p_module
from src.fraud_intel.features.channels import wire as wire_module
from src.fraud_intel.features.channels.online_banking import compute_online_banking_features
from src.fraud_intel.features.core import FeatureComputationContext, compute_shared_features
from src.fraud_intel.registry import CHANNEL_ADAPTERS

FORBIDDEN_FIELD_NAMES = {
    "synthetic_scenario_label",
    "scenario_id",
    "analyst_disposition",
    "outcome_status",
    "training_eligible",
}

FORBIDDEN_TYPE_NAMES = {"SyntheticGroundTruthLabel"}

PUBLIC_FEATURE_FUNCTIONS = [compute_shared_features] + [
    adapter.compute_features for adapter in CHANNEL_ADAPTERS.values()
]

_CHANNEL_MODULES = {
    "ach": ach_module, "wire": wire_module, "mobile_deposit": mobile_deposit_module,
    "online_banking": online_banking_module, "atm": atm_module, "debit_card": debit_card_module, "p2p": p2p_module,
}

_PAYLOAD_BY_CHANNEL = {
    "ach": ACHPayload(sec_code="PPD", originating_routing_number="1", receiving_routing_number="2", batch_id="B1", effective_entry_date=date(2026, 1, 1), company_id="C1"),
    "wire": WirePayload(wire_type="domestic", beneficiary_bank_id="B1", beneficiary_account="A1", purpose_code="GDS"),
    "mobile_deposit": MobileDepositPayload(duplicate_image_hash_flag=False, car_lar_mismatch_flag=False, signature_verification_flag=True, endorsement_present_flag=True, micr_consistency_flag=True, image_quality_score=0.9),
    "online_banking": OnlineBankingPayload(session_id="SESS1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"),
    "atm": ATMPayload(atm_id="ATM1", atm_geo_bucket="domestic", transaction_type="withdrawal", card_present_flag=True),
    "debit_card": DebitCardPayload(merchant_id="MER1", mcc_code="5411", pos_entry_mode="chip", card_present_flag=True, cross_border_flag=False),
    "p2p": P2PPayload(recipient_handle="@r1", network="zelle", memo_present_flag=True, recipient_is_new_flag=False),
}


def _event_for(channel: str) -> FraudEvent:
    return FraudEvent(
        channel=channel, customer_id="FIC1000", account_id="FIA100000",
        event_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), amount_minor_units=10_000,
        direction="debit", channel_payload=_PAYLOAD_BY_CHANNEL[channel],
    )


def test_no_feature_function_accepts_a_synthetic_ground_truth_label_parameter():
    for fn in PUBLIC_FEATURE_FUNCTIONS:
        for name, param in inspect.signature(fn).parameters.items():
            annotation = str(param.annotation)
            assert "SyntheticGroundTruthLabel" not in annotation, f"{fn.__name__}'s {name!r} parameter leaks a label type"
            assert "label" not in name.lower(), f"{fn.__name__} has a suspicious label-shaped parameter {name!r}"
            assert "disposition" not in name.lower()
            assert "outcome" not in name.lower()


def _referenced_identifiers(module) -> set[str]:
    """AST-level identifiers actually used as code (Name/Attribute nodes)
    in `module` -- deliberately ignores docstrings, comments, and string
    literals, so documentation that merely *names* a forbidden field (to
    explain that it is absent) cannot produce a false positive."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_feature_core_module_code_never_uses_forbidden_names_as_identifiers():
    """Structural check -- guarantees the leakage boundary is enforced by
    the actual code, not just "no test happens to exercise it". Only
    real Name/Attribute usages count; docstrings are exempt so the module
    can still document what it does NOT touch."""
    forbidden = FORBIDDEN_TYPE_NAMES | {"analyst_disposition", "outcome_status", "training_eligible"}
    core_identifiers = _referenced_identifiers(core_module)
    assert forbidden.isdisjoint(core_identifiers), f"core.py uses forbidden identifier(s): {forbidden & core_identifiers}"
    for channel, module in _CHANNEL_MODULES.items():
        identifiers = _referenced_identifiers(module)
        assert forbidden.isdisjoint(identifiers), f"{channel}.py uses forbidden identifier(s): {forbidden & identifiers}"


def test_feature_computation_context_fields_exclude_label_objects():
    field_types = {name: str(field.annotation) for name, field in FeatureComputationContext.model_fields.items()}
    for type_str in field_types.values():
        assert "SyntheticGroundTruthLabel" not in type_str


def test_shared_feature_output_never_contains_a_forbidden_key():
    event = FraudEvent(
        channel="online_banking",
        customer_id="FIC1000",
        account_id="FIA100000",
        event_timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        amount_minor_units=10_000,
        direction="debit",
        channel_payload=OnlineBankingPayload(
            session_id="SESS1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )
    ctx = FeatureComputationContext(
        current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp
    )
    f = compute_shared_features(ctx)
    assert set(f.keys()).isdisjoint(FORBIDDEN_FIELD_NAMES)

    ob_features = compute_online_banking_features(ctx)
    assert set(ob_features.keys()).isdisjoint(FORBIDDEN_FIELD_NAMES)


@pytest.mark.parametrize("channel", sorted(CHANNEL_ADAPTERS))
def test_every_channel_adapter_output_never_contains_a_forbidden_key(channel):
    adapter = CHANNEL_ADAPTERS[channel]
    event = _event_for(channel)
    ctx = FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)
    features = adapter.compute_features(ctx)
    assert set(features.keys()).isdisjoint(FORBIDDEN_FIELD_NAMES)
    assert set(adapter.feature_columns).isdisjoint(FORBIDDEN_FIELD_NAMES)


def test_ordered_feature_vector_output_columns_exclude_forbidden_names():
    from src.fraud_intel.features.channels.online_banking import ONLINE_BANKING_FEATURE_COLUMNS
    from src.fraud_intel.features.core import SHARED_FEATURE_COLUMNS

    assert set(SHARED_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FIELD_NAMES)
    assert set(ONLINE_BANKING_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FIELD_NAMES)
