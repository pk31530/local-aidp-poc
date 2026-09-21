"""Phase 7A corrective pass: audits the full runtime dispatch path for
all seven registered channels -- registry lookup, feature adapter,
entity extractor, rule/graph/ensemble policy loaders, MLflow naming
contract, and reason-code generation. No database, MLflow, Docker, or
network access anywhere in this file.
"""
from __future__ import annotations

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
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import EntityType, load_graph_policy
from src.fraud_intel.ensemble.policy import load_ensemble_policy
from src.fraud_intel.models.mlflow_naming import MODEL_COMPONENTS, UnknownModelComponentError, registered_model_name
from src.fraud_intel.models.training import _experiment_name
from src.fraud_intel.reason_codes.builder import REASON_CODE_VERSION, build_reason_codes
from src.fraud_intel.registry import CHANNEL_ADAPTERS, get_channel_adapter
from src.fraud_intel.rules.provider import LocalYamlRuleProvider, _load_rule_set_config

ALL_CHANNELS = sorted({"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"})

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _payload_for(channel: str):
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


# ---- registry lookup -----------------------------------------------------------------


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_registry_resolves_every_channel(channel):
    adapter = get_channel_adapter(channel)
    assert adapter.channel == channel


# ---- feature adapter + entity extractor -----------------------------------------------


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_feature_adapter_and_entity_extractor_resolve_for_every_channel(channel):
    adapter = get_channel_adapter(channel)
    event = _event_for(channel)
    ctx = FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)

    features = adapter.compute_features(ctx)
    assert set(features.keys()) == set(adapter.feature_columns)

    entities = adapter.extract_entities(event)
    import typing

    approved = set(typing.get_args(EntityType))
    for entity_type, _entity_id in entities:
        assert entity_type in approved


# ---- rule / graph / ensemble policy loaders --------------------------------------------


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_rule_set_loads_and_matches_channel(channel):
    config = _load_rule_set_config(channel)
    assert config.channel == channel
    assert config.rule_set_version


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_graph_policy_loads_and_matches_channel(channel):
    policy = load_graph_policy(channel)
    assert policy.channel == channel
    assert policy.graph_policy_version


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_ensemble_policy_loads_and_matches_channel(channel):
    policy = load_ensemble_policy(channel)
    assert policy.channel == channel
    assert policy.policy_version


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_rule_provider_evaluates_for_every_channel(channel):
    """LocalYamlRuleProvider resolves its own config from event.channel --
    no per-channel provider class, one shared implementation."""
    event = _event_for(channel)
    ctx = FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)
    adapter = get_channel_adapter(channel)
    features = adapter.compute_features(ctx)

    from src.fraud_intel.events.source_alert_context import SourceAlertContext
    import uuid

    source_alert = SourceAlertContext(
        source_alert_id=uuid.uuid4(), source_system="test", event_id=event.event_id, source_alert_created_at=T0,
        source_rule_ids=["R1"], source_rule_version="v1", source_alert_reason_codes=["REASON1"],
        generation_run_id="g", dataset_version="d", created_at=T0,
    )
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=source_alert, features=features)
    assert result.provider_status == "OK"
    assert result.rule_set_version


# ---- MLflow model/artifact naming contract ---------------------------------------------


def test_canonical_model_names_match_expected_pattern_for_all_seven_channels():
    """Phase 7B Stage 5 corrective pass: training, scoring's artifact
    loader, and promotion's verifier all call the SAME
    registered_model_name() -- there is no longer a separate
    training-side/scoring-side implementation to drift apart."""
    for channel in ALL_CHANNELS:
        for component in MODEL_COMPONENTS:
            name = registered_model_name(channel, component)
            assert name == f"fraud-detection-model-fraud-intel-{channel.replace('_', '-')}-{component}"


def test_lr_component_is_named_lr_shadow_never_lr():
    assert registered_model_name("online_banking", "lr-shadow") == "fraud-detection-model-fraud-intel-online-banking-lr-shadow"
    with pytest.raises(UnknownModelComponentError):
        registered_model_name("online_banking", "lr")


def test_mlflow_naming_is_distinct_per_channel():
    names = {registered_model_name(channel, "gbm") for channel in ALL_CHANNELS}
    assert len(names) == len(ALL_CHANNELS)
    experiment_names = {_experiment_name(channel) for channel in ALL_CHANNELS}
    assert len(experiment_names) == len(ALL_CHANNELS)


# ---- bundle lookup: real SQL is parameterized by channel, never hardcoded -------------


def test_operational_bundle_lookup_sql_is_channel_parameterized():
    import ast
    import inspect

    from src.cli import __main__ as cli_main

    source = inspect.getsource(cli_main._get_operational_bundle)
    assert "%s" in source  # parameterized query, not a hardcoded channel literal
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in ALL_CHANNELS:
            pytest.fail(f"_get_operational_bundle contains a hardcoded channel literal: {node.value!r}")


# ---- reason-code generation: already channel-agnostic, verified for all 7 ------------


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_reason_code_generation_resolves_for_every_channel(channel):
    from src.fraud_intel.rules.provider import RuleEvaluationResult
    from src.fraud_intel.events.source_alert_context import SourceAlertContext
    import uuid

    event = _event_for(channel)
    source_alert = SourceAlertContext(
        source_alert_id=uuid.uuid4(), source_system="test", event_id=event.event_id, source_alert_created_at=T0,
        source_rule_ids=["R1"], source_rule_version="v1", source_alert_reason_codes=["REASON1"],
        generation_run_id="g", dataset_version="d", created_at=T0,
    )
    rule_result = RuleEvaluationResult(
        provider_name="LocalYamlRuleProvider", provider_version="v1", rule_set_version="v1", fired_rule_ids=[],
        rule_categories={}, reason_codes=[], score_contribution=0.0, minimum_priority_band=None,
        evaluated_at=T0, latency_ms=0.0, provider_status="OK", provider_error_code=None,
    )
    reason_codes = build_reason_codes(
        rule_result=rule_result, source_alert=source_alert, feature_contributions=[], anomaly_score=0.0,
        graph_shared_device_count=0, graph_fan_in_count=0, graph_shortest_path=None,
        component_statuses={"rules": {"status": "OK", "error_code": None}},
    )
    assert isinstance(reason_codes, list)
    assert reason_codes  # at least the upstream reason code
    for rc in reason_codes:
        assert rc.code  # every reason code has a real code regardless of channel


def test_every_registered_channel_shares_the_same_reason_code_version():
    """No per-channel reason-code version fork -- REASON_CODE_VERSION is a
    single shared constant."""
    assert REASON_CODE_VERSION
    assert len(CHANNEL_ADAPTERS) == 7
