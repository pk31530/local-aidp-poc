"""Phase 3: RuleProvider / RuleEvaluationResult, the fixed operator
allowlist, versioned YAML validation for all seven channels, deterministic
ordering, the corrected no-score-manipulation failure path, and the
no-persistence / no-duplicate-source-alert design guarantee. No database,
Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import ast
import inspect
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml
from pydantic import ValidationError

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.rules import provider as provider_module
from src.fraud_intel.rules.provider import (
    RULE_CONFIG_LOAD_FAILED,
    RULE_EVALUATION_FAILED,
    RULE_PROVIDER_UNAVAILABLE_REASON_CODE,
    RULES_CONFIG_DIR,
    LocalYamlRuleProvider,
    Rule,
    RuleCondition,
    RuleSetConfig,
    _load_rule_set_config,
    _rule_matches,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"

CHANNELS = ["ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"]


def _source_alert(event_id: uuid.UUID) -> SourceAlertContext:
    return SourceAlertContext(
        source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id,
        source_alert_created_at=T0 - timedelta(seconds=1),
        source_rule_ids=["SIMULATED_RULE"],
        source_rule_version="v1",
        source_alert_reason_codes=["SIMULATED_REASON"],
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        created_at=T0 - timedelta(seconds=1),
    )


def _ach_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="ach",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=600_000,
        direction="debit",
        device_id="DEV-NEW",
        channel_payload=ACHPayload(
            sec_code="PPD",
            originating_routing_number="123456789",
            receiving_routing_number="987654321",
            batch_id="BATCH1",
            effective_entry_date=date(2026, 1, 1),
            company_id="COMP1",
        ),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _wire_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="wire",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=2_000_000,
        direction="debit",
        device_id="DEV-NEW",
        channel_payload=WirePayload(
            wire_type="international", beneficiary_bank_id="BB1", beneficiary_account="BA1", purpose_code="GDS"
        ),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _mobile_deposit_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="mobile_deposit",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=10_000,
        direction="credit",
        channel_payload=MobileDepositPayload(
            duplicate_image_hash_flag=True,
            car_lar_mismatch_flag=True,
            signature_verification_flag=False,
            endorsement_present_flag=False,
            micr_consistency_flag=False,
            image_quality_score=0.4,
        ),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _online_banking_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="online_banking",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=600_000,
        direction="debit",
        device_id="DEV-NEW",
        channel_payload=OnlineBankingPayload(
            session_id="SESS1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _atm_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="atm",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=40_000,
        direction="debit",
        channel_payload=ATMPayload(atm_id="ATM1", atm_geo_bucket="international", transaction_type="withdrawal", card_present_flag=True),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _debit_card_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="debit_card",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=5_000,
        direction="debit",
        device_id="DEV-NEW",
        channel_payload=DebitCardPayload(
            merchant_id="MER1", mcc_code="5411", pos_entry_mode="cnp", card_present_flag=False, cross_border_flag=True
        ),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


def _p2p_event(**overrides) -> FraudEvent:
    kwargs = dict(
        channel="p2p",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=300_000,
        direction="debit",
        channel_payload=P2PPayload(recipient_handle="@r1", network="Zelle-like", memo_present_flag=False, recipient_is_new_flag=True),
    )
    kwargs.update(overrides)
    return FraudEvent(**kwargs)


CHANNEL_EVENT_BUILDERS = {
    "ach": _ach_event,
    "wire": _wire_event,
    "mobile_deposit": _mobile_deposit_event,
    "online_banking": _online_banking_event,
    "atm": _atm_event,
    "debit_card": _debit_card_event,
    "p2p": _p2p_event,
}


# ---- per-channel mandatory-review fixtures (all conditions satisfied) ----------


MANDATORY_REVIEW_FEATURES_BY_CHANNEL = {
    "ach": {"is_new_device": True, "events_last_1h": 3},
    "wire": {"is_new_device": True},
    "mobile_deposit": {},
    "online_banking": {"profile_change_then_transfer_flag": True, "mfa_bypass_flag": True},
    "atm": {"events_last_1h": 3},
    "debit_card": {"events_last_10m": 3},
    "p2p": {"events_last_1h": 3},
}

EXPECTED_MANDATORY_RULE_ID = {
    "ach": "ACH_NEW_DEVICE_HIGH_AMOUNT_RAPID_VELOCITY",
    "wire": "WIRE_INTERNATIONAL_NEW_DEVICE_HIGH_AMOUNT",
    "mobile_deposit": "MOBILE_DEPOSIT_DUPLICATE_IMAGE_AND_CAR_LAR_MISMATCH",
    "online_banking": "ONLINE_BANKING_PROFILE_CHANGE_TRANSFER_MFA_BYPASS",
    "atm": "ATM_INTERNATIONAL_GEO_RAPID_WITHDRAWALS",
    "debit_card": "DEBIT_CARD_NOT_PRESENT_HIGH_VELOCITY",
    "p2p": "P2P_NEW_RECIPIENT_HIGH_AMOUNT_RAPID_VELOCITY",
}


@pytest.mark.parametrize("channel", CHANNELS)
def test_mandatory_review_rule_fires_for_each_channel(channel):
    event = CHANNEL_EVENT_BUILDERS[channel]()
    features = MANDATORY_REVIEW_FEATURES_BY_CHANNEL[channel]
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features=features)
    assert result.provider_status == "OK"
    expected_id = EXPECTED_MANDATORY_RULE_ID[channel]
    assert expected_id in result.fired_rule_ids
    assert result.rule_categories[expected_id] == "MANDATORY_REVIEW"


@pytest.mark.parametrize("channel", CHANNELS)
def test_no_rule_fires_on_a_quiet_fixture(channel):
    """A minimal/negative fixture -- no flags set, no elevated amounts --
    must fire zero rules for every channel."""
    quiet_overrides = {
        "ach": dict(amount_minor_units=1_000, device_id="DEV-KNOWN"),
        "wire": dict(amount_minor_units=1_000, device_id="DEV-KNOWN", channel_payload=WirePayload(wire_type="domestic", beneficiary_bank_id="B", beneficiary_account="A", purpose_code="GDS")),
        "mobile_deposit": dict(
            channel_payload=MobileDepositPayload(
                duplicate_image_hash_flag=False, car_lar_mismatch_flag=False, signature_verification_flag=True,
                endorsement_present_flag=True, micr_consistency_flag=True, image_quality_score=0.95,
            )
        ),
        "online_banking": dict(amount_minor_units=1_000, device_id="DEV-KNOWN", channel_payload=OnlineBankingPayload(session_id="S", login_method="password", mfa_used_flag=True, transaction_type="login", target_account="T")),
        "atm": dict(channel_payload=ATMPayload(atm_id="A1", atm_geo_bucket="urban", transaction_type="withdrawal", card_present_flag=True)),
        "debit_card": dict(device_id="DEV-KNOWN", channel_payload=DebitCardPayload(merchant_id="M", mcc_code="5411", pos_entry_mode="chip", card_present_flag=True, cross_border_flag=False)),
        "p2p": dict(amount_minor_units=1_000, channel_payload=P2PPayload(recipient_handle="@r", network="Zelle-like", memo_present_flag=True, recipient_is_new_flag=False)),
    }
    event = CHANNEL_EVENT_BUILDERS[channel](**quiet_overrides[channel])
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={"night_transaction_flag": False})
    assert result.fired_rule_ids == []
    assert result.score_contribution == 0.0
    assert result.provider_status == "OK"


def test_score_contributing_rule_adds_to_score_contribution():
    event = _ach_event(amount_minor_units=600_000, device_id="DEV-NEW")
    result = LocalYamlRuleProvider().evaluate(
        event=event, source_alert=_source_alert(event.event_id), features={"is_new_device": True, "events_last_1h": 0}
    )
    assert "ACH_NEW_DEVICE_HIGH_AMOUNT" in result.fired_rule_ids
    assert result.rule_categories["ACH_NEW_DEVICE_HIGH_AMOUNT"] == "SCORE_CONTRIBUTING"
    assert result.score_contribution == pytest.approx(0.15)


def test_informational_rule_adds_reason_code_but_no_score():
    event = _ach_event(amount_minor_units=1_000, device_id="DEV-KNOWN")
    result = LocalYamlRuleProvider().evaluate(
        event=event, source_alert=_source_alert(event.event_id), features={"is_new_device": False, "night_transaction_flag": True}
    )
    assert "ACH_NIGHT_TRANSACTION" in result.fired_rule_ids
    assert result.rule_categories["ACH_NIGHT_TRANSACTION"] == "INFORMATIONAL"
    assert result.score_contribution == 0.0


# ---- fixed operator allowlist ---------------------------------------------------


CONTEXT = {"amt": 100, "flag": True, "name": "x", "items": ["a", "b"]}


@pytest.mark.parametrize(
    "op,field,value,expected",
    [
        ("eq", "name", "x", True),
        ("eq", "name", "y", False),
        ("ne", "name", "y", True),
        ("ne", "name", "x", False),
        ("gt", "amt", 50, True),
        ("gt", "amt", 100, False),
        ("gte", "amt", 100, True),
        ("lt", "amt", 200, True),
        ("lt", "amt", 100, False),
        ("lte", "amt", 100, True),
        ("in", "name", ["x", "z"], True),
        ("in", "name", ["y", "z"], False),
        ("not_in", "name", ["y", "z"], True),
        ("not_in", "name", ["x", "z"], False),
        ("is_true", "flag", None, True),
        ("is_false", "flag", None, False),
    ],
)
def test_every_allowlisted_operator(op, field, value, expected):
    rule = Rule(id="T", category="INFORMATIONAL", when=[{"field": field, "op": op, "value": value}], reason_code="T")
    assert _rule_matches(rule, CONTEXT) is expected


def test_missing_field_resolves_to_none_and_never_crashes():
    rule = Rule(id="T", category="INFORMATIONAL", when=[{"field": "does_not_exist", "op": "gt", "value": 1}], reason_code="T")
    assert _rule_matches(rule, CONTEXT) is False


# ---- load-time validation: malformed YAML fails, not silently -----------------


def test_disallowed_operator_fails_at_load_time():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "eval", "value": 1})


def test_numeric_operator_with_non_numeric_value_fails():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "gt", "value": "not-a-number"})


def test_numeric_operator_rejects_bool_as_numeric_value():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "gt", "value": True})


def test_in_operator_requires_a_list_value():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "in", "value": "not-a-list"})


def test_is_true_operator_rejects_a_value():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "is_true", "value": True})


def test_eq_operator_requires_a_value():
    with pytest.raises(ValidationError):
        RuleCondition.model_validate({"field": "x", "op": "eq"})


def test_score_contributing_without_score_contribution_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate({"id": "R", "category": "SCORE_CONTRIBUTING", "when": [{"field": "x", "op": "is_true"}], "reason_code": "R"})


def test_informational_with_score_contribution_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {"id": "R", "category": "INFORMATIONAL", "when": [{"field": "x", "op": "is_true"}], "reason_code": "R", "score_contribution": 0.1}
        )


def test_negative_score_contribution_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {"id": "R", "category": "SCORE_CONTRIBUTING", "when": [{"field": "x", "op": "is_true"}], "reason_code": "R", "score_contribution": -0.1}
        )


def test_empty_reason_code_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate({"id": "R", "category": "INFORMATIONAL", "when": [{"field": "x", "op": "is_true"}], "reason_code": "  "})


def test_empty_when_list_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate({"id": "R", "category": "INFORMATIONAL", "when": [], "reason_code": "R"})


def test_disallowed_category_fails():
    with pytest.raises(ValidationError):
        Rule.model_validate({"id": "R", "category": "CRITICAL", "when": [{"field": "x", "op": "is_true"}], "reason_code": "R"})


def test_duplicate_rule_id_fails_at_configuration_load():
    with pytest.raises(ValidationError, match="duplicate rule id"):
        RuleSetConfig.model_validate(
            {
                "channel": "ach",
                "rule_set_version": "v1",
                "rules": [
                    {"id": "DUPLICATE", "category": "INFORMATIONAL", "when": [{"field": "x", "op": "is_true"}], "reason_code": "A"},
                    {"id": "DUPLICATE", "category": "INFORMATIONAL", "when": [{"field": "y", "op": "is_true"}], "reason_code": "B"},
                ],
            }
        )


def test_unknown_top_level_key_fails():
    with pytest.raises(ValidationError):
        RuleSetConfig.model_validate({"channel": "ach", "rule_set_version": "v1", "rules": [], "unexpected": True})


# ---- all seven real YAML files load and validate cleanly -----------------------


@pytest.mark.parametrize("channel", CHANNELS)
def test_real_channel_yaml_file_loads_and_validates(channel):
    config = _load_rule_set_config(channel)
    assert config.channel == channel
    assert config.rule_set_version == "v1"
    assert len(config.rules) >= 1
    categories = {rule.category for rule in config.rules}
    assert "MANDATORY_REVIEW" in categories


@pytest.mark.parametrize("channel", CHANNELS)
def test_real_channel_yaml_file_exists_on_disk(channel):
    path = RULES_CONFIG_DIR / f"rules_{channel}.yaml"
    assert path.is_file()
    with path.open() as f:
        raw = yaml.safe_load(f)
    assert raw["channel"] == channel


# ---- deterministic evaluation / result ordering --------------------------------


def test_fired_rule_ids_and_reason_codes_preserve_file_order_and_are_repeatable():
    event = _ach_event(amount_minor_units=600_000, device_id="DEV-NEW")
    features = {"is_new_device": True, "events_last_1h": 3, "night_transaction_flag": True}
    r1 = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features=features)
    r2 = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features=features)
    assert r1.fired_rule_ids == r2.fired_rule_ids
    assert r1.reason_codes == r2.reason_codes
    # File order: MANDATORY_REVIEW rule, then SCORE_CONTRIBUTING, then INFORMATIONAL.
    assert r1.fired_rule_ids == [
        "ACH_NEW_DEVICE_HIGH_AMOUNT_RAPID_VELOCITY",
        "ACH_NEW_DEVICE_HIGH_AMOUNT",
        "ACH_NIGHT_TRANSACTION",
    ]


# ---- failure behavior: no score manipulation, alert retained -------------------


def test_provider_failure_on_config_load_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("simulated missing rule config")

    monkeypatch.setattr(provider_module, "_load_rule_set_config", _raise)
    event = _ach_event()
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={})

    assert result.provider_status == "ERROR"
    assert result.provider_error_code == RULE_CONFIG_LOAD_FAILED
    assert result.minimum_priority_band == "MEDIUM"
    assert result.score_contribution == 0.0
    assert result.fired_rule_ids == []
    assert RULE_PROVIDER_UNAVAILABLE_REASON_CODE in result.reason_codes


def test_provider_failure_on_internal_evaluation_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(LocalYamlRuleProvider, "_build_condition_context", staticmethod(_raise))
    event = _ach_event()
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={})

    assert result.provider_status == "ERROR"
    assert result.provider_error_code == RULE_EVALUATION_FAILED
    assert result.minimum_priority_band == "MEDIUM"
    assert result.score_contribution == 0.0


def test_provider_error_code_never_leaks_the_raw_exception_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("/etc/secret/path leaked-detail postgres://user:pw@host")

    monkeypatch.setattr(provider_module, "_load_rule_set_config", _raise)
    event = _ach_event()
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={})
    assert result.provider_error_code in (RULE_CONFIG_LOAD_FAILED, RULE_EVALUATION_FAILED)
    assert "secret" not in result.provider_error_code
    assert "postgres" not in result.provider_error_code


def test_existing_source_alert_is_never_touched_by_a_failed_evaluation(monkeypatch):
    """The alert this evaluation is about existed before evaluate() ran and
    is passed in unchanged -- a failure must not mutate or replace it."""
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(provider_module, "_load_rule_set_config", _raise)
    event = _ach_event()
    alert = _source_alert(event.event_id)
    alert_before = alert.model_copy(deep=True)
    LocalYamlRuleProvider().evaluate(event=event, source_alert=alert, features={})
    assert alert == alert_before


# ---- no persistence / no duplicate source-alert generation ---------------------


def test_evaluate_return_type_never_includes_a_source_alert_or_id():
    event = _ach_event()
    result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={})
    dumped = result.model_dump()
    assert "source_alert_id" not in dumped
    assert "source_alert" not in dumped


def test_provider_module_never_imports_a_database_driver():
    source = inspect.getsource(provider_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"psycopg2", "src.common.db"}
    assert forbidden.isdisjoint(imported_names), f"provider.py imports a database driver: {forbidden & imported_names}"


def test_local_yaml_rule_provider_has_no_insert_or_write_method():
    members = {name for name, _ in inspect.getmembers(LocalYamlRuleProvider) if not name.startswith("__")}
    forbidden_substrings = ("insert", "write", "persist", "save", "create_source_alert", "create_alert")
    for name in members:
        for forbidden in forbidden_substrings:
            assert forbidden not in name.lower(), f"LocalYamlRuleProvider has a suspicious persistence-shaped member {name!r}"


# ---- label-leakage structural boundary ------------------------------------------


def test_evaluate_signature_has_no_label_shaped_parameter():
    sig = inspect.signature(LocalYamlRuleProvider.evaluate)
    for name, param in sig.parameters.items():
        annotation = str(param.annotation)
        assert "SyntheticGroundTruthLabel" not in annotation
        assert "label" not in name.lower()
        assert "disposition" not in name.lower()
        assert "outcome" not in name.lower()


def test_provider_module_code_never_uses_forbidden_names_as_identifiers():
    tree = ast.parse(inspect.getsource(provider_module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    forbidden = {"SyntheticGroundTruthLabel", "analyst_disposition", "outcome_status", "training_eligible"}
    assert forbidden.isdisjoint(names), f"provider.py uses forbidden identifier(s): {forbidden & names}"


# ---- RuleEvaluationResult contract shape ----------------------------------------


def test_rule_evaluation_result_provider_status_and_band_are_allowlisted():
    event = _ach_event()
    ok_result = LocalYamlRuleProvider().evaluate(event=event, source_alert=_source_alert(event.event_id), features={})
    assert ok_result.provider_status in ("OK", "UNAVAILABLE", "ERROR")
    assert ok_result.minimum_priority_band in (None, "LOW", "MEDIUM", "HIGH")
