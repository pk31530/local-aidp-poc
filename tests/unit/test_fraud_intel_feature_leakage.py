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
from datetime import datetime, timezone

from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.features import core as core_module
from src.fraud_intel.features.channels import online_banking as online_banking_module
from src.fraud_intel.features.channels.online_banking import compute_online_banking_features
from src.fraud_intel.features.core import FeatureComputationContext, compute_shared_features
from src.fraud_intel.events.base import FraudEvent

FORBIDDEN_FIELD_NAMES = {
    "synthetic_scenario_label",
    "scenario_id",
    "analyst_disposition",
    "outcome_status",
    "training_eligible",
}

FORBIDDEN_TYPE_NAMES = {"SyntheticGroundTruthLabel"}

PUBLIC_FEATURE_FUNCTIONS = [compute_shared_features, compute_online_banking_features]


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
    online_banking_identifiers = _referenced_identifiers(online_banking_module)
    assert forbidden.isdisjoint(core_identifiers), f"core.py uses forbidden identifier(s): {forbidden & core_identifiers}"
    assert forbidden.isdisjoint(online_banking_identifiers), (
        f"online_banking.py uses forbidden identifier(s): {forbidden & online_banking_identifiers}"
    )


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


def test_ordered_feature_vector_output_columns_exclude_forbidden_names():
    from src.fraud_intel.features.channels.online_banking import ONLINE_BANKING_FEATURE_COLUMNS
    from src.fraud_intel.features.core import SHARED_FEATURE_COLUMNS

    assert set(SHARED_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FIELD_NAMES)
    assert set(ONLINE_BANKING_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_FIELD_NAMES)
