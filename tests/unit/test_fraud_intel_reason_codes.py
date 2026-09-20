"""Phase 5: explainable reason-code contract (guide section 17). No
database, Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import ast
import inspect
import uuid
from datetime import datetime, timedelta, timezone

from src.fraud_intel.reason_codes import builder as reason_codes_module
from src.fraud_intel.reason_codes.builder import (
    MAX_GBM_CONTRIBUTION_CODES,
    MAX_TEXT_LENGTH,
    REASON_CODE_VERSION,
    build_reason_codes,
)
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.rules.provider import RuleEvaluationResult

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

OK_STATUSES = {
    "gbm": {"status": "OK", "error_code": None},
    "anomaly": {"status": "OK", "error_code": None},
    "graph": {"status": "OK", "error_code": None},
}


def _rule_result(**overrides) -> RuleEvaluationResult:
    base = dict(
        provider_name="LocalYamlRuleProvider", provider_version="v1", rule_set_version="v1",
        fired_rule_ids=["RULE_A", "RULE_B"],
        rule_categories={"RULE_A": "MANDATORY_REVIEW", "RULE_B": "SCORE_CONTRIBUTING"},
        reason_codes=["REASON_A", "REASON_B"], score_contribution=0.15,
        minimum_priority_band=None, evaluated_at=T0, latency_ms=1.0,
        provider_status="OK", provider_error_code=None,
    )
    base.update(overrides)
    return RuleEvaluationResult(**base)


def _source_alert(**overrides) -> SourceAlertContext:
    base = dict(
        source_system="LocalYamlRuleProvider (simulated upstream)", event_id=uuid.uuid4(),
        source_alert_created_at=T0 - timedelta(hours=1),
        source_rule_ids=["UPSTREAM_RULE"], source_rule_version="v1",
        source_alert_reason_codes=["VELOCITY_FLAG"],
        generation_run_id="genrun-1", dataset_version="dsv-1", created_at=T0 - timedelta(hours=1),
    )
    base.update(overrides)
    return SourceAlertContext(**base)


def _build(**overrides):
    kwargs = dict(
        rule_result=_rule_result(), source_alert=_source_alert(),
        feature_contributions=[("amount_zscore", 0.3), ("is_new_device", -0.1)],
        anomaly_score=0.9, graph_shared_device_count=3, graph_fan_in_count=4, graph_shortest_path=1,
        component_statuses=OK_STATUSES,
    )
    kwargs.update(overrides)
    return build_reason_codes(**kwargs)


def test_reason_code_version_defined():
    assert REASON_CODE_VERSION == "v1"


def test_merges_all_four_layers():
    codes = _build()
    layers = {c.layer for c in codes}
    assert layers == {"rule", "gbm", "anomaly", "graph"}


def test_upstream_and_triage_rule_codes_are_kept_distinguishable():
    codes = _build()
    codes_by_code = {c.code: c for c in codes}
    assert "UPSTREAM_VELOCITY_FLAG" in codes_by_code  # the upstream (simulated) alert's own reason code
    assert "REASON_A" in codes_by_code  # the triage layer's own evaluation, kept separate
    assert "REASON_B" in codes_by_code


def test_mandatory_rule_severity_preserved():
    codes = _build()
    by_code = {c.code: c for c in codes}
    assert by_code["REASON_A"].severity == "mandatory"
    assert by_code["REASON_B"].severity == "contributing"


def test_deterministic_layer_ordering_rule_gbm_anomaly_graph():
    codes = _build()
    layer_order = [c.layer for c in codes]
    seen_order = []
    for layer in layer_order:
        if not seen_order or seen_order[-1] != layer:
            seen_order.append(layer)
    # every layer present is contiguous and in rule -> gbm -> anomaly -> graph order
    assert seen_order == ["rule", "gbm", "anomaly", "graph"]


def test_repeated_calls_produce_identical_ordering():
    codes_a = _build()
    codes_b = _build()
    assert [c.code for c in codes_a] == [c.code for c in codes_b]


def test_gbm_contribution_codes_capped_at_max_and_ranked_by_magnitude():
    many_contributions = [(f"feature_{i}", float(i) / 10.0) for i in range(10)]
    codes = _build(feature_contributions=many_contributions)
    gbm_codes = [c for c in codes if c.layer == "gbm"]
    assert len(gbm_codes) == MAX_GBM_CONTRIBUTION_CODES
    # highest |contribution| first
    assert gbm_codes[0].code == "GBM_FEATURE_FEATURE_9"


def test_anomaly_flag_only_above_threshold():
    below = _build(anomaly_score=0.1)
    above = _build(anomaly_score=0.95)
    assert not any(c.code == "ANOMALY_SCORE_ELEVATED" for c in below)
    assert any(c.code == "ANOMALY_SCORE_ELEVATED" for c in above)


def test_graph_flags_use_configured_thresholds():
    codes = _build(graph_shared_device_count=5, graph_fan_in_count=5, graph_shortest_path=1)
    codes_present = {c.code for c in codes}
    assert "SHARED_DEVICE_RING" in codes_present
    assert "RAPID_RECIPIENT_FAN_IN" in codes_present
    assert "NEAR_FRAUD_LINKED_ENTITY" in codes_present


def test_graph_flags_absent_when_below_threshold():
    codes = _build(graph_shared_device_count=0, graph_fan_in_count=0, graph_shortest_path=None)
    codes_present = {c.code for c in codes}
    assert "SHARED_DEVICE_RING" not in codes_present
    assert "RAPID_RECIPIENT_FAN_IN" not in codes_present
    assert "NEAR_FRAUD_LINKED_ENTITY" not in codes_present


def test_deduplication_first_occurrence_wins():
    rule_result = _rule_result(
        fired_rule_ids=["RULE_A", "RULE_C"],
        rule_categories={"RULE_A": "MANDATORY_REVIEW", "RULE_C": "INFORMATIONAL"},
        reason_codes=["DUPLICATE_CODE", "DUPLICATE_CODE"],
    )
    codes = _build(rule_result=rule_result)
    matching = [c for c in codes if c.code == "DUPLICATE_CODE"]
    assert len(matching) == 1
    assert matching[0].severity == "mandatory"  # first occurrence (RULE_A) wins


def test_degraded_component_emits_unavailable_code_instead_of_normal_codes():
    statuses = {**OK_STATUSES, "gbm": {"status": "ERROR", "error_code": "GBM_SCORING_FAILED"}}
    codes = _build(component_statuses=statuses)
    codes_present = {c.code for c in codes}
    assert "GBM_UNAVAILABLE" in codes_present
    assert not any(c.code.startswith("GBM_FEATURE_") for c in codes)


def test_text_length_capped():
    codes = _build()
    for c in codes:
        assert len(c.text) <= MAX_TEXT_LENGTH


def test_reason_code_output_never_contains_scenario_id_or_synthetic_label():
    codes = _build()
    for c in codes:
        assert "scenario_id" not in c.text.lower()
        assert "synthetic_scenario_label" not in c.text.lower()
        assert "scenario_id" not in c.code.lower()


def test_builder_module_code_never_uses_forbidden_identifiers():
    tree = ast.parse(inspect.getsource(reason_codes_module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    forbidden = {"scenario_id", "synthetic_scenario_label", "SyntheticGroundTruthLabel", "analyst_disposition", "outcome_status"}
    assert forbidden.isdisjoint(names)


def test_reason_code_is_frozen():
    import pytest
    from pydantic import ValidationError

    codes = _build()
    with pytest.raises(ValidationError):
        codes[0].code = "CHANGED"
