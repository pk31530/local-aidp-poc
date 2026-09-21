"""RuleProvider interface and its local, YAML-driven implementation (guide
section 10). No database, Docker, or network access anywhere in this
module -- `LocalYamlRuleProvider.evaluate()` is a pure, in-memory
function.

Source-alert responsibility (explicit design decision, approved before
implementation): `RuleProvider` is a pure enrichment/evaluation
component. It receives an already-existing `SourceAlertContext`, the
current `FraudEvent`, and the validated Phase 2 feature vector, and
returns only a `RuleEvaluationResult`. It must NEVER create another
`SourceAlertContext`, generate a new `source_alert_id`, write to
`source_alerts`/`fraud_alerts`, or open a database connection -- there is
no persistence code anywhere in this file. For this local POC, Phase 1's
synthetic generators simulate upstream source-alert generation; in a real
bank environment, an adapter would supply `SourceAlertContext` directly.
The later scoring orchestrator (Phase 5) is responsible for linking this
evaluation's output to the existing `source_alert_id` -- this module never
performs that linkage itself.

Rules are evaluated only through a fixed allowlist of operators via a
Python dispatch table -- never `eval()`, `exec()`, dynamic imports, or a
general expression parser.
"""
from __future__ import annotations

import functools
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from src.common.config import PROJECT_ROOT
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext

RULES_CONFIG_DIR = PROJECT_ROOT / "config" / "fraud_intel"

RuleCategory = Literal["MANDATORY_REVIEW", "SCORE_CONTRIBUTING", "INFORMATIONAL"]
RuleOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "is_true", "is_false"]
ProviderStatus = Literal["OK", "UNAVAILABLE", "ERROR"]
PriorityBand = Literal["LOW", "MEDIUM", "HIGH"]

_NUMERIC_OPERATORS = {"gt", "gte", "lt", "lte"}
_LIST_OPERATORS = {"in", "not_in"}
_NO_VALUE_OPERATORS = {"is_true", "is_false"}

PROVIDER_NAME = "LocalYamlRuleProvider"
PROVIDER_VERSION = "v1"

RULE_PROVIDER_UNAVAILABLE_REASON_CODE = "RULE_PROVIDER_UNAVAILABLE"
FAILURE_MINIMUM_PRIORITY_BAND: PriorityBand = "MEDIUM"
UNKNOWN_RULE_SET_VERSION = "unknown"

# Stable, non-sensitive provider_error_code values -- never a raw exception
# message (which could leak file paths or internal detail).
RULE_CONFIG_LOAD_FAILED = "RULE_CONFIG_LOAD_FAILED"
RULE_EVALUATION_FAILED = "RULE_EVALUATION_FAILED"


class RuleEvaluationResult(BaseModel):
    """Exactly guide section 10's contract."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    provider_version: str
    rule_set_version: str
    fired_rule_ids: list[str]
    rule_categories: dict[str, RuleCategory]
    reason_codes: list[str]
    score_contribution: float
    minimum_priority_band: Optional[PriorityBand] = None
    evaluated_at: datetime
    latency_ms: float
    provider_status: ProviderStatus
    provider_error_code: Optional[str] = None


class RuleProvider(Protocol):
    def evaluate(
        self,
        *,
        event: FraudEvent,
        source_alert: SourceAlertContext,
        features: Mapping[str, Any],
    ) -> RuleEvaluationResult: ...


# ---- versioned YAML rule-set schema (validated at load time) ------------------


class RuleCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: RuleOperator
    value: Optional[Any] = None

    @model_validator(mode="after")
    def _validate_operand_type(self) -> "RuleCondition":
        if self.op in _NO_VALUE_OPERATORS:
            if self.value is not None:
                raise ValueError(f"condition on {self.field!r}: op {self.op!r} must not include a value")
        elif self.op in _NUMERIC_OPERATORS:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"condition on {self.field!r}: op {self.op!r} requires a numeric value")
        elif self.op in _LIST_OPERATORS:
            if not isinstance(self.value, list):
                raise ValueError(f"condition on {self.field!r}: op {self.op!r} requires a list value")
        else:  # eq, ne
            if self.value is None:
                raise ValueError(f"condition on {self.field!r}: op {self.op!r} requires a value")
        return self


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: RuleCategory
    when: list[RuleCondition]
    reason_code: str
    score_contribution: Optional[float] = None

    @field_validator("when")
    @classmethod
    def _when_non_empty(cls, value: list[RuleCondition]) -> list[RuleCondition]:
        if not value:
            raise ValueError("a rule's when list must not be empty")
        return value

    @field_validator("reason_code")
    @classmethod
    def _reason_code_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason_code must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_score_contribution_scope(self) -> "Rule":
        if self.category == "SCORE_CONTRIBUTING":
            if self.score_contribution is None:
                raise ValueError(f"rule {self.id!r}: SCORE_CONTRIBUTING rules require score_contribution")
            if self.score_contribution < 0:
                raise ValueError(f"rule {self.id!r}: score_contribution must be >= 0")
        elif self.score_contribution is not None:
            raise ValueError(f"rule {self.id!r}: score_contribution is only valid for SCORE_CONTRIBUTING rules")
        return self


class RuleSetConfig(BaseModel):
    """One channel's versioned rule set (config/fraud_intel/rules_<channel>.yaml)."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    rule_set_version: str
    rules: list[Rule]

    @model_validator(mode="after")
    def _validate_unique_rule_ids(self) -> "RuleSetConfig":
        ids = [rule.id for rule in self.rules]
        duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate rule id(s) in channel {self.channel!r}: {duplicates}")
        return self


@functools.lru_cache
def _load_rule_set_config(channel: str) -> RuleSetConfig:
    """yaml.safe_load only -- never yaml.load -- same convention as
    src.common.config's _load_yaml, with its own loader since these files
    live in a subdirectory that module does not support."""
    path = RULES_CONFIG_DIR / f"rules_{channel}.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RuleSetConfig.model_validate(raw)


# ---- fixed operator allowlist dispatch table -----------------------------------


def _op_eq(field_value: Any, value: Any) -> bool:
    return field_value == value


def _op_ne(field_value: Any, value: Any) -> bool:
    return field_value != value


def _op_gt(field_value: Any, value: Any) -> bool:
    return field_value is not None and field_value > value


def _op_gte(field_value: Any, value: Any) -> bool:
    return field_value is not None and field_value >= value


def _op_lt(field_value: Any, value: Any) -> bool:
    return field_value is not None and field_value < value


def _op_lte(field_value: Any, value: Any) -> bool:
    return field_value is not None and field_value <= value


def _op_in(field_value: Any, value: list) -> bool:
    return field_value in value


def _op_not_in(field_value: Any, value: list) -> bool:
    return field_value not in value


def _op_is_true(field_value: Any, value: Any) -> bool:
    return field_value is True


def _op_is_false(field_value: Any, value: Any) -> bool:
    return field_value is False


_OPERATOR_DISPATCH = {
    "eq": _op_eq,
    "ne": _op_ne,
    "gt": _op_gt,
    "gte": _op_gte,
    "lt": _op_lt,
    "lte": _op_lte,
    "in": _op_in,
    "not_in": _op_not_in,
    "is_true": _op_is_true,
    "is_false": _op_is_false,
}


def _rule_matches(rule: Rule, context: Mapping[str, Any]) -> bool:
    """AND of every condition. A field absent from `context` resolves to
    None, which every comparator treats as a safe non-match -- never a
    KeyError/crash."""
    return all(_OPERATOR_DISPATCH[condition.op](context.get(condition.field), condition.value) for condition in rule.when)


def _classify_error(exc: Exception) -> str:
    if isinstance(exc, (OSError, yaml.YAMLError, ValidationError)):
        return RULE_CONFIG_LOAD_FAILED
    return RULE_EVALUATION_FAILED


class LocalYamlRuleProvider:
    """Simulates, for this local POC, the rule-evaluation logic of a real
    bank's existing upstream alert-generation systems (guide section 4) --
    it is not new alert generation invented by v1.3. `evaluate()` never
    creates or writes a `SourceAlertContext`/`fraud_alerts` row, never
    generates a new `source_alert_id`, and never opens a database
    connection (see module docstring)."""

    provider_name = PROVIDER_NAME
    provider_version = PROVIDER_VERSION

    def evaluate(
        self,
        *,
        event: FraudEvent,
        source_alert: SourceAlertContext,
        features: Mapping[str, Any],
    ) -> RuleEvaluationResult:
        del source_alert  # accepted for interface completeness (see module
        # docstring) -- no Phase 3 channel rule currently references it;
        # nothing here writes to it or reads a database to resolve it.
        start = time.monotonic()
        try:
            config = _load_rule_set_config(event.channel)
            if config.channel != event.channel:
                raise ValueError(
                    f"rule config channel {config.channel!r} does not match event channel {event.channel!r}"
                )

            context = self._build_condition_context(event, features)

            fired_rule_ids: list[str] = []
            rule_categories: dict[str, RuleCategory] = {}
            reason_codes: list[str] = []
            score_contribution = 0.0

            # File order, preserved -- deterministic evaluation and result
            # ordering, never a set.
            for rule in config.rules:
                if _rule_matches(rule, context):
                    fired_rule_ids.append(rule.id)
                    rule_categories[rule.id] = rule.category
                    reason_codes.append(rule.reason_code)
                    if rule.category == "SCORE_CONTRIBUTING":
                        score_contribution += rule.score_contribution

            return RuleEvaluationResult(
                provider_name=self.provider_name,
                provider_version=self.provider_version,
                rule_set_version=config.rule_set_version,
                fired_rule_ids=fired_rule_ids,
                rule_categories=rule_categories,
                reason_codes=reason_codes,
                score_contribution=score_contribution,
                minimum_priority_band=None,
                evaluated_at=datetime.now(timezone.utc),
                latency_ms=(time.monotonic() - start) * 1000.0,
                provider_status="OK",
                provider_error_code=None,
            )
        except Exception as exc:
            # Corrected failure behavior (guide section 10): score_contribution
            # is NEVER manipulated to simulate a floor -- minimum_priority_band
            # is the sole, explicit, auditable mechanism. The existing source
            # alert is untouched either way; this method has no persistence
            # code path that could drop or duplicate it.
            return RuleEvaluationResult(
                provider_name=self.provider_name,
                provider_version=self.provider_version,
                rule_set_version=UNKNOWN_RULE_SET_VERSION,
                fired_rule_ids=[],
                rule_categories={},
                reason_codes=[RULE_PROVIDER_UNAVAILABLE_REASON_CODE],
                score_contribution=0.0,
                minimum_priority_band=FAILURE_MINIMUM_PRIORITY_BAND,
                evaluated_at=datetime.now(timezone.utc),
                latency_ms=(time.monotonic() - start) * 1000.0,
                provider_status="ERROR",
                provider_error_code=_classify_error(exc),
            )

    @staticmethod
    def _build_condition_context(event: FraudEvent, features: Mapping[str, Any]) -> dict[str, Any]:
        """Merges, in order (later keys win): the event's own channel
        payload fields (guide section 10's illustrative example references
        payload fields like duplicate_image_hash_flag directly), the
        validated Phase 2 feature vector, and a small fixed set of
        current-event intrinsic fields. Only FraudEvent/SourceAlertContext
        objects and this plain dict are ever touched -- no label,
        disposition, outcome, or training-eligibility field is readable
        from any of these three sources."""
        context: dict[str, Any] = dict(event.channel_payload.model_dump())
        context.update(features)
        context["amount_minor_units"] = event.amount_minor_units
        context["channel"] = event.channel
        context["direction"] = event.direction
        context["device_id_present"] = event.device_id is not None
        context["ip_address_present"] = event.ip_address is not None
        return context
