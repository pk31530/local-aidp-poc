"""Applies config/fraud_rules.yaml thresholds + reason-code rules to a
fraud probability and feature vector. The model score alone is never the
final business decision (guide section 16) — this is that business layer.
"""
from __future__ import annotations

from src.common.config import get_fraud_rules

_RULES = get_fraud_rules()
_THRESHOLDS = _RULES["decisioning"]

# Mirrors the human-readable conditions documented in
# config/fraud_rules.yaml's reason_codes section (kept as plain numbers
# here rather than eval'd from YAML, to avoid evaluating untrusted
# expressions).
_HIGH_AMOUNT_RATIO = 5.0
_MULTIPLE_FAILED_ATTEMPTS = 2
_HIGH_VELOCITY_COUNT = 3


def classify(fraud_probability: float) -> tuple[str, str]:
    """Returns (risk_level, decision) per the configured thresholds:
    0.00-0.40 APPROVE, 0.40-0.75 MONITOR, 0.75-0.90 REVIEW, 0.90-1.00 BLOCK.
    """
    if fraud_probability <= _THRESHOLDS["approve_max"]:
        return "LOW", "APPROVE"
    if fraud_probability <= _THRESHOLDS["monitor_max"]:
        return "MEDIUM", "MONITOR"
    if fraud_probability <= _THRESHOLDS["review_max"]:
        return "HIGH", "REVIEW"
    return "HIGH", "BLOCK"


def reason_codes(features: dict) -> list[str]:
    codes = []
    if features.get("new_device_flag"):
        codes.append("NEW_DEVICE")
    if features.get("new_country_flag"):
        codes.append("NEW_COUNTRY")
    if features.get("amount_vs_customer_average", 0) >= _HIGH_AMOUNT_RATIO:
        codes.append("HIGH_AMOUNT_VS_AVERAGE")
    if features.get("failed_attempts_last_1h", 0) >= _MULTIPLE_FAILED_ATTEMPTS:
        codes.append("MULTIPLE_RECENT_ATTEMPTS")
    if features.get("night_transaction_flag"):
        codes.append("NIGHT_TRANSACTION")
    if features.get("transactions_last_10m", 0) >= _HIGH_VELOCITY_COUNT:
        codes.append("HIGH_VELOCITY")
    return codes
