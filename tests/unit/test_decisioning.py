import pytest

from src.decisioning.engine import classify, reason_codes


@pytest.mark.parametrize(
    "probability,expected_risk,expected_decision",
    [
        (0.0, "LOW", "APPROVE"),
        (0.40, "LOW", "APPROVE"),
        (0.41, "MEDIUM", "MONITOR"),
        (0.75, "MEDIUM", "MONITOR"),
        (0.76, "HIGH", "REVIEW"),
        (0.90, "HIGH", "REVIEW"),
        (0.91, "HIGH", "BLOCK"),
        (1.0, "HIGH", "BLOCK"),
    ],
)
def test_classify_thresholds(probability, expected_risk, expected_decision):
    risk, decision = classify(probability)
    assert risk == expected_risk
    assert decision == expected_decision


def test_reason_codes_fires_on_each_condition_independently():
    assert reason_codes({"new_device_flag": True}) == ["NEW_DEVICE"]
    assert reason_codes({"new_country_flag": True}) == ["NEW_COUNTRY"]
    assert reason_codes({"amount_vs_customer_average": 5.0}) == ["HIGH_AMOUNT_VS_AVERAGE"]
    assert reason_codes({"amount_vs_customer_average": 4.99}) == []
    assert reason_codes({"failed_attempts_last_1h": 2}) == ["MULTIPLE_RECENT_ATTEMPTS"]
    assert reason_codes({"failed_attempts_last_1h": 1}) == []
    assert reason_codes({"night_transaction_flag": True}) == ["NIGHT_TRANSACTION"]
    assert reason_codes({"transactions_last_10m": 3}) == ["HIGH_VELOCITY"]
    assert reason_codes({"transactions_last_10m": 2}) == []


def test_reason_codes_combines_and_matches_guide_worked_example():
    features = {
        "new_device_flag": True,
        "new_country_flag": True,
        "amount_vs_customer_average": 10.25,
        "night_transaction_flag": False,
        "failed_attempts_last_1h": 0,
        "transactions_last_10m": 0,
    }
    assert reason_codes(features) == ["NEW_DEVICE", "NEW_COUNTRY", "HIGH_AMOUNT_VS_AVERAGE"]


def test_reason_codes_empty_for_clean_transaction():
    features = {
        "new_device_flag": False,
        "new_country_flag": False,
        "amount_vs_customer_average": 1.1,
        "night_transaction_flag": False,
        "failed_attempts_last_1h": 0,
        "transactions_last_10m": 1,
    }
    assert reason_codes(features) == []
