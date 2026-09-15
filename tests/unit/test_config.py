from src.common.config import get_app_settings, get_fraud_rules, get_settings


def test_settings_load_with_defaults():
    settings = get_settings()
    assert settings.postgres_db == "aidp"
    assert settings.postgres_test_db == "aidp_test"
    assert settings.mlflow_tracking_uri.endswith(":5001")


def test_settings_dsn_uses_test_db():
    settings = get_settings()
    assert settings.postgres_test_dsn.endswith(f"/{settings.postgres_test_db}")
    assert settings.postgres_dsn.endswith(f"/{settings.postgres_db}")


def test_app_settings_yaml_has_generation_defaults():
    app_settings = get_app_settings()
    gen = app_settings["generation"]
    assert gen["historical_transactions"] == 50000  # fix H5
    assert 0 < gen["fraud_ratio"] < 1
    assert app_settings["timezone"] == "Asia/Kolkata"


def test_fraud_rules_thresholds_are_ordered():
    rules = get_fraud_rules()["decisioning"]
    assert rules["approve_max"] < rules["monitor_max"] < rules["review_max"] < 1.0
