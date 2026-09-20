from dataclasses import dataclass, field
from types import SimpleNamespace

import src.common.mlflow_setup as mlflow_setup_module
from src.common.mlflow_setup import get_model_alias_info, load_champion_model


@dataclass
class _FakeModelVersion:
    run_id: str = "run-1"
    version: str = "3"


@dataclass
class _FakeRunData:
    params: dict = field(default_factory=dict)


@dataclass
class _FakeRun:
    data: _FakeRunData


class _FakeMlflowClient:
    def __init__(self, model_type: str | None):
        self._model_type = model_type
        self.requested_aliases: list = []

    def get_model_version_by_alias(self, model_name, alias):
        self.requested_aliases.append((model_name, alias))
        return _FakeModelVersion()

    def get_run(self, run_id):
        params = {} if self._model_type is None else {"model_type": self._model_type}
        return _FakeRun(data=_FakeRunData(params=params))


def _build_fake_mlflow(model_type: str | None, calls: list):
    """A fully isolated stand-in for the `mlflow` module, patched onto
    src.common.mlflow_setup only (fix: mlflow.xgboost/mlflow.sklearn are
    real LazyLoader objects that don't reliably stay patched via
    monkeypatch.setattr across calls — replacing the whole `mlflow` name
    used inside mlflow_setup.py avoids touching real mlflow internals)."""

    def xgboost_load_model(uri):
        calls.append(("xgboost", uri))
        return "xgb-model"

    def sklearn_load_model(uri):
        calls.append(("sklearn", uri))
        return "sk-model"

    client = _FakeMlflowClient(model_type)
    fake_mlflow = SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=lambda: client),
        xgboost=SimpleNamespace(load_model=xgboost_load_model),
        sklearn=SimpleNamespace(load_model=sklearn_load_model),
    )
    fake_mlflow.client = client  # exposed for tests that need to inspect calls
    return fake_mlflow


def test_xgboost_model_type_routes_to_xgboost_loader(monkeypatch):
    calls: list = []
    monkeypatch.setattr(mlflow_setup_module, "mlflow", _build_fake_mlflow("xgboost", calls))

    model, version = load_champion_model("fraud-detection-model")

    assert model == "xgb-model"
    assert version == "3"
    assert calls == [("xgboost", "models:/fraud-detection-model@champion")]


def test_random_forest_fallback_model_type_routes_to_sklearn_loader(monkeypatch):
    calls: list = []
    monkeypatch.setattr(mlflow_setup_module, "mlflow", _build_fake_mlflow("random_forest_fallback", calls))

    model, version = load_champion_model("fraud-detection-model")

    assert model == "sk-model"
    assert version == "3"
    assert calls == [("sklearn", "models:/fraud-detection-model@champion")]


def test_missing_model_type_param_defaults_to_xgboost(monkeypatch):
    calls: list = []
    monkeypatch.setattr(mlflow_setup_module, "mlflow", _build_fake_mlflow(None, calls))

    model, version = load_champion_model("fraud-detection-model")

    assert model == "xgb-model"
    assert calls == [("xgboost", "models:/fraud-detection-model@champion")]


# ---- get_model_alias_info: metadata only, never loads weights ---------------------


def test_get_model_alias_info_returns_only_the_safe_fields(monkeypatch):
    calls: list = []
    fake_mlflow = _build_fake_mlflow("xgboost", calls)
    monkeypatch.setattr(mlflow_setup_module, "mlflow", fake_mlflow)

    info = get_model_alias_info("fraud-detection-model", "champion")

    assert info == {
        "model_name": "fraud-detection-model",
        "alias": "champion",
        "version": "3",
        "model_type": "xgboost",
        "run_id": "run-1",
    }
    assert calls == []  # never loaded weights


def test_get_model_alias_info_requests_the_given_alias(monkeypatch):
    calls: list = []
    fake_mlflow = _build_fake_mlflow("xgboost", calls)
    monkeypatch.setattr(mlflow_setup_module, "mlflow", fake_mlflow)

    get_model_alias_info("fraud-detection-model", "some-other-alias")

    assert fake_mlflow.client.requested_aliases == [("fraud-detection-model", "some-other-alias")]


def test_get_model_alias_info_defaults_model_type_to_xgboost_when_param_missing(monkeypatch):
    calls: list = []
    fake_mlflow = _build_fake_mlflow(None, calls)
    monkeypatch.setattr(mlflow_setup_module, "mlflow", fake_mlflow)

    info = get_model_alias_info("fraud-detection-model", "champion")

    assert info["model_type"] == "xgboost"
