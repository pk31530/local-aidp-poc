from dataclasses import dataclass, field
from types import SimpleNamespace

import src.common.mlflow_setup as mlflow_setup_module
from src.common.mlflow_setup import load_champion_model


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

    def get_model_version_by_alias(self, model_name, alias):
        assert alias == "champion"
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
    return SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=lambda: client),
        xgboost=SimpleNamespace(load_model=xgboost_load_model),
        sklearn=SimpleNamespace(load_model=sklearn_load_model),
    )


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
