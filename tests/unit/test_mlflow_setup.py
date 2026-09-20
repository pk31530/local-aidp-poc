from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from mlflow.exceptions import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    PERMISSION_DENIED,
    TEMPORARILY_UNAVAILABLE,
    UNAUTHENTICATED,
    MlflowException,
)

import src.common.mlflow_setup as mlflow_setup_module
from src.common.mlflow_setup import ModelAliasNotFoundError, get_model_alias_info, load_champion_model


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


# ---- get_model_alias_info: structured not-found vs. operational failures ----------


class _RaisingMlflowClient:
    """Raises a given MlflowException from get_model_version_by_alias, so
    no real MLflow client or connection is ever created."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def get_model_version_by_alias(self, model_name, alias):
        raise self._exc

    def get_run(self, run_id):
        raise AssertionError("must not be reached when get_model_version_by_alias raises")


def _fake_mlflow_with_raising_client(exc: Exception):
    client = _RaisingMlflowClient(exc)
    return SimpleNamespace(tracking=SimpleNamespace(MlflowClient=lambda: client))


def test_missing_alias_raises_model_alias_not_found_error(monkeypatch):
    # The real error code MLflow's SQLAlchemy-backed model registry store
    # raises for a missing alias (verified against
    # SqlAlchemyStore.get_model_version_by_alias) — not RESOURCE_DOES_NOT_EXIST.
    # error_code must be the actual protobuf enum value, not its string name —
    # MlflowException.__init__ silently falls back to INTERNAL_ERROR otherwise.
    not_found = MlflowException("Registered model alias nope not found.", error_code=INVALID_PARAMETER_VALUE)
    assert not_found.error_code == "INVALID_PARAMETER_VALUE"  # guards the fixture itself
    monkeypatch.setattr(mlflow_setup_module, "mlflow", _fake_mlflow_with_raising_client(not_found))

    with pytest.raises(ModelAliasNotFoundError):
        get_model_alias_info("fraud-detection-model", "nope")


@pytest.mark.parametrize(
    "error_code",
    [TEMPORARILY_UNAVAILABLE, UNAUTHENTICATED, INTERNAL_ERROR, PERMISSION_DENIED],
)
def test_other_mlflow_exceptions_are_not_converted_and_propagate(monkeypatch, error_code):
    """Connection/auth/server failures must remain plain MlflowException,
    not be reclassified as ModelAliasNotFoundError — the CLI relies on this
    to treat them as operational failures (exit 3), not user error (exit 2)."""
    operational = MlflowException("mlflow server unavailable", error_code=error_code)
    monkeypatch.setattr(mlflow_setup_module, "mlflow", _fake_mlflow_with_raising_client(operational))

    with pytest.raises(MlflowException) as exc_info:
        get_model_alias_info("fraud-detection-model", "champion")

    assert not isinstance(exc_info.value, ModelAliasNotFoundError)


def test_no_real_mlflow_client_is_ever_constructed(monkeypatch):
    """Guards the fake itself: MlflowClient() must resolve to our fake
    lambda, never mlflow.tracking.MlflowClient's real implementation."""
    constructed = []
    fake_mlflow = SimpleNamespace(
        tracking=SimpleNamespace(
            MlflowClient=lambda: constructed.append(1)
            or _RaisingMlflowClient(MlflowException("not found", error_code=INVALID_PARAMETER_VALUE))
        )
    )
    monkeypatch.setattr(mlflow_setup_module, "mlflow", fake_mlflow)

    with pytest.raises(ModelAliasNotFoundError):
        get_model_alias_info("fraud-detection-model", "nope")

    assert constructed == [1]
