import inspect
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.common.features import MODEL_FEATURE_COLUMNS
from src.control_plane.config import TrainingRunConfig
from src.control_plane.runs import RunRecord
from src.ml import train as train_module
from src.ml.train import _split


def _dataset(n_train: int, n_val: int, n_test: int) -> pd.DataFrame:
    n = n_train + n_val + n_test
    data = {col: [1.0] * n for col in MODEL_FEATURE_COLUMNS}
    data["is_fraud"] = [i % 2 for i in range(n)]
    data["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return pd.DataFrame(data)


def test_split_partition_sizes_match_persisted_split_column_exactly():
    """Finding 1 fix: train.py must read the "split" column persisted by
    src.common.splits.assign_split rather than re-deriving its own
    train_test_split, so evaluation uses exactly the partition the
    merchant/country risk lookups were fit on."""
    df = _dataset(n_train=70, n_val=15, n_test=15)

    X_train, X_val, X_test, y_train, y_val, y_test = _split(df)

    assert len(X_train) == len(y_train) == (df["split"] == "train").sum() == 70
    assert len(X_val) == len(y_val) == (df["split"] == "val").sum() == 15
    assert len(X_test) == len(y_test) == (df["split"] == "test").sum() == 15


def test_split_ignores_no_other_partitioning_logic_even_when_uneven():
    """An intentionally lopsided persisted split (not a 70/15/15 shape) must
    still be honored verbatim — proving _split never re-derives its own
    proportions independently of the persisted column."""
    df = _dataset(n_train=50, n_val=40, n_test=10)

    X_train, X_val, X_test, y_train, y_val, y_test = _split(df)

    assert len(X_train) == 50
    assert len(X_val) == 40
    assert len(X_test) == 10
    assert len(X_train) + len(X_val) + len(X_test) == len(df)


def test_split_uses_exact_model_feature_columns():
    df = _dataset(n_train=10, n_val=5, n_test=5)
    X_train, _, _, _, _, _ = _split(df)
    assert list(X_train.columns) == MODEL_FEATURE_COLUMNS


# ---- Phase 3: lifecycle integration — no real training, no real MLflow/Postgres ----


class _FakeRunStore:
    """In-memory stand-in for _PostgresRunStore — no database touched, same
    contract used by tests/unit/test_control_plane_runs.py."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id,
            "trigger_source": None,
            "git_sha": None,
            "config_snapshot": None,
            "config_hash": None,
            "dataset_version": None,
            "model_version": None,
            "error_type": None,
            "error_message": None,
            "started_at": datetime.now(timezone.utc),
            "heartbeat_at": None,
            "completed_at": None,
            **row,
        }
        self.rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self.rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self.rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        return [RunRecord(**r) for r in list(self.rows.values())[:limit]]


class _FakeModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


_FIXED_METRICS = {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "roc_auc": 1.0,
    "false_positive_rate": 0.0,
    "false_negative_rate": 0.0,
    "accuracy": 1.0,
    "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
}


class _FakeMlflowRunContext:
    def __init__(self, run_id):
        self.info = type("Info", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeModelInfo:
    def __init__(self, version):
        self.registered_model_version = version


class _FakeMlflowClient:
    def __init__(self):
        self.aliases_set = []

    def set_registered_model_alias(self, model_name, alias, version):
        self.aliases_set.append((model_name, alias, version))


class _FakeMlflow:
    """Replaces train_module.mlflow entirely — no real MLflow server or
    file store is ever touched by these tests."""

    def __init__(self):
        self.client = _FakeMlflowClient()
        self.xgboost = type("M", (), {"log_model": staticmethod(lambda *a, **k: _FakeModelInfo("7"))})()
        self.sklearn = type("M", (), {"log_model": staticmethod(lambda *a, **k: _FakeModelInfo("7"))})()
        self.models = type("M", (), {"infer_signature": staticmethod(lambda *a, **k: "fake-signature")})()
        self.tracking = type("M", (), {"MlflowClient": lambda self_=None: self.client})()

    def set_experiment(self, name):
        pass

    def start_run(self, run_name=None):
        return _FakeMlflowRunContext("fake-mlflow-run-id")

    def log_param(self, key, value):
        pass

    def log_params(self, params):
        pass

    def log_metric(self, key, value):
        pass

    def log_artifact(self, path):
        pass


def _valid_features_dataframe(n_train=4, n_val=1, n_test=1) -> pd.DataFrame:
    n = n_train + n_val + n_test
    data = {col: [1.0] * n for col in MODEL_FEATURE_COLUMNS}
    data["is_fraud"] = [i % 2 for i in range(n)]
    data["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def _fake_lifecycle_and_mlflow(monkeypatch):
    """Replaces train_module.RunLifecycle with a factory backed by an
    in-memory store (no database) and train_module.mlflow with a fully
    fake stand-in (no MLflow server, no real logging/registration)."""
    from src.control_plane.runs import RunLifecycle as RealRunLifecycle

    store_holder = {"store": _FakeRunStore()}
    monkeypatch.setattr(train_module, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store_holder["store"]))
    monkeypatch.setattr(train_module, "mlflow", _FakeMlflow())
    monkeypatch.setattr(train_module, "configure_mlflow", lambda: None)
    monkeypatch.setattr(train_module, "_train_xgboost", lambda *a, **k: (_FakeModel(), {"n_estimators": 1}, "xgboost"))
    monkeypatch.setattr(train_module, "_evaluate", lambda *a, **k: dict(_FIXED_METRICS))
    monkeypatch.setattr(train_module, "_record_model_version_in_postgres", lambda **kwargs: None)
    yield store_holder


def _write_features_parquet(path, **kwargs):
    import polars as pl

    pl.from_pandas(_valid_features_dataframe(**kwargs)).write_parquet(path)


# ---- legacy signature unchanged, no sentinel in public API -------------------------


def test_train_signature_unchanged():
    sig = inspect.signature(train_module.train)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["features_path"]
    assert params[0].default == train_module.DEFAULT_FEATURES_PATH


_ORDINARY_DEFAULT_TYPES = (type(None), str, bool, int, float)


def test_no_sentinel_object_in_public_signatures():
    for fn in (train_module.train, train_module.train_configured):
        for param in inspect.signature(fn).parameters.values():
            if param.default is inspect.Parameter.empty:
                continue
            assert isinstance(param.default, _ORDINARY_DEFAULT_TYPES) or param.default == train_module.DEFAULT_FEATURES_PATH


def test_legacy_wrapper_builds_expected_config_and_delegates_once(monkeypatch, tmp_path):
    captured = {}

    def _fake_configured(config, *, trigger_source="legacy"):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(train_module, "train_configured", _fake_configured)

    features_path = tmp_path / "features.parquet"
    result = train_module.train(features_path)

    assert result == {"ok": True}
    assert captured["config"] == TrainingRunConfig(features_path=features_path)
    assert captured["trigger_source"] == "legacy"


# ---- successful run: dataset/model version recorded, no real training -------------


def test_successful_run_records_dataset_and_model_version(tmp_path, _fake_lifecycle_and_mlflow):
    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path)

    config = TrainingRunConfig(features_path=features_path)
    result = train_module.train_configured(config, trigger_source="test")

    assert result["model_type"] == "xgboost"
    store = _fake_lifecycle_and_mlflow["store"]
    assert len(store.rows) == 1  # exactly one run record for this invocation
    record = next(iter(store.rows.values()))
    assert record["status"] == "SUCCESS"
    assert record["trigger_source"] == "test"
    assert record["dataset_version"] is not None
    assert record["model_version"] == "7"
    assert record["artifacts"]["mlflow_run_id"] == "fake-mlflow-run-id"


# ---- a failure after dataset_version is known: FAILED with original exception -----


def test_failure_mid_training_records_failed_and_reraises(tmp_path, _fake_lifecycle_and_mlflow, monkeypatch):
    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("evaluate exploded")

    monkeypatch.setattr(train_module, "_evaluate", _boom)

    config = TrainingRunConfig(features_path=features_path)
    with pytest.raises(RuntimeError, match="evaluate exploded"):
        train_module.train_configured(config)

    store = _fake_lifecycle_and_mlflow["store"]
    assert len(store.rows) == 1
    record = next(iter(store.rows.values()))
    assert record["status"] == "FAILED"
    assert record["error_type"] == "RuntimeError"
    assert record["dataset_version"] is not None  # computed before the failure


# ---- a missing features file must still record FAILED, not crash before begin() ---


def test_missing_features_file_records_failed_not_a_bare_crash(tmp_path, _fake_lifecycle_and_mlflow):
    missing_path = tmp_path / "does_not_exist.parquet"
    config = TrainingRunConfig(features_path=missing_path)

    with pytest.raises(FileNotFoundError):
        train_module.train_configured(config)

    store = _fake_lifecycle_and_mlflow["store"]
    assert len(store.rows) == 1
    record = next(iter(store.rows.values()))
    assert record["status"] == "FAILED"
    assert record["dataset_version"] is None  # never computed — the file read failed first


def test_legacy_call_creates_exactly_one_run_record(tmp_path, _fake_lifecycle_and_mlflow):
    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path)

    train_module.train(features_path)

    store = _fake_lifecycle_and_mlflow["store"]
    assert len(store.rows) == 1
    assert next(iter(store.rows.values()))["trigger_source"] == "legacy"
