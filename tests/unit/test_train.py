import pandas as pd

from src.common.features import MODEL_FEATURE_COLUMNS
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
