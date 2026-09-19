import polars as pl
import pytest

from src.common.splits import TEST_SIZE, TRAIN_SIZE, VAL_SIZE, assign_split


def _labels(n_fraud: int, n_legit: int) -> list[bool]:
    return [True] * n_fraud + [False] * n_legit


def _df(labels: list[bool]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transaction_id": [f"TX{i}" for i in range(len(labels))],
            "is_fraud": labels,
        }
    )


def test_assign_split_is_deterministic_given_same_seed():
    df = _df(_labels(n_fraud=60, n_legit=340))
    first = assign_split(df, seed=42)
    second = assign_split(df, seed=42)
    assert first["split"].to_list() == second["split"].to_list()


def test_assign_split_different_seed_can_change_assignment():
    df = _df(_labels(n_fraud=60, n_legit=340))
    a = assign_split(df, seed=1)
    b = assign_split(df, seed=2)
    assert a["split"].to_list() != b["split"].to_list()


def test_assign_split_produces_correctly_stratified_proportions():
    n_fraud, n_legit = 100, 900
    df = _df(_labels(n_fraud=n_fraud, n_legit=n_legit))
    out = assign_split(df, seed=42)

    assert set(out["split"].to_list()) == {"train", "val", "test"}

    counts = {row["split"]: row["count"] for row in out["split"].value_counts().to_dicts()}
    total = out.height
    assert counts["train"] == pytest.approx(total * TRAIN_SIZE, abs=2)
    assert counts["val"] == pytest.approx(total * VAL_SIZE, abs=2)
    assert counts["test"] == pytest.approx(total * TEST_SIZE, abs=2)

    overall_fraud_rate = n_fraud / total
    for split_name in ("train", "val", "test"):
        split_df = out.filter(pl.col("split") == split_name)
        split_fraud_rate = split_df["is_fraud"].sum() / split_df.height
        assert split_fraud_rate == pytest.approx(overall_fraud_rate, abs=0.03)


def test_assign_split_preserves_row_count_and_existing_columns():
    n = 20
    df = pl.DataFrame(
        {
            "transaction_id": [f"TX{i}" for i in range(n)],
            "merchant": ["Grocery", "Electronics"] * (n // 2),
            "is_fraud": [i % 2 == 0 for i in range(n)],
        }
    )
    out = assign_split(df, seed=42)
    assert out.height == df.height
    assert "merchant" in out.columns
    assert "transaction_id" in out.columns
