"""Phase 4: assign_chronological_split (src.common.splits, additive --
assign_split() itself is untested here since it is unchanged). No
database, Docker, or network access anywhere in this file.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.common.splits import assign_chronological_split, assign_split, realized_split_fractions

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _df(rows: list[tuple[datetime, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"ts": [r[0] for r in rows], "id": [r[1] for r in rows]}
    )


def _evenly_spaced(n: int, step_minutes: int = 10) -> pl.DataFrame:
    rows = [(BASE + timedelta(minutes=step_minutes * i), f"id{i:04d}") for i in range(n)]
    return _df(rows)


def test_assign_split_is_unchanged_and_still_importable():
    """Phase 4 must not modify or remove assign_split()."""
    assert callable(assign_split)


def test_split_adds_expected_labels_only():
    df = _evenly_spaced(30)
    result = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)
    assert set(result["split"].to_list()) <= {"train", "calibration", "test"}


def test_result_preserves_chronological_order_within_each_split():
    df = _evenly_spaced(30)
    result = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)
    for name in ("train", "calibration", "test"):
        part = result.filter(pl.col("split") == name)["ts"].to_list()
        assert part == sorted(part)


def test_train_rows_are_strictly_before_calibration_and_test():
    df = _evenly_spaced(30)
    result = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)
    max_train = result.filter(pl.col("split") == "train")["ts"].max()
    min_calib = result.filter(pl.col("split") == "calibration")["ts"].min()
    max_calib = result.filter(pl.col("split") == "calibration")["ts"].max()
    min_test = result.filter(pl.col("split") == "test")["ts"].min()
    assert max_train < min_calib
    assert max_calib < min_test


def test_tie_break_sort_is_deterministic_by_id_for_equal_timestamps():
    # A block of three rows sharing one timestamp, inserted out of id
    # order, plus enough distinct earlier/later timestamps to make the
    # split itself valid.
    rows = [(BASE - timedelta(minutes=i), f"pre{i:03d}") for i in range(1, 11)]
    rows += [(BASE, "z"), (BASE, "a"), (BASE, "m")]
    rows += [(BASE + timedelta(minutes=i), f"post{i:03d}") for i in range(1, 11)]
    df = _df(rows)
    result = assign_chronological_split(df, "ts", "id", 0.4, 0.3, 0.3)
    # Whatever the partition assignment, the block sharing BASE's
    # timestamp must appear in id-ascending order -- proving the tie-break
    # is (timestamp, id), never original insertion order.
    block = result.filter(pl.col("ts") == BASE)["id"].to_list()
    assert block == sorted(block) == ["a", "m", "z"]


def test_rows_sharing_a_timestamp_never_split_across_two_partitions():
    """A block of many rows at the SAME timestamp, positioned right where
    the raw row-count fraction would otherwise cut through the middle of
    it, must all land in the same partition."""
    rows = []
    # 10 early rows, then a big block of 20 rows sharing one timestamp
    # (this is exactly where a 0.5 boundary would naively fall mid-block),
    # then 10 late rows.
    for i in range(10):
        rows.append((BASE + timedelta(minutes=i), f"early{i:03d}"))
    block_ts = BASE + timedelta(minutes=100)
    for i in range(20):
        rows.append((block_ts, f"block{i:03d}"))
    for i in range(10):
        rows.append((BASE + timedelta(minutes=200 + i), f"late{i:03d}"))
    df = _df(rows)

    result = assign_chronological_split(df, "ts", "id", 0.5, 0.25, 0.25)
    block_rows = result.filter(pl.col("ts") == block_ts)
    assert block_rows.height == 20
    assert block_rows["split"].n_unique() == 1


def test_purge_gap_removes_a_buffer_from_both_sides_of_each_boundary():
    df = _evenly_spaced(40, step_minutes=60)  # one row per hour, 40 hours total
    purge_gap = timedelta(hours=2, minutes=30)
    result_no_purge = assign_chronological_split(df, "ts", "id", 0.5, 0.25, 0.25)
    result_purged = assign_chronological_split(df, "ts", "id", 0.5, 0.25, 0.25, purge_gap=purge_gap)

    assert result_purged.height < result_no_purge.height

    max_train = result_purged.filter(pl.col("split") == "train")["ts"].max()
    min_calib = result_purged.filter(pl.col("split") == "calibration")["ts"].min()
    assert (min_calib - max_train) >= purge_gap


def test_purge_gap_zero_behaves_like_a_group_aligned_plain_split():
    df = _evenly_spaced(30)
    zero_purge = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15, purge_gap=timedelta(0))
    no_arg = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)
    assert zero_purge["id"].to_list() == no_arg["id"].to_list()
    assert zero_purge["split"].to_list() == no_arg["split"].to_list()


def test_empty_dataframe_raises():
    df = pl.DataFrame({"ts": pl.Series([], dtype=pl.Datetime(time_zone="UTC")), "id": pl.Series([], dtype=pl.Utf8)})
    with pytest.raises(ValueError, match="at least one row"):
        assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)


def test_single_timestamp_group_raises():
    df = _df([(BASE, "a"), (BASE, "b"), (BASE, "c")])
    with pytest.raises(ValueError, match="distinct timestamp"):
        assign_chronological_split(df, "ts", "id", 0.34, 0.33, 0.33)


def test_two_timestamp_groups_insufficient_for_three_partitions():
    df = _df([(BASE, "a"), (BASE, "b"), (BASE + timedelta(minutes=1), "c")])
    with pytest.raises(ValueError, match="three distinct timestamp"):
        assign_chronological_split(df, "ts", "id", 0.34, 0.33, 0.33)


def test_fractions_must_sum_to_one():
    df = _evenly_spaced(10)
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_chronological_split(df, "ts", "id", 0.5, 0.3, 0.3)


def test_negative_fraction_rejected():
    df = _evenly_spaced(10)
    with pytest.raises(ValueError, match="non-negative"):
        assign_chronological_split(df, "ts", "id", 1.2, -0.1, -0.1)


def test_purge_gap_that_empties_one_partition_raises_a_specific_error():
    # A tiny, cheap calibration slice (short time span) flanked by wide
    # train/test spans: a moderate purge gap wipes out just the
    # calibration partition while train/test still retain rows.
    rows = [(BASE + timedelta(minutes=i), f"train{i:03d}") for i in range(20)]
    rows += [(BASE + timedelta(minutes=100), "calib000"), (BASE + timedelta(minutes=101), "calib001")]
    rows += [(BASE + timedelta(minutes=200 + i), f"test{i:03d}") for i in range(20)]
    df = _df(rows)
    with pytest.raises(ValueError, match="'calibration' partition"):
        assign_chronological_split(df, "ts", "id", 0.48, 0.04, 0.48, purge_gap=timedelta(minutes=2))


def test_purge_gap_that_removes_every_row_raises():
    df = _evenly_spaced(6, step_minutes=60)  # 6 hourly rows total
    with pytest.raises(ValueError, match="removed every row"):
        assign_chronological_split(df, "ts", "id", 0.34, 0.33, 0.33, purge_gap=timedelta(days=10))


def test_realized_split_fractions_reflect_actual_row_counts():
    df = _evenly_spaced(30)
    result = assign_chronological_split(df, "ts", "id", 0.7, 0.15, 0.15)
    fractions = realized_split_fractions(result)
    assert set(fractions.keys()) == {"train", "calibration", "test"}
    assert fractions["train"] + fractions["calibration"] + fractions["test"] == pytest.approx(1.0)
    train_count = result.filter(pl.col("split") == "train").height
    assert fractions["train"] == pytest.approx(train_count / result.height)


def test_realized_split_fractions_on_empty_frame():
    df = pl.DataFrame({"split": pl.Series([], dtype=pl.Utf8)})
    fractions = realized_split_fractions(df)
    assert fractions == {"train": 0.0, "calibration": 0.0, "test": 0.0}
