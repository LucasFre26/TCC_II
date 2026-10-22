"""Agregação ponderada por cpu_num e coorte homogênea."""

import numpy as np
import pandas as pd
import pytest

from energy_prediction.data.alibaba import build_grid_table
from energy_prediction.preprocessing.cluster import (
    CohortTooSmallError,
    UnknownAggregationError,
    aggregate,
    aggregate_weighted,
    get_strategy,
    resolve_machine_info,
    select_homogeneous_cohort,
)
from synthetic import make_meta_frame, make_usage_frame, write_usage_archive


def _grid(rows):
    frame = pd.DataFrame(rows, columns=["machine_id", "bin", "cpu"])
    for column in ("mem", "mem_gps", "net_in", "net_out", "disk"):
        frame[column] = 1.0
    return frame


def _info(cpu_nums, active=None):
    active = active or {}
    return pd.DataFrame(
        {
            "machine_id": list(cpu_nums),
            "cpu_num": [float(v) for v in cpu_nums.values()],
            "cpu_num_conflict": False,
            "active": [active.get(m, True) for m in cpu_nums],
        }
    )


def test_known_weights_give_40():
    grid = _grid([("a", 0, 10.0), ("b", 0, 50.0)])
    series, _ = aggregate_weighted(grid, _info({"a": 32, "b": 96}), step_seconds=300.0)
    assert series["cpu"].iloc[0] == pytest.approx((32 * 10 + 96 * 50) / 128)
    assert series["timestamp"].iloc[0] == 0.0


def test_absent_machine_is_out_of_numerator_and_denominator():
    grid = _grid([("a", 0, 10.0), ("b", 0, 50.0), ("a", 1, 20.0)])
    series, _ = aggregate_weighted(grid, _info({"a": 32, "b": 96}), step_seconds=300.0)
    assert series["cpu"].iloc[1] == pytest.approx(20.0)
    assert series["n_machines"].tolist() == [2, 1]


def test_coverage_is_half_when_half_the_capacity_is_missing():
    grid = _grid([("a", 0, 10.0), ("b", 0, 50.0), ("a", 1, 20.0)])
    series, info = aggregate_weighted(grid, _info({"a": 64, "b": 64}), step_seconds=300.0)
    assert series["coverage"].tolist() == pytest.approx([1.0, 0.5])
    assert info["coverage_min"] == pytest.approx(0.5)
    assert info["coverage_mean"] == pytest.approx(0.75)


def test_missing_bin_is_kept_as_nan_with_zero_coverage():
    grid = _grid([("a", 0, 10.0), ("a", 2, 30.0)])
    series, _ = aggregate_weighted(grid, _info({"a": 32}), step_seconds=300.0)
    assert len(series) == 3
    assert np.isnan(series["cpu"].iloc[1]) and series["coverage"].iloc[1] == 0.0


def test_same_weights_for_exogenous_and_variable_specific_denominator():
    grid = _grid([("a", 0, 10.0), ("b", 0, 50.0)])
    grid.loc[grid["machine_id"] == "b", "net_in"] = np.nan
    grid.loc[grid["machine_id"] == "a", "net_in"] = 7.0
    series, _ = aggregate_weighted(grid, _info({"a": 32, "b": 96}), step_seconds=300.0)
    assert series["net_in"].iloc[0] == pytest.approx(7.0)
    assert series["mem"].iloc[0] == pytest.approx(1.0)


def test_machine_without_metadata_is_excluded_and_counted():
    grid = _grid([("a", 0, 10.0), ("ghost", 0, 90.0)])
    series, info = aggregate_weighted(grid, _info({"a": 32}), step_seconds=300.0)
    assert series["cpu"].iloc[0] == pytest.approx(10.0)
    assert info["machines_without_metadata_excluded"] == 1


def test_cpu_num_conflict_uses_most_frequent_then_smallest():
    meta = pd.DataFrame(
        {
            "machine_id": ["a", "a", "a", "b", "b"],
            "time_stamp": [0, 1, 2, 0, 1],
            "cpu_num": [32, 96, 96, 32, 96],
            "status": "USING",
        }
    )
    info = resolve_machine_info(meta, active_statuses=["USING"]).set_index("machine_id")
    assert info.loc["a", "cpu_num"] == 96 and info.loc["a", "cpu_num_conflict"]
    assert info.loc["b", "cpu_num"] == 32


def test_active_requires_all_status_records_in_active_set():
    meta = pd.DataFrame(
        {
            "machine_id": ["a", "a", "b"],
            "time_stamp": [0, 5, 0],
            "cpu_num": 32,
            "status": ["USING", "OFFLINE", "USING"],
        }
    )
    info = resolve_machine_info(meta, active_statuses=["USING"]).set_index("machine_id")
    assert not info.loc["a", "active"] and info.loc["b", "active"]
    early = resolve_machine_info(meta, active_statuses=["USING"], window_end_seconds=5).set_index("machine_id")
    assert early.loc["a", "active"]


def test_cohort_selects_same_cpu_num_and_active_only():
    info = _info({"a": 96, "b": 96, "c": 32, "d": 96}, active={"d": False})
    chosen, cpu_num = select_homogeneous_cohort(info, cpu_num=None, min_machines=2)
    assert chosen == ["a", "b"] and cpu_num == 96.0
    chosen, cpu_num = select_homogeneous_cohort(info, cpu_num=32, min_machines=1)
    assert chosen == ["c"]


def test_tiny_cohort_raises_with_size():
    info = _info({"a": 96, "b": 32})
    with pytest.raises(CohortTooSmallError, match="1 máquinas"):
        select_homogeneous_cohort(info, cpu_num=96, min_machines=5)


def test_cohort_series_uses_only_cohort_machines():
    grid = _grid([("a", 0, 10.0), ("b", 0, 30.0), ("c", 0, 90.0)])
    info = _info({"a": 96, "b": 96, "c": 32})
    series, meta = aggregate("homogeneous_cohort", grid, info, step_seconds=300.0, cohort_min_machines=2)
    assert series["cpu"].iloc[0] == pytest.approx(20.0)
    assert meta["cohort_size"] == 2 and meta["cohort_cpu_num"] == 96.0


def test_unknown_strategy_raises_before_processing():
    with pytest.raises(UnknownAggregationError):
        get_strategy("media_simples")
    assert set(["cluster_weighted_cpu_num", "homogeneous_cohort"]) <= set(
        __import__("energy_prediction.preprocessing.cluster", fromlist=["x"]).AGGREGATION_STRATEGIES
    )


def test_asynchronous_machines_share_the_same_bin(tmp_path):
    machines = {"m_1": 32, "m_2": 96}
    usage = make_usage_frame(machines, n_steps=10, async_offsets=True, seed=5)
    path = write_usage_archive(tmp_path / "u.tar.gz", usage)
    grid, _ = build_grid_table(path, step_seconds=10.0, chunk_rows=1000)
    meta = make_meta_frame(machines)
    info = resolve_machine_info(meta, active_statuses=["USING"])
    series, _ = aggregate_weighted(grid, info, step_seconds=10.0)
    assert (series["n_machines"] == 2).all()
    expected = usage.assign(bin=(usage["time_stamp"] // 10).astype(int))
    row = expected.loc[expected["bin"] == 3].set_index("machine_id")["cpu_util_percent"]
    assert series.loc[series["bin"] == 3, "cpu"].iloc[0] == pytest.approx((32 * row["m_1"] + 96 * row["m_2"]) / 128)


def test_no_cross_interval_leakage(tmp_path):
    usage = make_usage_frame({"m_1": 32}, n_steps=6, async_offsets=False)
    path = write_usage_archive(tmp_path / "u.tar.gz", usage)
    base, _ = build_grid_table(path, step_seconds=10.0, chunk_rows=1000)
    changed = usage.copy()
    changed.loc[changed["time_stamp"] >= 30.0, "cpu_util_percent"] = 99.0
    path2 = write_usage_archive(tmp_path / "v.tar.gz", changed)
    other, _ = build_grid_table(path2, step_seconds=10.0, chunk_rows=1000)
    assert base.loc[base["bin"] < 3, "cpu"].tolist() == other.loc[other["bin"] < 3, "cpu"].tolist()


def test_equivalent_resamplings_do_not_change_a_gapless_series(tmp_path):
    usage = make_usage_frame({"m_1": 32, "m_2": 96}, n_steps=8, async_offsets=False, seed=2)
    path = write_usage_archive(tmp_path / "u.tar.gz", usage)
    info = resolve_machine_info(make_meta_frame({"m_1": 32, "m_2": 96}), active_statuses=["USING"])
    small, _ = build_grid_table(path, step_seconds=10.0, chunk_rows=3)
    big, _ = build_grid_table(path, step_seconds=10.0, chunk_rows=10_000)
    one, _ = aggregate_weighted(small, info, step_seconds=10.0)
    two, _ = aggregate_weighted(big, info, step_seconds=10.0)
    pd.testing.assert_frame_equal(one, two)
