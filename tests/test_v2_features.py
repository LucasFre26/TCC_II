"""Features causais, ablação aninhada, horizontes e igualdade de informação."""

import numpy as np
import pandas as pd
import pytest

from energy_prediction.features.cpu_features import (
    CALENDAR_FEATURES,
    PERSISTENCE_INPUT,
    InformationSetError,
    ablation_columns,
    assert_equal_information,
    build_feature_table,
    build_group_specs,
    drop_incomplete_origins,
    horizon_steps,
    parse_duration_steps,
    target_column,
)
from synthetic import make_series

STEP = 300.0
GROUPS = {
    "cpu": {"columns": ["cpu"], "lags": ["5min", "10min", "30min"], "windows": ["15min", "30min"]},
    "mem": {"columns": ["mem", "mem_gps"], "lags": ["5min"], "windows": ["15min"]},
    "net": {"columns": ["net_in", "net_out"], "lags": ["5min"], "windows": []},
    "disk": {"columns": ["disk"], "lags": ["5min"], "windows": []},
}
ABLATION = {"A": ["cpu"], "B": ["cpu", "mem"], "C": ["cpu", "mem", "net", "disk"]}
HORIZONS = ["5min", "15min", "30min"]


def _setup(n=200, seed=0):
    series = make_series(n, step_seconds=STEP, seed=seed)
    specs = build_group_specs(GROUPS, STEP)
    steps = horizon_steps(HORIZONS, STEP)
    return series, specs, steps, build_feature_table(series, specs, steps)


def test_duration_conversion():
    assert parse_duration_steps("15min", 300.0) == 3
    assert parse_duration_steps("300s", 300.0) == 1
    assert parse_duration_steps("1h", 300.0) == 12
    assert parse_duration_steps(4, 300.0) == 4
    assert parse_duration_steps("15min", 10.0) == 90


def test_duration_not_multiple_of_step_is_an_error():
    with pytest.raises(ValueError, match="múltiplo"):
        parse_duration_steps("7min", 300.0)
    with pytest.raises(ValueError):
        parse_duration_steps("abc", 300.0)
    with pytest.raises(ValueError):
        parse_duration_steps("1min", 300.0)


def test_horizons_follow_the_grid_and_are_not_hardcoded():
    assert horizon_steps(["5min", "15min", "30min"], 300.0) == [1, 3, 6]
    assert horizon_steps(["5min", "15min", "30min"], 60.0) == [5, 15, 30]


def test_cpu_features_match_naive_loop():
    series, specs, steps, table = _setup()
    cpu = series["cpu"].to_numpy()
    for origin in (40, 77, 150):
        row = table.loc[table["origin_index"] == origin].iloc[0]
        assert row["cpu_t"] == pytest.approx(cpu[origin])
        for lag in specs["cpu"].lags:
            assert row[f"cpu_lag_{lag}"] == pytest.approx(cpu[origin - lag])
        for window in specs["cpu"].windows:
            past = cpu[origin - window + 1 : origin + 1]
            assert row[f"cpu_roll_mean_{window}"] == pytest.approx(past.mean())
            assert row[f"cpu_roll_std_{window}"] == pytest.approx(past.std(ddof=0))


def test_group_columns_are_present_per_group():
    _, specs, _, table = _setup()
    for spec in specs.values():
        assert set(spec.feature_names()) <= set(table.columns)
    assert {"net_in_t", "net_out_t", "net_in_lag_1", "net_out_lag_1"} <= set(specs["net"].feature_names())
    assert "disk_t" in specs["disk"].feature_names()


def test_calendar_is_periodic_over_24h_and_not_civil():
    series = make_series(400, step_seconds=STEP)
    specs = build_group_specs(GROUPS, STEP)
    table = build_feature_table(series, specs, [1])
    period = int(86400 / STEP)
    a, b = table.iloc[10], table.iloc[10 + period]
    assert a["sin_day"] == pytest.approx(b["sin_day"]) and a["cos_day"] == pytest.approx(b["cos_day"])
    assert table.iloc[0]["sin_day"] == pytest.approx(0.0, abs=1e-12)


def test_ablation_sets_are_nested_with_identical_calendar():
    _, specs, _, _ = _setup()
    cols = ablation_columns(specs, ABLATION)
    a, b, c = set(cols["A"]), set(cols["B"]), set(cols["C"])
    assert a < b < c
    assert b - a == set(specs["mem"].feature_names())
    assert c - b == set(specs["net"].feature_names()) | set(specs["disk"].feature_names())
    for name in cols:
        assert set(CALENDAR_FEATURES) <= set(cols[name])
        assert PERSISTENCE_INPUT in cols[name]


def test_non_nested_or_cpuless_ablation_is_rejected():
    _, specs, _, _ = _setup()
    with pytest.raises(ValueError, match="aninhadas"):
        ablation_columns(specs, {"A": ["cpu", "mem"], "B": ["cpu", "net"]})
    with pytest.raises(ValueError, match="CPU"):
        ablation_columns(specs, {"A": ["mem"], "B": ["mem", "net"]})
    with pytest.raises(ValueError, match="desconhecidos"):
        ablation_columns(specs, {"A": ["cpu", "xyz"]})


def test_targets_are_shifted_cpu_and_last_origins_are_removed():
    series, specs, steps, table = _setup(n=100)
    max_h = max(steps)
    assert table["origin_index"].max() == len(series) - max_h - 1
    for h in steps:
        row = table.iloc[20]
        assert row[target_column(h)] == pytest.approx(series["cpu"].iloc[int(row["origin_index"]) + h])
    assert table[[target_column(h) for h in steps]].notna().all().all()


def test_incomplete_origins_use_a_common_set_for_all_configurations():
    series, specs, steps, _ = _setup(n=200)
    series.loc[100, "net_in"] = np.nan
    table = build_feature_table(series, specs, steps)
    cols = ablation_columns(specs, ABLATION)
    common = drop_incomplete_origins(table, cols["C"], steps)
    assert 100 not in set(common["origin_index"])
    assert 101 not in set(common["origin_index"])  # net_in_lag_1 usa a observação ausente
    only_a = drop_incomplete_origins(table, cols["A"], steps)
    assert len(common) < len(only_a)
    assert common["origin_index"].min() >= max(specs["cpu"].max_lookback, 1)


def _perturb_future(series: pd.DataFrame, origin: int) -> pd.DataFrame:
    changed = series.copy()
    for column in ("cpu", "mem", "mem_gps", "net_in", "net_out", "disk"):
        changed.loc[origin + 1 :, column] = 999.0
    return changed


def test_leakage_canary_future_and_target_do_not_change_features():
    series, specs, steps, table = _setup(n=200, seed=1)
    feature_cols = [c for c in table.columns if not c.startswith("y_h") and c != "origin_index"]
    for origin in (60, 100, 150):
        changed = _perturb_future(series, origin)
        other = build_feature_table(changed, specs, steps)
        left = table.loc[table["origin_index"] == origin, feature_cols].iloc[0]
        right = other.loc[other["origin_index"] == origin, feature_cols].iloc[0]
        pd.testing.assert_series_equal(left, right)
    changed_target_only = series.copy()
    changed_target_only.loc[100 + max(steps), "cpu"] = 555.0
    other = build_feature_table(changed_target_only, specs, steps)
    left = table.loc[table["origin_index"] == 100, feature_cols].iloc[0]
    right = other.loc[other["origin_index"] == 100, feature_cols].iloc[0]
    pd.testing.assert_series_equal(left, right)


def test_persistence_input_is_in_every_configuration():
    _, specs, _, _ = _setup()
    cols = ablation_columns(specs, ABLATION)
    assert_equal_information([PERSISTENCE_INPUT], cols)


def test_information_set_regression_is_detected():
    _, specs, _, _ = _setup()
    cols = ablation_columns(specs, ABLATION)
    cols["B"] = [c for c in cols["B"] if c != "cpu_t"]
    with pytest.raises(InformationSetError, match="cpu_t"):
        assert_equal_information([PERSISTENCE_INPUT], cols)
