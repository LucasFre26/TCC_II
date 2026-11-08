"""V3, grupo 3: atributos transversais (configuração D) e casos fechados de Fan, Jensen e integral."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_prediction.energy.aggregate_effect import aggregate_effect
from energy_prediction.energy.fan import FanModel, shape_second_derivative
from energy_prediction.energy.integrate import IncompleteTrajectoryError, horizon_energies, trapezoid_energy
from energy_prediction.energy.substitute import SubstituteSourceError, load_substitute
from energy_prediction.experiments import v3_context
from energy_prediction.experiments.v3_context import build_context
from energy_prediction.features.distribution import distribution_frame
from v3_helpers import ROOT, smoke_config

DIST = dict(elevated_threshold=80, change_steps=6, skewness=True, skewness_max_undefined_fraction=0.5)


def _grid():
    rows = []
    for b in range(20):
        for i, m in enumerate(["a", "b", "c", "d"]):
            rows.append({"machine_id": m, "bin": b, "cpu": 10.0 + 25 * i + b})
    return pd.DataFrame(rows)


def test_distribution_uses_only_present_machines_at_the_origin():
    grid = _grid()
    grid.loc[(grid["machine_id"] == "d") & (grid["bin"] == 5), "cpu"] = np.nan
    frame, info = distribution_frame(grid, ["a", "b", "c", "d"], np.arange(20), **DIST)
    row = frame.set_index("bin").loc[5]
    present = np.array([15.0, 40.0, 65.0])
    assert row["dist_valid_frac"] == pytest.approx(0.75)
    assert row["dist_std"] == pytest.approx(present.std())
    assert row["dist_p50"] == pytest.approx(40.0)
    assert row["dist_high_frac"] == pytest.approx(0.0)
    assert frame.set_index("bin").loc[12, "dist_high_frac"] == pytest.approx(0.25)  # 85+12 >= 80
    assert info["elevated_threshold"] == 80


def test_future_values_do_not_change_the_origin_vector():
    grid = _grid()
    before, _ = distribution_frame(grid, ["a", "b", "c", "d"], np.arange(20), **DIST)
    changed = grid.copy()
    changed.loc[changed["bin"] > 10, "cpu"] = 99.0  # muda percentil e agregado futuros
    after, _ = distribution_frame(changed, ["a", "b", "c", "d"], np.arange(20), **DIST)
    pd.testing.assert_frame_equal(before[before["bin"] <= 10], after[after["bin"] <= 10])


def test_threshold_is_a_constant_not_estimated():
    grid = _grid()
    low, _ = distribution_frame(grid, ["a", "b", "c", "d"], np.arange(20), **{**DIST, "elevated_threshold": 50})
    assert low.set_index("bin").loc[0, "dist_high_frac"] == pytest.approx(0.5)  # 60 e 85 >= 50


def test_skewness_dropped_when_mostly_undefined():
    grid = _grid()
    two = grid[grid["machine_id"].isin(["a", "b"])]
    frame, info = distribution_frame(two, ["a", "b"], np.arange(20), **DIST)
    assert "dist_skew" not in frame and info["skewness"]["included"] is False and "reason" in info["skewness"]


def test_missing_d_value_keeps_origin_in_abc_and_drops_it_from_d(monkeypatch):
    original = v3_context.distribution_frame

    def with_hole(*args, **kwargs):
        frame, info = original(*args, **kwargs)
        frame.loc[frame["bin"] == 600, "dist_p90"] = np.nan
        return frame, info

    monkeypatch.setattr(v3_context, "distribution_frame", with_hole)
    ctx = build_context(smoke_config(), ROOT)
    assert 600 in set(ctx.table["origin_index"])
    assert 600 not in set(ctx.d_table["origin_index"])
    assert len(ctx.d_table) == len(ctx.table) - 1
    assert ctx.d_splits.test_range == ctx.splits.test_range
    extra = [c for c in ctx.columns["D"] if c not in ctx.columns["C"]]
    assert set(ctx.columns["C"]) < set(ctx.columns["D"]) and all(c.startswith("dist_") for c in extra)


def test_fan_linear_recovery_and_invalid_counting():
    fan = FanModel(p_idle=100.0, p_busy=200.0)
    power, invalid = fan.power([0.0, 50.0, 100.0], 1.0)
    assert power.tolist() == [100.0, 150.0, 200.0] and invalid == 0
    power, invalid = fan.power([-1.0, 101.0, 50.0, np.nan], 1.4)
    assert np.isnan(power[:2]).all() and invalid == 2 and np.isnan(power[3])
    for r in (1.0, 1.4, 1.8):
        p, _ = fan.power([0.0, 100.0], r)
        assert p.tolist() == [100.0, 200.0]


def test_two_machine_jensen_gap_is_computed_numerically():
    fan = FanModel(p_idle=100.0, p_busy=200.0)
    grid = pd.DataFrame({"machine_id": ["a", "b"], "bin": [0, 0], "cpu": [20.0, 80.0]})
    info = pd.DataFrame({"machine_id": ["a", "b"], "cpu_num": [96.0, 96.0], "cpu_num_conflict": False, "active": True})
    table, summary = aggregate_effect(grid, info, fan, [1.0, 1.8], unstable_below=0.05)
    linear = table[table["r"] == 1.0].iloc[0]
    assert linear["p_of_mean_w"] == pytest.approx(linear["mean_of_p_w"])
    curved = table[table["r"] == 1.8].iloc[0]
    p = lambda u: 100 + 100 * (2 * u - u**1.8)  # noqa: E731
    assert curved["p_of_mean_w"] == pytest.approx(p(0.5))
    assert curved["mean_of_p_w"] == pytest.approx((p(0.2) + p(0.8)) / 2)
    assert curved["sum_of_p_w"] == pytest.approx(p(0.2) + p(0.8))
    assert curved["jensen_gap_w"] < 0
    approx = p(0.5) + 0.5 * 100 * shape_second_derivative(np.array([0.5]), 1.8)[0] * 0.09
    assert curved["second_order_mean_of_p_w_diagnostic"] == pytest.approx(approx)


def test_unstable_intervals_are_counted_apart():
    fan = FanModel(p_idle=100.0, p_busy=200.0)
    grid = pd.DataFrame({"machine_id": ["a", "b", "a", "b"], "bin": [0, 0, 1, 1], "cpu": [1.0, 3.0, 40.0, 60.0]})
    info = pd.DataFrame({"machine_id": ["a", "b"], "cpu_num": [96.0, 96.0], "cpu_num_conflict": False, "active": True})
    table, summary = aggregate_effect(grid, info, fan, [1.0, 1.4], unstable_below=0.05)
    counts = {e["r"]: e["unstable_bins"] for e in summary["per_exponent"]}
    assert counts == {1.0: 0, 1.4: 1}


def test_trapezoid_constant_path_and_incomplete_horizon():
    assert trapezoid_energy(np.full(7, 150.0), 300.0) == pytest.approx(150.0 * 6 * 300.0)
    with pytest.raises(IncompleteTrajectoryError):
        trapezoid_energy([150.0, np.nan, 150.0], 300.0)
    origin = np.array([100.0, 100.0])
    steps = np.array([[100.0, 100.0, 100.0], [100.0, np.nan, 100.0]])
    energies = horizon_energies(origin, steps, [1, 3], 300.0)
    assert energies[1].tolist() == [30000.0, 30000.0]
    assert energies[3][0] == pytest.approx(90000.0) and np.isnan(energies[3][1])
    left = horizon_energies(np.array([200.0]), np.array([[100.0]]), [1], 300.0)[1]
    assert left[0] == pytest.approx(300.0 * 150.0)  # extremo esquerdo na origem


def test_substitute_without_reference_fails(tmp_path: Path):
    skeleton = {k: None for k in ("result_id", "source", "system", "published", "p_idle_w", "p_busy_w", "limitation")}
    path = tmp_path / "sub.yaml"
    path.write_text(yaml.safe_dump(skeleton))
    with pytest.raises(SubstituteSourceError, match="nenhuma energia"):
        load_substitute(path)
    watts_only = {**skeleton, "p_idle_w": 50, "p_busy_w": 400}
    path.write_text(yaml.safe_dump(watts_only))
    with pytest.raises(SubstituteSourceError):
        load_substitute(path)


def test_shipped_substitute_has_source_and_limitation():
    record, fan = load_substitute(ROOT / "config/specpower_substitute.yaml")
    assert record["result_id"].startswith("power_ssj2008-") and record["source"].startswith("https://www.spec.org/")
    assert record["system"] and str(record["published"]).startswith(("2017", "2018")) and record["sockets"] == 2
    assert 0 < fan.p_idle < fan.p_busy
    assert "hardware" in record["limitation"] and "carga" in record["limitation"]
