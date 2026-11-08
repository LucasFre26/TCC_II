"""Unidades da V3 Parte 2: protocolo por dia, estatística, painel, grupos, reconciliação, pesos,
regimes, calibração, energia e configuração. Nada aqui lê o Alibaba v2018 nem exige GPU."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.data.synthetic_fleet import make_synthetic_fleet
from energy_prediction.energy.fan import FanModel
from energy_prediction.energy.heterogeneous import assign_substitutes
from energy_prediction.energy.routes import decompose, fixed_sets, group_means, grouped_route_power, horizon_energy_wh, machine_route_power, observed_paths
from energy_prediction.energy.substitute import SubstituteSourceError, load_substitutes
from energy_prediction.evaluation.adaptive_conformal import aci_offsets, conformal_quantile, rolling_offsets, sort_quantiles
from energy_prediction.evaluation.day_forward import evaluate_block, make_day_blocks
from energy_prediction.evaluation.day_stats import BETTER, NULL, WORSE, compare_by_day, decide, holm, stratified_bootstrap_means
from energy_prediction.evaluation.nested import InfeasibleInnerSplitError
from energy_prediction.evaluation.reconciliation import origin_shares, reconcile, shrunk_covariance
from energy_prediction.experiments.v3p2_config import V3P2ConfigError, validate_config
from energy_prediction.models.regime import FALL, RISE, STABLE, VOLATILE, classify, combine, fit_weights, regime_signals
from energy_prediction.models.sample_weights import recency_weights, rise_weights
from energy_prediction.preprocessing.cluster import aggregate_subset, aggregate_weighted, reconstruct_from_groups, resolve_machine_info
from energy_prediction.preprocessing.grouping import RESIDUAL, assign_groups, group_series
from energy_prediction.preprocessing.panel import PanelCheckError, build_panel, check_against_series, load_or_build_panel, train_statistics
from energy_prediction.runtime.output import PROTECTED_RESULTS, ProtectedOutputError, guard_output_root
from v3p2_helpers import ROOT, v3p2_config

ORIGINS = np.arange(12, 2298)


def blocks(**kw):
    params = dict(development_days=[3, 4, 5, 6], inherited_day=7, steps_per_day=288, max_horizon=6, purge_steps=18, inner_validation_steps=144, inner_purge_steps=18, min_inner_train_steps=576, expected_inherited_range=[2010, 2297])
    params.update(kw)
    return make_day_blocks(ORIGINS, **params)[0]


# --- 1.1 guarda de saída --------------------------------------------------------------------------

def test_guard_default_unchanged_and_v3_protected_when_asked(tmp_path):
    guard_output_root(tmp_path, tmp_path / "results/v3")  # padrão da V3 aceita results/v3
    with pytest.raises(ProtectedOutputError):
        guard_output_root(tmp_path, tmp_path / "results/v3/run", (*PROTECTED_RESULTS, "results/v3"))
    assert not (tmp_path / "results").exists()


def test_v3p2_modules_do_not_import_torch():
    code = "import sys; import energy_prediction.experiments.v3p2_config, energy_prediction.evaluation.day_forward, energy_prediction.preprocessing.panel, energy_prediction.energy.routes; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == "False"


# --- 1.3 / 1.4 configuração -----------------------------------------------------------------------

def test_unknown_key_rejected():
    with pytest.raises(V3P2ConfigError):
        validate_config(v3p2_config(**{"protocol.extra": 1}))


def test_pretrained_requires_provenance_and_rejects_alibaba():
    with pytest.raises(V3P2ConfigError):
        validate_config(v3p2_config(**{"pretrained.enabled": True}))
    cfg = v3p2_config(**{"pretrained.enabled": True, "pretrained.model": "x", "pretrained.version": "1", "pretrained.corpus_source": "Alibaba Cluster Trace 2018, Azure", "pretrained.excludes_alibaba_2018": True})
    with pytest.raises(V3P2ConfigError, match="Alibaba"):
        validate_config(cfg)


def test_full_mode_requires_preregistration_and_smoke_does_not():
    validate_config(v3p2_config())
    with pytest.raises(V3P2ConfigError):
        validate_config(v3p2_config(mode="full", preregistration=None, **{"data.source": "alibaba"}))


# --- 1.5 frota sintética --------------------------------------------------------------------------

def test_synthetic_defaults_unchanged():
    a = make_synthetic_fleet(n_machines=8, n_steps=400, seed=3)[0]
    b = make_synthetic_fleet(n_machines=8, n_steps=400, seed=3, common_factor=0.0, level_shift_last_day=0.0, failure_domains=0)[0]
    pd.testing.assert_frame_equal(a, b)


# --- 2.1 / 2.2 blocos por dia -----------------------------------------------------------------------

def test_inherited_block_matches_v3_test_and_targets_inside_day():
    bl = blocks()
    day7 = next(b for b in bl if b.day == 7)
    assert day7.evaluation_range == (2010, 2297) and day7.inspection == 4
    for b in bl:
        lo, hi = b.evaluation_range
        assert lo + 6 >= 288 * b.day and hi + 6 <= 288 * b.day + 287


def test_future_block_labels_do_not_change_earlier_blocks():
    bl = blocks()
    y = np.sin(np.arange(len(ORIGINS)) / 7.0)
    b3 = bl[0]

    def fp(g, train, predict, **_):
        return np.full(len(predict), y[train].mean() + g), {}

    first = evaluate_block(2, fp, y, b3, metric=lambda a, b: float(np.mean(np.abs(a - b))))
    day5 = (ORIGINS >= 288 * 5 - 6) & (ORIGINS <= 288 * 5 + 281)
    y2 = y.copy()
    y2[day5] += 100.0
    second = evaluate_block(2, lambda g, t, p, **_: (np.full(len(p), y2[t].mean() + g), {}), y2, b3, metric=lambda a, b: float(np.mean(np.abs(a - b))))
    assert first.best == second.best and np.allclose(first.block_prediction, second.block_prediction)
    assert ORIGINS[b3.train].max() + 6 < 288 * 5 - 6


def test_infeasible_block_reports_sizes():
    with pytest.raises(InfeasibleInnerSplitError, match="exigidos 5000"):
        blocks(min_inner_train_steps=5000)


# --- 2.3 estatística por dia ------------------------------------------------------------------------

def test_bootstrap_never_crosses_days():
    means = stratified_bootstrap_means({3: np.zeros(288), 4: np.ones(288)}, block=12, n_boot=200, seed=1)
    assert np.allclose(means, 0.5)


def test_decision_rule_four_cases():
    assert decide(-2, -1, [-1, -1, -1, 1], 3) == BETTER
    assert decide(1, 2, [1, 1, 1, -1], 3) == WORSE
    assert decide(-2, -1, [-1, -1, 1, 1], 3) == NULL
    assert decide(-1, 1, [-1, -1, -1, -1], 3) == NULL


def test_seed_reproduces_interval_and_holm_monotone():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"day": np.repeat([3, 4], 100), "origin_index": np.arange(200), "loss": rng.normal(1, 1, 200)})
    other = frame.assign(loss=frame["loss"] + 0.5)
    a = compare_by_day(frame, other, blocks=[12], primary_block=12, n_boot=300, seed=7, min_favorable_days=2, meta={})
    b = compare_by_day(frame, other, blocks=[12], primary_block=12, n_boot=300, seed=7, min_favorable_days=2, meta={})
    assert a[0][0]["ci_low"] == b[0][0]["ci_low"] and a[0][0]["decision"] == BETTER
    adjusted = holm([0.01, 0.04, 0.03])
    assert np.all(adjusted >= [0.01, 0.04, 0.03]) and adjusted.max() <= 1


# --- 2.4 – 2.6 painel e agregação por subconjunto ---------------------------------------------------

@pytest.fixture(scope="module")
def fleet():
    grid, meta, _ = make_synthetic_fleet(n_machines=10, n_steps=600, seed=5, failure_domains=3)
    info = resolve_machine_info(meta, active_statuses=["USING"])
    series, _ = aggregate_weighted(grid, info, step_seconds=300.0)
    population = sorted(set(grid["machine_id"]) & set(info["machine_id"]))
    panel = build_panel(grid, population, series["bin"].to_numpy(), key="t")
    weights = info.set_index("machine_id")["cpu_num"].reindex(panel.machines).to_numpy(float)
    return grid, meta, info, series, panel, weights


def test_panel_missing_interval_stays_missing_and_reproduces_series(fleet):
    grid, _, _, series, panel, weights = fleet
    col = panel.machines.index("m_001")
    assert np.isnan(panel.values["cpu"][100:105, col]).all()
    assert check_against_series(panel, weights, series, tolerance=1e-4)["passed"]
    broken = series.assign(cpu=series["cpu"] + 1.0)
    with pytest.raises(PanelCheckError):
        check_against_series(panel, weights, broken, tolerance=1e-4)


def test_panel_rebuilt_when_hash_differs(fleet, tmp_path):
    grid, _, _, series, panel, _ = fleet
    _, first = load_or_build_panel(grid, panel.machines, series["bin"].to_numpy(), source_key="a", cache_dir=tmp_path)
    _, again = load_or_build_panel(grid, panel.machines, series["bin"].to_numpy(), source_key="a", cache_dir=tmp_path)
    _, other = load_or_build_panel(grid, panel.machines, series["bin"].to_numpy(), source_key="b", cache_dir=tmp_path)
    assert first["rebuilt"] and not again["rebuilt"] and other["rebuilt"]


def test_train_statistics_ignore_evaluation_data(fleet):
    grid, _, _, series, panel, _ = fleet
    stats = train_statistics(panel, last_bin=300, step_seconds=300.0, min_coverage=0.5)
    altered = build_panel(grid.assign(cpu=np.where(grid["bin"] > 300, 99.0, grid["cpu"])), panel.machines, series["bin"].to_numpy(), key="x")
    pd.testing.assert_frame_equal(stats, train_statistics(altered, last_bin=300, step_seconds=300.0, min_coverage=0.5))


def test_subset_aggregation_example_and_partition_identity(fleet):
    grid = pd.DataFrame({"machine_id": ["a", "b"], "bin": [0, 0], "cpu": [10.0, 50.0]})
    info = pd.DataFrame({"machine_id": ["a", "b"], "cpu_num": [32.0, 96.0]})
    out = aggregate_subset(grid, info, ["a", "b"], step_seconds=300.0)
    assert out["cpu"].iloc[0] == pytest.approx(40.0) and out["weight_present"].iloc[0] == 128.0
    grid, _, info, series, _, _ = fleet
    rng = np.random.default_rng(1)
    machines = sorted(info["machine_id"])
    labels = rng.integers(0, 3, len(machines))
    parts = [aggregate_subset(grid, info, [m for m, g in zip(machines, labels) if g == k], step_seconds=300.0) for k in range(3)]
    rebuilt = reconstruct_from_groups(np.column_stack([p["cpu"] for p in parts]), np.column_stack([p["weight_present"] for p in parts]))
    assert np.nanmax(np.abs(rebuilt - series["cpu"].to_numpy())) < 1e-9


# --- 4.1 / 4.2 pesos e ablação ---------------------------------------------------------------------

def test_sample_weights():
    w = recency_weights(np.array([0, 144]), 144)
    assert w[0] / w[1] == pytest.approx(0.5)
    assert np.all(rise_weights(np.array([1.0, 5.0]), np.array([0.0, 0.0]), lam=0, sigma=2) == 1)
    assert list(rise_weights(np.array([1.0, 5.0]), np.array([0.0, 0.0]), lam=3, sigma=2)) == [1.0, 4.0]


def test_window_normalization_shift_invariance_and_causality():
    torch = pytest.importorskip("torch")
    from energy_prediction.models.neural.dlinear import DLinear
    from energy_prediction.models.neural.window_norm import WindowNormalized

    torch.manual_seed(0)
    model = WindowNormalized(DLinear(3, 36, 1)).eval()
    x = torch.randn(4, 3, 36)
    shifted = x.clone()
    shifted[:, 0, :] += 7.0
    with torch.no_grad():
        assert torch.allclose(model(shifted), model(x) + 7.0, atol=1e-4)


# --- 5.1 – 5.3 grupos e reconciliação ----------------------------------------------------------------

def test_grouping_partition_and_single_group(fleet):
    grid, meta, _, series, panel, weights = fleet
    stats = train_statistics(panel, last_bin=300, step_seconds=300.0, min_coverage=0.5)
    domains = meta.dropna(subset=["failure_domain_2"]).drop_duplicates("machine_id")[["machine_id", "failure_domain_2"]]
    for criterion, k in (("mean", 3), ("volatility", 2), ("profile", 2), ("failure_domain", None)):
        a = assign_groups(stats, criterion, k, domains=domains, min_group_size=2, n_init=2)
        assert sorted(a["machine_id"]) == sorted(panel.machines) and not a["machine_id"].duplicated().any()
    one = assign_groups(stats, "mean", 1)
    labels, values, present = group_series(panel, weights, one)
    ref = series["cpu"].to_numpy()
    assert labels == ["q00"] and np.nanmax(np.abs(values[:, 0] - ref)) < 1e-4


def test_residual_group_for_machines_without_training(fleet):
    _, _, _, _, panel, _ = fleet
    stats = train_statistics(panel, last_bin=300, step_seconds=300.0, min_coverage=1.1)
    assert (assign_groups(stats, "mean", 2)["group"] == RESIDUAL).all()


def test_reconciliation_coherence_and_mint_identity_equals_ols():
    rng = np.random.default_rng(2)
    total, groups = rng.normal(40, 1, 20), rng.normal(40, 3, (20, 3))
    shares = origin_shares(rng.uniform(1, 2, (20, 3)))
    for method in ("bottom_up", "ols", "mint_shrink"):
        W, _ = shrunk_covariance(rng.normal(0, 1, (50, 4)))
        t, g = reconcile(total, groups, shares, method, W)
        assert np.allclose(t, np.sum(g * shares, axis=1))
    a, _ = reconcile(total, groups, shares, "ols")
    b, _ = reconcile(total, groups, shares, "mint_shrink", np.eye(4))
    assert np.allclose(a, b)


# --- 6.1 – 6.3 modelo global -----------------------------------------------------------------------

def test_machine_features_are_causal_and_absent_machine_is_excluded(fleet):
    from energy_prediction.features.cpu_features import build_group_specs
    from energy_prediction.features.machine_features import build_machine_features, sample_machines, training_pairs
    from energy_prediction.models.global_xgb import aggregate_predictions

    _, meta, _, series, panel, weights = fleet
    spec = build_group_specs({"cpu": {"columns": ["cpu"], "lags": ["5min", "60min"], "windows": ["30min"]}}, 300.0)["cpu"]
    stats = train_statistics(panel, last_bin=300, step_seconds=300.0, min_coverage=0.5)
    domains = meta.drop_duplicates("machine_id")[["machine_id", "failure_domain_2"]]
    fleet_cpu = series["cpu"].to_numpy()
    feats = build_machine_features(panel, spec, stats, domains, fleet_cpu, step_seconds=300.0)
    altered = fleet_cpu.copy()
    altered[206] += 50.0
    feats2 = build_machine_features(panel, spec, stats, domains, altered, step_seconds=300.0)
    o, c = training_pairs(np.array([200]), np.arange(panel.n_machines), 1)
    assert np.array_equal(feats.design(o, c), feats2.design(o, c), equal_nan=True)
    assert sample_machines(stats, 5, seed=1) == sample_machines(stats, 5, seed=1)

    class Echo:
        def predict(self, X):
            return np.full(len(X), 50.0)

    col = panel.machines.index("m_001")
    agg, used, *_ = aggregate_predictions(feats, Echo(), np.array([102]), np.arange(panel.n_machines), weights, 1)
    assert used[0] == panel.n_machines - 1 and agg[0] == pytest.approx(50.0)
    assert np.isnan(feats.cpu[102, col])


# --- 7.1 / 7.2 regimes ------------------------------------------------------------------------------

def test_regime_detector_is_causal_with_precedence():
    cpu = np.r_[np.zeros(30), np.arange(30) * 2.0, np.zeros(30)]
    slope, vol = regime_signals(cpu, slope_steps=6, volatility_steps=12)
    cpu2 = cpu.copy()
    cpu2[51:] += 99
    slope2, vol2 = regime_signals(cpu2, slope_steps=6, volatility_steps=12)
    assert slope[50] == slope2[50] and vol[50] == vol2[50]
    limits = {"volatility": 0.5, "rise": 1.0, "fall": -1.0}
    out = classify(np.array([5.0, 5.0, -5.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.arange(4), limits)
    assert list(out) == [VOLATILE, RISE, FALL, STABLE]


def test_combiner_weights_sum_to_one_and_fallback():
    rng = np.random.default_rng(3)
    y, P = rng.normal(0, 1, 40), rng.normal(0, 1, (40, 3))
    regimes = np.array([STABLE] * 35 + [RISE] * 5, dtype=object)
    for method in ("select", "inverse_mae", "mean", "global_inverse_mae"):
        w, fallback = fit_weights(y, P, regimes, method=method, min_origins=10)
        assert all(np.isclose(v.sum(), 1.0) for v in w.values())
        if method in ("select", "inverse_mae"):
            assert RISE in fallback
        assert combine(P, regimes, w).shape == (40,)


def test_lambda_rule_respects_mae_constraint():
    from energy_prediction.experiments.v3p2_families.peaks import choose_lambda

    chosen, allowed = choose_lambda({0.0: (1.0, 3.0), 1.0: (1.04, 2.0), 3.0: (1.08, 1.0)}, 0.05)
    assert chosen == 1.0 and 3.0 not in allowed


# --- 8.1 / 8.2 calibração ---------------------------------------------------------------------------

def test_quantile_crossings_are_counted_and_sorted():
    Q, crossed = sort_quantiles(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 4.0]]))
    assert crossed == 1 and np.all(np.diff(Q, axis=1) >= 0)


@pytest.mark.parametrize("method", ["rolling", "aci"])
def test_delayed_feedback_and_label_tampering(method):
    rng = np.random.default_rng(4)
    initial, scores, origins, h = rng.normal(0, 1, 50), rng.normal(0, 1, 40), np.arange(100, 140), 6

    def offsets(s):
        if method == "rolling":
            return rolling_offsets(initial, s, origins, h, 0.1, 30)
        return aci_offsets(initial, s, origins, h, 0.1, 0.05)[0]

    base = offsets(scores)
    tampered = scores.copy()
    tampered[10] += 50.0  # alvo da origem 110 alterado
    after = offsets(tampered)
    assert np.allclose(base[: 10 + h], after[: 10 + h])
    assert not np.allclose(base, after)
    assert conformal_quantile(np.arange(10.0), 0.0) == np.inf


# --- 9.1 – 9.4 energia ------------------------------------------------------------------------------

def test_substitute_without_source_fails(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("primary: x\nlimitation: carga e hardware\nsubstitutes:\n  - {result_id: x, system: s, published: '2017', p_idle_w: 1, p_busy_w: 2, threads: 96, role: r}\n", encoding="utf-8")
    with pytest.raises(SubstituteSourceError):
        load_substitutes(bad)
    record, entries, fans = load_substitutes(ROOT / "config/specpower_substitutes.yaml")
    assert record["primary"] in fans and len(entries) == 3


def test_routes_collapse_for_linear_model_and_order_for_concave():
    rng = np.random.default_rng(5)
    cpu = rng.uniform(10, 90, (20, 6))
    origins = np.arange(0, 10)
    F = fixed_sets(cpu, origins, 6)
    U = observed_paths(cpu, origins, 6)
    w = np.full(6, 96.0)
    zero = np.zeros(6, dtype=object)
    subs = np.full(6, "s", dtype=object)
    fans = {"s": FanModel(50.0, 400.0)}
    mean = group_means(U, F, w, zero, [0])
    for r, check in ((1.0, "equal"), (1.4, "mean_ge")):
        ref, _ = machine_route_power(U, F, subs, fans, r)
        route, _ = grouped_route_power(mean, F, zero, [0], subs, fans, r)
        if check == "equal":
            assert np.allclose(ref, route)
        else:
            assert np.all(route >= ref - 1e-9)
    e = horizon_energy_wh(np.full((2, 7), 150.0), 6, 300.0)
    assert np.allclose(e, 150.0 * 1800 / 3600)
    d = decompose(np.array([1.0]), np.array([2.0]), np.array([4.0]))
    assert d["closure"][0] == 0 and d["total"][0] == 3.0


def test_heterogeneous_assignment_is_deterministic():
    a = assign_substitutes(["m_1", "m_2", "m_3"], ["x", "y"])
    assert a == assign_substitutes(["m_1", "m_2", "m_3"], ["x", "y"]) and set(a.values()) <= {"x", "y"}


def test_power_bounds_are_ordered_for_registered_exponents():
    fan = FanModel(49.6, 461.0)
    low, high = np.array([0.1, 0.4, 0.7]), np.array([0.2, 0.6, 1.0])
    for r in (1.0, 1.4, 1.8):
        assert np.all(fan.power_from_u(low, r) <= fan.power_from_u(high, r))
