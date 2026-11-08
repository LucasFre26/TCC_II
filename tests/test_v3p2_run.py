"""V3 Parte 2 de ponta a ponta no smoke sintético: famílias, proteção, portões, reprodução e relatório."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("statsmodels")

from energy_prediction.experiments import run_v3p2  # noqa: E402
from energy_prediction.experiments.run_v3p2 import MissingFamilyError, PreregistrationChangedError, execute  # noqa: E402
from energy_prediction.experiments.v3p2_config import load_v3p2_config  # noqa: E402
from energy_prediction.experiments.v3p2_context import build_context  # noqa: E402
from energy_prediction.runtime.output import ProtectedOutputError  # noqa: E402
from v3p2_helpers import fast_config, write_project  # noqa: E402

FAMILIES = ["integrity", "baselines", "robustness", "hierarchical", "global_model", "regimes", "peaks", "calibration", "energy", "report"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v3p2")
    cfg = fast_config(**{"families.global_model.tcn.enabled": True, "families.global_model.tcn.gate_ratio": 0.0, "families.calibration.spci.enabled": True, "families.calibration.spci.budget_minutes": 0})
    manifest = execute(write_project(tmp, cfg), run_id="full")
    return Path(manifest["output_dir"]), manifest


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for k, v in value.items():
            found.add(str(k).lower())
            _keys(v, found)
    elif isinstance(value, list):
        for v in value:
            _keys(v, found)
    return found


def test_all_families_and_manifest(full_run):
    run, manifest = full_run
    assert list(manifest["families"]) == FAMILIES
    stored = json.loads((run / "manifest.json").read_text())
    assert stored["seed"] == 42 and stored["not_result"] and stored["reference_hashes"]["unchanged"]
    assert any(e["family"] == "pretrained" for e in stored["excluded"])
    assert any(g["gate"] == "tcn_global" and not g["passed"] for g in stored["gates"])
    assert any(e["item"] == "TCN global" for e in stored["excluded"])
    assert any(e["item"] == "SPCI" for e in stored["excluded"]) and any(g["gate"] == "spci_budget" and not g["passed"] for g in stored["gates"])


def test_blocks_panel_and_inherited_label(full_run):
    run, _ = full_run
    blocks = json.loads((run / "integrity/blocks.json").read_text())["blocks"]
    day7 = next(b for b in blocks if b["day"] == 7)
    assert day7["evaluation_range"] == [2010, 2297] and day7["inspection_number"] == 4
    assert json.loads((run / "integrity/panel_check.json").read_text())["passed"]


def test_baselines_one_prediction_per_origin(full_run):
    run, _ = full_run
    p = pd.read_csv(run / "baselines/predictions.csv")
    assert not p.duplicated(["model", "variant", "horizon_steps", "block", "phase", "origin_index"]).any()
    assert set(p["model"]) == {"persistence", "xgboost_v2", "xgboost_delta", "statespace", "hybrid"}
    paths = pd.read_csv(run / "baselines/paths.csv")
    assert sorted(paths["step"].unique()) == [1, 2, 3, 4, 5, 6]


def test_family_outputs(full_run):
    run, _ = full_run
    for rel in ("robustness/shift_bias.csv", "robustness/seed_spread.csv", "hierarchical/group_metrics.csv", "hierarchical/weight_drift.csv", "hierarchical/structure.csv", "global_model/sample_summary.csv", "regimes/confusion_detected_vs_realized.csv", "peaks/tradeoff.csv", "calibration/coverage_windows.csv", "energy/cumulative_24h.csv"):
        assert (run / rel).is_file(), rel
    summary = pd.read_csv(run / "global_model/sample_summary.csv")
    assert {"n_examples_train_h1", "n_machines_sampled", "n_distinct_origins_train", "effective_sample_estimate"} <= set(summary.columns)
    info = json.loads((run / "peaks/family_info.json").read_text())
    assert info["lambda0_reproduces_baseline_max_abs_diff"] < 1e-9
    assert "empíricas" in json.loads((run / "calibration/family_info.json").read_text())["statement"]
    reg = pd.read_csv(run / "regimes/predictions.csv")
    assert set(reg["regime"]) <= {"estavel", "subida", "queda", "alta_volatilidade"}


def test_energy_has_no_pue_nor_measured_consumption(full_run):
    run, _ = full_run
    info = json.loads((run / "energy/energy_info.json").read_text())
    assert info["max_abs_decomposition_closure_wh"] < 1e-6
    assert not any("pue" == k for k in _keys(info))
    for path in (run / "energy").glob("*.csv"):
        header = path.read_text(encoding="utf-8").splitlines()[0].lower()
        assert "pue" not in header.split(",") and "consumo medido" not in header
    assert (run / "energy/decomposition_heterogeneo_sintetico.csv").is_file()


def test_report_is_regenerable_from_files(full_run, tmp_path):
    from energy_prediction.experiments.v3p2_report import generate

    run, _ = full_run
    generate(run, tmp_path / "again")
    assert (tmp_path / "again/relatorio.md").read_text() == (run / "report/relatorio.md").read_text()
    text = (run / "report/relatorio.md").read_text()
    assert "quarta inspeção" in text and "confirmação" in text and "PUE não foi calculado" in text
    prim = pd.read_csv(run / "report/primary_comparisons.csv")
    assert {"H1", "H2", "H3", "H4", "H5", "H6", "H8a", "H8b", "H8c"} <= set(prim["hypothesis"])


def test_atlas_only_reads_family_files(full_run, tmp_path):
    import shutil

    from energy_prediction.analysis.v3p2_atlas import generate as atlas

    run, _ = full_run
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    before = {f: f.stat().st_mtime_ns for f in copy.rglob("*") if f.is_file()}
    summary = atlas(copy, tmp_path)
    assert "07_ranking_desenvolvimento" in summary["figures"] and "14_energia_sensibilidade" in summary["figures"]
    assert all(f.stat().st_mtime_ns == t for f, t in before.items())  # nada fora de atlas/ foi reescrito
    metrics = pd.read_csv(copy / "atlas/tables/metricas_consolidadas.csv")
    ref = metrics[(metrics["model"] == "xgboost_v2") & (metrics["variant"] == "A")]
    assert np.allclose(ref["gain_vs_xgboost_v2_pct"], 0.0)
    for record in json.loads((copy / "atlas/figures.json").read_text()):
        assert (copy / "atlas" / record["data"]).is_file() and record["sources"]


def test_missing_dependency_names_family(tmp_path):
    with pytest.raises(MissingFamilyError, match="baselines"):
        execute(write_project(tmp_path, fast_config()), family="energy", run_id="x")


def test_output_into_v3_refused(tmp_path):
    path = write_project(tmp_path, fast_config(run_root="results/v3/inside"))
    with pytest.raises(ProtectedOutputError):
        execute(path, family="integrity", run_id="x")
    assert not (path.parent.parent / "results").exists()


def test_exception_leaves_no_partial_family(tmp_path, monkeypatch):
    path = write_project(tmp_path, fast_config())
    execute(path, family="integrity", run_id="r")
    from energy_prediction.experiments.v3p2_families import baselines

    monkeypatch.setattr(baselines, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("falha")))
    with pytest.raises(RuntimeError):
        execute(path, family="baselines", run_id="r")
    run = path.parent.parent / "results/v3-part2-smoke/r"
    assert (run / "integrity").is_dir() and not (run / "baselines").exists() and not list(run.glob(".*tmp"))


def test_preregistration_change_interrupts(tmp_path, monkeypatch):
    path = write_project(tmp_path, fast_config(preregistration="docs/v3p2-preregistro.md"), preregistration=True)
    doc = path.parent.parent / "docs/v3p2-preregistro.md"
    from energy_prediction.experiments.v3p2_families import integrity

    original = integrity.run

    def tamper(*a, **k):
        out = original(*a, **k)
        doc.write_text(doc.read_text() + "\nalterado\n", encoding="utf-8")
        return out

    monkeypatch.setattr(integrity, "run", tamper)
    with pytest.raises(PreregistrationChangedError):
        execute(path, family="baselines", run_id="p")
    manifest = json.loads((path.parent.parent / "results/v3-part2-smoke/p/manifest.json").read_text())
    assert manifest["preregistration"]["sha256"]


def test_reproduction_detects_injected_deviation(tmp_path):
    from energy_prediction.experiments.v3p2_families.integrity import ReproductionError, reproduce_v2_xgboost
    from energy_prediction.features.cpu_features import target_column
    from energy_prediction.models.registry import XGBoostModel

    cfg = load_v3p2_config(write_project(tmp_path, fast_config()))
    ctx = build_context(cfg, tmp_path / "proj")
    block = next(b for b in ctx.blocks if b.role == "inherited_test")
    params = {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 30}
    X = ctx.table[ctx.columns["A"]].to_numpy(float)
    rows, selections = [], []
    for steps in ctx.steps:
        y = ctx.table[target_column(steps)].to_numpy(float)
        pred = XGBoostModel(params, seed=ctx.seed, n_jobs=1).fit(X[block.train], y[block.train]).predict(X[block.evaluation])
        selections.append({"model": "xgboost", "config": "A", "horizon_steps": steps, "best_params": params})
        rows.append(pd.DataFrame({"split": "test_final", "model": "xgboost", "config": "A", "horizon_steps": steps, "origin_index": ctx.table["origin_index"].to_numpy()[block.evaluation], "y_pred": pred.astype(float)}))
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "selections.json").write_text(json.dumps(selections))
    preds = pd.concat(rows)
    preds.to_csv(v2 / "predictions.csv", index=False)
    assert reproduce_v2_xgboost(ctx, v2, 1e-6)["passed"]
    preds.assign(y_pred=preds["y_pred"] + np.where(preds["origin_index"] == 2100, 0.5, 0.0)).to_csv(v2 / "predictions.csv", index=False)
    with pytest.raises(ReproductionError):
        reproduce_v2_xgboost(ctx, v2, 1e-6)




@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ctx")
    cfg = load_v3p2_config(write_project(tmp, fast_config()))
    return build_context(cfg, tmp / "proj")


def test_calendar_ablation_only_removes_calendar(ctx):
    from energy_prediction.features.cpu_features import CALENDAR_FEATURES

    columns = ctx.columns["A"]
    no_calendar = [c for c in columns if c not in CALENDAR_FEATURES]
    assert set(columns) - set(no_calendar) == {"sin_day", "cos_day"}


def test_network_windows_and_group_design_are_causal(ctx):
    import copy

    from energy_prediction.experiments.v3p2_families.hierarchical import group_design
    from energy_prediction.experiments.v3p2_families.networks import NetworkData

    data = NetworkData(ctx, 36)
    other = copy.copy(ctx)
    other.series = ctx.series.copy()
    other.series.loc[other.series["bin"] > 1000, "cpu"] += 50.0
    data2 = NetworkData(other, 36)
    row = int(np.flatnonzero(ctx.table["origin_index"].to_numpy() == 1000)[0])
    assert np.array_equal(data.X[row], data2.X[row])
    values = ctx.series["cpu"].to_numpy().copy()
    values[500:503] = np.nan
    X, _ = group_design(ctx, values, [1])
    values2 = values.copy()
    values2[503:] += 40.0
    X2, _ = group_design(ctx, values2, [1])
    assert np.array_equal(X[502], X2[502], equal_nan=True)


def test_global_training_pairs_stay_before_purge(ctx):
    from energy_prediction.features.machine_features import training_pairs

    origins = ctx.table["origin_index"].to_numpy()
    for block in ctx.blocks:
        o, _ = training_pairs(origins[block.train], np.arange(5), 2)
        assert o.max() + max(ctx.steps) < block.evaluation_range[0]


def test_hierarchical_candidate_ignores_evaluated_block(tmp_path, monkeypatch):
    from energy_prediction.experiments import v3_context

    cfg = fast_config(**{"protocol.development_days": [3, 4], "protocol.inherited_test_day": None, "protocol.expected_inherited_range": None})
    path_a = write_project(tmp_path / "a", cfg)
    execute(path_a, family="baselines", run_id="h")
    first = execute(path_a, family="hierarchical", run_id="h")
    original = v3_context.load_fleet

    def altered(cfg_, root):
        grid, meta, profile = original(cfg_, root)
        mask = grid["bin"].between(288 * 4, 288 * 5 - 1)
        grid.loc[mask, "cpu"] = np.clip(grid.loc[mask, "cpu"] + 15.0, 0, 100)
        return grid, meta, profile

    monkeypatch.setattr("energy_prediction.experiments.v3p2_context.load_fleet", altered)
    path_b = write_project(tmp_path / "b", cfg)
    execute(path_b, family="baselines", run_id="h")
    second = execute(path_b, family="hierarchical", run_id="h")
    a = pd.read_csv(Path(first["output_dir"]) / "hierarchical/candidates.csv")
    b = pd.read_csv(Path(second["output_dir"]) / "hierarchical/candidates.csv")
    pd.testing.assert_frame_equal(a[a["block"] == "day_4"].reset_index(drop=True), b[b["block"] == "day_4"].reset_index(drop=True))


@pytest.mark.cuda
def test_global_tcn_moves_only_batches_to_gpu(ctx):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA indisponível")
    from energy_prediction.features.machine_features import training_pairs
    from energy_prediction.models.neural.global_tcn import WindowSource, train_global_tcn

    panel = ctx.panel()
    cpu = panel.values["cpu"]
    source = WindowSource(cpu, panel.bins, ctx.step_seconds, 36, float(np.nanstd(cpu)))
    block = ctx.blocks[0]
    o, c = training_pairs(ctx.table["origin_index"].to_numpy()[block.inner.train], np.arange(panel.n_machines), 1)
    y = cpu[np.minimum(o + 1, len(cpu) - 1), c]
    ok = source.eligible(o, c) & np.isfinite(y)
    params = {"channels": 4, "dilations": [1, 2, 4, 8], "kernel_size": 3, "dropout": 0.1, "learning_rate": 0.001, "batch_size": 64, "examples_per_epoch": 512, "max_epochs": 1, "patience": 1}
    result = train_global_tcn(source, (o[ok], c[ok], y[ok]), None, params, device="cuda", seed=1, epochs=1)
    assert 0 < result["peak_cuda_bytes"] and not any(t.numel() == cpu.size for t in result["model"].parameters())
