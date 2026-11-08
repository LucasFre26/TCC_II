"""V3 de ponta a ponta no smoke sintético: famílias, energia, relatório, falhas e smoke real ausente."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("statsmodels")

from energy_prediction.data.load_genai import MissingArchiveError  # noqa: E402
from energy_prediction.energy.substitute import SubstituteSourceError  # noqa: E402
from energy_prediction.experiments import run_v3  # noqa: E402
from energy_prediction.experiments.run_v3 import execute  # noqa: E402
from energy_prediction.models.v3_registry import UnknownV3ModelError  # noqa: E402
from v3_helpers import ROOT, smoke_config, write_project  # noqa: E402


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full")
    config_path = write_project(tmp, smoke_config())
    manifest = execute(config_path, run_id="full")
    return Path(manifest["output_dir"]), manifest


def _all_keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key).lower())
            _all_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _all_keys(item, found)
    return found


def test_full_run_writes_every_family_manifest_and_report(full_run):
    run, manifest = full_run
    assert list(manifest["families"]) == ["integrity", "residual", "neural", "quantile", "energy"]
    stored = json.loads((run / "manifest.json").read_text())
    assert stored["seed"] == 42 and stored["device"]["effective"] == "cpu" and stored["config_sha256"]
    assert stored["data"]["source"] == "synthetic"
    assert any(e["family"] == "energy" and "ridge_v2" in e["item"] for e in stored["excluded"])
    for name in ("execution_report.md", "limitations.md"):
        assert (run / "report" / name).is_file()
    limitations = (run / "report/limitations.md").read_text()
    for phrase in ("já inspecionado na V2", "coorte", "servidor substituto", "herdam", "empírica", "PUE não foi calculado"):
        assert phrase in limitations
    figures = {p.name for p in (run / "report/figures").glob("*.png")}
    assert {"observed_vs_predicted_A.png", "mae_by_horizon_A.png", "fan_power_curves.png", "energy_error_vs_r.png", "energy_reference_vs_projected.png", "residual_histograms.png"} <= figures
    assert not list(run.glob(".*.tmp"))


def test_energy_outputs_are_modeled_not_measured_and_have_no_pue(full_run):
    run, _ = full_run
    energy = run / "energy"
    info = json.loads((energy / "energy_info.json").read_text())
    assert not any("pue" in key for key in _all_keys(info))
    assert "modelada" in info["quantity"] and "consumo medido" not in info["quantity"]
    assert "Não é erro contra medidor" in info["propagation_statement"]
    assert "intervalo de confiança" not in json.dumps(info, ensure_ascii=False).lower()
    for path in energy.glob("*.csv"):
        columns = " ".join(pd.read_csv(path, nrows=1).columns).lower()
        assert "pue" not in columns and "measured" not in columns, path
    assert not any("pue" in p.name.lower() for p in run.rglob("*"))
    summary = pd.read_csv(energy / "energy_summary.csv")
    assert (summary.groupby(["model", "horizon_steps"])["r"].nunique() == 3).all()
    assert set(summary["quantity"]) == {info["quantity"]}


def test_inherited_steps_are_marked_and_trajectories_complete(full_run):
    run, _ = full_run
    info = json.loads((run / "energy/energy_info.json").read_text())
    assert info["inheritance"]["map_step_to_scored_horizon"] == {"1": 1, "2": 3, "3": 3, "4": 6, "5": 6, "6": 6}
    trajectories = pd.read_csv(run / "energy/trajectories.csv")
    delta = trajectories[trajectories["model"] == "xgboost_delta"]
    assert set(delta.loc[delta["inherited_params"], "step"]) == {2, 4, 5}
    assert not trajectories.loc[trajectories["model"].isin(["persistence", "dlinear"]), "inherited_params"].any()
    for model in ("xgboost_delta", "hybrid", "statespace"):
        diff = info["availability"][model]["consistency"]["max_abs_difference_vs_stored_test_predictions"]
        assert max(diff.values()) < 1e-6
    effect = pd.read_csv(run / "energy/aggregate_effect.csv")
    assert {"p_of_mean_w", "mean_of_p_w", "sum_of_p_w", "jensen_gap_w", "unstable_approximation", "second_order_mean_of_p_w_diagnostic"} <= set(effect.columns)


def test_report_regenerates_without_fitting(full_run):
    run, _ = full_run
    code = (
        "import sys\n"
        "from energy_prediction.experiments.v3_report import generate\n"
        f"generate({str(run)!r})\n"
        "bad = [m for m in sys.modules if m.startswith(('xgboost', 'torch', 'statsmodels', 'sklearn', 'energy_prediction.models', 'energy_prediction.experiments.v3_families'))]\n"
        "print(bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": str(ROOT / "src")})
    assert out.stdout.strip().splitlines()[-1] == "[]"
    assert (run / "report/execution_report.md").read_text().startswith("# Relatório de execução da V3")


def test_failure_in_one_family_keeps_completed_ones(tmp_path: Path, monkeypatch):
    config_path = write_project(tmp_path, smoke_config())
    execute(config_path, family="quantile", run_id="r")
    real_runner = run_v3._runner

    def broken(family):
        if family == "neural":
            def fail(*_, **__):
                raise RuntimeError("falha simulada na família neural")
            return fail
        return real_runner(family)

    monkeypatch.setattr(run_v3, "_runner", broken)
    with pytest.raises(RuntimeError, match="simulada"):
        execute(config_path, family="neural", run_id="r")
    run = tmp_path / "proj/results/v3-smoke/r"
    assert (run / "quantile/metrics.csv").is_file()
    assert not (run / "neural").exists() and not (run / ".neural.tmp").exists()
    assert list(json.loads((run / "manifest.json").read_text())["families"]) == ["integrity", "quantile"]


def test_energy_without_source_fails_and_writes_no_energy(tmp_path: Path):
    config_path = write_project(tmp_path, smoke_config(), substitute=False)
    skeleton = yaml.safe_load((ROOT / "config/v3.yaml").read_text())  # só para garantir que o caminho é o mesmo
    assert skeleton["families"]["energy"]["substitute"] == "config/specpower_substitute.yaml"
    (tmp_path / "proj/config/specpower_substitute.yaml").write_text("result_id: null\np_idle_w: 50\np_busy_w: 400\n")
    with pytest.raises(SubstituteSourceError):
        execute(config_path, family="energy", run_id="e")
    assert not (tmp_path / "proj/results/v3-smoke/e/energy").exists()


def test_unknown_model_fails_before_any_output(tmp_path: Path):
    config = smoke_config()
    config["families"]["neural"]["models"] = ["dlinear", "transformer"]
    config_path = write_project(tmp_path, config)
    with pytest.raises(UnknownV3ModelError):
        execute(config_path, family="neural", run_id="u")
    assert not (tmp_path / "proj/results").exists()


def test_real_smoke_without_archive_fails_explicitly(tmp_path: Path):
    config = yaml.safe_load((ROOT / "config/v3-smoke-real.yaml").read_text(encoding="utf-8"))
    config_path = write_project(tmp_path, config)
    with pytest.raises(MissingArchiveError, match="machine_usage"):
        execute(config_path, family="integrity", run_id="s")
    assert not (tmp_path / "proj/results").exists()
    assert not (tmp_path / "proj/data/processed").exists()


def test_synthetic_suite_never_reads_the_archive(monkeypatch, tmp_path: Path):
    from energy_prediction.data import alibaba

    def forbidden(*_, **__):
        raise AssertionError("o smoke sintético não pode ler machine_usage")

    monkeypatch.setattr(alibaba, "iter_usage_chunks", forbidden)
    monkeypatch.setattr(alibaba, "file_sha256", forbidden)
    execute(write_project(tmp_path, smoke_config()), family="integrity", run_id="n")
