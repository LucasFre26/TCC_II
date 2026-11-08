"""Execução ponta a ponta da V2 com arquivos sintéticos pequenos."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_prediction.data.load_genai import MissingArchiveError
from energy_prediction.experiments.run_v2 import ConfigError, execute, load_config
from synthetic import make_meta_frame, make_usage_frame, write_meta_archive, write_usage_archive

ROOT = Path(__file__).resolve().parents[1]
MACHINES = {f"m_{i}": (32 if i < 6 else 96) for i in range(12)}


def _project(tmp_path: Path, **overrides) -> Path:
    """Projeto de código mínimo com dados sintéticos; devolve o caminho do config."""
    proj = tmp_path / "proj"
    (proj / "config").mkdir(parents=True)
    (proj / "data/alibaba-v2018").mkdir(parents=True)
    usage = make_usage_frame(MACHINES, n_steps=1200, step=300.0, seed=4)
    t = usage["time_stamp"].to_numpy()
    usage["cpu_util_percent"] = np.clip(usage["cpu_util_percent"] + 10 * np.sin(2 * np.pi * t / 86400), 1, 99)
    write_usage_archive(proj / "data/alibaba-v2018/machine_usage.tar.gz", usage)
    write_meta_archive(proj / "data/alibaba-v2018/machine_meta.tar.gz", make_meta_frame(MACHINES))
    config = yaml.safe_load((ROOT / "config/v2.yaml").read_text(encoding="utf-8"))
    config["data"]["machines"]["sample_size"] = None
    config["data"]["grid_step"] = "auto"
    config["data"]["granularity"]["sample_rows"] = 100_000
    config["cohort"]["min_machines"] = 3
    config["validation"].update(n_folds=3, validation_size="8h", test_size="16h", min_train_size="24h")
    config["models"]["ridge"] = {"alphas": [1.0, 10.0]}
    config["models"]["xgboost"] = {
        "grid": {"max_depth": [2], "n_estimators": [20], "learning_rate": [0.1]},
        "fixed": {},
        "n_jobs": 1,
    }
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = proj / "config/v2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_shipped_config_is_accepted_by_validation():
    config = load_config(ROOT / "config/v2.yaml")
    assert config["models"]["names"] == ["persistence", "ridge", "xgboost"]
    assert set(config["features"]["ablation"]) == {"A", "B", "C"}


def test_missing_and_unknown_keys_fail_without_writing(tmp_path: Path):
    config_path = _project(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    broken = {k: v for k, v in config.items() if k != "seed"}
    config_path.write_text(yaml.safe_dump(broken))
    with pytest.raises(ConfigError, match="seed"):
        execute(config_path)
    unknown = {**config, "extra": 1}
    config_path.write_text(yaml.safe_dump(unknown))
    with pytest.raises(ConfigError, match="extra"):
        execute(config_path)
    nested = yaml.safe_load(yaml.safe_dump(config))
    nested["validation"]["nova"] = 1
    config_path.write_text(yaml.safe_dump(nested))
    with pytest.raises(ConfigError, match="validation.nova"):
        execute(config_path)
    assert not (tmp_path / "proj/results").exists()


def test_missing_data_gives_clear_error_and_no_output(tmp_path: Path):
    config_path = _project(tmp_path, **{"data.usage_archive": "data/alibaba-v2018/ausente.tar.gz"})
    with pytest.raises(MissingArchiveError, match="fetchData.sh"):
        execute(config_path)
    assert not (tmp_path / "proj/results").exists()


def test_disallowed_model_fails_before_reading_data(tmp_path: Path):
    config_path = _project(tmp_path, **{"models.names": ["persistence", "lstm"]})
    with pytest.raises(ValueError, match="lstm"):
        execute(config_path)
    assert not (tmp_path / "proj/results").exists()
    assert not (tmp_path / "proj/data/processed").exists()


def test_unknown_aggregation_strategy_fails_early(tmp_path: Path):
    config_path = _project(tmp_path, **{"aggregation.strategy": "media_simples"})
    with pytest.raises(ValueError, match="media_simples"):
        execute(config_path)
    assert not (tmp_path / "proj/data/processed").exists()


def test_output_dir_cannot_be_results_v1(tmp_path: Path):
    config_path = _project(tmp_path, output_dir="results/v1")
    with pytest.raises(ConfigError, match="results/v1"):
        execute(config_path)


def test_end_to_end_artifacts_v1_untouched_and_no_energy(tmp_path: Path, monkeypatch):
    config_path = _project(tmp_path)
    v1 = tmp_path / "proj/results/v1"
    v1.mkdir(parents=True)
    sentinel = v1 / "metrics.csv"
    sentinel.write_text("model,mae\npersistence,0.5\n")
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    summary = execute(config_path)
    out = tmp_path / "proj/results/v2"
    assert summary["output_dir"] == str(out)
    assert sentinel.read_text() == "model,mae\npersistence,0.5\n"
    assert not (tmp_path / "proj/results/v2.tmp").exists()

    for name in ("config_effective.yaml", "dataset_profile.json", "run_summary.json"):
        assert (out / name).is_file()
    for analysis in ("main", "cohort"):
        for name in ("metrics.csv", "metrics_summary.md", "folds.json", "predictions.csv", "selections.json", "series_profile.json", "error_extremes.csv", "bootstrap_ci.csv", "exogenous_lead.json"):
            assert (out / analysis / name).is_file(), (analysis, name)
        for figure in ("real_vs_predicted.png", "horizon_comparison.png", "ablation_comparison.png"):
            assert (out / analysis / "figures" / figure).stat().st_size > 0

    profile = json.loads((out / "dataset_profile.json").read_text())
    assert profile["step_seconds"] == 300.0
    assert profile["granularity"]["step_seconds"] == pytest.approx(300.0)
    assert profile["n_machines"] == 12
    effective = yaml.safe_load((out / "config_effective.yaml").read_text())
    assert effective["resolved"]["horizon_steps"] == [1, 3, 6]
    assert effective["resolved"]["purge_steps"] >= effective["resolved"]["required_purge_steps"]
    folds = json.loads((out / "main/folds.json").read_text())
    assert [f["split"] for f in folds] == ["fold_0", "fold_1", "fold_2", "test_final"]

    metrics = pd.read_csv(out / "main/metrics.csv")
    assert set(metrics["model"]) == {"persistence", "ridge", "xgboost"}
    assert set(metrics["config"]) == {"-", "A", "B", "C"}
    assert {"test_final", "aggregate_mean", "aggregate_std", "fold_0"} <= set(metrics["split"])
    cohort_profile = json.loads((out / "cohort/series_profile.json").read_text())
    assert cohort_profile["aggregation"]["cohort_size"] == 6

    produced = " ".join(p.name.lower() for p in out.rglob("*"))
    forbidden = ("energy", "energia", "power", "potencia", "pue", "lstm", "tcn")
    assert not any(word in produced for word in forbidden)
    assert not any(word in " ".join(metrics.columns).lower() for word in forbidden)
    assert (out / "run_summary.json").read_text().count("não foram calculados") == 1


def test_failed_run_leaves_no_partial_results(tmp_path: Path):
    config_path = _project(tmp_path, **{"validation.test_size": "400h"})
    with pytest.raises(ValueError, match="exige pelo menos"):
        execute(config_path)
    assert not (tmp_path / "proj/results").exists()


def test_repeated_run_gives_equivalent_metrics(tmp_path: Path):
    config_path = _project(tmp_path)
    execute(config_path)
    first = pd.read_csv(tmp_path / "proj/results/v2/main/metrics.csv")
    execute(config_path)
    second = pd.read_csv(tmp_path / "proj/results/v2/main/metrics.csv")
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-9, atol=1e-9)
    profile = json.loads((tmp_path / "proj/results/v2/dataset_profile.json").read_text())
    assert profile["cache"]["hit"] is True


def test_cohort_results_do_not_overwrite_main(tmp_path: Path):
    config_path = _project(tmp_path)
    execute(config_path)
    main = pd.read_csv(tmp_path / "proj/results/v2/main/metrics.csv")
    cohort = pd.read_csv(tmp_path / "proj/results/v2/cohort/metrics.csv")
    assert list(main.columns) == list(cohort.columns)
    assert len(main) == len(cohort)
    assert not np.allclose(main["mae"].to_numpy(), cohort["mae"].to_numpy())


def test_cohort_disabled_writes_only_main(tmp_path: Path):
    config_path = _project(tmp_path, **{"cohort.enabled": False})
    execute(config_path)
    assert (tmp_path / "proj/results/v2/main").is_dir()
    assert not (tmp_path / "proj/results/v2/cohort").exists()
