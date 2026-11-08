"""Testes da prova de conceito. Não escrevem em data/ nem acessam a rede."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.data.load_genai import (
    MissingArchiveError,
    file_sha256,
    load_metric_archive,
    require_archives,
)
from energy_prediction.evaluation.metrics import mae, mase, rmse, skill_score
from energy_prediction.experiments.run_v1 import execute
from energy_prediction.features.supervised import build_supervised, split_origins
from energy_prediction.models.baselines import moving_average_forecast, persistence_forecast
from energy_prediction.models.classical import predict_hist_gradient_boosting, select_ridge_alpha
from energy_prediction.preprocessing.aggregate import (
    DuplicateTimestampError,
    aggregate_by_timestamp,
    join_series,
    sort_unique,
)
from energy_prediction.preprocessing.checks import IrregularStepError, assert_expected_step
from energy_prediction.preprocessing.profile import profile_joined
from energy_prediction.preprocessing.scale import apply_scaler, fit_scaler
from energy_prediction.visualization.plots import plot_gpu_series, plot_horizon_forecasts

ROOT = Path(__file__).resolve().parents[1]
GPU_ARCHIVE = ROOT / "data/clusterdata-master/cluster-trace-v2026-GenAI/pod_gpu_duty_cycle_anon.tar.gz"
MEMORY_ARCHIVE = ROOT / "data/clusterdata-master/cluster-trace-v2026-GenAI/pod_memory_util_anon.tar.gz"
HORIZONS = [1, 5, 15]
GPU_LAGS = [1, 2, 3, 6, 12]
MEMORY_LAGS = [0, 1, 6]
WINDOWS = [6, 12]


def _regular_frame(n: int, step: float = 57.0) -> pd.DataFrame:
    index = np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": index * step,
            "gpu": index.astype(float),
            "memory": index.astype(float) / 10.0,
            "n_containers_gpu": np.ones(n),
            "n_containers_memory": np.ones(n),
        }
    )


def test_missing_archive_does_not_create_output(tmp_path: Path):
    missing = tmp_path / "ausente.tar.gz"
    output = tmp_path / "results"
    with pytest.raises(MissingArchiveError):
        require_archives(missing, missing)
    assert not output.exists()


def test_real_archives_keep_hash_and_row_counts():
    if not GPU_ARCHIVE.is_file() or not MEMORY_ARCHIVE.is_file():
        pytest.skip("Arquivos GenAI ausentes.")
    before = (file_sha256(GPU_ARCHIVE), file_sha256(MEMORY_ARCHIVE))
    gpu = load_metric_archive(GPU_ARCHIVE)
    memory = load_metric_archive(MEMORY_ARCHIVE)
    after = (file_sha256(GPU_ARCHIVE), file_sha256(MEMORY_ARCHIVE))
    assert before == after
    assert len(gpu) == 157417
    assert len(memory) == 214890
    assert list(gpu.columns) == ["timestamp", "value", "container_id"]


def test_aggregate_mean_and_count():
    frame = pd.DataFrame(
        {
            "timestamp": [57.0, 57.0, 114.0],
            "value": [1.0, 3.0, 10.0],
            "container_id": ["a", "b", "a"],
        }
    )
    aggregated = aggregate_by_timestamp(frame)
    row = aggregated.loc[aggregated["timestamp"] == 57.0].iloc[0]
    assert row["value"] == pytest.approx(2.0)
    assert row["n_containers"] == 2


def test_inner_join_counts_exclusive_timestamp():
    gpu = pd.DataFrame({"timestamp": [0.0, 57.0], "value": [1.0, 2.0], "n_containers": [1, 1]})
    memory = pd.DataFrame({"timestamp": [57.0, 114.0], "value": [0.1, 0.2], "n_containers": [1, 1]})
    joined, excluded = join_series(gpu, memory)
    assert excluded == 2
    assert list(joined["timestamp"]) == [57.0]
    assert joined.iloc[0]["gpu"] == pytest.approx(2.0)
    assert joined.iloc[0]["memory"] == pytest.approx(0.1)


def test_zero_and_negative_memory_are_kept():
    gpu = pd.DataFrame({"timestamp": [0.0], "value": [0.0], "n_containers": [1]})
    memory = pd.DataFrame({"timestamp": [0.0], "value": [-0.2], "n_containers": [1]})
    joined, excluded = join_series(gpu, memory)
    assert excluded == 0
    assert joined.iloc[0]["gpu"] == 0.0
    assert joined.iloc[0]["memory"] == pytest.approx(-0.2)


def test_profile_three_timestamps():
    frame = pd.DataFrame(
        {
            "timestamp": [0.0, 57.0, 114.0],
            "gpu": [0.0, 10.0, 20.0],
            "memory": [0.1, 0.2, 0.3],
            "n_containers_gpu": [1, 1, 1],
            "n_containers_memory": [1, 1, 1],
        }
    )
    profile = profile_joined(frame, excluded_timestamps=0)
    assert profile["n_timestamps"] == 3
    assert profile["median_step_seconds"] == pytest.approx(57.0)
    assert profile["gpu_zero_fraction"] == pytest.approx(1 / 3)
    assert profile["gpu_min"] == 0.0
    assert profile["gpu_median"] == 10.0
    assert profile["gpu_mean"] == pytest.approx(10.0)
    assert profile["gpu_max"] == 20.0
    assert profile["contemporaneous_correlation"] == pytest.approx(1.0)
    assert profile["gpu_outside_0_100"] == 0


def test_step_60_fails_and_57_passes():
    with pytest.raises(IrregularStepError):
        assert_expected_step(pd.Series([0.0, 60.0, 120.0]), 57.0)
    assert assert_expected_step(pd.Series([0.0, 57.0, 114.0]), 57.0) == pytest.approx(57.0)


def test_duplicate_timestamp_is_rejected():
    frame = pd.DataFrame(
        {
            "timestamp": [10.0, 10.0],
            "gpu": [1.0, 2.0],
            "memory": [0.1, 0.2],
            "n_containers_gpu": [1, 1],
            "n_containers_memory": [1, 1],
        }
    )
    with pytest.raises(DuplicateTimestampError):
        sort_unique(frame)
    ordered = sort_unique(
        pd.DataFrame(
            {
                "timestamp": [114.0, 0.0, 57.0],
                "gpu": [3.0, 1.0, 2.0],
                "memory": [0.3, 0.1, 0.2],
                "n_containers_gpu": [1, 1, 1],
                "n_containers_memory": [1, 1, 1],
            }
        )
    )
    assert list(ordered["timestamp"]) == [0.0, 57.0, 114.0]


def test_scaler_uses_train_mean_on_constant_column():
    train = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    test = pd.DataFrame({"x": [8.0]})
    scaler = fit_scaler(train)
    assert apply_scaler(scaler, train).ravel() == pytest.approx([0.0, 0.0, 0.0])
    assert apply_scaler(scaler, test).ravel() == pytest.approx([3.0])


def test_direct_target_drops_last_origin_for_horizon_1():
    frame = _regular_frame(40)
    table = build_supervised(
        frame,
        gpu_lags=[1],
        memory_lags=[0],
        rolling_windows=[6],
        horizons=[1],
    )
    assert table["y_1"].iloc[0] == pytest.approx(frame["gpu"].iloc[int(table["origin_index"].iloc[0]) + 1])
    assert table["origin_index"].max() == 38
    assert 39 not in set(table["origin_index"])


def test_future_change_does_not_alter_features():
    frame = _regular_frame(80)
    original = build_supervised(
        frame,
        gpu_lags=GPU_LAGS,
        memory_lags=MEMORY_LAGS,
        rolling_windows=WINDOWS,
        horizons=HORIZONS,
    )
    changed = frame.copy()
    origin = 20
    changed.loc[origin + 15, "gpu"] = 999.0
    changed.loc[origin + 1, "gpu"] = 999.0
    changed.loc[origin + 15, "memory"] = 999.0
    updated = build_supervised(
        changed,
        gpu_lags=GPU_LAGS,
        memory_lags=MEMORY_LAGS,
        rolling_windows=WINDOWS,
        horizons=HORIZONS,
    )
    feature_columns = [
        column
        for column in original.columns
        if column.startswith(("gpu_lag_", "gpu_roll_", "memory_lag_", "sin_", "cos_"))
    ]
    left = original.loc[original["origin_index"] == origin, feature_columns].iloc[0]
    right = updated.loc[updated["origin_index"] == origin, feature_columns].iloc[0]
    pd.testing.assert_series_equal(left, right)


def test_split_keeps_test_after_train_and_outside_train_targets():
    table = build_supervised(
        _regular_frame(127),
        gpu_lags=GPU_LAGS,
        memory_lags=MEMORY_LAGS,
        rolling_windows=WINDOWS,
        horizons=HORIZONS,
    )
    parts = split_origins(len(table), test_fraction=0.2, validation_fraction=0.2, purge_steps=27)
    train_origins = table.iloc[parts["train"]]["origin_index"].to_numpy()
    test_origins = table.iloc[parts["test"]]["origin_index"].to_numpy()
    assert train_origins.max() < test_origins.min()
    first_test = test_origins.min()
    assert np.all(train_origins + 15 < first_test)
    assert np.all(train_origins < first_test)
    assert len(parts["purge"]) == 27


def test_persistence_and_moving_average():
    assert persistence_forecast(np.array([10.0, 20.0, 30.0])).tolist() == [10.0, 20.0, 30.0]
    series = np.arange(20, dtype=float)
    forecast = moving_average_forecast(series, np.array([15]), 12)
    assert forecast[0] == pytest.approx(np.arange(3, 15).mean())


def test_ridge_alpha_is_the_validation_minimum():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 2))
    y_train = 3.0 * x_train[:, 0]
    x_validation = rng.normal(size=(15, 2))
    y_validation = 3.0 * x_validation[:, 0]
    alphas = [0.1, 1.0, 10.0]
    chosen, _ = select_ridge_alpha(x_train, y_train, x_validation, y_validation, alphas)
    from sklearn.linear_model import Ridge

    errors = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        errors.append(mae(y_validation, model.predict(x_validation)))
    assert chosen == alphas[int(np.argmin(errors))]


def test_hist_gradient_boosting_is_deterministic():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=50), "y_1": rng.normal(size=50)})
    train["y_1"] = 0.5 * train["x"]
    test = pd.DataFrame({"x": rng.normal(size=10)})
    first = predict_hist_gradient_boosting(train, test, ["x"], "y_1", seed=42)
    second = predict_hist_gradient_boosting(train, test, ["x"], "y_1", seed=42)
    assert first == pytest.approx(second)


def test_mae_rmse_mase_and_skill_score():
    observed = np.zeros(4)
    predicted = np.ones(4)
    assert mae(observed, predicted) == pytest.approx(1.0)
    assert rmse(observed, predicted) == pytest.approx(1.0)
    assert mase(1.0, 1.0) == pytest.approx(1.0)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_figures_are_nonempty(tmp_path: Path):
    frame = _regular_frame(30)
    series_path = tmp_path / "gpu_series.png"
    forecast_path = tmp_path / "horizon1_observed_vs_predicted.png"
    plot_gpu_series(frame, series_path)
    plot_horizon_forecasts(
        frame["timestamp"],
        frame["gpu"],
        {
            "persistence": frame["gpu"],
            "moving_average": frame["gpu"],
            "ridge": frame["gpu"],
            "hist_gradient_boosting": frame["gpu"],
        },
        forecast_path,
    )
    assert series_path.stat().st_size > 0
    assert forecast_path.stat().st_size > 0


def test_execute_missing_config_creates_no_results(tmp_path: Path):
    config = tmp_path / "config" / "v1.yaml"
    config.parent.mkdir()
    config.write_text(
        "\n".join(
            [
                "seed: 42",
                "gpu_archive: ausente-gpu.tar.gz",
                "memory_archive: ausente-mem.tar.gz",
                "horizons: [1, 5, 15]",
                "gpu_lags: [1, 2, 3, 6, 12]",
                "memory_lags: [0, 1, 6]",
                "rolling_windows: [6, 12]",
                "moving_average_window: 12",
                "test_fraction: 0.2",
                "validation_fraction: 0.2",
                "purge_steps: 27",
                "expected_step_seconds: 57",
                "alphas: [0.1, 1.0, 10.0]",
                "output_dir: results/v1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(MissingArchiveError):
        execute(config)
    assert not (tmp_path / "results").exists()
