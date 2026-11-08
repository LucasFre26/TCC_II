"""V3, grupo 5: laço de treino, janela, TCN causal, N-BEATSx, orçamento, portão e medições (CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from energy_prediction.experiments.v3_context import build_context  # noqa: E402
from energy_prediction.experiments.v3_families import neural  # noqa: E402
from energy_prediction.models.neural.dlinear import DLinear  # noqa: E402
from energy_prediction.models.neural.loop import LoopConfig, predict, train  # noqa: E402
from energy_prediction.models.neural.nbeatsx import NBEATSx  # noqa: E402
from energy_prediction.models.neural.tcn import TCN, feasible_dilations  # noqa: E402
from energy_prediction.models.neural.windows import channel_matrix, config_channels, receptive_field, window_tensor  # noqa: E402
from v3_helpers import ROOT, smoke_config  # noqa: E402

CPU = {"requested": "cpu", "effective": "cpu"}


@pytest.fixture(scope="module")
def ctx():
    return build_context(smoke_config(), ROOT)


def _toy(n=200, window=12, horizon=1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3, window)).astype(np.float32)
    Y = X[:, 0, -1:].repeat(horizon, axis=1) + 0.1 * rng.normal(size=(n, horizon))
    return X, Y


def test_loop_on_cpu_is_finite_and_records_history():
    X, Y = _toy()
    cfg = LoopConfig(max_epochs=8, patience=3, batch_size=64, learning_rate=1e-2)
    result = train(lambda: DLinear(3, 12, 1, moving_average=5), X[:150], Y[:150], X[150:], Y[150:], cfg=cfg, device="cpu", seed=42)
    assert 1 <= result.best_epoch <= result.epochs_run <= 8
    assert len(result.history) == result.epochs_run and all(np.isfinite(h["val_loss"]) for h in result.history)
    assert result.precision == "fp32" and result.peak_cuda_bytes == 0 and result.n_params > 0
    pred, seconds = predict(result.model, X[150:], device="cpu")
    assert np.isfinite(pred).all() and pred.shape == (50, 1) and seconds >= 0
    refit = train(lambda: DLinear(3, 12, 1, moving_average=5), X, Y, None, None, cfg=cfg, device="cpu", seed=42, epochs=result.best_epoch)
    assert refit.epochs_run == result.best_epoch


def test_loop_is_reproducible_on_cpu():
    X, Y = _toy()
    cfg = LoopConfig(max_epochs=3, batch_size=32)
    runs = [predict(train(lambda: NBEATSx(3, 12, 1, width=16, layers=2), X, Y, None, None, cfg=cfg, device="cpu", seed=7, epochs=3).model, X, device="cpu")[0] for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize("arch", ["dlinear", "tcn", "nbeatsx", "nhits", "patchtst"])
def test_every_architecture_trains_on_cpu(arch):
    X, Y = _toy(window=24, horizon=6)
    params = {
        "dlinear": {"linear": "individual", "moving_average": 5},
        "tcn": {"channels": 8, "dilations": [1, 2, 4], "kernel_size": 3, "dropout": 0.1},
        "nbeatsx": {"width": 16, "layers": 2, "trend_degree": 2, "harmonics": 2},
        "nhits": {"pool_sizes": [4, 2, 1], "width": 16, "layers": 2},
        "patchtst": {"patch_length": 8, "stride": 4, "d_model": 8, "heads": 2, "layers": 1, "dropout": 0.1},
    }[arch]
    result = train(lambda: neural.NeuralFamily.build(arch, params, 3, 24, 6), X, Y, None, None, cfg=LoopConfig(max_epochs=1), device="cpu", seed=1, epochs=1)
    pred, _ = predict(result.model, X, device="cpu")
    assert pred.shape == (len(X), 6) and np.isfinite(pred).all()


def test_tcn_stack_removed_when_receptive_field_exceeds_window():
    assert receptive_field(3, [1, 2, 4, 8]) == 31 and receptive_field(3, [1, 2, 4, 8, 16]) == 63
    kept, removed = feasible_dilations([[1, 2, 4, 8], [1, 2, 4, 8, 16]], 3, 36)
    assert kept == [[1, 2, 4, 8]] and removed[0]["receptive_field"] == 63
    with pytest.raises(ValueError):
        TCN(1, 36, 1, channels=4, dilations=[1, 2, 4, 8, 16])


def test_tcn_convolutions_are_causal():
    torch.manual_seed(0)
    model = TCN(2, 36, 1, channels=4, dilations=[1, 2, 4, 8], dropout=0.0).eval()
    x = torch.randn(1, 2, 36)
    changed = x.clone()
    changed[:, :, 20:] += 5.0
    with torch.no_grad():
        a, b = model.blocks(x), model.blocks(changed)
    assert torch.allclose(a[:, :, :20], b[:, :, :20])
    assert not torch.allclose(a[:, :, 20:], b[:, :, 20:])


@pytest.mark.parametrize("config", ["B", "C"])
def test_future_exogenous_values_do_not_reach_the_network(ctx, config):
    channels = config_channels(ctx.specs, ctx.cfg["features"]["ablation"][config])
    assert channels[0] == "cpu" and len(channels) > 3
    origin = 700
    matrix, _, _ = channel_matrix(ctx.series, channels)
    changed_series = ctx.series.copy()
    exogenous = [c for c in channels[1:] if c in changed_series]
    changed_series.loc[changed_series.index > origin, exogenous] += 40.0
    changed, _, _ = channel_matrix(changed_series, channels)
    before = window_tensor(matrix, np.array([origin]), 48)
    after = window_tensor(changed, np.array([origin]), 48)
    assert np.array_equal(before, after)
    torch.manual_seed(0)
    model = NBEATSx(len(channels), 48, 1, width=16, layers=2).eval()
    with torch.no_grad():
        assert torch.equal(model(torch.as_tensor(before, dtype=torch.float32)), model(torch.as_tensor(after, dtype=torch.float32)))


def test_infeasible_window_is_discarded_with_sizes(ctx):
    ctx.cfg["families"]["neural"]["window_candidates"] = [24, 600]
    try:
        family = neural.NeuralFamily(ctx, "cpu")
        window, record = neural.select_window(family, neural.Budget(1))
    finally:
        ctx.cfg["families"]["neural"]["window_candidates"] = [24, 48]
    assert window == 24
    bad = next(c for c in record["candidates"] if c["window_steps"] == 600)
    assert bad["feasible"] is False and "exigidos 96" in bad["reason"] and "teste" not in bad["reason"]


def test_budget_exhaustion_is_recorded_not_silent(ctx, tmp_path: Path):
    ctx.cfg["families"]["neural"]["budget_hours"] = 1e-9
    try:
        summary = neural.run(ctx, tmp_path, device=CPU)
    finally:
        ctx.cfg["families"]["neural"]["budget_hours"] = 1
    info = json.loads((tmp_path / "family_info.json").read_text())
    core = {u.label for u in neural.core_units(["dlinear", "tcn", "nbeatsx"], "dlinear", True)}
    assert {i["item"] for i in info["infeasible"]} >= core
    assert not info["core_completed"] and summary["excluded"]
    assert pd.read_csv(tmp_path / "metrics.csv").empty if (tmp_path / "metrics.csv").stat().st_size > 1 else True


@pytest.fixture(scope="module")
def neural_run(ctx, tmp_path_factory):
    out = tmp_path_factory.mktemp("neural")
    summary = neural.run(ctx, out, device=CPU)
    return out, summary


def test_direct_table_is_primary_and_path_secondary(neural_run, ctx):
    out, summary = neural_run
    assert summary["core_completed"] and summary["window_steps"] in (24, 48)
    metrics = pd.read_csv(out / "metrics.csv")
    test = metrics[metrics["split"] == "test_final"]
    direct = test[test["mode"] == "direct"]
    assert (direct.groupby(["model", "config", "horizon_steps"]).size() == 1).all()
    assert set(direct.loc[direct["model"] != "persistence", "config"]) == {"A", "B", "C"}
    assert set(test.loc[test["mode"] == "path", "table_role"]) == {"secundaria"}
    assert set(direct["table_role"]) == {"principal"}
    predictions = pd.read_csv(out / "predictions.csv")
    tests = predictions[predictions["split"] == "test_final"]
    expected = set(ctx.table["origin_index"].to_numpy()[ctx.splits.test])
    for (_, _, _, _), part in tests.groupby(["model", "config", "mode", "horizon_steps"]):
        assert set(part["origin_index"]) == expected
    path = tests[(tests["mode"] == "path") & (tests["model"] == "dlinear")]
    assert sorted(path["horizon_steps"].unique()) == [1, 2, 3, 4, 5, 6]
    d = pd.read_csv(out / "d_contrast_metrics.csv")
    assert set(d["model"]) == {"dlinear"} and set(d["config"]) == {"A", "D"}
    assert (out / "checkpoints/dlinear_A_direct_h1.pt").is_file()


def test_measurements_on_cpu(neural_run):
    out, _ = neural_run
    profile = pd.read_csv(out / "training_profile.csv")
    assert (profile["peak_cuda_bytes_max"] == 0).all() and (profile["device"] == "cpu").all()
    assert (profile["n_params"] > 0).all() and (profile["inference_seconds_test"] >= 0).all()
    history = pd.read_csv(out / "epoch_history.csv")
    assert {"arch", "phase", "epoch", "train_loss", "val_loss"} <= set(history.columns) and len(history) > 0


def test_failed_gate_writes_record_without_optional_metrics(neural_run, monkeypatch, ctx, tmp_path: Path):
    out, summary = neural_run
    record = json.loads((out / "optional_architectures.json").read_text())
    assert "gate" in record and "teste final não entra" in record["gate"]["criterion"]
    if not record["gate"]["passed"]:
        assert record["fitted"] == [] and "reason" in record
        assert not {"nhits", "patchtst"} & set(pd.read_csv(out / "metrics.csv")["model"])

    monkeypatch.setattr(neural, "persistence_inner_mae", lambda *a, **k: float("inf"))
    forced = tmp_path / "forced"
    forced.mkdir()
    neural.run(ctx, forced, device=CPU)
    passed = json.loads((forced / "optional_architectures.json").read_text())
    assert passed["gate"]["passed"] and passed["fitted"] == ["nhits"]
    assert "patchtst_skipped" in passed  # orçamento de 1 h não deixa folga de mais de 2 h
    assert "nhits" in set(pd.read_csv(forced / "metrics.csv")["model"])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponível")
def test_cuda_epoch_runs_and_measures_memory():
    X, Y = _toy()
    result = train(lambda: DLinear(3, 12, 1, moving_average=5), X, Y, X[:50], Y[:50], cfg=LoopConfig(max_epochs=2), device="cuda", seed=42)
    assert result.peak_cuda_bytes > 0 and result.precision in ("bf16-mixed", "fp16-mixed", "fp32")
    pred, _ = predict(result.model, X, device="cuda")
    assert np.isfinite(pred).all()
