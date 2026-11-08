"""Apoio dos testes da V3: configuração do smoke sintético e projeto temporário. Não lê o Alibaba v2018."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def smoke_config(**overrides) -> dict:
    config = yaml.safe_load((ROOT / "config/v3-smoke.yaml").read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config


def write_project(tmp_path: Path, config: dict, *, substitute: bool = True) -> Path:
    """Projeto mínimo (config/ e results/) num diretório temporário; devolve o caminho do YAML."""
    project = tmp_path / "proj"
    (project / "config").mkdir(parents=True, exist_ok=True)
    if substitute:
        shutil.copy(ROOT / "config/specpower_substitute.yaml", project / "config/specpower_substitute.yaml")
    path = project / "config/v3.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def fake_v2_reference(ctx, directory: Path, *, perfect: bool = True) -> Path:
    """Pasta no formato de results/v2-full/main com XGBoost e Ridge A no teste (previsão perfeita ou persistência)."""
    from energy_prediction.features.cpu_features import target_column
    from energy_prediction.validation.rolling_origin import describe_splits

    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    test = ctx.table.iloc[ctx.splits.test]
    for steps in ctx.steps:
        y = test[target_column(steps)].to_numpy()
        pred = y if perfect else test["cpu_t"].to_numpy()
        for model in ("xgboost", "ridge"):
            frames.append(
                pd.DataFrame(
                    {
                        "split": "test_final",
                        "model": model,
                        "config": "A",
                        "horizon_steps": steps,
                        "origin_index": test["origin_index"].to_numpy(),
                        "timestamp": test["timestamp"].to_numpy(),
                        "y_true": y,
                        "y_pred": pred,
                    }
                )
            )
    pd.concat(frames).to_csv(directory / "predictions.csv", index=False)
    (directory / "folds.json").write_text(json.dumps(describe_splits(ctx.table, ctx.splits, target_column(ctx.steps[0]))))
    selections = []
    for steps in ctx.steps:
        selections.append({"model": "ridge", "config": "A", "horizon_steps": steps, "best_params": {"alpha": 1.0}})
        selections.append({"model": "xgboost", "config": "A", "horizon_steps": steps, "best_params": {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 20}})
    (directory / "selections.json").write_text(json.dumps(selections))
    metrics = pd.DataFrame(
        [{"model": "xgboost", "config": "A", "horizon_steps": s, "horizon_seconds": s * 300.0, "split": "test_final", "n": len(test), "mae": 0.0} for s in ctx.steps]
    )
    metrics.to_csv(directory / "metrics.csv", index=False)
    return directory


def tree_bytes(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(Path(path).rglob("*")) if p.is_file()}


def constant_fit_predict(constants):
    """Modelo de teste: o ponto g da grade prevê a constante constants[g]."""

    def fit_predict(g, train, predict, **_):
        return np.full(len(predict), float(constants[g])), {}

    return fit_predict
