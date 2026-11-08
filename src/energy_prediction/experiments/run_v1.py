"""Executa a prova de conceito V1 de ponta a ponta."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from energy_prediction.data.load_genai import file_sha256, load_metric_archive, require_archives
from energy_prediction.evaluation.metrics import mae, mase, rmse, skill_score
from energy_prediction.features.supervised import build_supervised, feature_columns, split_origins
from energy_prediction.models.baselines import moving_average_forecast, persistence_forecast
from energy_prediction.models.classical import (
    ALLOWED_MODELS,
    predict_hist_gradient_boosting,
    predict_ridge,
)
from energy_prediction.preprocessing.aggregate import aggregate_by_timestamp, join_series, sort_unique
from energy_prediction.preprocessing.checks import assert_expected_step
from energy_prediction.preprocessing.profile import profile_joined
from energy_prediction.visualization.plots import plot_gpu_series, plot_horizon_forecasts

PRELIMINARY_STATEMENT = (
    "Estes são resultados preliminares. A série não é CPU do Alibaba, "
    "não é utilização de CPU do Alibaba, não é energia do Alibaba, "
    "não há medição elétrica, e o modelo de potência não foi aplicado. "
    "O pacote não calcula energia."
)


def project_root_from_config(config_path: Path) -> Path:
    """config/v1.yaml vive em Tcc-sourcecode/config; a raiz do código é o pai."""
    return config_path.resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = [
        "seed",
        "gpu_archive",
        "memory_archive",
        "horizons",
        "gpu_lags",
        "memory_lags",
        "rolling_windows",
        "moving_average_window",
        "test_fraction",
        "validation_fraction",
        "purge_steps",
        "expected_step_seconds",
        "alphas",
        "output_dir",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Configuração sem chaves: {missing}")
    return config


def _subset(frame: pd.DataFrame, positions) -> pd.DataFrame:
    return frame.iloc[positions].reset_index(drop=True)


def execute(config_path: Path) -> dict:
    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    gpu_path = root / config["gpu_archive"]
    memory_path = root / config["memory_archive"]
    output_dir = root / config["output_dir"]
    require_archives(gpu_path, memory_path)

    hash_before = {"gpu": file_sha256(gpu_path), "memory": file_sha256(memory_path)}
    gpu_raw = load_metric_archive(gpu_path)
    memory_raw = load_metric_archive(memory_path)
    hash_after = {"gpu": file_sha256(gpu_path), "memory": file_sha256(memory_path)}
    if hash_before != hash_after:
        raise RuntimeError("O hash dos arquivos brutos mudou durante a leitura.")

    gpu_agg = aggregate_by_timestamp(gpu_raw).rename(columns={"n_containers": "n_containers"})
    memory_agg = aggregate_by_timestamp(memory_raw)
    joined, excluded = join_series(gpu_agg, memory_agg)
    joined = sort_unique(joined)
    observed_step = assert_expected_step(joined["timestamp"], float(config["expected_step_seconds"]))
    profile = profile_joined(joined, excluded)
    profile["observed_step_seconds"] = observed_step
    profile["gpu_rows_raw"] = int(len(gpu_raw))
    profile["memory_rows_raw"] = int(len(memory_raw))
    profile["gpu_containers"] = int(gpu_raw["container_id"].nunique(dropna=True))
    profile["memory_containers"] = int(memory_raw["container_id"].nunique(dropna=True))
    profile["gpu_raw_zero_fraction"] = float((gpu_raw["value"] == 0).mean())
    profile["memory_raw_min"] = float(memory_raw["value"].min())
    profile["memory_negative_rows"] = int((memory_raw["value"] < 0).sum())
    profile["memory_duplicate_key_rows"] = int(
        memory_raw.duplicated(["timestamp", "container_id"]).sum()
    )

    supervised = build_supervised(
        joined,
        gpu_lags=list(config["gpu_lags"]),
        memory_lags=list(config["memory_lags"]),
        rolling_windows=list(config["rolling_windows"]),
        horizons=list(config["horizons"]),
    )
    parts = split_origins(
        len(supervised),
        test_fraction=float(config["test_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        purge_steps=int(config["purge_steps"]),
    )
    train = _subset(supervised, parts["train"])
    validation = _subset(supervised, parts["validation"])
    test = _subset(supervised, parts["test"])
    columns = feature_columns(
        list(config["gpu_lags"]),
        list(config["rolling_windows"]),
        list(config["memory_lags"]),
    )

    rows = []
    horizon_one_forecasts: dict[str, pd.Series] | None = None
    for horizon in config["horizons"]:
        target = f"y_{horizon}"
        y_test = test[target].to_numpy(dtype=float)
        persistence = persistence_forecast(test["gpu_at_origin"].to_numpy(dtype=float))
        moving = moving_average_forecast(
            joined["gpu"].to_numpy(dtype=float),
            test["origin_index"].to_numpy(dtype=int),
            int(config["moving_average_window"]),
        )
        persistence_mae = mae(y_test, persistence)
        persistence_rmse = rmse(y_test, persistence)
        predictions = {
            ("persistence", "gpu_only"): (persistence, None),
            ("moving_average", "gpu_only"): (moving, None),
        }
        for variant, variant_columns in columns.items():
            ridge_pred, alpha = predict_ridge(
                train, validation, test, variant_columns, target, list(config["alphas"])
            )
            boosting_pred = predict_hist_gradient_boosting(
                train, test, variant_columns, target, int(config["seed"])
            )
            predictions[("ridge", variant)] = (ridge_pred, alpha)
            predictions[("hist_gradient_boosting", variant)] = (boosting_pred, None)

        if horizon == 1:
            horizon_one_forecasts = {
                "persistence": pd.Series(persistence),
                "moving_average": pd.Series(moving),
                "ridge": pd.Series(predictions[("ridge", "gpu_memory")][0]),
                "hist_gradient_boosting": pd.Series(
                    predictions[("hist_gradient_boosting", "gpu_memory")][0]
                ),
            }

        for (model_name, variant), (pred, alpha) in predictions.items():
            model_mae = mae(y_test, pred)
            model_rmse = rmse(y_test, pred)
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "variant": variant,
                    "mae": model_mae,
                    "rmse": model_rmse,
                    "mase": mase(model_mae, persistence_mae),
                    "skill_score": skill_score(model_rmse, persistence_rmse),
                    "alpha": alpha,
                }
            )

    metrics = pd.DataFrame(rows)
    unknown = set(metrics["model"]).difference(ALLOWED_MODELS)
    if unknown or set(metrics["model"]) != set(ALLOWED_MODELS):
        raise RuntimeError(f"Conjunto de modelos inesperado: {sorted(metrics['model'].unique())}")

    elapsed = time.perf_counter() - started
    sanity = _sanity(train, validation, test, columns["gpu_memory"], metrics)
    summary = {
        "seed": int(config["seed"]),
        "observed_step_seconds": observed_step,
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_purge": int(len(parts["purge"])),
        "n_test": int(len(test)),
        "gpu_zero_fraction": profile["gpu_zero_fraction"],
        "preliminary_statement": PRELIMINARY_STATEMENT,
        "elapsed_seconds": elapsed,
        "profile": profile,
        "archive_sha256": hash_after,
        "config": config,
        "sanity": sanity,
    }

    figures = output_dir / "figures"
    tables = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "metrics.csv", index=False)
    (tables / "metrics.md").write_text(_markdown_table(metrics), encoding="utf-8")
    (output_dir / "relatorio-preliminar.md").write_text(
        _report(summary, metrics),
        encoding="utf-8",
    )
    plot_gpu_series(joined, figures / "gpu_series.png")
    if horizon_one_forecasts is None:
        raise RuntimeError("Horizonte 1 não foi avaliado.")
    plot_horizon_forecasts(
        test["timestamp"],
        test["y_1"],
        horizon_one_forecasts,
        figures / "horizon1_observed_vs_predicted.png",
    )
    return summary


def _sanity(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    metrics: pd.DataFrame,
) -> dict:
    correlations = {}
    for column in feature_names:
        correlations[column] = float(train[column].corr(train["y_1"]))
    finite = [abs(value) for value in correlations.values() if value == value]
    max_abs = max(finite) if finite else None
    high_skill = metrics.loc[
        metrics["skill_score"] > 0.95, ["model", "horizon", "variant", "skill_score"]
    ]
    return {
        "max_abs_feature_target_corr_train_h1": max_abs,
        "feature_target_corr_train_h1": correlations,
        "suspicious_skill_rows": json.loads(high_skill.to_json(orient="records")),
        "y1_mean_train": float(train["y_1"].mean()),
        "y1_mean_validation": float(validation["y_1"].mean()),
        "y1_mean_test": float(test["y_1"].mean()),
    }


def _markdown_table(metrics: pd.DataFrame) -> str:
    shown = metrics.copy()
    for column in ("mae", "rmse", "mase", "skill_score"):
        shown[column] = shown[column].map(lambda value: f"{value:.6f}")
    shown["alpha"] = shown["alpha"].map(lambda value: "" if value is None or pd.isna(value) else f"{value}")
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in shown.to_numpy())
    return "\n".join([header, separator, body]) + "\n"


def _report(summary: dict, metrics: pd.DataFrame) -> str:
    profile = summary["profile"]
    lines = [
        "# Resultados preliminares da V1",
        "",
        PRELIMINARY_STATEMENT,
        "",
        "A comparação abaixo descreve esta execução. Ela não altera hipóteses nem conclusões do TCC.",
        "",
        "## Subset",
        "",
        f"- Linhas brutas de GPU: {profile['gpu_rows_raw']}",
        f"- Linhas brutas de memória: {profile['memory_rows_raw']}",
        f"- Contêineres de GPU: {profile['gpu_containers']}",
        f"- Contêineres de memória: {profile['memory_containers']}",
        f"- Instantes após o inner join: {profile['n_timestamps']}",
        f"- Instantes excluídos por existir em só uma série: {profile['excluded_timestamps']}",
        f"- Passo mediano observado: {summary['observed_step_seconds']} s",
        f"- Fração de zeros nas linhas brutas de GPU: {profile['gpu_raw_zero_fraction']:.6f}",
        f"- Fração de zeros da GPU agregada: {profile['gpu_zero_fraction']:.6f}",
        f"- Linhas de memória com valor negativo: {profile['memory_negative_rows']}",
        f"- Linhas de memória com par instante-contêiner repetido: {profile['memory_duplicate_key_rows']}",
        f"- Média do alvo no horizonte 1, treino / validação / teste: {summary['sanity']['y1_mean_train']:.6f} / {summary['sanity']['y1_mean_validation']:.6f} / {summary['sanity']['y1_mean_test']:.6f}",
        f"- Treino / validação / purga / teste: {summary['n_train']} / {summary['n_validation']} / {summary['n_purge']} / {summary['n_test']}",
        f"- Duração da execução: {summary['elapsed_seconds']:.3f} s",
        f"- Semente: {summary['seed']}",
        "",
        "## Métricas no teste",
        "",
        _markdown_table(metrics),
        "",
        "## Leitura descritiva",
        "",
        _comparison(metrics),
        "Os modelos não observam o duty cycle na origem; a persistência observa. "
        "A média do alvo no horizonte 1 cai entre treino e teste, então a vantagem da persistência descreve esta execução.",
        "",
        "## Sanidade",
        "",
        f"- Maior correlação absoluta, no treino, entre feature e alvo do horizonte 1: {summary['sanity']['max_abs_feature_target_corr_train_h1']}",
        f"- Linhas com skill score acima de 0,95: {summary['sanity']['suspicious_skill_rows']}",
        "",
    ]
    return "\n".join(lines)


def _comparison(metrics: pd.DataFrame) -> str:
    sentences = []
    for horizon, block in metrics.groupby("horizon"):
        best = block.sort_values(["mae", "model"]).iloc[0]
        persistence = block[(block["model"] == "persistence") & (block["variant"] == "gpu_only")].iloc[0]
        sentences.append(
            f"No horizonte {int(horizon)}, a menor MAE desta execução foi {best['mae']:.6f} "
            f"({best['model']}, variante {best['variant']}). "
            f"A persistência teve MAE {persistence['mae']:.6f} e MASE 1 por construção."
        )
    return " ".join(sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prova de conceito V1 de previsão de utilização.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = execute(args.config)
    print(
        "V1 concluída. "
        f"teste={summary['n_test']} passo={summary['observed_step_seconds']}s "
        f"duracao={summary['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
