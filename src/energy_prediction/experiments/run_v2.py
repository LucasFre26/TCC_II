"""Executa a V2: previsão de CPU no Alibaba Cluster Trace v2018 (machine_usage).

Ingestão -> agregação -> features -> folds rolling-origin com purga -> modelos -> métricas -> artefatos.
Os resultados só são gravados em `results/v2/` quando a execução inteira termina com sucesso.
Este módulo não calcula potência, energia nem PUE e não ajusta modelos sequenciais.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from energy_prediction.data.alibaba import load_or_build_grid
from energy_prediction.evaluation.error_analysis import bootstrap_table, default_pairs, error_extremes_table
from energy_prediction.evaluation.exogenous_lead import lead_correlations
from energy_prediction.evaluation.protocol import run_protocol
from energy_prediction.features.cpu_features import (
    TARGET_SERIES,
    PERSISTENCE_INPUT,
    ablation_columns,
    assert_equal_information,
    build_feature_table,
    build_group_specs,
    drop_incomplete_origins,
    horizon_steps,
    max_window_steps,
    parse_duration_steps,
    target_column,
)
from energy_prediction.models.registry import validate_model_names
from energy_prediction.preprocessing.cluster import aggregate, get_strategy, resolve_machine_info
from energy_prediction.validation.rolling_origin import describe_splits, make_splits, required_purge_steps
from energy_prediction.visualization.v2_plots import (
    plot_ablation_comparison,
    plot_horizon_comparison,
    plot_real_vs_predicted,
)

STATEMENT = (
    "Série de utilização de CPU do Alibaba Cluster Trace v2018 (machine_usage). "
    "Não há medição elétrica; potência, energia e PUE não foram calculados nesta versão."
)

FREE = "free"  # dicionário de conteúdo livre (validado por quem o consome)
SCHEMA = {
    "seed": None,
    "output_dir": None,
    "data": {
        "usage_archive": None,
        "meta_archive": None,
        "cache_dir": None,
        "chunk_rows": None,
        "window": {"start_seconds": None, "end_seconds": None},
        "machines": {"ids": None, "sample_size": None},
        "granularity": {"sample_rows": None, "min_conformity": None},
        "grid_step": None,
    },
    "aggregation": {"strategy": None, "active_statuses": None},
    "cohort": {"enabled": None, "cpu_num": None, "min_machines": None},
    "features": {"groups": FREE, "ablation": FREE},
    "horizons": None,
    "validation": {
        "strategy": None,
        "n_folds": None,
        "validation_size": None,
        "test_size": None,
        "min_train_size": None,
        "purge": None,
    },
    "models": {"names": None, "ridge": FREE, "xgboost": FREE},
    "error_analysis": {
        "peak_quantile": None,
        "bootstrap": {"block_steps": None, "n_boot": None, "sensitivity_blocks": None},
    },
}


class ConfigError(ValueError):
    """Chave obrigatória ausente ou desconhecida na configuração da V2."""


def _check_keys(config, schema, path: str = "") -> None:
    if not isinstance(config, dict):
        raise ConfigError(f"A chave {path or 'raiz'} deve ser um mapeamento.")
    missing = [key for key in schema if key not in config]
    if missing:
        raise ConfigError(f"Configuração sem chaves: {[f'{path}{key}' for key in missing]}")
    unknown = [key for key in config if key not in schema]
    if unknown:
        raise ConfigError(f"Configuração com chaves desconhecidas: {[f'{path}{key}' for key in unknown]}")
    for key, sub in schema.items():
        if isinstance(sub, dict):
            _check_keys(config[key], sub, f"{path}{key}.")


def load_config(config_path: Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _check_keys(config, SCHEMA)
    return config


def project_root_from_config(config_path: Path) -> Path:
    """config/v2.yaml vive em Tcc-sourcecode/config; a raiz do código é o pai."""
    return Path(config_path).resolve().parent.parent


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo não serializável: {type(value)}")


def _dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    for column in ("mae", "rmse", "mase", "skill_score"):
        if column in shown:
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in shown.to_numpy())
    return "\n".join([header, separator, body]) + "\n"


def _guard_output_dir(root: Path, output_dir: Path) -> None:
    v1 = (root / "results" / "v1").resolve()
    target = output_dir.resolve()
    if target == v1 or v1 in target.parents or target in v1.parents:
        raise ConfigError("output_dir da V2 não pode coincidir com results/v1 nem conter/estar dentro dele.")


def _series_profile(series: pd.DataFrame, agg_info: dict) -> dict:
    return {
        "n_points": int(len(series)),
        "period_seconds": [float(series["timestamp"].min()), float(series["timestamp"].max())],
        "cpu_missing_points": int(series["cpu"].isna().sum()),
        "coverage_min": float(series["coverage"].min()),
        "coverage_mean": float(series["coverage"].mean()),
        "aggregation": agg_info,
    }


def _run_analysis(name: str, strategy: str, grid, machine_info, cfg: dict, resolved: dict, seed: int, **strategy_params) -> dict:
    step_seconds = resolved["step_seconds"]
    series, agg_info = aggregate(strategy, grid, machine_info, step_seconds=step_seconds, **strategy_params)
    specs = resolved["specs"]
    steps = resolved["horizon_steps"]
    columns = resolved["columns"]
    table = build_feature_table(series, specs, steps)
    table = drop_incomplete_origins(table, columns[list(columns)[-1]], steps)
    splits = make_splits(
        table["origin_index"].to_numpy(),
        n_folds=int(cfg["validation"]["n_folds"]),
        validation_steps=resolved["validation_steps"],
        test_steps=resolved["test_steps"],
        min_train_steps=resolved["min_train_steps"],
        purge_steps=resolved["purge_steps"],
        max_horizon_steps=max(steps),
        max_window_steps=resolved["max_window_steps"],
        strategy=cfg["validation"]["strategy"],
    )
    result = run_protocol(table, columns, steps, splits, cfg["models"], seed=seed, step_seconds=step_seconds)
    return {
        "name": name,
        "strategy": strategy,
        "series": series,
        "series_profile": _series_profile(series, agg_info),
        "table_rows": int(len(table)),
        "folds": describe_splits(table, splits, target_column(steps[0])),
        "result": result,
    }


def _error_analysis(analysis: dict, cfg: dict, resolved: dict, seed: int) -> dict:
    """Erros extremos, IC por block bootstrap e antecedência das exógenas; não altera a seleção."""
    predictions = analysis["result"].predictions
    settings = cfg["error_analysis"]
    boot = settings["bootstrap"]
    pairs = default_pairs(predictions)
    blocks = list(dict.fromkeys([int(boot["block_steps"]), *[int(b) for b in boot["sensitivity_blocks"]]]))
    intervals = pd.concat(
        [bootstrap_table(predictions, pairs, block=block, n_boot=int(boot["n_boot"]), seed=seed) for block in blocks],
        ignore_index=True,
    )
    exogenous = [column for spec in resolved["specs"].values() for column in spec.columns if column != TARGET_SERIES]
    return {
        "extremes": error_extremes_table(predictions, peak_quantile=float(settings["peak_quantile"])),
        "intervals": intervals,
        "lead": lead_correlations(analysis["series"], target=TARGET_SERIES, exogenous=exogenous, horizons=resolved["horizon_steps"]),
    }


def _write_analysis(directory: Path, analysis: dict, resolved: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result = analysis["result"]
    result.metrics.to_csv(directory / "metrics.csv", index=False)
    result.predictions.to_csv(directory / "predictions.csv", index=False)
    errors = analysis["errors"]
    errors["extremes"].to_csv(directory / "error_extremes.csv", index=False)
    errors["intervals"].to_csv(directory / "bootstrap_ci.csv", index=False)
    _dump_json(directory / "exogenous_lead.json", errors["lead"])
    _dump_json(directory / "folds.json", analysis["folds"])
    _dump_json(directory / "selections.json", result.selections)
    _dump_json(directory / "series_profile.json", analysis["series_profile"])
    show = ["model", "config", "horizon_seconds", "split", "n", "mae", "rmse", "mase", "skill_score"]
    metrics = result.metrics
    summary = [
        f"# Resultados V2 ({analysis['name']})",
        "",
        STATEMENT,
        "",
        "## Teste final (avaliado uma vez)",
        "",
        _markdown_table(metrics[metrics["split"] == "test_final"], show),
        "## Média entre folds de desenvolvimento",
        "",
        _markdown_table(metrics[metrics["split"] == "aggregate_mean"], show),
    ]
    (directory / "metrics_summary.md").write_text("\n".join(summary), encoding="utf-8")
    figures = directory / "figures"
    steps = resolved["horizon_steps"]
    last_config = list(resolved["columns"])[-1]
    plot_real_vs_predicted(result.predictions, figures / "real_vs_predicted.png", config=last_config, horizon_steps=steps[0], step_seconds=resolved["step_seconds"])
    plot_horizon_comparison(metrics, figures / "horizon_comparison.png", config=last_config)
    plot_ablation_comparison(metrics, figures / "ablation_comparison.png")


def execute(config_path: Path) -> dict:
    started = time.perf_counter()
    config_path = Path(config_path)
    cfg = load_config(config_path)
    root = project_root_from_config(config_path)
    output_dir = _resolve(root, cfg["output_dir"])
    _guard_output_dir(root, output_dir)
    seed = int(cfg["seed"])

    # Validações que não dependem dos dados: falham antes de qualquer processamento.
    validate_model_names(list(cfg["models"]["names"]))
    get_strategy(cfg["aggregation"]["strategy"])
    if cfg["cohort"]["enabled"]:
        get_strategy("homogeneous_cohort")

    data = cfg["data"]
    grid, meta, profile = load_or_build_grid(
        _resolve(root, data["usage_archive"]),
        _resolve(root, data["meta_archive"]),
        cache_dir=_resolve(root, data["cache_dir"]),
        grid_step=data["grid_step"],
        chunk_rows=int(data["chunk_rows"]),
        granularity_sample_rows=int(data["granularity"]["sample_rows"]),
        granularity_min_conformity=float(data["granularity"]["min_conformity"]),
        start_seconds=data["window"]["start_seconds"],
        end_seconds=data["window"]["end_seconds"],
        machine_ids=data["machines"]["ids"],
        machine_sample_size=data["machines"]["sample_size"],
        seed=seed,
    )
    step_seconds = float(profile["step_seconds"])
    machine_info = resolve_machine_info(
        meta,
        active_statuses=list(cfg["aggregation"]["active_statuses"]),
        window_end_seconds=data["window"]["end_seconds"],
    )

    specs = build_group_specs(cfg["features"]["groups"], step_seconds)
    columns = ablation_columns(specs, cfg["features"]["ablation"])
    assert_equal_information([PERSISTENCE_INPUT], columns)
    steps = horizon_steps(list(cfg["horizons"]), step_seconds)
    window = max_window_steps(specs)
    required = required_purge_steps(max(steps), window)
    purge = required if cfg["validation"]["purge"] == "auto" else parse_duration_steps(cfg["validation"]["purge"], step_seconds)
    resolved = {
        "step_seconds": step_seconds,
        "specs": specs,
        "columns": columns,
        "horizon_steps": steps,
        "max_window_steps": window,
        "purge_steps": purge,
        "required_purge_steps": required,
        "validation_steps": parse_duration_steps(cfg["validation"]["validation_size"], step_seconds),
        "test_steps": parse_duration_steps(cfg["validation"]["test_size"], step_seconds),
        "min_train_steps": parse_duration_steps(cfg["validation"]["min_train_size"], step_seconds),
    }

    analyses = [
        _run_analysis(
            "main",
            cfg["aggregation"]["strategy"],
            grid,
            machine_info,
            cfg,
            resolved,
            seed,
        )
    ]
    if cfg["cohort"]["enabled"]:
        analyses.append(
            _run_analysis(
                "cohort",
                "homogeneous_cohort",
                grid,
                machine_info,
                cfg,
                resolved,
                seed,
                cohort_cpu_num=cfg["cohort"]["cpu_num"],
                cohort_min_machines=int(cfg["cohort"]["min_machines"]),
            )
        )

    effective = {
        "config": cfg,
        "resolved": {
            "step_seconds": step_seconds,
            "horizon_steps": steps,
            "max_window_steps": window,
            "purge_steps": purge,
            "required_purge_steps": required,
            "validation_steps": resolved["validation_steps"],
            "test_steps": resolved["test_steps"],
            "min_train_steps": resolved["min_train_steps"],
            "feature_columns": columns,
            "group_lags_steps": {name: list(spec.lags) for name, spec in specs.items()},
            "group_windows_steps": {name: list(spec.windows) for name, spec in specs.items()},
        },
    }
    summary = {
        "seed": seed,
        "statement": STATEMENT,
        "elapsed_seconds": time.perf_counter() - started,
        "analyses": [a["name"] for a in analyses],
        "granularity_detected": profile.get("granularity"),
        "step_seconds": step_seconds,
        "n_machines": profile["n_machines"],
        "n_grid_points": profile["n_grid_points"],
        "period_seconds": profile["period_seconds"],
        "folds": {a["name"]: a["folds"] for a in analyses},
    }

    # Grava em diretório temporário e só troca ao final, para nunca deixar resultado parcial.
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        (temporary / "config_effective.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
        _dump_json(temporary / "dataset_profile.json", profile)
        _dump_json(temporary / "run_summary.json", summary)
        for analysis in analyses:
            analysis["errors"] = _error_analysis(analysis, cfg, resolved, seed)
            _write_analysis(temporary / analysis["name"], analysis, resolved)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary["output_dir"] = str(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="V2: previsão de CPU no Alibaba Cluster Trace v2018.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = execute(args.config)
    print(
        "V2 concluída. "
        f"analises={summary['analyses']} passo={summary['step_seconds']}s "
        f"maquinas={summary['n_machines']} duracao={summary['elapsed_seconds']:.2f}s "
        f"saida={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
