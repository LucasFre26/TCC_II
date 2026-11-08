"""Dados, série principal, tabela de atributos, folds e divisão interna comuns às famílias da V3.

A tabela A/B/C descarta origens pelo conjunto de C, como a V2; a tabela de D é um subconjunto
dela (descarte pelo conjunto de D), com os mesmos limites de tempo dos folds e do teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.data.alibaba import load_or_build_grid
from energy_prediction.data.synthetic_fleet import make_synthetic_fleet
from energy_prediction.evaluation.nested import (
    InnerSplit,
    build_inner_splits,
    splits_on_subset,
)
from energy_prediction.experiments.v3_config import nested_enabled
from energy_prediction.features.cpu_features import (
    PERSISTENCE_INPUT,
    ablation_columns,
    assert_equal_information,
    build_feature_table,
    build_group_specs,
    drop_incomplete_origins,
    horizon_steps,
    max_window_steps,
    parse_duration_steps,
)
from energy_prediction.features.distribution import DIST_GROUP, distribution_frame, distribution_group
from energy_prediction.preprocessing.cluster import aggregate, get_strategy, resolve_machine_info
from energy_prediction.validation.rolling_origin import Splits, make_splits, required_purge_steps


def resolve_path(root: Path, value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass
class V3Context:
    cfg: dict
    root: Path
    seed: int
    step_seconds: float
    grid: pd.DataFrame
    meta: pd.DataFrame
    machine_info: pd.DataFrame
    profile: dict
    population: list[str]
    series: pd.DataFrame
    aggregation_info: dict
    specs: dict
    columns: dict[str, list[str]]
    steps: list[int]
    path_steps: list[int]
    purge_steps: int
    table: pd.DataFrame
    splits: Splits
    inner: dict[str, InnerSplit] | None
    inner_dropped: list[dict]
    d_table: pd.DataFrame | None
    d_splits: Splits | None
    d_inner: dict[str, InnerSplit] | None
    dist_info: dict | None
    resolved: dict = field(default_factory=dict)

    @property
    def nested(self) -> bool:
        return self.inner is not None

    def nested_params(self) -> dict:
        nested = self.cfg["validation"]["nested"]
        return {
            "validation_steps": parse_duration_steps(nested["validation_size"], self.step_seconds),
            "purge_steps": parse_duration_steps(nested["purge"], self.step_seconds),
            "min_train_steps": parse_duration_steps(nested["min_train_size"], self.step_seconds),
            "infeasible_policy": nested["infeasible_policy"],
        }

    def inner_for(self, table: pd.DataFrame, splits: Splits, eligible: np.ndarray | None = None) -> dict[str, InnerSplit] | None:
        """Divisão interna de uma tabela (principal ou D), opcionalmente restrita a origens elegíveis."""
        if not nested_enabled(self.cfg):
            return None
        inner, _ = build_inner_splits(table["origin_index"].to_numpy(), splits, eligible=eligible, **self.nested_params())
        return inner

    @property
    def v2_dir(self) -> Path | None:
        return resolve_path(self.root, self.cfg["reference"]["v2_dir"])

    @property
    def configs(self) -> list[str]:
        return [name for name in self.columns if name != "D"]


def validate_before_data(cfg: dict) -> None:
    """Falhas que não dependem dos dados ocorrem antes de ler qualquer arquivo."""
    get_strategy(cfg["aggregation"]["strategy"])
    if cfg["cohort"]["enabled"]:
        get_strategy("homogeneous_cohort")


def load_fleet(cfg: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = cfg["data"]
    if data["source"] == "synthetic":
        params = dict(data["synthetic"])
        return make_synthetic_fleet(**params)
    grid, meta, profile = load_or_build_grid(
        resolve_path(root, data["usage_archive"]),
        resolve_path(root, data["meta_archive"]),
        cache_dir=resolve_path(root, data["cache_dir"]),
        grid_step=data["grid_step"],
        chunk_rows=int(data["chunk_rows"]),
        granularity_sample_rows=int(data["granularity"]["sample_rows"]),
        granularity_min_conformity=float(data["granularity"]["min_conformity"]),
        start_seconds=data["window"]["start_seconds"],
        end_seconds=data["window"]["end_seconds"],
        machine_ids=data["machines"]["ids"],
        machine_sample_size=data["machines"]["sample_size"],
        seed=int(cfg["seed"]),
    )
    profile["source"] = "alibaba"
    return grid, meta, profile


def build_context(cfg: dict, root: Path) -> V3Context:
    validate_before_data(cfg)
    seed = int(cfg["seed"])
    grid, meta, profile = load_fleet(cfg, root)
    step_seconds = float(profile["step_seconds"])
    machine_info = resolve_machine_info(
        meta,
        active_statuses=list(cfg["aggregation"]["active_statuses"]),
        window_end_seconds=cfg["data"]["window"]["end_seconds"],
    )
    population = sorted(set(grid["machine_id"].astype(str).unique()) & set(machine_info["machine_id"]))
    series, aggregation_info = aggregate(cfg["aggregation"]["strategy"], grid, machine_info, step_seconds=step_seconds)

    specs = build_group_specs(cfg["features"]["groups"], step_seconds)
    ablation = {name: list(groups) for name, groups in cfg["features"]["ablation"].items()}
    if list(ablation) != ["A", "B", "C"]:
        raise ValueError("A V3 pré-registra as configurações A, B e C (D é acrescentada a partir de C).")
    dist_cfg = cfg["features"]["distribution"]
    dist_info = None
    if dist_cfg["enabled"]:
        dist, dist_info = distribution_frame(
            grid,
            population,
            series["bin"].to_numpy(),
            elevated_threshold=float(dist_cfg["elevated_cpu_threshold"]),
            change_steps=int(dist_cfg["dispersion_change_steps"]),
            skewness=bool(dist_cfg["skewness"]),
            skewness_max_undefined_fraction=float(dist_cfg["skewness_max_undefined_fraction"]),
        )
        series = series.merge(dist, on="bin", how="left")
        specs[DIST_GROUP] = distribution_group(dist_info["columns"])
        ablation["D"] = [*ablation["C"], DIST_GROUP]
    columns = ablation_columns(specs, ablation)
    assert_equal_information([PERSISTENCE_INPUT], columns)

    steps = horizon_steps(list(cfg["horizons"]), step_seconds)
    path_steps = list(range(1, int(cfg["trajectory_steps"]) + 1))
    if max(steps) > len(path_steps):
        raise ValueError("trajectory_steps precisa cobrir o maior horizonte.")
    window = max_window_steps(specs)
    required = required_purge_steps(max(steps), window)
    validation = cfg["validation"]
    purge = required if validation["purge"] == "auto" else parse_duration_steps(validation["purge"], step_seconds)

    full = build_feature_table(series, specs, path_steps)
    table = drop_incomplete_origins(full, columns["C"], steps)
    splits = make_splits(
        table["origin_index"].to_numpy(),
        n_folds=int(validation["n_folds"]),
        validation_steps=parse_duration_steps(validation["validation_size"], step_seconds),
        test_steps=parse_duration_steps(validation["test_size"], step_seconds),
        min_train_steps=parse_duration_steps(validation["min_train_size"], step_seconds),
        purge_steps=purge,
        max_horizon_steps=max(steps),
        max_window_steps=window,
        strategy=validation["strategy"],
    )
    ctx = V3Context(
        cfg=cfg,
        root=root,
        seed=seed,
        step_seconds=step_seconds,
        grid=grid,
        meta=meta,
        machine_info=machine_info,
        profile=profile,
        population=population,
        series=series,
        aggregation_info=aggregation_info,
        specs=specs,
        columns=columns,
        steps=steps,
        path_steps=path_steps,
        purge_steps=purge,
        table=table,
        splits=splits,
        inner=None,
        inner_dropped=[],
        d_table=None,
        d_splits=None,
        d_inner=None,
        dist_info=dist_info,
    )
    if nested_enabled(cfg):
        ctx.inner, ctx.inner_dropped = build_inner_splits(table["origin_index"].to_numpy(), splits, **ctx.nested_params())
    if "D" in columns:
        ctx.d_table = drop_incomplete_origins(table, columns["D"], steps)
        ctx.d_splits = splits_on_subset(ctx.d_table["origin_index"].to_numpy(), splits)
        ctx.d_inner = ctx.inner_for(ctx.d_table, ctx.d_splits)
    ctx.resolved = {
        "step_seconds": step_seconds,
        "horizon_steps": steps,
        "trajectory_steps": path_steps,
        "max_window_steps": window,
        "purge_steps": purge,
        "required_purge_steps": required,
        "nested": ctx.nested_params() if nested_enabled(cfg) else None,
        "feature_columns": columns,
        "n_origins_abc": int(len(table)),
        "n_origins_d": None if ctx.d_table is None else int(len(ctx.d_table)),
        "test_range": list(splits.test_range),
        "development_range": list(splits.development_range),
    }
    return ctx
