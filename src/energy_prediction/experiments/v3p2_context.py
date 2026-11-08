"""Contexto da V3 Parte 2: série agregada, tabela de atributos, blocos por dia e painel por máquina.

A série e a tabela seguem exatamente a construção da V3 (mesma agregação, mesmos atributos,
origens incompletas em C descartadas), para que o teste herdado reproduza a V2. O painel por
máquina é construído só quando uma família o pede.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.data.alibaba import read_machine_domains
from energy_prediction.evaluation.day_forward import DayBlock, make_day_blocks
from energy_prediction.experiments.v3_context import load_fleet, resolve_path
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
from energy_prediction.preprocessing.cluster import aggregate, resolve_machine_info
from energy_prediction.preprocessing.panel import MachinePanel, check_against_series, load_or_build_panel
from energy_prediction.validation.rolling_origin import PurgeTooSmallError, required_purge_steps


@dataclass
class V3P2Context:
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
    blocks: list[DayBlock]
    dropped_blocks: list[dict]
    resolved: dict = field(default_factory=dict)
    _panel: MachinePanel | None = None
    _panel_record: dict | None = None
    _domains: pd.DataFrame | None = None

    @property
    def development_blocks(self) -> list[DayBlock]:
        return [b for b in self.blocks if b.role == "development"]

    def block(self, day: int) -> DayBlock:
        for b in self.blocks:
            if b.day == int(day):
                return b
        raise KeyError(f"Bloco do dia {day} não configurado.")

    @property
    def weights(self) -> pd.Series:
        return self.machine_info.set_index("machine_id")["cpu_num"].astype(float)

    def panel(self) -> MachinePanel:
        if self._panel is None:
            data = self.cfg["data"]
            persist = bool(data.get("persist_panel")) and self.profile.get("source", "alibaba") != "synthetic"
            cache = self.profile.get("cache") or {}
            source_key = str(cache.get("key") or f"synthetic-{self.profile.get('synthetic')}")
            cache_dir = resolve_path(self.root, data["cache_dir"]) if persist else None
            bins = self.series["bin"].to_numpy(dtype=int)
            self._panel, self._panel_record = load_or_build_panel(self.grid, self.population, bins, source_key=source_key, cache_dir=cache_dir)
        return self._panel

    def panel_weights(self) -> np.ndarray:
        panel = self.panel()
        return self.weights.reindex(panel.machines).to_numpy(dtype=float)

    def check_panel(self) -> dict:
        report = check_against_series(self.panel(), self.panel_weights(), self.series, tolerance=float(self.cfg["panel"]["tolerance_pp"]))
        return {**report, "panel": self._panel_record, "panel_info": self.panel().info}

    def domains(self) -> pd.DataFrame:
        """failure_domain_2 por máquina da população (do metadado sintético ou do arquivo real)."""
        if self._domains is None:
            if "failure_domain_2" in self.meta.columns:
                frame = self.meta.dropna(subset=["failure_domain_2"]).groupby("machine_id")["failure_domain_2"].agg(lambda s: sorted(s.astype(str))[0]).reset_index()
            else:
                frame = read_machine_domains(resolve_path(self.root, self.cfg["data"]["meta_archive"]))[["machine_id", "failure_domain_2"]]
            frame["machine_id"] = frame["machine_id"].astype(str)
            self._domains = pd.DataFrame({"machine_id": self.population}).merge(frame, on="machine_id", how="left")
        return self._domains

    def bin_position(self, bin_value: int) -> int:
        return int(bin_value) - int(self.series["bin"].iloc[0])


def build_context(cfg: dict, root: Path) -> V3P2Context:
    seed = int(cfg["seed"])
    grid, meta, profile = load_fleet(cfg, root)
    step_seconds = float(profile["step_seconds"])
    machine_info = resolve_machine_info(meta, active_statuses=list(cfg["aggregation"]["active_statuses"]), window_end_seconds=cfg["data"]["window"]["end_seconds"])
    population = sorted(set(grid["machine_id"].astype(str).unique()) & set(machine_info["machine_id"]))
    series, aggregation_info = aggregate(cfg["aggregation"]["strategy"], grid, machine_info, step_seconds=step_seconds)

    specs = build_group_specs(cfg["features"]["groups"], step_seconds)
    ablation = {name: list(groups) for name, groups in cfg["features"]["ablation"].items()}
    columns = ablation_columns(specs, ablation)
    assert_equal_information([PERSISTENCE_INPUT], columns)
    steps = horizon_steps(list(cfg["horizons"]), step_seconds)
    path_steps = list(range(1, int(cfg["trajectory_steps"]) + 1))
    window = max_window_steps(specs)
    proto = cfg["protocol"]
    purge = parse_duration_steps(proto["purge"], step_seconds)
    required = required_purge_steps(max(steps), window)
    if purge < required:
        raise PurgeTooSmallError(f"Purga de {purge} passos menor que o mínimo {required}.")
    full = build_feature_table(series, specs, path_steps)
    table = drop_incomplete_origins(full, columns[list(columns)[-1]], steps)
    inner = proto["inner"]
    blocks, dropped = make_day_blocks(
        table["origin_index"].to_numpy(),
        development_days=[int(d) for d in proto["development_days"]],
        inherited_day=proto["inherited_test_day"],
        steps_per_day=int(proto["steps_per_day"]),
        max_horizon=max(steps),
        purge_steps=purge,
        inner_validation_steps=parse_duration_steps(inner["validation_size"], step_seconds),
        inner_purge_steps=parse_duration_steps(inner["purge"], step_seconds),
        min_inner_train_steps=parse_duration_steps(inner["min_train_size"], step_seconds),
        infeasible_policy=inner["infeasible_policy"],
        expected_inherited_range=proto["expected_inherited_range"],
    )
    ctx = V3P2Context(
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
        blocks=blocks,
        dropped_blocks=dropped,
    )
    ctx.resolved = {
        "step_seconds": step_seconds,
        "horizon_steps": steps,
        "trajectory_steps": path_steps,
        "max_window_steps": window,
        "purge_steps": purge,
        "required_purge_steps": required,
        "n_origins": int(len(table)),
        "blocks": [b.describe() for b in blocks],
        "dropped_blocks": dropped,
        "feature_columns": columns,
    }
    return ctx
