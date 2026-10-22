"""Série de cluster ponderada por cpu_num e coorte homogênea (unidade experimental da V2)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from energy_prediction.data.alibaba import SERIES_COLUMNS

SERIES_NAMES = tuple(SERIES_COLUMNS)


class CohortTooSmallError(ValueError):
    """A coorte homogênea tem menos máquinas que o mínimo configurado."""


class UnknownAggregationError(ValueError):
    """A estratégia de agregação configurada não existe."""


def resolve_machine_info(
    meta: pd.DataFrame,
    *,
    active_statuses: list[str],
    window_end_seconds: float | None = None,
) -> pd.DataFrame:
    """Um cpu_num por máquina e o indicador de atividade.

    Regra de conflito: cpu_num mais frequente nos registros da máquina; empate fica com o menor.
    Ativa = todos os registros de status anteriores ao fim da janela estão em `active_statuses`.
    """
    records = meta.copy()
    if window_end_seconds is not None:
        in_time = records["time_stamp"].isna() | (records["time_stamp"] < window_end_seconds)
        records = records.loc[in_time]
    rows = []
    for machine, group in records.groupby("machine_id", sort=True):
        counts = group["cpu_num"].value_counts()
        top = counts[counts == counts.max()].index.min()
        rows.append(
            {
                "machine_id": str(machine),
                "cpu_num": float(top),
                "cpu_num_conflict": bool(group["cpu_num"].nunique() > 1),
                "active": bool(group["status"].isin(active_statuses).all()),
            }
        )
    return pd.DataFrame(rows, columns=["machine_id", "cpu_num", "cpu_num_conflict", "active"])


def aggregate_weighted(grid: pd.DataFrame, machine_info: pd.DataFrame, *, step_seconds: float) -> tuple[pd.DataFrame, dict]:
    """Média ponderada por cpu_num das máquinas presentes em cada intervalo, com cobertura.

    O denominador de cada variável soma só os pesos das máquinas com valor naquele intervalo.
    A cobertura usa a presença de CPU sobre o peso total das máquinas consideradas.
    """
    weights = machine_info.set_index("machine_id")["cpu_num"]
    known = grid["machine_id"].isin(weights.index)
    excluded_machines = int(grid.loc[~known, "machine_id"].nunique())
    frame = grid.loc[known].copy()
    if frame.empty:
        raise ValueError("Nenhuma máquina do machine_usage possui metadado em machine_meta.")
    frame["w"] = frame["machine_id"].map(weights).astype(float)
    considered = machine_info.loc[machine_info["machine_id"].isin(frame["machine_id"].unique()), "cpu_num"].sum()

    bins = np.arange(int(frame["bin"].min()), int(frame["bin"].max()) + 1)
    out = pd.DataFrame({"bin": bins})
    grouped = frame.groupby("bin")
    for name in SERIES_NAMES:
        present = frame[name].notna()
        numerator = (frame[name].fillna(0.0) * frame["w"]).where(present, 0.0).groupby(frame["bin"]).sum()
        denominator = frame["w"].where(present, 0.0).groupby(frame["bin"]).sum()
        value = numerator / denominator.where(denominator > 0)
        out[name] = out["bin"].map(value)
        if name == "cpu":
            out["coverage"] = out["bin"].map(denominator / considered).fillna(0.0)
    out["n_machines"] = out["bin"].map(frame.loc[frame["cpu"].notna()].groupby("bin")["machine_id"].nunique()).fillna(0).astype(int)
    out["timestamp"] = out["bin"] * step_seconds
    info = {
        "machines_without_metadata_excluded": excluded_machines,
        "machines_aggregated": int(frame["machine_id"].nunique()),
        "total_weight": float(considered),
        "coverage_min": float(out["coverage"].min()),
        "coverage_mean": float(out["coverage"].mean()),
        "cpu_num_conflicts": int(machine_info["cpu_num_conflict"].sum()),
    }
    ordered = ["bin", "timestamp", *SERIES_NAMES, "coverage", "n_machines"]
    return out[ordered], info


def select_homogeneous_cohort(
    machine_info: pd.DataFrame,
    *,
    cpu_num: float | None,
    min_machines: int,
    present_machines: set[str] | None = None,
) -> tuple[list[str], float]:
    """Máquinas ativas com o mesmo cpu_num. Padrão: o cpu_num mais comum entre as ativas (empate: menor)."""
    candidates = machine_info.loc[machine_info["active"]]
    if present_machines is not None:
        candidates = candidates.loc[candidates["machine_id"].isin(present_machines)]
    if candidates.empty:
        raise CohortTooSmallError("Nenhuma máquina ativa disponível para a coorte homogênea (0 encontradas).")
    if cpu_num is None:
        counts = candidates["cpu_num"].value_counts()
        cpu_num = float(counts[counts == counts.max()].index.min())
    chosen = candidates.loc[candidates["cpu_num"] == float(cpu_num), "machine_id"].tolist()
    if len(chosen) < min_machines:
        raise CohortTooSmallError(
            f"Coorte homogênea com cpu_num={cpu_num} tem {len(chosen)} máquinas; mínimo {min_machines}."
        )
    return sorted(chosen), float(cpu_num)


def _strategy_weighted(grid, machine_info, *, step_seconds, **_) -> tuple[pd.DataFrame, dict]:
    return aggregate_weighted(grid, machine_info, step_seconds=step_seconds)


def _strategy_cohort(grid, machine_info, *, step_seconds, cohort_cpu_num=None, cohort_min_machines=1, **_):
    present = set(grid["machine_id"].unique())
    chosen, cpu_num = select_homogeneous_cohort(
        machine_info, cpu_num=cohort_cpu_num, min_machines=cohort_min_machines, present_machines=present
    )
    subset_info = machine_info.loc[machine_info["machine_id"].isin(chosen)]
    series, info = aggregate_weighted(grid.loc[grid["machine_id"].isin(chosen)], subset_info, step_seconds=step_seconds)
    info.update({"cohort_cpu_num": cpu_num, "cohort_size": len(chosen)})
    return series, info


AGGREGATION_STRATEGIES: dict[str, Callable] = {
    "cluster_weighted_cpu_num": _strategy_weighted,
    "homogeneous_cohort": _strategy_cohort,
}


def get_strategy(name: str) -> Callable:
    if name not in AGGREGATION_STRATEGIES:
        raise UnknownAggregationError(
            f"Estratégia de agregação desconhecida: {name!r}. Disponíveis: {sorted(AGGREGATION_STRATEGIES)}"
        )
    return AGGREGATION_STRATEGIES[name]


def aggregate(name: str, grid: pd.DataFrame, machine_info: pd.DataFrame, *, step_seconds: float, **params):
    return get_strategy(name)(grid, machine_info, step_seconds=step_seconds, **params)
