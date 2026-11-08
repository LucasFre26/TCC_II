"""Atributos causais de CPU e exógenas, alvos diretos e conjuntos de ablação da V2.

Toda feature da origem t usa só observações em t ou antes. Os nomes seguem a série de
trabalho (`cpu`, `mem`, `mem_gps`, `net_in`, `net_out`, `disk`), sem referência a GPU.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

CALENDAR_FEATURES = ("sin_day", "cos_day")
SECONDS_PER_DAY = 86_400.0
TARGET_SERIES = "cpu"
PERSISTENCE_INPUT = "cpu_t"

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|min|h)\s*$")
_UNIT_SECONDS = {"s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0}


class InformationSetError(ValueError):
    """A persistência usa informação que algum modelo não recebe."""


def parse_duration_steps(value, step_seconds: float) -> int:
    """Duração em passos da grade. Inteiro = passos; texto ('15min', '300s', '1h') = tempo."""
    if isinstance(value, bool):
        raise ValueError(f"Duração inválida: {value!r}")
    if isinstance(value, (int, np.integer)):
        steps = int(value)
    else:
        match = _DURATION.match(str(value))
        if not match:
            raise ValueError(f"Duração inválida: {value!r}. Use passos (inteiro) ou '15min', '300s', '1h'.")
        seconds = float(match.group(1)) * _UNIT_SECONDS[match.group(2)]
        ratio = seconds / step_seconds
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Duração {value!r} ({seconds:g} s) não é múltiplo do passo da grade ({step_seconds:g} s)."
            )
        steps = int(round(ratio))
    if steps < 1:
        raise ValueError(f"Duração {value!r} resulta em menos de 1 passo.")
    return steps


@dataclass(frozen=True)
class GroupSpec:
    name: str
    columns: tuple[str, ...]
    lags: tuple[int, ...]
    windows: tuple[int, ...]

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for column in self.columns:
            names.append(f"{column}_t")
            names.extend(f"{column}_lag_{lag}" for lag in self.lags)
            for window in self.windows:
                names.append(f"{column}_roll_mean_{window}")
                names.append(f"{column}_roll_std_{window}")
        return names

    @property
    def max_lookback(self) -> int:
        return max([0, *self.lags, *self.windows])


def build_group_specs(groups: dict, step_seconds: float) -> dict[str, GroupSpec]:
    specs = {}
    for name, cfg in groups.items():
        specs[name] = GroupSpec(
            name=name,
            columns=tuple(cfg["columns"]),
            lags=tuple(sorted({parse_duration_steps(v, step_seconds) for v in cfg.get("lags", [])})),
            windows=tuple(sorted({parse_duration_steps(v, step_seconds) for v in cfg.get("windows", [])})),
        )
    return specs


def horizon_steps(horizons: list, step_seconds: float) -> list[int]:
    steps = [parse_duration_steps(h, step_seconds) for h in horizons]
    if len(set(steps)) != len(steps):
        raise ValueError("Horizontes duplicados depois da conversão em passos.")
    return sorted(steps)


def target_column(steps: int) -> str:
    return f"y_h{steps}"


def max_window_steps(specs: dict[str, GroupSpec]) -> int:
    return max(spec.max_lookback for spec in specs.values())


def _assert_regular(series: pd.DataFrame) -> None:
    if "bin" in series.columns and not (series["bin"].diff().dropna() == 1).all():
        raise ValueError("A série de trabalho precisa de grade contígua (um intervalo por passo).")


def build_feature_table(series: pd.DataFrame, specs: dict[str, GroupSpec], steps: list[int]) -> pd.DataFrame:
    """Uma linha por origem com todas as features de todos os grupos e um alvo por horizonte.

    Origens sem alvo completo para o maior horizonte são removidas de todos os horizontes.
    Valores ausentes permanecem NaN; a política de origens incompletas é aplicada à parte.
    """
    _assert_regular(series)
    series = series.reset_index(drop=True)
    n = len(series)
    max_h = max(steps)
    if n <= max_h:
        raise ValueError("Série curta demais para o maior horizonte.")
    columns: dict[str, np.ndarray | pd.Series] = {
        "origin_index": np.arange(n),
        "timestamp": series["timestamp"].to_numpy(dtype=float),
    }
    for spec in specs.values():
        for column in spec.columns:
            values = series[column].astype(float)
            columns[f"{column}_t"] = values
            for lag in spec.lags:
                columns[f"{column}_lag_{lag}"] = values.shift(lag)
            for window in spec.windows:
                rolling = values.rolling(window, min_periods=window)
                columns[f"{column}_roll_mean_{window}"] = rolling.mean()
                columns[f"{column}_roll_std_{window}"] = rolling.std(ddof=0)
    phase = 2.0 * math.pi * (series["timestamp"].to_numpy(dtype=float) % SECONDS_PER_DAY) / SECONDS_PER_DAY
    columns["sin_day"] = np.sin(phase)
    columns["cos_day"] = np.cos(phase)
    target = series[TARGET_SERIES].astype(float)
    for step in steps:
        columns[target_column(step)] = target.shift(-step)
    table = pd.DataFrame(columns)
    return table.iloc[: n - max_h].reset_index(drop=True)


def ablation_columns(specs: dict[str, GroupSpec], ablation: dict[str, list[str]]) -> dict[str, list[str]]:
    """Colunas por configuração (A, B, C...) como listas ordenadas de grupos, mais o calendário.

    As configurações precisam ser aninhadas: cada uma é prefixo da seguinte.
    """
    unknown = {g for groups in ablation.values() for g in groups} - set(specs)
    if unknown:
        raise ValueError(f"Grupos de features desconhecidos na ablação: {sorted(unknown)}")
    ordered = list(ablation.items())
    for (_, smaller), (_, larger) in zip(ordered, ordered[1:]):
        if list(larger[: len(smaller)]) != list(smaller) or len(larger) <= len(smaller):
            raise ValueError("As configurações de ablação precisam ser aninhadas (cada uma estende a anterior).")
    for name, groups in ablation.items():
        if not groups or groups[0] != TARGET_SERIES or TARGET_SERIES not in specs[groups[0]].columns:
            raise ValueError(f"A configuração {name} precisa começar pelo grupo de CPU.")
    result = {}
    for name, groups in ablation.items():
        cols: list[str] = []
        for group in groups:
            cols.extend(specs[group].feature_names())
        result[name] = cols + list(CALENDAR_FEATURES)
    return result


def drop_incomplete_origins(table: pd.DataFrame, columns: list[str], step_list: list[int]) -> pd.DataFrame:
    """Origens com NaN em qualquer feature usada ou em algum alvo saem de todas as configurações.

    Não há imputação; o conjunto de origens é o mesmo para A, B, C e para a persistência.
    """
    needed = list(dict.fromkeys([*columns, *[target_column(s) for s in step_list]]))
    keep = table[needed].notna().all(axis=1)
    return table.loc[keep].reset_index(drop=True)


def assert_equal_information(persistence_inputs: list[str], model_columns: dict[str, list[str]]) -> None:
    """A entrada da persistência precisa fazer parte das features de todos os modelos."""
    for name, columns in model_columns.items():
        missing = [c for c in persistence_inputs if c not in columns]
        if missing:
            raise InformationSetError(
                f"A configuração {name} não contém {missing}, usado pela persistência; "
                "modelos e baseline devem ter o mesmo conjunto de informação na origem."
            )
