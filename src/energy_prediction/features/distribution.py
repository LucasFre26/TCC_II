"""Estatísticas transversais da frota na origem (grupo `dist` da configuração D).

Calculadas na grade por máquina já cacheada, intervalo a intervalo, só com as máquinas da
população principal que têm CPU naquele intervalo (mesma regra de ausência da agregação).
Cada máquina conta uma vez (estatística da distribuição entre máquinas, sem peso por cpu_num).
O valor da origem t usa só o intervalo t; a variação da dispersão usa t e t-k. Nenhum percentil
futuro da CPU agregada entra como atributo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.features.cpu_features import GroupSpec

DIST_GROUP = "dist"
BASE_COLUMNS = (
    "dist_std",
    "dist_p50",
    "dist_p90",
    "dist_p95",
    "dist_valid_frac",
    "dist_high_frac",
    "dist_std_change",
)
SKEW_COLUMN = "dist_skew"


def distribution_frame(
    grid: pd.DataFrame,
    population: list[str],
    bins: np.ndarray,
    *,
    elevated_threshold: float,
    change_steps: int,
    skewness: bool,
    skewness_max_undefined_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    """Uma linha por intervalo de `bins` (na ordem da série agregada) com as estatísticas transversais."""
    threshold = float(elevated_threshold)
    members = set(population)
    frame = grid.loc[grid["machine_id"].isin(members) & grid["cpu"].notna(), ["bin", "cpu"]]
    grouped = frame.groupby("bin")["cpu"]
    quantiles = grouped.quantile([0.5, 0.9, 0.95]).unstack()
    stats = pd.DataFrame(
        {
            "dist_std": grouped.std(ddof=0),
            "dist_p50": quantiles[0.5],
            "dist_p90": quantiles[0.9],
            "dist_p95": quantiles[0.95],
            "n_present": grouped.size(),
            "n_high": (frame["cpu"] >= threshold).groupby(frame["bin"]).sum(),
        }
    )
    if skewness:
        stats[SKEW_COLUMN] = grouped.skew()
    out = pd.DataFrame({"bin": np.asarray(bins, dtype=int)}).merge(stats, left_on="bin", right_index=True, how="left")
    n_population = max(len(members), 1)
    present = out["n_present"].fillna(0.0)
    out["dist_valid_frac"] = present / n_population
    out["dist_high_frac"] = (out["n_high"] / present).where(present > 0)
    out["dist_std_change"] = out["dist_std"] - out["dist_std"].shift(int(change_steps))
    info = {
        "population_size": int(len(members)),
        "elevated_threshold": threshold,
        "change_steps": int(change_steps),
        "weighting": "cada máquina conta uma vez",
        "bins_without_machines": int((present == 0).sum()),
        "skewness": {"requested": bool(skewness), "included": False},
    }
    columns = list(BASE_COLUMNS)
    if skewness:
        undefined = float(out[SKEW_COLUMN].isna().mean())
        info["skewness"].update(undefined_fraction=undefined, limit=float(skewness_max_undefined_fraction))
        if undefined <= float(skewness_max_undefined_fraction):
            columns.append(SKEW_COLUMN)
            info["skewness"]["included"] = True
        else:
            info["skewness"]["reason"] = "fração de origens com assimetria indefinida acima do limite"
    info["columns"] = columns
    return out[["bin", *columns]].reset_index(drop=True), info


def distribution_group(columns: list[str]) -> GroupSpec:
    """Grupo sem defasagens nem janelas: o valor de cada estatística na origem."""
    return GroupSpec(name=DIST_GROUP, columns=tuple(columns), lags=(), windows=())
