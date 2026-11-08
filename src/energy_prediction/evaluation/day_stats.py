"""Inferência pareada estratificada por dia (V3 Parte 2).

A diferença de perda (candidato − referência) é reamostrada por blocos móveis dentro de cada dia
de desenvolvimento; as reamostragens de cada dia são concatenadas e a média do conjunto é a
estatística. Nenhum bloco atravessa a fronteira entre dias. Regra de decisão pré-registrada:

- melhor: IC de 95% inteiro abaixo de zero e diferença favorável (< 0) em pelo menos
  `min_favorable_days` dias;
- pior: IC inteiro acima de zero e diferença desfavorável (> 0) em pelo menos esse número de dias;
- não distinguível: qualquer outro caso.

O p-valor é bicaudal, das mesmas reamostragens, com a correção (k + 1)/(B + 1). Holm é aplicado
sobre o conjunto primário e, separadamente, sobre o conjunto exploratório de cada horizonte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.evaluation.error_analysis import block_indices
from energy_prediction.evaluation.nested import AlignmentError

BETTER = "melhor"
WORSE = "pior"
NULL = "não distinguível"


def stratified_bootstrap_means(diff_by_day: dict[int, np.ndarray], *, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Médias de B reamostragens: blocos móveis dentro de cada dia, dias concatenados."""
    rng = np.random.default_rng(int(seed))
    total = sum(len(v) for v in diff_by_day.values())
    sums = np.zeros(int(n_boot))
    for day in sorted(diff_by_day):
        values = np.asarray(diff_by_day[day], dtype=float)
        size = min(int(block), len(values))
        index = block_indices(len(values), size, rng, int(n_boot))
        sums += values[index].sum(axis=1)
    return sums / total


def two_sided_p(means: np.ndarray) -> float:
    means = np.asarray(means, dtype=float)
    below = (np.sum(means <= 0) + 1) / (len(means) + 1)
    above = (np.sum(means >= 0) + 1) / (len(means) + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def decide(ci_low: float, ci_high: float, per_day: list[float], min_favorable_days: int) -> str:
    favorable = sum(1 for d in per_day if d < 0)
    unfavorable = sum(1 for d in per_day if d > 0)
    if ci_high < 0 and favorable >= int(min_favorable_days):
        return BETTER
    if ci_low > 0 and unfavorable >= int(min_favorable_days):
        return WORSE
    return NULL


def compare_by_day(
    loss_a: pd.DataFrame,
    loss_b: pd.DataFrame,
    *,
    blocks: list[int],
    primary_block: int,
    n_boot: int,
    seed: int,
    min_favorable_days: int,
    meta: dict,
) -> tuple[list[dict], list[dict]]:
    """`loss_a`/`loss_b`: colunas day, origin_index, loss. Devolve linhas agregadas (uma por bloco) e por dia."""
    a = loss_a.sort_values(["day", "origin_index"]).reset_index(drop=True)
    b = loss_b.sort_values(["day", "origin_index"]).reset_index(drop=True)
    if len(a) != len(b) or not (a["origin_index"].to_numpy() == b["origin_index"].to_numpy()).all() or not (a["day"].to_numpy() == b["day"].to_numpy()).all():
        raise AlignmentError(f"Origens diferentes no par {meta}.")
    diff = a["loss"].to_numpy(dtype=float) - b["loss"].to_numpy(dtype=float)
    days = a["day"].to_numpy()
    by_day = {int(d): diff[days == d] for d in np.unique(days)}
    per_day_rows = []
    per_day_means = []
    for d in sorted(by_day):
        mask = days == d
        per_day_means.append(float(by_day[d].mean()))
        per_day_rows.append(
            {
                **meta,
                "day": int(d),
                "n": int(mask.sum()),
                "loss_a": float(a.loc[mask, "loss"].mean()),
                "loss_b": float(b.loc[mask, "loss"].mean()),
                "mean_diff": float(by_day[d].mean()),
            }
        )
    rows = []
    for block in blocks:
        means = stratified_bootstrap_means(by_day, block=int(block), n_boot=int(n_boot), seed=int(seed))
        low, high = np.percentile(means, [2.5, 97.5])
        rows.append(
            {
                **meta,
                "block_steps": int(block),
                "primary_block": bool(int(block) == int(primary_block)),
                "n_boot": int(n_boot),
                "n": int(len(diff)),
                "n_days": int(len(by_day)),
                "loss_a": float(a["loss"].mean()),
                "loss_b": float(b["loss"].mean()),
                "mean_diff": float(diff.mean()),
                "relative_diff": float(diff.mean() / b["loss"].mean()) if b["loss"].mean() else np.nan,
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": two_sided_p(means),
                "days_favorable": int(sum(1 for m in per_day_means if m < 0)),
                "days_unfavorable": int(sum(1 for m in per_day_means if m > 0)),
                "decision": decide(float(low), float(high), per_day_means, min_favorable_days),
            }
        )
    return rows, per_day_rows


def holm(p_values) -> np.ndarray:
    """P-valores ajustados por Holm (passo descendente), na ordem de entrada."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def apply_holm(frame: pd.DataFrame, *, group: list[str] | None = None, column: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    """Acrescenta p_holm e holm_reject; com `group`, a correção é feita dentro de cada grupo."""
    out = frame.copy()
    out["p_holm"] = np.nan
    if out.empty:
        out["holm_reject"] = []
        return out
    by = group[0] if group and len(group) == 1 else group
    groups = [((), out.index)] if not group else list(out.groupby(by).groups.items())
    for _, index in groups:
        out.loc[index, "p_holm"] = holm(out.loc[index, column].to_numpy())
    out["holm_reject"] = out["p_holm"] < float(alpha)
    return out
