"""Formato comum das previsões, métricas e comparações por dia das famílias da V3 Parte 2.

Previsões: uma linha por (modelo, variante, horizonte, bloco, fase, origem), com `phase` igual a
`inner` (validação interna do bloco) ou `block` (bloco avaliado). Erro = observado − previsto
(positivo = subestimação). Comparações primárias usam só os blocos de desenvolvimento; o teste
herdado (dia 7) é reportado à parte, com o número da inspeção.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.day_forward import DEVELOPMENT, DayBlock
from energy_prediction.evaluation.day_stats import apply_holm, compare_by_day
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT, target_column
from energy_prediction.runtime.output import dump_json

INNER = "inner"
BLOCK = "block"


def block_frame(table: pd.DataFrame, block: DayBlock, positions, *, phase: str, model: str, variant: str, steps: int, y_pred, **extra) -> pd.DataFrame:
    part = table.iloc[np.asarray(positions, dtype=int)]
    frame = pd.DataFrame(
        {
            "model": model,
            "variant": variant,
            "horizon_steps": int(steps),
            "block": block.label,
            "day": int(block.day),
            "role": block.role,
            "phase": phase,
            "origin_index": part["origin_index"].to_numpy(),
            "timestamp": part["timestamp"].to_numpy(),
            "cpu_t": part[PERSISTENCE_INPUT].to_numpy(dtype=float),
            "y_true": part[target_column(steps)].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def point_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    e = y - p
    return {
        "n": int(len(e)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias_obs_minus_pred": float(np.mean(e)),
        "max_abs_error": float(np.max(np.abs(e))) if len(e) else np.nan,
        "underestimation_fraction": float(np.mean(e > 0)) if len(e) else np.nan,
    }


def metrics_table(predictions: pd.DataFrame, keys=("model", "variant", "horizon_steps")) -> pd.DataFrame:
    """Métricas por bloco (fase `block`), média dos blocos de desenvolvimento e teste herdado."""
    rows = []
    block = predictions[predictions["phase"] == BLOCK]
    persistence = (block["y_true"] - block["cpu_t"]).abs()
    block = block.assign(_pers=persistence)
    for key, part in block.groupby(list(keys) + ["block", "day", "role"], sort=False):
        base = dict(zip(list(keys) + ["block", "day", "role"], key))
        m = point_metrics(part["y_true"], part["y_pred"])
        m["persistence_mae_same_origins"] = float(part["_pers"].mean())
        m["mase_vs_persistence"] = m["mae"] / m["persistence_mae_same_origins"] if m["persistence_mae_same_origins"] else np.nan
        m["train_level_shift"] = float(part["level_shift"].iloc[0]) if "level_shift" in part else np.nan
        rows.append({**base, **m})
    for key, part in block[block["role"] == DEVELOPMENT].groupby(list(keys), sort=False):
        base = dict(zip(list(keys), key if isinstance(key, tuple) else (key,)))
        m = point_metrics(part["y_true"], part["y_pred"])
        m["persistence_mae_same_origins"] = float(part["_pers"].mean())
        m["mase_vs_persistence"] = m["mae"] / m["persistence_mae_same_origins"] if m["persistence_mae_same_origins"] else np.nan
        rows.append({**base, "block": "development_pooled", "day": -1, "role": DEVELOPMENT, **m})
    return pd.DataFrame(rows)


def loss_frame(predictions: pd.DataFrame, *, model: str, variant: str, steps: int, role: str = DEVELOPMENT, loss=None, extra_filter=None) -> pd.DataFrame:
    mask = (predictions["model"] == model) & (predictions["variant"] == variant) & (predictions["horizon_steps"] == int(steps)) & (predictions["phase"] == BLOCK) & (predictions["role"] == role)
    if extra_filter is not None:
        mask &= extra_filter(predictions)
    part = predictions.loc[mask]
    if part.empty:
        raise KeyError(f"Sem previsões para {model}/{variant}/h{steps}/{role}.")
    if part.duplicated(["day", "origin_index"]).any():
        raise ValueError(f"Previsões duplicadas para {model}/{variant}/h{steps}; use a média das sementes antes da comparação.")
    values = loss(part) if loss is not None else (part["y_true"] - part["y_pred"]).abs()
    return pd.DataFrame({"day": part["day"].to_numpy(), "origin_index": part["origin_index"].to_numpy(), "loss": np.asarray(values, dtype=float)})


def stats_settings(cfg: dict) -> dict:
    st = cfg["statistics"]
    primary = int(st["primary_block_steps"])
    return {
        "blocks": list(dict.fromkeys([primary, *[int(b) for b in st["sensitivity_blocks"]]])),
        "primary_block": primary,
        "n_boot": int(st["n_boot"]),
        "seed": int(cfg["seed"]),
        "min_favorable_days": int(cfg["protocol"]["min_favorable_days"]),
    }


def compare(loss_a: pd.DataFrame, loss_b: pd.DataFrame, cfg: dict, meta: dict) -> tuple[list[dict], list[dict]]:
    return compare_by_day(loss_a, loss_b, meta=meta, **stats_settings(cfg))


def finalize_comparisons(rows: list[dict], *, alpha: float) -> pd.DataFrame:
    """Holm: primário sobre o conjunto primário da família (bloco primário); exploratório por horizonte."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = np.nan
    frame["holm_reject"] = False
    for scope, group in (("primary", None), ("exploratory", ["horizon_steps"])):
        mask = (frame["scope"] == scope) & frame["primary_block"]
        if mask.any():
            part = apply_holm(frame.loc[mask], group=group, alpha=alpha)
            frame.loc[mask, "p_holm"] = part["p_holm"]
            frame.loc[mask, "holm_reject"] = part["holm_reject"]
    return frame


def inherited_rows(predictions: pd.DataFrame, pairs: list[tuple[tuple[str, str], tuple[str, str]]], steps_list: list[int]) -> pd.DataFrame:
    """Diferença de MAE no teste herdado (dia 7), sem inferência: só continuidade com a V3."""
    rows = []
    block = predictions[(predictions["phase"] == BLOCK) & (predictions["role"] == "inherited_test")]
    if block.empty:
        return pd.DataFrame()
    for (ma, va), (mb, vb) in pairs:
        for steps in steps_list:
            a = block[(block["model"] == ma) & (block["variant"] == va) & (block["horizon_steps"] == steps)]
            b = block[(block["model"] == mb) & (block["variant"] == vb) & (block["horizon_steps"] == steps)]
            if a.empty or b.empty:
                continue
            rows.append(
                {
                    "model_a": ma, "variant_a": va, "model_b": mb, "variant_b": vb, "horizon_steps": int(steps),
                    "mae_a": float((a["y_true"] - a["y_pred"]).abs().mean()), "mae_b": float((b["y_true"] - b["y_pred"]).abs().mean()),
                    "role": "inherited_test", "inspection_number": 4,
                    "note": "teste herdado da V3 (quarta inspeção); continuidade, não confirmação",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mae_diff"] = out["mae_a"] - out["mae_b"]
    return out


def write_tables(out: Path, **tables) -> None:
    for name, frame in tables.items():
        if frame is None:
            continue
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(Path(out) / f"{name}.csv", index=False)
        else:
            dump_json(Path(out) / f"{name}.json", frame)


def read_family_csv(run_dir, family: str, name: str) -> pd.DataFrame:
    path = run_dir.family_path(family) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{family}/{name}.csv ausente na corrida {run_dir.run_id}.")
    return pd.read_csv(path)


def read_family_json(run_dir, family: str, name: str):
    path = run_dir.family_path(family) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{family}/{name}.json ausente na corrida {run_dir.run_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def level_shift_by_block(ctx, steps: int = 1) -> dict[str, float]:
    """Nível médio do alvo no bloco menos o do treino do bloco (diagnóstico de H8)."""
    y = ctx.table[target_column(steps)].to_numpy(dtype=float)
    return {b.label: float(np.mean(y[b.evaluation]) - np.mean(y[b.train])) for b in ctx.blocks}
