"""Comparações pré-registradas da V3 e block bootstrap pareado (função da V2).

Cada par compara a perda por origem de dois preditores nas mesmas origens do teste final.
O intervalo de 95% vem dos percentis 2,5 e 97,5. Só o bloco primário (1 h) sustenta afirmação;
os outros blocos são sensibilidade. Intervalo que contém zero não é significativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.evaluation.error_analysis import moving_block_bootstrap_diff
from energy_prediction.evaluation.nested import AlignmentError


def loss_series(predictions: pd.DataFrame, *, model: str, config: str, horizon_steps: int, split: str = "test_final", loss=None, **filters) -> pd.Series:
    """Perda por origem (padrão: erro absoluto), indexada por origin_index."""
    mask = (
        (predictions["model"] == model)
        & (predictions["config"] == config)
        & (predictions["horizon_steps"] == horizon_steps)
        & (predictions["split"] == split)
    )
    for column, value in filters.items():
        mask &= predictions[column] == value
    part = predictions.loc[mask]
    if part.empty:
        raise KeyError(f"Sem previsões para {model}/{config}/h{horizon_steps}/{split} {filters}.")
    values = loss(part) if loss is not None else (part["y_true"] - part["y_pred"]).abs()
    return pd.Series(np.asarray(values, dtype=float), index=part["origin_index"].to_numpy()).sort_index()


def compare_losses(loss_a: pd.Series, loss_b: pd.Series, *, blocks: list[int], primary_block: int, n_boot: int, seed: int, meta: dict) -> list[dict]:
    if not loss_a.index.equals(loss_b.index):
        raise AlignmentError(f"Origens diferentes no par {meta}.")
    rows = []
    for block in blocks:
        stats = moving_block_bootstrap_diff(loss_a.to_numpy(), loss_b.to_numpy(), block=int(block), n_boot=int(n_boot), seed=int(seed))
        rows.append(
            {
                **meta,
                "block_steps": int(block),
                "primary_block": bool(int(block) == int(primary_block)),
                "n_boot": int(n_boot),
                "n": int(len(loss_a)),
                **stats,
                "claim_allowed": bool(stats["significant"] and int(block) == int(primary_block)),
            }
        )
    return rows


def bootstrap_blocks(cfg: dict) -> tuple[list[int], int, int]:
    boot = cfg["comparisons"]["bootstrap"]
    primary = int(boot["primary_block_steps"])
    blocks = list(dict.fromkeys([primary, *[int(b) for b in boot["sensitivity_blocks"]]]))
    return blocks, primary, int(boot["n_boot"])


def load_v2_reference(v2_dir, *, config: str = "A") -> dict:
    """Previsões do XGBoost da V2 no teste final, sem reajuste. None se a pasta não estiver configurada."""
    if v2_dir is None:
        return {}
    predictions = pd.read_csv(v2_dir / "predictions.csv")
    test = predictions[(predictions["split"] == "test_final") & (predictions["model"] == "xgboost") & (predictions["config"] == config)]
    return {int(h): part.copy() for h, part in test.groupby("horizon_steps")}
