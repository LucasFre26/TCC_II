"""Erros extremos, erro nos picos e block bootstrap da diferença de erro absoluto.

Opera sobre a tabela de previsões da V2 (uma linha por origem) e não altera a seleção de modelos.
Convenção de sinal: erro = y_true - y_pred; positivo significa que o modelo subestimou.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSISTENCE = ("persistence", "-")


def extreme_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Erro absoluto máximo, P95 do erro absoluto, maior subestimação e fração de subestimações."""
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    absolute = np.abs(error)
    return {
        "max_abs_error": float(absolute.max()),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "max_underestimation": float(max(error.max(), 0.0)),
        "underestimation_fraction": float(np.mean(error > 0)),
    }


def peak_errors(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> dict:
    """MAE e viés médio nas origens em que o valor real atinge o quantil indicado."""
    truth = np.asarray(y_true, dtype=float)
    error = truth - np.asarray(y_pred, dtype=float)
    threshold = float(np.quantile(truth, quantile))
    peak = truth >= threshold
    return {
        "peak_threshold": threshold,
        "peak_n": int(peak.sum()),
        "peak_mae": float(np.mean(np.abs(error[peak]))),
        "peak_bias": float(np.mean(error[peak])),
    }


def block_indices(n: int, block: int, rng: np.random.Generator, n_boot: int = 1) -> np.ndarray:
    """Índices (n_boot, n) de reamostragens por blocos móveis contíguos, truncadas em n."""
    if not 1 <= block <= n:
        raise ValueError(f"Bloco de {block} passos inválido para série de {n} pontos.")
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def moving_block_bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, *, block: int, n_boot: int, seed: int) -> dict:
    """Média de loss_a - loss_b e IC 95% por percentis do block bootstrap."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("As séries de erro precisam ter o mesmo tamanho e as mesmas origens.")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[block_indices(len(diff), block, rng, n_boot)].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "significant": bool(high < 0 or low > 0)}


def default_pairs(predictions: pd.DataFrame) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Cada modelo ajustado contra a persistência e, em cada modelo, B e C contra A."""
    fitted = predictions.loc[predictions["model"] != PERSISTENCE[0], ["model", "config"]].drop_duplicates()
    pairs = [((m, c), PERSISTENCE) for m, c in fitted.itertuples(index=False)]
    for model, group in fitted.groupby("model"):
        configs = sorted(group["config"])
        for config in configs[1:]:
            pairs.append(((model, config), (model, configs[0])))
    return pairs


def _absolute_errors(frame: pd.DataFrame, model: str, config: str) -> pd.Series:
    part = frame[(frame["model"] == model) & (frame["config"] == config)]
    return (part["y_true"] - part["y_pred"]).abs().set_axis(part["origin_index"].to_numpy()).sort_index()


def bootstrap_table(predictions: pd.DataFrame, pairs, *, block: int, n_boot: int, seed: int, split: str = "test_final") -> pd.DataFrame:
    """IC 95% da diferença de MAE para cada par, por horizonte, nas mesmas origens."""
    rows = []
    data = predictions[predictions["split"] == split]
    for steps, frame in data.groupby("horizon_steps"):
        for (model_a, config_a), (model_b, config_b) in pairs:
            loss_a = _absolute_errors(frame, model_a, config_a)
            loss_b = _absolute_errors(frame, model_b, config_b)
            if not loss_a.index.equals(loss_b.index):
                raise ValueError(f"Origens diferentes entre {model_a}/{config_a} e {model_b}/{config_b}.")
            stats = moving_block_bootstrap_diff(loss_a.to_numpy(), loss_b.to_numpy(), block=block, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "split": split,
                    "horizon_steps": int(steps),
                    "model_a": model_a,
                    "config_a": config_a,
                    "model_b": model_b,
                    "config_b": config_b,
                    "block_steps": int(block),
                    "n_boot": int(n_boot),
                    "n": int(len(loss_a)),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def error_extremes_table(predictions: pd.DataFrame, *, peak_quantile: float, split: str = "test_final") -> pd.DataFrame:
    """Erros extremos e de pico por horizonte, modelo e configuração."""
    rows = []
    data = predictions[predictions["split"] == split]
    for (steps, model, config), frame in data.groupby(["horizon_steps", "model", "config"], sort=True):
        y_true, y_pred = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
        rows.append(
            {
                "split": split,
                "horizon_steps": int(steps),
                "model": model,
                "config": config,
                "n": int(len(frame)),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                **extreme_errors(y_true, y_pred),
                **peak_errors(y_true, y_pred, peak_quantile),
            }
        )
    return pd.DataFrame(rows)
