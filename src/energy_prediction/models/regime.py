"""Detector causal de regime e combinação de especialistas (H3).

Na origem t, só com a série agregada até t:
- inclinação = média dos últimos `slope_steps` passos − média dos `slope_steps` anteriores;
- volatilidade = desvio das últimas `volatility_steps` variações de um passo.
Limiares: quantis pré-registrados no treino do bloco. Alta volatilidade tem precedência; depois
subida (inclinação ≥ q_subida) e queda (inclinação ≤ q_queda); o resto é estável.
Combinadores: seleção do especialista de menor MAE no regime, ponderação pelo inverso do MAE no
regime, e as referências sem regime (média simples e inverso do MAE global). Regime com menos
origens que o mínimo usa os pesos globais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STABLE, RISE, FALL, VOLATILE = "estavel", "subida", "queda", "alta_volatilidade"
REGIMES = (STABLE, RISE, FALL, VOLATILE)


def regime_signals(cpu: np.ndarray, *, slope_steps: int, volatility_steps: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(np.asarray(cpu, dtype=float))
    recent = s.rolling(int(slope_steps), min_periods=int(slope_steps)).mean()
    slope = recent - recent.shift(int(slope_steps))
    volatility = s.diff().rolling(int(volatility_steps), min_periods=int(volatility_steps)).std(ddof=0)
    return slope.to_numpy(), volatility.to_numpy()


def thresholds(slope: np.ndarray, volatility: np.ndarray, train_origins: np.ndarray, *, volatility_quantile: float, rise_quantile: float, fall_quantile: float) -> dict:
    o = np.asarray(train_origins, dtype=int)
    return {
        "volatility": float(np.nanquantile(volatility[o], volatility_quantile)),
        "rise": float(np.nanquantile(slope[o], rise_quantile)),
        "fall": float(np.nanquantile(slope[o], fall_quantile)),
    }


def classify(slope: np.ndarray, volatility: np.ndarray, origins: np.ndarray, limits: dict) -> np.ndarray:
    o = np.asarray(origins, dtype=int)
    sl, vo = slope[o], volatility[o]
    out = np.full(len(o), STABLE, dtype=object)
    out[sl <= limits["fall"]] = FALL
    out[sl >= limits["rise"]] = RISE
    out[vo > limits["volatility"]] = VOLATILE
    return out


def _mae_cols(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(P - y[:, None]), axis=0)


def fit_weights(y: np.ndarray, P: np.ndarray, regimes: np.ndarray, *, method: str, min_origins: int) -> tuple[dict[str, np.ndarray], list[str]]:
    """Pesos por regime (soma 1). `method`: select | inverse_mae | mean | global_inverse_mae."""
    y = np.asarray(y, dtype=float)
    P = np.asarray(P, dtype=float)
    n_e = P.shape[1]
    if method == "mean":
        w = np.full(n_e, 1.0 / n_e)
        return {r: w for r in REGIMES}, []
    global_mae = _mae_cols(y, P)
    global_inv = (1.0 / np.maximum(global_mae, 1e-9)) / np.sum(1.0 / np.maximum(global_mae, 1e-9))
    if method == "global_inverse_mae":
        return {r: global_inv for r in REGIMES}, []
    weights, fallback = {}, []
    for r in REGIMES:
        mask = regimes == r
        if mask.sum() < int(min_origins):
            weights[r] = global_inv if method == "inverse_mae" else np.eye(n_e)[int(np.argmin(global_mae))]
            fallback.append(r)
            continue
        m = _mae_cols(y[mask], P[mask])
        if method == "select":
            weights[r] = np.eye(n_e)[int(np.argmin(m))]
        elif method == "inverse_mae":
            inv = 1.0 / np.maximum(m, 1e-9)
            weights[r] = inv / inv.sum()
        else:
            raise ValueError(f"Combinador desconhecido: {method!r}")
    return weights, fallback


def combine(P: np.ndarray, regimes: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    W = np.vstack([weights[r] for r in regimes])
    return np.sum(np.asarray(P, dtype=float) * W, axis=1)
