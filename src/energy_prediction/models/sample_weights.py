"""Pesos por exemplo de treino para o XGBoost (V3 Parte 2).

- Recência (H8): w_i = 2^(−Δt_i / meia-vida), com Δt medido em passos até a última origem de treino.
- Subida (H4): w_i = 1 + λ quando a variação do alvo até o horizonte, y(s+h) − y(s), é ≥ σ_h, e
  1 nos demais exemplos; σ_h é o desvio dessas variações no próprio treino. O peso usa só alvos de
  treino e não existe na previsão.
"""

from __future__ import annotations

import numpy as np


def recency_weights(origins: np.ndarray, half_life_steps: float) -> np.ndarray:
    origins = np.asarray(origins, dtype=float)
    if half_life_steps <= 0:
        raise ValueError("A meia-vida precisa ser positiva.")
    age = origins.max() - origins
    return np.power(2.0, -age / float(half_life_steps))


def rise_sigma(y_target: np.ndarray, y_origin: np.ndarray) -> float:
    change = np.asarray(y_target, dtype=float) - np.asarray(y_origin, dtype=float)
    return float(np.nanstd(change))


def rise_weights(y_target: np.ndarray, y_origin: np.ndarray, *, lam: float, sigma: float) -> np.ndarray:
    change = np.asarray(y_target, dtype=float) - np.asarray(y_origin, dtype=float)
    if float(lam) < 0:
        raise ValueError("λ não pode ser negativo.")
    return np.where(change >= float(sigma), 1.0 + float(lam), 1.0)
