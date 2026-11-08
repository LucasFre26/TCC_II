"""Calibração conformal cronológica dos limites de CPU (H5).

Escores: limite superior s = y − q̂_sup; limite inferior s = q̂_inf − y. O limite calibrado é
q̂_sup + d (ou q̂_inf − d), com d o quantil empírico de nível ⌈(n+1)(1−α)⌉/n de um conjunto de
escores. O estado inicial de cada bloco são os escores da validação interna. Durante o bloco, na
origem t só entram escores de previsões feitas em s com s + h ≤ t (retorno atrasado em h).

Métodos:
- `static`: só os escores da validação interna (réplica da regra da V3);
- `rolling`: os últimos n escores resolvidos (validação interna + bloco);
- `aci`: inferência conformal adaptativa (Gibbs e Candès, 2021; referência a conferir): o nível
  efetivo α_t é atualizado por α ← α + γ(α_nominal − erro) a cada escore resolvido, com erro = 1
  quando o limite emitido para aquela origem não cobriu o alvo; o quantil usa o conjunto
  expansivo de escores resolvidos; α_t ≤ 0 dá limite infinito (contado);
- `spci`: regressão quantílica (XGBoost) do próximo escore sobre as `lags` últimas defasagens dos
  escores resolvidos, reajustada a cada `refit_every` origens (próxima de Xu e Xie, 2023; a
  conferir).
Nenhum método tem garantia de cobertura sob dependência temporal: a cobertura é empírica.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.inf
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    rank = math.ceil((n + 1) * (1.0 - float(alpha)))
    return float(s[rank - 1]) if rank <= n else np.inf


def _resolved_order(origins: np.ndarray, steps: int):
    """Para cada posição i, quantos escores do bloco (em ordem) já estão resolvidos em origins[i]."""
    o = np.asarray(origins, dtype=int)
    return np.searchsorted(o + int(steps), o, side="right")


def static_offsets(initial: np.ndarray, n_block: int, alpha: float) -> np.ndarray:
    return np.full(int(n_block), conformal_quantile(initial, alpha))


def rolling_offsets(initial: np.ndarray, block_scores: np.ndarray, origins: np.ndarray, steps: int, alpha: float, window: int) -> np.ndarray:
    resolved = _resolved_order(origins, steps)
    pool = np.concatenate([np.asarray(initial, dtype=float), np.asarray(block_scores, dtype=float)])
    n0 = len(initial)
    out = np.empty(len(origins))
    for i, k in enumerate(resolved):
        stop = n0 + int(k)
        out[i] = conformal_quantile(pool[max(0, stop - int(window)) : stop], alpha)
    return out


def aci_offsets(initial: np.ndarray, block_scores: np.ndarray, origins: np.ndarray, steps: int, alpha: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Deslocamentos e níveis efetivos α_t; o erro de cada origem usa o limite emitido para ela."""
    resolved = _resolved_order(origins, steps)
    scores = np.asarray(block_scores, dtype=float)
    pool = list(np.asarray(initial, dtype=float))
    alpha_t = float(alpha)
    offsets = np.empty(len(origins))
    levels = np.empty(len(origins))
    done = 0
    for i, k in enumerate(resolved):
        while done < int(k):
            err = 1.0 if scores[done] > offsets[done] else 0.0  # escore acima do deslocamento emitido = não cobriu
            alpha_t = alpha_t + float(gamma) * (float(alpha) - err)
            pool.append(scores[done])
            done += 1
        levels[i] = alpha_t
        offsets[i] = conformal_quantile(np.asarray(pool), alpha_t)
    return offsets, levels


def spci_offsets(initial: np.ndarray, block_scores: np.ndarray, origins: np.ndarray, steps: int, alpha: float, *, lags: int, n_estimators: int, refit_every: int, seed: int) -> np.ndarray:
    from xgboost import XGBRegressor

    resolved = _resolved_order(origins, steps)
    series = np.concatenate([np.asarray(initial, dtype=float), np.asarray(block_scores, dtype=float)])
    n0 = len(initial)
    out = np.empty(len(origins))
    model, fitted_at = None, -1
    for i, k in enumerate(resolved):
        available = series[: n0 + int(k)]
        if len(available) <= lags + 10:
            out[i] = conformal_quantile(available, alpha)
            continue
        if model is None or i - fitted_at >= int(refit_every):
            X = np.lib.stride_tricks.sliding_window_view(available[:-1], lags)
            y = available[lags:]
            model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=1.0 - float(alpha), n_estimators=int(n_estimators), max_depth=3, learning_rate=0.1, tree_method="hist", random_state=int(seed), n_jobs=1, verbosity=0).fit(X, y)
            fitted_at = i
        out[i] = float(model.predict(available[-lags:][None, :])[0])
    return out


def interval_score(y, lower, upper, alpha: float) -> np.ndarray:
    """Interval score (Winkler) do intervalo bilateral de nível 1 − α."""
    y, lower, upper = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    width = upper - lower
    return width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)


def sort_quantiles(Q: np.ndarray) -> tuple[np.ndarray, int]:
    """Ordena os quantis previstos por origem; devolve a matriz ordenada e o número de origens com cruzamento."""
    Q = np.asarray(Q, dtype=float)
    crossed = int(np.any(np.diff(Q, axis=1) < 0, axis=1).sum())
    return np.sort(Q, axis=1), crossed
