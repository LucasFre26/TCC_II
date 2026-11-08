"""Reconstrução e reconciliação da previsão agregada a partir dos grupos (H1).

O agregado é uma média ponderada pelos pesos presentes, então a matriz de agregação depende da
origem: S(t) = [s(t)ᵀ; I_K], com s_g(t) = peso presente do grupo / peso presente total na origem.
Previsões base: ŷ = [ŷ_total, ŷ_1..ŷ_K]. Métodos:
- de baixo para cima: total = s(t)·ŷ_g;
- OLS: ỹ = S (SᵀS)⁻¹ Sᵀ ŷ;
- MinT com encolhimento: ỹ = S (SᵀW⁻¹S)⁻¹ SᵀW⁻¹ ŷ, com W a covariância dos erros base na
  validação interna encolhida para a diagonal (intensidade de Schäfer–Strimmer).
Grupos sem peso na origem saem de s(t); grupos sem previsão base finita também saem, e s(t) é
renormalizado. Toda previsão reconciliada é coerente: total = s(t)·ỹ_g.
"""

from __future__ import annotations

import numpy as np

METHODS = ("bottom_up", "ols", "mint_shrink")


def origin_shares(weights_at_origin: np.ndarray) -> np.ndarray:
    w = np.asarray(weights_at_origin, dtype=float)
    total = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, w / np.where(total > 0, total, 1.0), np.nan)


def shrunk_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariância dos resíduos (linhas = origens) encolhida para a diagonal; devolve W e λ."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r).all(axis=1)]
    n, p = r.shape
    if n < 3:
        return np.eye(p), 1.0
    centered = r - r.mean(axis=0)
    sample = centered.T @ centered / n
    sd = np.sqrt(np.diag(sample))
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = centered / sd
    corr = z.T @ z / n
    w = np.einsum("ni,nj->nij", z, z)
    var_corr = n / (n - 1) ** 3 * ((w - corr[None]) ** 2).sum(axis=0)
    off = ~np.eye(p, dtype=bool)
    denom = float((corr[off] ** 2).sum())
    lam = 1.0 if denom <= 0 else float(np.clip(var_corr[off].sum() / denom, 0.0, 1.0))
    target = np.diag(np.diag(sample))
    shrunk = lam * target + (1.0 - lam) * sample
    return shrunk + 1e-9 * np.eye(p), lam


def reconcile(base_total: np.ndarray, base_groups: np.ndarray, shares: np.ndarray, method: str, W: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Total e grupos reconciliados por origem. `base_groups` e `shares` têm forma (origens, K)."""
    if method not in METHODS:
        raise ValueError(f"Método desconhecido: {method!r}")
    total_hat = np.asarray(base_total, dtype=float)
    groups_hat = np.asarray(base_groups, dtype=float)
    shares = np.asarray(shares, dtype=float)
    n, k = groups_hat.shape
    out_total = np.full(n, np.nan)
    out_groups = np.full((n, k), np.nan)
    for i in range(n):
        usable = np.isfinite(groups_hat[i]) & np.isfinite(shares[i]) & (shares[i] > 0)
        if not usable.any():
            continue
        s = shares[i, usable] / shares[i, usable].sum()
        g = groups_hat[i, usable]
        if method == "bottom_up" or not np.isfinite(total_hat[i]):
            out_groups[i, usable] = g
            out_total[i] = float(s @ g)
            continue
        S = np.vstack([s[None, :], np.eye(usable.sum())])
        y = np.concatenate([[total_hat[i]], g])
        if method == "ols":
            P = np.linalg.solve(S.T @ S, S.T)
        else:
            index = np.concatenate([[0], 1 + np.flatnonzero(usable)])
            Wi = np.linalg.inv((np.eye(k + 1) if W is None else W)[np.ix_(index, index)])
            P = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
        bottom = P @ y
        out_groups[i, usable] = bottom
        out_total[i] = float(s @ bottom)
    return out_total, out_groups
