"""Diagnóstico temporal: anotação das origens, erro por regime e faixa, resíduos e janelas objetivas.

Regimes, faixas e janelas dependem só da série observada (nunca do erro de um modelo):
- regime em h passos: subida se y(t+h) − y(t) >= sigma_h, queda se <= −sigma_h, estável caso contrário,
  com sigma_h o desvio-padrão dessas variações no período de desenvolvimento;
- faixas de CPU: quantis da CPU observada no desenvolvimento, aplicados ao valor observado no alvo;
- janelas: duração fixa, escolhidas em sequência sem sobreposição.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

REGIMES = ["queda", "estável", "subida"]


def change_sigma(cpu: pd.Series, horizons: list[int], dev_end: int) -> dict[int, float]:
    dev = cpu.loc[:dev_end]
    return {h: float((dev.shift(-h) - dev).loc[: dev_end - h].std()) for h in horizons}


def band_edges(cpu: pd.Series, dev_end: int, n_bands: int) -> np.ndarray:
    values = cpu.loc[:dev_end].dropna().to_numpy()
    inner = np.quantile(values, np.linspace(0, 1, n_bands + 1)[1:-1])
    return np.r_[-np.inf, inner, np.inf]


def band_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for k in range(len(edges) - 1):
        low = "mín" if not np.isfinite(edges[k]) else f"{edges[k]:.1f}"
        high = "máx" if not np.isfinite(edges[k + 1]) else f"{edges[k + 1]:.1f}"
        labels.append(f"Q{k + 1} [{low}; {high})")
    return labels


def annotate(predictions: pd.DataFrame, cpu: pd.Series, *, dev_end: int, horizons: list[int], n_bands: int, change_bins: list[float]) -> tuple[pd.DataFrame, dict]:
    """Acrescenta CPU na origem, variações recentes e realizadas, regime e faixa de CPU."""
    sigma = change_sigma(cpu, horizons, dev_end)
    edges = band_edges(cpu, dev_end, n_bands)
    labels = band_labels(edges)
    out = predictions.copy()
    origin = out["origin_index"].to_numpy()
    values = cpu.reindex(np.arange(int(cpu.index.max()) + 1))
    at = lambda idx: values.reindex(idx).to_numpy()  # noqa: E731
    out["cpu_at_origin_pct"] = at(origin)
    out["recent_change_1_step_pp"] = out["cpu_at_origin_pct"] - at(origin - 1)
    out["recent_change_6_steps_pp"] = out["cpu_at_origin_pct"] - at(origin - 6)
    out["realized_change_pp"] = out["y_obs_cpu_pct"] - out["cpu_at_origin_pct"]
    s = out["horizon_steps"].map(sigma).to_numpy()
    out["regime_sigma_pp"] = s
    out["regime"] = np.select([out["realized_change_pp"] >= s, out["realized_change_pp"] <= -s], ["subida", "queda"], "estável")
    out["cpu_band"] = pd.cut(out["y_obs_cpu_pct"], edges, labels=labels, right=False).astype(str)
    bins = [float(b) for b in change_bins]
    def edge(value: float) -> str:
        return "-∞" if value <= -1e8 else ("+∞" if value >= 1e8 else f"{value:g}")

    bin_labels = [f"[{edge(b)}; {edge(c)})" for b, c in zip(bins[:-1], bins[1:])]
    out["recent_change_1_step_bin"] = pd.cut(out["recent_change_1_step_pp"], bins, labels=bin_labels, right=False).astype(str)
    out["realized_change_bin"] = pd.cut(out["realized_change_pp"], bins, labels=bin_labels, right=False).astype(str)
    info = {"regime_sigma_pp": sigma, "cpu_band_edges_pct": edges.tolist(), "cpu_band_labels": labels, "change_bin_labels": bin_labels, "dev_end_index": dev_end}
    return out, info


def errors_by_group(annotated: pd.DataFrame, groupings: list[str]) -> pd.DataFrame:
    rows = []
    keys = ["model", "config", "mode", "horizon_steps", "horizon_min"]
    for grouping in groupings:
        for key, part in annotated.groupby(keys + [grouping], sort=False, observed=True):
            e = part["error_obs_minus_pred_pp"].to_numpy()
            rows.append({**dict(zip(keys, key[:-1])), "grouping": grouping, "group": key[-1], "n": len(e), "mae_pp": float(np.mean(np.abs(e))), "bias_obs_minus_pred_pp": float(np.mean(e)), "rmse_pp": float(np.sqrt(np.mean(e**2)))})
    return pd.DataFrame(rows)


def error_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    """Quantis do erro assinado e do erro absoluto, formato longo."""
    rows = []
    levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    keys = ["population", "model", "config", "mode", "split", "horizon_steps", "horizon_min"]
    for key, part in predictions.groupby(keys, sort=False):
        e = part["error_obs_minus_pred_pp"].to_numpy()
        for kind, values in (("erro_assinado_obs_menos_prev", e), ("erro_absoluto", np.abs(e))):
            q = np.quantile(values, levels)
            for level, value in zip(levels, q):
                rows.append({**dict(zip(keys, key)), "error_kind": kind, "statistic": f"p{int(round(level * 100)):02d}", "value_pp": float(value)})
            rows.append({**dict(zip(keys, key)), "error_kind": kind, "statistic": "media", "value_pp": float(values.mean())})
            rows.append({**dict(zip(keys, key)), "error_kind": kind, "statistic": "maximo", "value_pp": float(values.max())})
    return pd.DataFrame(rows)


def acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=float) - np.mean(values)
    denom = float(np.sum(x**2))
    return np.array([1.0] + [float(np.sum(x[k:] * x[:-k]) / denom) for k in range(1, max_lag + 1)])


def residual_acf(test: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """ACF dos resíduos de teste por modelo e horizonte, banda ±1,96/√n e Ljung–Box (6, 12, 24).

    Em previsão de h passos com origens consecutivas, autocorrelação até h−1 é esperada
    mesmo para um preditor ótimo; ela não é, sozinha, sinal de modelo mal especificado.
    """
    rows = []
    for key, part in test.groupby(["model", "config", "mode", "horizon_steps"], sort=False):
        e = part.sort_values("origin_index")["error_obs_minus_pred_pp"].to_numpy()
        n = len(e)
        r = acf(e, max_lag)
        lb = {}
        for m in (6, 12, 24):
            q = n * (n + 2) * np.sum(r[1 : m + 1] ** 2 / (n - np.arange(1, m + 1)))
            lb[m] = (float(q), float(stats.chi2.sf(q, m)))
        for lag in range(1, max_lag + 1):
            rows.append(
                {
                    "model": key[0], "config": key[1], "mode": key[2], "horizon_steps": int(key[3]), "lag_steps": lag, "lag_min": lag * 5.0,
                    "acf": r[lag], "band_95": 1.96 / np.sqrt(n), "n": n,
                    "ljung_box_q_6": lb[6][0], "ljung_box_p_6": lb[6][1], "ljung_box_q_12": lb[12][0], "ljung_box_p_12": lb[12][1],
                    "ljung_box_q_24": lb[24][0], "ljung_box_p_24": lb[24][1],
                }
            )
    return pd.DataFrame(rows)


def objective_windows(cpu: pd.Series, start: int, end: int, *, length: int, change_steps: int, step_seconds: float) -> pd.DataFrame:
    """Janelas de diagnóstico sobre a série observada em [start, end], sem usar previsões."""
    y = cpu.loc[start:end]
    idx = y.index.to_numpy()
    values = y.to_numpy(dtype=float)
    starts = np.arange(0, len(values) - length + 1)

    def score(kind: str, s: int) -> float:
        w = values[s : s + length]
        if kind in ("volatilidade", "estabilidade"):
            return float(np.std(np.diff(w)))
        moves = w[change_steps:] - w[:-change_steps]
        return float(moves.max() if kind == "subida" else moves.min())

    rules = {
        "volatilidade": ("maior desvio-padrão da primeira diferença na janela", max),
        "subida": (f"maior subida em {change_steps} passos dentro da janela", max),
        "queda": (f"maior queda em {change_steps} passos dentro da janela", min),
        "estabilidade": ("menor desvio-padrão da primeira diferença na janela", min),
    }
    taken: list[tuple[int, int]] = []
    rows = []
    for kind, (rule, pick) in rules.items():
        candidates = [(score(kind, s), s) for s in starts if all(s + length <= a or s >= b for a, b in taken)]
        if not candidates:
            continue
        value, s = pick(candidates, key=lambda c: c[0])
        taken.append((s, s + length))
        rows.append(
            {
                "window": kind, "rule": rule, "criterion_value_pp": value, "start_index": int(idx[s]), "end_index": int(idx[s + length - 1]),
                "start_time_h": idx[s] * step_seconds / 3600.0, "end_time_h": idx[s + length - 1] * step_seconds / 3600.0, "length_steps": length,
                "selection_order": len(rows) + 1,
            }
        )
    return pd.DataFrame(rows)
