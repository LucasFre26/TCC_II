"""Pinball, cobertura empírica, subestimação e erro nos picos para previsões quantílicas.

Convenção: erro = y - q̂; positivo significa que o quantil ficou abaixo do observado (subestimação).
Métricas de pico usam o P90 da CPU observada no próprio bloco avaliado e ficam em tabela separada.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.evaluation.error_analysis import extreme_errors, peak_errors


def pinball_loss(y_true, y_quantile, quantile: float) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_quantile, dtype=float)
    q = float(quantile)
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def quantile_mean_metrics(y_true, y_quantile, quantile: float) -> dict:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_quantile, dtype=float)
    error = truth - pred
    under = error > 0
    return {
        "n": int(len(truth)),
        "pinball": pinball_loss(truth, pred, quantile),
        "coverage_below_quantile": float(np.mean(truth <= pred)),
        "underestimation_frequency": float(np.mean(under)),
        "underestimation_magnitude": float(np.mean(error[under])) if under.any() else 0.0,
        "mae": float(np.mean(np.abs(error))),
    }


def quantile_peak_metrics(y_true, y_quantile, peak_quantile: float) -> dict:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_quantile, dtype=float)
    peaks = peak_errors(truth, pred, peak_quantile)
    extremes = extreme_errors(truth, pred)
    return {
        "peak_quantile": float(peak_quantile),
        **peaks,
        "max_abs_error": extremes["max_abs_error"],
        "p95_abs_error": extremes["p95_abs_error"],
    }


def interval_metrics(y_true, lower, upper) -> dict:
    truth = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return {
        "mean_width": float(np.mean(hi - lo)),
        "interval_coverage": float(np.mean((truth >= lo) & (truth <= hi))),
        "upper_coverage": float(np.mean(truth <= hi)),
        "crossing_fraction": float(np.mean(hi < lo)),
    }
