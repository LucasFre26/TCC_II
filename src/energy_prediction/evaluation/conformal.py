"""Ajuste conformal do teto do intervalo, calibrado só na validação externa.

Escore de calibração: s = y - q̂ (resíduo do quantil superior) nas origens da validação externa,
concatenadas em ordem cronológica. O ajuste é o quantil empírico de nível ⌈(n+1)(1-α)⌉/n desses
escores, somado ao quantil previsto no teste. O rótulo do teste não entra. Sob dependência
temporal a cobertura é empírica, sem garantia teórica. Abaixo do mínimo de resíduos, o ajuste
não é aplicado.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_offset(scores, alpha: float, *, min_residuals: int) -> dict:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    record = {"alpha": float(alpha), "n_residuals": n, "min_residuals": int(min_residuals)}
    if n < int(min_residuals):
        return {**record, "applied": False, "offset": None, "reason": "resíduos de calibração abaixo do mínimo"}
    rank = math.ceil((n + 1) * (1.0 - float(alpha)))
    if rank > n:
        return {**record, "applied": True, "offset": float("inf"), "rank": rank, "reason": "posto acima de n; teto infinito"}
    offset = float(np.sort(values)[rank - 1])
    return {**record, "applied": True, "offset": offset, "rank": rank}


def adjusted_upper(q_upper_test, calibration: dict) -> np.ndarray | None:
    """Teto ajustado; None quando a calibração não foi aplicada (não se publica cobertura teórica)."""
    if not calibration["applied"]:
        return None
    return np.asarray(q_upper_test, dtype=float) + float(calibration["offset"])
