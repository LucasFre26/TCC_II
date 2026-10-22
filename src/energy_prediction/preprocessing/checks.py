"""Conferência do passo temporal observado."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class IrregularStepError(ValueError):
    """O passo mediano não é o valor configurado."""


def median_step_seconds(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        raise IrregularStepError("Série curta demais para medir o passo.")
    deltas = np.diff(timestamps.to_numpy(dtype=float))
    return float(np.median(deltas))


def assert_expected_step(timestamps: pd.Series, expected_step_seconds: float) -> float:
    observed = median_step_seconds(timestamps)
    if not math.isclose(observed, expected_step_seconds, rel_tol=0.0, abs_tol=1e-6):
        raise IrregularStepError(
            f"Passo mediano {observed} s difere do esperado {expected_step_seconds} s."
        )
    return observed
