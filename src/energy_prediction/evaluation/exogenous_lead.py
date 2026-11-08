"""Correlações que indicam se as variáveis exógenas antecedem a CPU na série agregada.

Mede, para cada exógena x e horizonte h (em passos da grade):
  - nível: corr(cpu_t, x_t);
  - antecedência: corr(x_t - x_{t-1}, cpu_{t+h} - cpu_t), isto é, se a variação recente de x
    antecipa a variação futura da CPU;
  - persistência da própria CPU: corr(cpu_t, cpu_{t-h}).
A série precisa estar numa grade regular; pontos ausentes são ignorados par a par.
"""

from __future__ import annotations

import pandas as pd


def lead_correlations(series: pd.DataFrame, *, target: str, exogenous: list[str], horizons: list[int]) -> dict:
    cpu = series[target].astype(float)
    future_change = {h: cpu.shift(-h) - cpu for h in horizons}
    out = {
        "target": target,
        "horizon_steps": [int(h) for h in horizons],
        "target_autocorrelation": {str(h): float(cpu.autocorr(h)) for h in horizons},
        "exogenous": {},
    }
    for name in exogenous:
        x = series[name].astype(float)
        recent_change = x.diff()
        out["exogenous"][name] = {
            "level_correlation": float(cpu.corr(x)),
            "lead_correlation": {str(h): float(recent_change.corr(future_change[h])) for h in horizons},
            "std": float(x.std()),
        }
    out["target_std"] = float(cpu.std())
    return out
