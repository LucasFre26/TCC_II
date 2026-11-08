"""Calibração dos quantis Q50, Q90 e Q95 e dos tetos ajustados.

- Cobertura é empírica (fração de y <= q); sob dependência temporal não há garantia teórica.
- Q90 e Q95 são quantis unilaterais. A margem q95 − q50 não é largura de intervalo bilateral:
  não existe quantil inferior abaixo de Q50 nesta corrida.
- O teto ajustado foi calibrado nos resíduos da validação externa; por isso só o teste o avalia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.analysis.statistics import pinball

VARIANTS = [
    ("Q50", "bruto", 0.50, "q50_cpu_pct"),
    ("Q90", "bruto", 0.90, "q90_cpu_pct"),
    ("Q95", "bruto", 0.95, "q95_cpu_pct"),
    ("Q90", "ajustado", 0.90, "q90_adjusted_cpu_pct"),
    ("Q95", "ajustado", 0.95, "q95_adjusted_cpu_pct"),
]


def _scores(y: np.ndarray, q: np.ndarray, q50: np.ndarray, nominal: float, peak: np.ndarray) -> dict:
    covered = y <= q
    excess = y - q
    return {
        "n": int(len(y)), "nominal_coverage": nominal, "empirical_coverage": float(covered.mean()), "coverage_gap": float(covered.mean() - nominal),
        "exceedance_rate": float(1.0 - covered.mean()), "pinball_pp": float(pinball(excess, nominal).mean()),
        "mean_exceedance_pp": float(excess[~covered].mean()) if (~covered).any() else np.nan, "max_exceedance_pp": float(excess.max()) if len(excess) else np.nan,
        "mean_upper_margin_above_q50_pp": float(np.mean(q - q50)) if nominal > 0.5 else np.nan,
        "n_peak": int(peak.sum()), "peak_coverage": float(covered[peak].mean()) if peak.any() else np.nan,
        "peak_mean_exceedance_pp": float(excess[peak & ~covered].mean()) if (peak & ~covered).any() else np.nan,
    }


def calibration_table(qt: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
    rows = []
    frames = {"validation_pooled": qt[qt["split"].str.startswith("fold_")], "test_final": qt[qt["split"] == "test_final"]}
    for split, frame in frames.items():
        for (config, h), part in frame.groupby(["config", "horizon_steps"]):
            y = part["y_obs_cpu_pct"].to_numpy()
            peak = y >= peak_threshold
            for name, variant, nominal, column in VARIANTS:
                if column not in part or part[column].isna().all():
                    continue
                rows.append({"split": split, "config": config, "horizon_steps": int(h), "horizon_min": h * 5.0, "quantile": name, "variant": variant, "peak_threshold_cpu_pct": peak_threshold, **_scores(y, part[column].to_numpy(), part["q50_cpu_pct"].to_numpy(), nominal, peak)})
    return pd.DataFrame(rows)


def pinball_check(qt: pd.DataFrame, metrics: pd.DataFrame | None, tolerance: float) -> dict:
    if metrics is None:
        return {"check": "reproducao_pinball", "passed": False, "detail": "quantile/metrics.csv ausente"}
    diffs = []
    stored = metrics[metrics["split"].str.match(r"^(fold_\d+|test_final)$")]
    for _, r in stored.iterrows():
        column = f"q{int(round(r['quantile'] * 100))}_cpu_pct"
        part = qt[(qt["config"] == r["config"]) & (qt["horizon_steps"] == r["horizon_steps"]) & (qt["split"] == r["split"])]
        value = pinball(part["y_obs_cpu_pct"].to_numpy() - part[column].to_numpy(), float(r["quantile"])).mean()
        diffs.append(abs(value - r["pinball"]))
    diff = float(max(diffs))
    return {"check": "reproducao_pinball", "passed": diff <= tolerance, "detail": f"{len(diffs)} linhas de quantile/metrics.csv", "max_abs_diff": diff, "tolerance": tolerance}
