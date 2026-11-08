"""Tabela longa de previsões pontuais da V2 e da V3, alinhada ao instante previsto.

Convenção: erro = observado − previsto (positivo quando o modelo subestima), a mesma de
`evaluation/error_analysis.py`. O tempo é contado em segundos desde o início do trace,
não em horário civil. O instante previsto é a origem mais o horizonte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.analysis.loader import RunArtifacts

V2_NAMES = {"ridge": "ridge_v2", "xgboost": "xgboost_v2", "persistence": "persistence"}
COLUMNS = [
    "version", "family", "model", "config", "mode", "population", "analysis_set", "split",
    "horizon_steps", "horizon_min", "origin_index", "origin_time_s", "target_index", "target_time_s", "target_time_h",
    "y_obs_cpu_pct", "y_pred_cpu_pct", "error_obs_minus_pred_pp", "abs_error_pp",
]


def step_seconds(art: RunArtifacts) -> float:
    profile = art.doc("integrity/series_profile")
    if profile and "resolved" in profile:
        return float(profile["resolved"]["step_seconds"])
    return float(art.manifest["config"]["data"]["grid_step"].rstrip("s")) if "config" in art.manifest else 300.0


def _frame(raw: pd.DataFrame, *, version: str, family: str, population: str, analysis_set: str, step: float, rename=None, config_map=None) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "version": version,
            "family": family,
            "model": raw["model"].map(rename).fillna(raw["model"]) if rename else raw["model"],
            "config": raw["config"].map(config_map).fillna(raw["config"]) if config_map else raw["config"],
            "mode": raw["mode"] if "mode" in raw else "direct",
            "population": population,
            "analysis_set": analysis_set,
            "split": raw["split"],
            "horizon_steps": raw["horizon_steps"].astype(int),
            "origin_index": raw["origin_index"].astype(int),
            "origin_time_s": raw["timestamp"].astype(float),
            "y_obs_cpu_pct": raw["y_true"].astype(float),
            "y_pred_cpu_pct": raw["y_pred"].astype(float),
        }
    )
    out["horizon_min"] = out["horizon_steps"] * step / 60.0
    out["target_index"] = out["origin_index"] + out["horizon_steps"]
    out["target_time_s"] = out["origin_time_s"] + out["horizon_steps"] * step
    out["target_time_h"] = out["target_time_s"] / 3600.0
    out["error_obs_minus_pred_pp"] = out["y_obs_cpu_pct"] - out["y_pred_cpu_pct"]
    out["abs_error_pp"] = out["error_obs_minus_pred_pp"].abs()
    return out[COLUMNS]


def family_of(model: str, families: dict) -> str:
    return families.get(model, "V3 quantílico" if model == "xgboost_quantile" else "outro")


def predictions_table(art: RunArtifacts, families: dict) -> pd.DataFrame:
    """Uma linha por previsão pontual; persistência aparece uma vez por população."""
    step = step_seconds(art)
    parts = []
    residual = art.require("residual/predictions")
    parts.append(_frame(residual, version="V3", family="", population="main", analysis_set="main", step=step))
    neural = art.require("neural/predictions")
    parts.append(_frame(neural[neural["model"] != "persistence"], version="V3", family="", population="main", analysis_set="main", step=step))
    v2 = art.frame("v2_main/predictions")
    if v2 is not None:
        parts.append(_frame(v2[v2["model"] != "persistence"], version="V2", family="", population="main", analysis_set="main", step=step, rename=V2_NAMES))
    cohort = art.frame("v2_cohort/predictions")
    if cohort is not None:
        parts.append(_frame(cohort, version="V2", family="", population="cohort", analysis_set="main", step=step, rename=V2_NAMES))
    for key in ("residual/d_contrast_predictions", "neural/d_contrast_predictions"):
        frame = art.frame(key)
        if frame is not None:
            parts.append(_frame(frame, version="V3", family="", population="main", analysis_set="d_contrast", step=step, config_map={"A": "A_on_D"}))
    table = pd.concat(parts, ignore_index=True)
    table["family"] = table["model"].map(lambda m: family_of(m, families))
    return table.sort_values(["population", "analysis_set", "model", "config", "mode", "split", "horizon_steps", "origin_index"], kind="stable").reset_index(drop=True)


def quantile_table(art: RunArtifacts) -> pd.DataFrame | None:
    """Quantis e tetos ajustados em formato largo: uma linha por configuração, horizonte e origem."""
    raw = art.frame("quantile/predictions")
    if raw is None:
        return None
    step = step_seconds(art)
    raw = raw.copy()
    raw["column"] = np.where(raw["mode"] == "conformal_upper", "q" + (raw["quantile"] * 100).round().astype(int).astype(str) + "_adjusted_cpu_pct", "q" + (raw["quantile"] * 100).round().astype(int).astype(str) + "_cpu_pct")
    wide = raw.pivot_table(index=["config", "horizon_steps", "split", "origin_index", "timestamp", "y_true"], columns="column", values="y_pred", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"timestamp": "origin_time_s", "y_true": "y_obs_cpu_pct"})
    wide.insert(0, "model", "xgboost_quantile")
    wide.insert(1, "population", "main")
    wide["horizon_min"] = wide["horizon_steps"] * step / 60.0
    wide["target_index"] = wide["origin_index"] + wide["horizon_steps"]
    wide["target_time_s"] = wide["origin_time_s"] + wide["horizon_steps"] * step
    wide["target_time_h"] = wide["target_time_s"] / 3600.0
    value_columns = [c for c in ["q50_cpu_pct", "q90_cpu_pct", "q95_cpu_pct", "q90_adjusted_cpu_pct", "q95_adjusted_cpu_pct"] if c in wide]
    wide["upper_margin_q95_minus_q50_pp"] = wide["q95_cpu_pct"] - wide["q50_cpu_pct"]
    ordered = ["model", "population", "config", "split", "horizon_steps", "horizon_min", "origin_index", "origin_time_s", "target_index", "target_time_s", "target_time_h", "y_obs_cpu_pct", *value_columns, "upper_margin_q95_minus_q50_pp"]
    return wide[ordered].sort_values(["config", "horizon_steps", "split", "origin_index"]).reset_index(drop=True)


def observed_series(predictions: pd.DataFrame, series: pd.Series | None = None) -> pd.Series:
    """CPU agregada observada por índice de intervalo.

    Usa a série reconstruída quando disponível; senão, junta a CPU na origem (previsão da
    persistência) e os valores observados nos instantes previstos da população principal.
    """
    if series is not None:
        return series.astype(float)
    main = predictions[(predictions["population"] == "main") & (predictions["analysis_set"] == "main")]
    at_target = main.groupby("target_index")["y_obs_cpu_pct"].first()
    persistence = main[main["model"] == "persistence"].groupby("origin_index")["y_pred_cpu_pct"].first()
    combined = pd.concat([at_target, persistence])
    return combined[~combined.index.duplicated()].sort_index()
