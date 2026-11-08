"""Métricas pontuais expandidas, por fold, validação agregada e teste, e rankings.

MASE segue a definição do projeto: MAE do modelo / MAE da persistência nas mesmas origens
(não é a MASE escalada pelo erro ingênuo no treino). Skill score = 1 − RMSE/RMSE da persistência.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = ["version", "family", "model", "config", "mode", "population", "analysis_set", "split", "horizon_steps", "horizon_min"]
FOLD_PATTERN = r"^fold_\d+$"


def point_metrics(error: np.ndarray) -> dict:
    """Indicadores de um vetor de erros (observado − previsto), em pontos percentuais de CPU."""
    e = np.asarray(error, dtype=float)
    a = np.abs(e)
    return {
        "n": int(len(e)),
        "mae_pp": float(a.mean()),
        "rmse_pp": float(np.sqrt(np.mean(e**2))),
        "bias_obs_minus_pred_pp": float(e.mean()),
        "error_std_pp": float(e.std(ddof=1)) if len(e) > 1 else np.nan,
        "median_abs_error_pp": float(np.median(a)),
        "p90_abs_error_pp": float(np.quantile(a, 0.90)),
        "p95_abs_error_pp": float(np.quantile(a, 0.95)),
        "p99_abs_error_pp": float(np.quantile(a, 0.99)),
        "max_abs_error_pp": float(a.max()),
        "underestimation_fraction": float(np.mean(e > 0)),
    }


def _with_references(predictions: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta o erro da persistência e do XGBoost da V2 (mesma configuração) nas mesmas origens."""
    keys = ["population", "split", "horizon_steps", "origin_index"]
    base = predictions[(predictions["analysis_set"] == "main") & (predictions["mode"] == "direct")]
    persistence = base[base["model"] == "persistence"][keys + ["error_obs_minus_pred_pp"]].rename(columns={"error_obs_minus_pred_pp": "ref_persistence_error"})
    xgb = base[base["model"] == "xgboost_v2"][keys + ["config", "error_obs_minus_pred_pp"]].rename(columns={"error_obs_minus_pred_pp": "ref_xgboost_v2_error"})
    out = predictions.merge(persistence, on=keys, how="left")
    return out.merge(xgb, on=keys + ["config"], how="left")


def _summarize(frame: pd.DataFrame) -> dict:
    row = point_metrics(frame["error_obs_minus_pred_pp"].to_numpy())
    pers = frame["ref_persistence_error"].to_numpy(dtype=float)
    xgb = frame["ref_xgboost_v2_error"].to_numpy(dtype=float)
    if np.isfinite(pers).all():
        mae_p, rmse_p = np.mean(np.abs(pers)), np.sqrt(np.mean(pers**2))
        row.update(
            persistence_mae_same_origins_pp=float(mae_p),
            mase_vs_persistence=row["mae_pp"] / mae_p,
            skill_score_rmse=1.0 - row["rmse_pp"] / rmse_p,
            improvement_vs_persistence_pct=100.0 * (1.0 - row["mae_pp"] / mae_p),
        )
    if np.isfinite(xgb).all():
        mae_x = np.mean(np.abs(xgb))
        row.update(xgboost_v2_mae_same_origins_pp=float(mae_x), improvement_vs_xgboost_v2_pct=100.0 * (1.0 - row["mae_pp"] / mae_x))
    return row


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(GROUP, sort=False):
        rows.append({**dict(zip(GROUP, key)), **_summarize(part)})
    return pd.DataFrame(rows)


def metrics_tables(predictions: pd.DataFrame, inner: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(métricas de validação agregada e teste, métricas por fold).

    `validation_pooled` junta as 720 origens dos cinco folds; média e desvio entre folds ficam
    em colunas próprias. O MAE interno de seleção vem dos metrics.csv, sem recálculo.
    """
    data = _with_references(predictions)
    folds = data[data["split"].str.match(FOLD_PATTERN)]
    by_fold = _aggregate(folds)
    pooled = _aggregate(folds.assign(split="validation_pooled"))
    test = _aggregate(data[data["split"] == "test_final"])
    keys = [k for k in GROUP if k != "split"]
    spread = by_fold.groupby(keys, sort=False).agg(
        mae_fold_mean_pp=("mae_pp", "mean"), mae_fold_std_pp=("mae_pp", "std"), rmse_fold_mean_pp=("rmse_pp", "mean"), rmse_fold_std_pp=("rmse_pp", "std"), n_folds=("mae_pp", "size")
    ).reset_index()
    pooled = pooled.merge(spread, on=keys, how="left")
    metrics = pd.concat([pooled, test], ignore_index=True)
    if inner is not None and not inner.empty:
        metrics = metrics.merge(inner, on=["model", "config", "mode", "horizon_steps", "population", "analysis_set"], how="left")
    by_fold = by_fold.sort_values(["population", "analysis_set", "model", "config", "mode", "horizon_steps", "split"]).reset_index(drop=True)
    return metrics.sort_values(["population", "analysis_set", "split", "model", "config", "mode", "horizon_steps"]).reset_index(drop=True), by_fold


def inner_selection(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """MAE interno de seleção gravado pela V3 (não recalculado)."""
    rows = []
    for frame in frames:
        if frame is None:
            continue
        part = frame[frame["split"] == "inner_selection_mae"]
        for _, r in part.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "config": r["config"],
                    "mode": r.get("mode", "direct") if isinstance(r.get("mode", "direct"), str) else "direct",
                    "horizon_steps": int(r["horizon_steps"]),
                    "population": "main",
                    "analysis_set": "main",
                    "inner_selection_mae_pp": float(r["mae"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["model", "config", "mode", "horizon_steps"]) if not out.empty else out


def rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    """Posição por MAE e RMSE dentro de cada configuração e horizonte (população principal, modo direto).

    A persistência (configuração "-") entra em todo escopo como referência.
    """
    base = metrics[(metrics["population"] == "main") & (metrics["analysis_set"] == "main") & (metrics["mode"] == "direct")]
    persistence = base[base["model"] == "persistence"]
    rows = []
    for (split, config, h), part in base[base["config"] != "-"].groupby(["split", "config", "horizon_steps"]):
        scope = pd.concat([part, persistence[(persistence["split"] == split) & (persistence["horizon_steps"] == h)]])
        for metric in ("mae_pp", "rmse_pp"):
            ordered = scope.sort_values(metric, kind="stable")
            for rank, (_, r) in enumerate(ordered.iterrows(), start=1):
                rows.append(
                    {
                        "split": split, "config_scope": config, "horizon_steps": int(h), "horizon_min": r["horizon_min"],
                        "metric": metric, "rank": rank, "model": r["model"], "config": r["config"], "value_pp": r[metric], "n_models": len(ordered),
                    }
                )
    return pd.DataFrame(rows).sort_values(["split", "config_scope", "metric", "horizon_steps", "rank"]).reset_index(drop=True)


def fold_wins(by_fold: pd.DataFrame, reference: str = "xgboost_v2") -> pd.DataFrame:
    """Em quantos folds cada modelo teve MAE menor que a referência da mesma configuração (A para a persistência)."""
    base = by_fold[(by_fold["population"] == "main") & (by_fold["analysis_set"] == "main") & (by_fold["mode"] == "direct")]
    ref = base[base["model"] == reference][["config", "split", "horizon_steps", "mae_pp"]].rename(columns={"mae_pp": "reference_mae_pp"})
    rows = []
    for (model, config, h), part in base.groupby(["model", "config", "horizon_steps"]):
        ref_config = "A" if config == "-" else config
        joined = part.merge(ref[ref["config"] == ref_config].drop(columns="config"), on=["split", "horizon_steps"], how="inner")
        if joined.empty:
            continue
        rows.append({"model": model, "config": config, "horizon_steps": int(h), "reference_model": reference, "reference_config": ref_config, "n_folds": len(joined), "folds_better_than_reference": int((joined["mae_pp"] < joined["reference_mae_pp"]).sum())})
    return pd.DataFrame(rows)
