"""Verificações feitas antes de qualquer indicador: origens, unicidade e reprodução das métricas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.analysis.loader import RunArtifacts

PRED_KEYS = ["model", "config", "mode", "split", "horizon_steps", "origin_index"]


class IntegrityError(RuntimeError):
    """Uma verificação de integridade falhou; a análise não continua."""


def check(name: str, passed: bool, detail: str, **values) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail, **values}


def _keys(frame: pd.DataFrame) -> list[str]:
    return [k for k in PRED_KEYS if k in frame.columns]


def uniqueness(frame: pd.DataFrame, label: str) -> dict:
    keys = _keys(frame)
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        first = frame.loc[duplicated, keys].iloc[0].to_dict()
        return check(f"unicidade/{label}", False, f"{int(duplicated.sum())} linhas duplicadas; primeira combinação: {first}")
    return check(f"unicidade/{label}", True, f"{len(frame)} linhas, uma por {'/'.join(keys)}")


def test_origins(frame: pd.DataFrame, label: str, expected: tuple[int, int]) -> dict:
    test = frame[frame["split"] == "test_final"]
    wanted = set(range(expected[0], expected[1] + 1))
    group_keys = [k for k in ["model", "config", "mode", "horizon_steps", "quantile"] if k in test.columns]
    bad = []
    for key, part in test.groupby(group_keys):
        if set(part["origin_index"]) != wanted:
            bad.append(key)
    detail = f"origens do teste {expected[0]}–{expected[1]} em {test.groupby(group_keys).ngroups} grupos" if not bad else f"grupos fora do teste esperado: {bad[:5]}"
    return check(f"origens_teste/{label}", not bad, detail)


def metric_reproduction(metrics: pd.DataFrame, predictions: pd.DataFrame, label: str, tolerance: float) -> dict:
    """MAE e RMSE recalculados das previsões contra os valores gravados (folds e teste)."""
    keys = [k for k in ["model", "config", "mode", "horizon_steps", "split"] if k in metrics.columns and k in predictions.columns]
    scored = metrics[metrics["split"].str.match(r"^(fold_\d+|test_final)$")]
    error = predictions["y_true"] - predictions["y_pred"]
    recomputed = (
        predictions.assign(abs_error=error.abs(), sq_error=error**2)
        .groupby(keys)
        .agg(mae_recomputed=("abs_error", "mean"), mse=("sq_error", "mean"), n_recomputed=("abs_error", "size"))
        .reset_index()
    )
    recomputed["rmse_recomputed"] = np.sqrt(recomputed["mse"])
    joined = scored.merge(recomputed, on=keys, how="left")
    missing = int(joined["mae_recomputed"].isna().sum())
    diff = float(np.nanmax(np.abs(np.r_[joined["mae"] - joined["mae_recomputed"], joined["rmse"] - joined["rmse_recomputed"]]))) if len(joined) else np.nan
    passed = missing == 0 and np.isfinite(diff) and diff <= tolerance and (joined["n"] == joined["n_recomputed"]).all()
    return check(f"reproducao_metricas/{label}", passed, f"{len(joined)} linhas de metrics.csv; sem previsão: {missing}", max_abs_diff=diff, tolerance=tolerance)


def persistence_identical(art: RunArtifacts, tolerance: float) -> dict:
    v3 = art.require("residual/predictions")
    v2 = art.frame("v2_main/predictions")
    neural = art.require("neural/predictions")
    a = v3[v3["model"] == "persistence"].set_index(["split", "horizon_steps", "origin_index"])["y_pred"].sort_index()
    b = neural[neural["model"] == "persistence"].set_index(["split", "horizon_steps", "origin_index"])["y_pred"].sort_index()
    diffs = [float(np.max(np.abs(a - b.reindex(a.index))))]
    if v2 is not None:
        c = v2[v2["model"] == "persistence"].set_index(["split", "horizon_steps", "origin_index"])["y_pred"].sort_index()
        diffs.append(float(np.max(np.abs(a - c.reindex(a.index)))))
    diff = max(diffs)
    detail = "persistência da V2, da família residual e da família neural nas mesmas origens (a grade da V3 foi reconstruída; diferenças de arredondamento abaixo da tolerância)"
    return check("persistencia_identica_v2_v3", diff <= tolerance, detail, max_abs_diff=diff, tolerance=tolerance)


def observed_consistent(art: RunArtifacts, tolerance: float) -> dict:
    """O valor observado de cada (split, horizonte, origem) é o mesmo em todas as famílias da população principal."""
    frames = [art.require("residual/predictions"), art.require("neural/predictions")]
    if art.frame("v2_main/predictions") is not None:
        frames.append(art.frame("v2_main/predictions"))
    stacked = pd.concat([f[["split", "horizon_steps", "origin_index", "y_true"]] for f in frames])
    spread = stacked.groupby(["split", "horizon_steps", "origin_index"])["y_true"].agg(lambda s: s.max() - s.min())
    diff = float(spread.max())
    return check("observado_consistente", diff <= tolerance, f"{len(spread)} pares (split, horizonte, origem) entre V2 e V3", max_abs_diff=diff, tolerance=tolerance)


def substitute_consistent(art: RunArtifacts, substitute: dict) -> dict:
    info = art.doc("energy/energy_info")
    if info is None:
        return check("servidor_substituto", False, "energy_info.json ausente")
    stored = info["substitute"]
    same = all(stored.get(k) == substitute.get(k) for k in ("result_id", "p_idle_w", "p_busy_w"))
    return check(
        "servidor_substituto",
        same,
        f"{stored.get('result_id')}: P_idle={stored.get('p_idle_w')} W, P_busy={stored.get('p_busy_w')} W em energy_info.json e em specpower_substitute.yaml",
    )


def run_checks(art: RunArtifacts, tolerances: dict, substitute: dict) -> list[dict]:
    expected = tuple(art.doc("integrity/alignment")["test_range"]) if art.doc("integrity/alignment") else (2010, 2297)
    out = []
    sources = {
        "v3_residual": ("residual/predictions", "residual/metrics"),
        "v3_neural": ("neural/predictions", "neural/metrics"),
        "v3_residual_d": ("residual/d_contrast_predictions", "residual/d_contrast_metrics"),
        "v3_neural_d": ("neural/d_contrast_predictions", "neural/d_contrast_metrics"),
        "v2_main": ("v2_main/predictions", "v2_main/metrics"),
        "v2_cohort": ("v2_cohort/predictions", "v2_cohort/metrics"),
    }
    for label, (pred_key, metric_key) in sources.items():
        predictions, metrics = art.frame(pred_key), art.frame(metric_key)
        if predictions is None:
            continue
        out.append(uniqueness(predictions, label))
        out.append(test_origins(predictions, label, expected))
        if metrics is not None:
            out.append(metric_reproduction(metrics, predictions, label, float(tolerances["metric"])))
    quantiles = art.frame("quantile/predictions")
    if quantiles is not None:
        out.append(uniqueness(quantiles.assign(mode=quantiles["mode"] + "_" + quantiles["quantile"].astype(str)), "v3_quantile"))
        out.append(test_origins(quantiles, "v3_quantile", expected))
    out.append(persistence_identical(art, float(tolerances["series_pp"])))
    out.append(observed_consistent(art, float(tolerances["series_pp"])))
    out.append(substitute_consistent(art, substitute))
    return out


def raise_on_failure(checks: list[dict]) -> None:
    failed = [c for c in checks if not c["passed"]]
    if failed:
        lines = "\n".join(f"- {c['check']}: {c['detail']} (diferença {c.get('max_abs_diff')})" for c in failed)
        raise IntegrityError(f"Verificações de integridade falharam:\n{lines}")
