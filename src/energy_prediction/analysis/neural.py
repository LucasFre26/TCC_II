"""Diagnóstico das redes neurais com os registros da corrida, sem novo treino.

A loss gravada é o MSE do alvo padronizado pela média e pelo desvio do treino de cada ajuste
(unidade adimensional). Os reajustes (fold externo e teste) usam o número de épocas do melhor
estado e não têm loss de validação: a ausência é declarada, não preenchida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _selection_key(record: dict) -> tuple[str, str, str, str]:
    config = record["config"]
    if record.get("contrast") == "d_vs_a" and config == "A":
        config = "A_on_D"
    horizon = "path" if record.get("mode") == "path" else str(int(record["horizon_steps"]))
    return record["model"], config, record.get("mode", "direct"), horizon


def history_table(history: pd.DataFrame, selections: list[dict]) -> pd.DataFrame:
    best = {_selection_key(r): int(r["best_index"]) for r in selections}
    out = history.rename(columns={"arch": "model", "horizon": "horizon", "grid": "grid_index", "train_loss": "train_loss_mse_standardized", "val_loss": "val_loss_mse_standardized"}).copy()
    out["selected_grid_index"] = [best.get((m, c, mo, str(h))) for m, c, mo, h in zip(out["model"], out["config"], out["mode"], out["horizon"])]
    out["is_selected_grid"] = out["grid_index"] == out["selected_grid_index"]
    out["has_validation_loss"] = out["val_loss_mse_standardized"].notna()
    return out


def unit_diagnostics(history: pd.DataFrame) -> pd.DataFrame:
    """Indicadores por unidade (rede, configuração, modo, horizonte, fold) no ponto selecionado da grade."""
    sel = history[(history["phase"] == "selection") & history["is_selected_grid"]]
    rows = []
    for key, part in sel.groupby(["model", "config", "mode", "horizon", "fold"], sort=False):
        part = part.sort_values("epoch")
        val = part["val_loss_mse_standardized"].to_numpy()
        train = part["train_loss_mse_standardized"].to_numpy()
        b = int(np.nanargmin(val))
        rows.append(
            {
                "model": key[0], "config": key[1], "mode": key[2], "horizon": key[3], "fold": int(key[4]),
                "epochs_run": int(part["epoch"].max()), "best_epoch": int(part["epoch"].iloc[b]),
                "train_loss_at_best": float(train[b]), "val_loss_at_best": float(val[b]), "val_over_train_at_best": float(val[b] / train[b]),
                "train_loss_last": float(train[-1]), "val_loss_last": float(val[-1]), "val_increase_after_best": float(val[-1] / val[b] - 1.0),
                "train_loss_first": float(train[0]), "val_loss_first": float(val[0]),
                "train_reduction_fraction": float(1.0 - train[b] / train[0]),
            }
        )
    return pd.DataFrame(rows)


def unit_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Média entre folds dos indicadores e sinais descritivos (não causais) de ajuste."""
    keys = ["model", "config", "mode", "horizon"]
    agg = diagnostics.groupby(keys).agg(
        n_folds=("fold", "size"), best_epoch_mean=("best_epoch", "mean"), best_epoch_min=("best_epoch", "min"), epochs_run_mean=("epochs_run", "mean"),
        val_over_train_at_best_mean=("val_over_train_at_best", "mean"), val_increase_after_best_mean=("val_increase_after_best", "mean"),
        train_loss_at_best_mean=("train_loss_at_best", "mean"), val_loss_at_best_mean=("val_loss_at_best", "mean"),
    ).reset_index()
    agg["early_best_signal"] = agg["best_epoch_mean"] <= 5
    agg["generalization_gap_signal"] = agg["val_over_train_at_best_mean"] >= 1.5
    agg["note"] = (
        "sinal descritivo: melhor época cedo indica que a loss de validação parou de cair nas primeiras épocas; "
        "razão validação/treino alta indica distância entre ajuste e validação interna; nenhum dos dois prova sobreajuste"
    )
    return agg


def generalization(metrics: pd.DataFrame, persistence_inner: dict[int, float]) -> pd.DataFrame:
    """MAE de seleção, validação externa agregada e teste; razão teste/seleção normalizada pela persistência.

    Base de seleção: MAE interno (V3) ou média dos folds externos (V2, que selecionou assim).
    Uma razão normalizada maior que 1 indica que o modelo perdeu mais do que a persistência ao
    passar do período de seleção para o teste.
    """
    base = metrics[(metrics["population"] == "main") & (metrics["analysis_set"] == "main") & (metrics["mode"] == "direct") & (metrics["config"].isin(["A", "-"]))]
    pooled = base[base["split"] == "validation_pooled"].set_index(["model", "horizon_steps"])
    test = base[base["split"] == "test_final"].set_index(["model", "horizon_steps"])
    rows = []
    pers_fold_mean = pooled.xs("persistence", level="model")["mae_fold_mean_pp"]
    for (model, h), t in test.iterrows():
        v = pooled.loc[(model, h)]
        if model == "persistence":
            selection, basis, ref = persistence_inner.get(int(h), np.nan), "validacao_interna", persistence_inner.get(int(h), np.nan)
        elif pd.notna(t.get("inner_selection_mae_pp")):
            selection, basis, ref = t["inner_selection_mae_pp"], "validacao_interna", persistence_inner.get(int(h), np.nan)
        else:
            selection, basis, ref = v["mae_fold_mean_pp"], "media_folds_externos", pers_fold_mean.loc[h]
        pers_test = test.loc[("persistence", h), "mae_pp"]
        ratio = t["mae_pp"] / selection
        rows.append(
            {
                "model": model, "horizon_steps": int(h), "selection_basis": basis, "selection_mae_pp": selection, "validation_pooled_mae_pp": v["mae_pp"],
                "validation_fold_std_pp": v["mae_fold_std_pp"], "test_mae_pp": t["mae_pp"], "test_over_selection": ratio,
                "persistence_test_over_selection": pers_test / ref, "normalized_generalization_ratio": ratio / (pers_test / ref),
            }
        )
    return pd.DataFrame(rows)


def window_table(window_selection: dict) -> pd.DataFrame:
    rows = []
    for c in window_selection.get("candidates", []):
        for h, value in (c.get("inner_mae_by_horizon") or {}).items():
            rows.append({"model": "dlinear", "config": "A", "window_steps": c["window_steps"], "window_hours": c["window_steps"] * 5 / 60, "horizon_steps": int(h), "inner_mae_pp": value, "inner_mae_sum_pp": c.get("inner_mae_sum"), "chosen": c["window_steps"] == window_selection.get("chosen_window_steps"), "rule": window_selection.get("rule")})
    return pd.DataFrame(rows)
