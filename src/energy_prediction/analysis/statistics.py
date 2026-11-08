"""Inferência sobre diferenças pareadas de perda com o block bootstrap da V2.

- As reamostragens são as mesmas de `moving_block_bootstrap_diff` (mesmo gerador, semente e
  ordem de sorteio); o p-valor bicaudal sai da mesma distribuição que dá o intervalo de 95%.
- Comparações pré-registradas: recalculadas e conferidas contra os bootstrap_ci.csv da V3.
  Holm é acrescentado como coluna informativa sobre todas as linhas do bloco primário.
- Matriz exploratória: todos os pares de modelos da configuração A, por horizonte, com Holm por
  horizonte. Não sustenta conclusão nova; serve para descrever o conjunto.
- As origens do teste são consecutivas e dependentes; o teste final é o mesmo inspecionado na V2.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from energy_prediction.analysis.loader import RunArtifacts
from energy_prediction.evaluation.error_analysis import block_indices


def bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, *, block: int, n_boot: int, seed: int) -> dict:
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[block_indices(len(diff), block, rng, n_boot)].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    below, above = int(np.sum(means <= 0)), int(np.sum(means >= 0))
    p = min(1.0, 2.0 * min(below + 1, above + 1) / (n_boot + 1))
    return {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "significant": bool(high < 0 or low > 0), "p_value_bootstrap": float(p)}


def holm(pvalues) -> np.ndarray:
    """P-valores ajustados por Holm (passo descendente, monótonos)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def pinball(error: np.ndarray, quantile: float) -> np.ndarray:
    e = np.asarray(error, dtype=float)
    return np.maximum(quantile * e, (quantile - 1.0) * e)


def _loss(art: RunArtifacts, family: str, row: pd.Series) -> tuple[pd.Series, pd.Series]:
    def series(model: str, config: str) -> pd.Series:
        if model == "xgboost_v2":
            frame = art.require("v2_main/predictions")
            frame = frame[(frame["model"] == "xgboost") & (frame["config"] == config)]
        else:
            key = f"{family}/d_contrast_predictions" if row["comparison"] == "d_vs_a" else f"{family}/predictions"
            frame = art.require(key)
            frame = frame[(frame["model"] == model) & (frame["config"] == config)]
            if "mode" in frame.columns and family == "neural":
                frame = frame[frame["mode"] == "direct"]
            if family == "quantile":
                frame = frame[(frame["mode"] == "quantile") & np.isclose(frame["quantile"], row["quantile"])]
        frame = frame[(frame["split"] == "test_final") & (frame["horizon_steps"] == row["horizon_steps"])].sort_values("origin_index")
        error = (frame["y_true"] - frame["y_pred"]).to_numpy()
        loss = pinball(error, float(row["quantile"])) if family == "quantile" else np.abs(error)
        return pd.Series(loss, index=frame["origin_index"].to_numpy())

    return series(row["model_a"], row["config_a"]), series(row["model_b"], row["config_b"])


def reproduce_preregistered(art: RunArtifacts, *, tolerance: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    for family in ("residual", "neural", "quantile"):
        stored = art.frame(f"{family}/bootstrap_ci")
        if stored is None:
            continue
        cache: dict = {}
        for _, row in stored.iterrows():
            key = (row["comparison"], row["model_a"], row["config_a"], row["model_b"], row["config_b"], int(row["horizon_steps"]), row.get("quantile"))
            if key not in cache:
                cache[key] = _loss(art, family, row)
            a, b = cache[key]
            if not a.index.equals(b.index):
                raise ValueError(f"Origens diferentes ao recalcular {key}.")
            result = bootstrap(a.to_numpy(), b.to_numpy(), block=int(row["block_steps"]), n_boot=int(row["n_boot"]), seed=int(art.manifest.get("seed", 42)))
            reference_loss = float(b.mean())
            rows.append(
                {
                    "analysis_type": "pre_registrada", "family": family, "comparison": row["comparison"], "loss": row.get("loss", "erro_absoluto") if isinstance(row.get("loss"), str) else "erro_absoluto",
                    "quantile": row.get("quantile", np.nan), "horizon_steps": int(row["horizon_steps"]), "model_a": row["model_a"], "config_a": row["config_a"],
                    "model_b": row["model_b"], "config_b": row["config_b"], "block_steps": int(row["block_steps"]), "primary_block": bool(row["primary_block"]),
                    "n_boot": int(row["n_boot"]), "n": int(len(a)), **result, "loss_a_mean": float(a.mean()), "loss_b_mean": reference_loss,
                    "stored_mean_diff": row["mean_diff"], "stored_ci_low": row["ci_low"], "stored_ci_high": row["ci_high"], "stored_significant": bool(row["significant"]),
                    "reproduction_max_abs_diff": float(max(abs(result["mean_diff"] - row["mean_diff"]), abs(result["ci_low"] - row["ci_low"]), abs(result["ci_high"] - row["ci_high"]))),
                }
            )
    table = pd.DataFrame(rows)
    diff = float(table["reproduction_max_abs_diff"].max()) if len(table) else np.nan
    same_decision = bool((table["significant"] == table["stored_significant"]).all()) if len(table) else False
    check = {
        "check": "reproducao_bootstrap_pre_registrado", "passed": bool(len(table) and diff <= tolerance and same_decision),
        "detail": f"{len(table)} linhas de bootstrap_ci.csv (residual, neural, quantílico) recalculadas", "max_abs_diff": diff, "tolerance": tolerance,
    }
    return table, check


def exploratory(predictions: pd.DataFrame, models: list[str], *, config: str, blocks: list[int], n_boot: int, seed: int) -> pd.DataFrame:
    """Todos os pares de modelos (configuração A e persistência), teste final, por horizonte e bloco."""
    test = predictions[(predictions["split"] == "test_final") & (predictions["population"] == "main") & (predictions["analysis_set"] == "main") & (predictions["mode"] == "direct")]
    test = test[test["config"].isin([config, "-"])]
    rows = []
    for h, part in test.groupby("horizon_steps"):
        losses = {m: g.sort_values("origin_index").set_index("origin_index")["abs_error_pp"] for m, g in part.groupby("model") if m in models}
        present = [m for m in models if m in losses]
        for a, b in itertools.combinations(present, 2):
            la, lb = losses[a], losses[b]
            if not la.index.equals(lb.index):
                raise ValueError(f"Origens diferentes entre {a} e {b}.")
            for block in blocks:
                result = bootstrap(la.to_numpy(), lb.to_numpy(), block=int(block), n_boot=n_boot, seed=seed)
                rows.append(
                    {
                        "analysis_type": "exploratoria", "family": "todas", "comparison": "par_exploratorio", "loss": "erro_absoluto", "quantile": np.nan,
                        "horizon_steps": int(h), "model_a": a, "config_a": config if a != "persistence" else "-", "model_b": b, "config_b": config if b != "persistence" else "-",
                        "block_steps": int(block), "n_boot": n_boot, "n": int(len(la)), **result, "loss_a_mean": float(la.mean()), "loss_b_mean": float(lb.mean()),
                    }
                )
    return pd.DataFrame(rows)


def decorate(table: pd.DataFrame, *, primary_block: int, practical: float, watts_per_pp: float) -> pd.DataFrame:
    """Diferença relativa, relevância prática, equivalente em watts e Holm por família."""
    out = table.copy()
    out["primary_block"] = out["block_steps"] == primary_block
    out["relative_diff"] = out["mean_diff"] / out["loss_b_mean"]
    out["practical_relevance"] = np.where(out["relative_diff"].abs() >= practical, "relevante", "pequena")
    is_abs = out["loss"] == "erro_absoluto"
    out["watts_per_substitute_server_equivalent"] = np.where(is_abs, out["mean_diff"] * watts_per_pp, np.nan)
    out["holm_family"] = np.where(out["analysis_type"] == "pre_registrada", "pre_registradas_bloco_primario", "exploratoria_h" + out["horizon_steps"].astype(str))
    out["p_holm"] = np.nan
    primary = out["primary_block"]
    for family, idx in out[primary].groupby("holm_family").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_value_bootstrap"].to_numpy())
    out["holm_significant"] = out["p_holm"] < 0.05
    keys = ["analysis_type", "family", "comparison", "quantile", "horizon_steps", "model_a", "config_a", "model_b", "config_b"]
    agreement = out.groupby(keys, dropna=False)["significant"].agg(lambda s: "todos" if s.all() else ("nenhum" if not s.any() else "parcial")).rename("significant_across_blocks")
    return out.merge(agreement.reset_index(), on=keys, how="left")


def summary_table(decorated: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por comparação (bloco primário), com decisão, Holm, relevância e concordância entre blocos."""
    cols = [
        "analysis_type", "family", "comparison", "loss", "quantile", "horizon_steps", "model_a", "config_a", "model_b", "config_b", "n",
        "loss_a_mean", "loss_b_mean", "mean_diff", "relative_diff", "ci_low", "ci_high", "significant", "p_value_bootstrap", "p_holm", "holm_significant",
        "significant_across_blocks", "practical_relevance", "watts_per_substitute_server_equivalent",
    ]
    return decorated[decorated["primary_block"]][cols].reset_index(drop=True)
