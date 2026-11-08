"""Reajustes verificados dos modelos clássicos, TreeSHAP nativo do XGBoost e tempo em CPU.

Cada reajuste usa os hiperparâmetros já selecionados e a mesma região de desenvolvimento do
teste final. Antes de qualquer uso, as previsões de teste do reajuste precisam reproduzir as
gravadas; senão o modelo é recusado e o motivo vai para o manifesto. A importância por TreeSHAP
descreve contribuição para a previsão; não implica relação causal entre atributo e CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUPS = (
    ("cpu_lag", "CPU defasada"), ("cpu_roll", "CPU janela móvel"), ("cpu_t", "CPU na origem"), ("mem", "memória"),
    ("net", "rede"), ("disk", "disco"), ("sin_day", "calendário"), ("cos_day", "calendário"), ("dist_", "distribuição da frota"),
)


def feature_group(name: str) -> str:
    for prefix, label in GROUPS:
        if name.startswith(prefix):
            return label
    return "outro"


@dataclass
class Refit:
    model: str
    config: str
    horizon_steps: int
    prediction: np.ndarray
    stored: np.ndarray
    fit_seconds: float
    predict_seconds: float
    booster: object | None
    X_test: np.ndarray | None
    columns: list[str]
    complexity: dict

    @property
    def max_abs_diff(self) -> float:
        return float(np.max(np.abs(self.prediction - self.stored)))


def _tree_complexity(booster) -> dict:
    trees = booster.trees_to_dataframe()
    return {"n_trees": int(trees["Tree"].nunique()), "n_leaves": int((trees["Feature"] == "Leaf").sum()), "n_nodes": int(len(trees))}


def _timed(fn, repeats: int):
    times, result = [], None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


class Refitter:
    def __init__(self, ctx, art, *, repeats: int):
        self.ctx = ctx
        self.art = art
        self.repeats = repeats
        self.table = ctx.table
        self.dev = ctx.splits.development
        self.test = ctx.splits.test
        self.origins = self.table["origin_index"].to_numpy()

    def _stored(self, source: str, model: str, config: str, h: int) -> np.ndarray:
        frame = self.art.require(source)
        part = frame[(frame["model"] == model) & (frame["config"] == config) & (frame["split"] == "test_final") & (frame["horizon_steps"] == h)]
        if "mode" in part.columns:
            part = part[part["mode"] == "direct"]
        return part.set_index("origin_index")["y_pred"].reindex(self.origins[self.test]).to_numpy(dtype=float)

    def _params(self, source: str, model: str, config: str, h: int) -> dict:
        for record in self.art.doc(source):
            if record["model"] == model and record["config"] == config and int(record["horizon_steps"]) == h and record.get("contrast") is None and record.get("mode", "direct") == "direct":
                return record["best_params"]
        raise KeyError(f"Sem seleção de {model}/{config}/h{h} em {source}.")

    def _xy(self, config: str, h: int):
        from energy_prediction.features.cpu_features import target_column

        columns = self.ctx.columns[config]
        X = self.table[columns].to_numpy(dtype=float)
        y = self.table[target_column(h)].to_numpy(dtype=float)
        train = self.dev[np.isfinite(y[self.dev])]
        return columns, X, y, train

    def xgboost_v2(self, config: str, h: int) -> Refit:
        from energy_prediction.models.registry import XGBoostModel

        params = self._params("v2_main/selections", "xgboost", config, h)
        xgb = self.ctx.cfg["families"]["energy"]["xgboost_v2"]
        columns, X, y, train = self._xy(config, h)
        model = XGBoostModel(params, seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
        _, fit_s = _timed(lambda: model.fit(X[train], y[train]), self.repeats)
        pred, pred_s = _timed(lambda: model.predict(X[self.test]), self.repeats)
        booster = model._model.get_booster()
        return Refit("xgboost_v2", config, h, pred, self._stored("v2_main/predictions", "xgboost", config, h), fit_s, pred_s, booster, X[self.test], columns, {**_tree_complexity(booster), "params": params})

    def xgboost_delta(self, config: str, h: int) -> Refit:
        from energy_prediction.models.delta import DeltaXGBoost

        params = self._params("residual/selections", "xgboost_delta", config, h)
        xgb = self.ctx.cfg["families"]["residual"]["xgboost"]
        columns, X, y, train = self._xy(config, h)
        model = DeltaXGBoost(params, feature_names=columns, seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
        _, fit_s = _timed(lambda: model.fit(X[train], y[train]), self.repeats)
        pred, pred_s = _timed(lambda: model.predict(X[self.test]), self.repeats)
        booster = model._model._model.get_booster()
        return Refit("xgboost_delta", config, h, pred, self._stored("residual/predictions", "xgboost_delta", config, h), fit_s, pred_s, booster, X[self.test], columns, {**_tree_complexity(booster), "params": params})

    def ridge_v2(self, config: str, h: int) -> Refit:
        from energy_prediction.models.registry import RidgeModel

        params = self._params("v2_main/selections", "ridge", config, h)
        columns, X, y, train = self._xy(config, h)
        model = RidgeModel(params["alpha"])
        _, fit_s = _timed(lambda: model.fit(X[train], y[train]), self.repeats)
        pred, pred_s = _timed(lambda: model.predict(X[self.test]), self.repeats)
        return Refit("ridge_v2", config, h, pred, self._stored("v2_main/predictions", "ridge", config, h), fit_s, pred_s, None, None, columns, {"n_coefficients": len(columns) + 1, "params": params})

    def _spec(self, h: int) -> str:
        return self.art.doc("residual/family_info")["statespace_specification_by_horizon"][str(h)]

    def statespace(self, config: str, h: int) -> Refit:
        from energy_prediction.models.statespace import fit_parameters, filtered_forecasts

        spec = self._spec(h)
        series = self.ctx.series["cpu"].to_numpy(dtype=float)
        dev_end = int(self.origins[self.dev].max())
        maxiter = int(self.ctx.cfg["families"]["residual"]["statespace"]["maxiter"])
        params, fit_s = _timed(lambda: fit_parameters(series[: dev_end + 1], spec, maxiter=maxiter), self.repeats)
        forecasts, pred_s = _timed(lambda: filtered_forecasts(series, spec, params, [h]), self.repeats)
        pred = forecasts.by_horizon[h][self.origins[self.test]]
        return Refit("statespace", "A", h, pred, self._stored("residual/predictions", "statespace", "A", h), fit_s, pred_s, None, None, ["cpu"], {"n_parameters": int(np.size(params)), "specification": spec})

    def hybrid(self, config: str, h: int) -> Refit:
        from energy_prediction.models.hybrid import HybridCorrector
        from energy_prediction.models.statespace import StateSpaceForecaster

        params = self._params("residual/selections", "hybrid", config, h)
        xgb = self.ctx.cfg["families"]["residual"]["xgboost"]
        columns, X, y, train = self._xy(config, h)
        maxiter = int(self.ctx.cfg["families"]["residual"]["statespace"]["maxiter"])
        dev_end = int(self.origins[self.dev].max())

        def fit():
            forecaster = StateSpaceForecaster(self.ctx.series["cpu"].to_numpy(), horizons=[h], maxiter=maxiter)
            forecasts = forecaster.forecasts(self._spec(h), dev_end)
            base = forecasts.by_horizon[h][self.origins]
            corrector = HybridCorrector(params, seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
            corrector.fit(X[train], y[train], base[train], forecasts)
            return corrector, base

        (corrector, base), fit_s = _timed(fit, self.repeats)
        pred, pred_s = _timed(lambda: corrector.predict(X[self.test], base[self.test]), self.repeats)
        booster = corrector._model._model.get_booster()
        return Refit("hybrid", config, h, pred, self._stored("residual/predictions", "hybrid", config, h), fit_s, pred_s, booster, X[self.test], columns, {**_tree_complexity(booster), "statespace_parameters": 2, "params": params})


def tree_shap(refit: Refit) -> pd.DataFrame:
    """Média do |SHAP| no teste por atributo (TreeSHAP do próprio XGBoost, `pred_contribs`)."""
    import xgboost as xgb

    contributions = refit.booster.predict(xgb.DMatrix(refit.X_test), pred_contribs=True)
    values = contributions[:, :-1]
    mean_abs = np.abs(values).mean(axis=0)
    out = pd.DataFrame({"feature": refit.columns, "mean_abs_shap_pp": mean_abs, "mean_shap_pp": values.mean(axis=0)})
    out["group"] = out["feature"].map(feature_group)
    out["share"] = out["mean_abs_shap_pp"] / out["mean_abs_shap_pp"].sum()
    out["rank"] = out["mean_abs_shap_pp"].rank(ascending=False, method="first").astype(int)
    out.insert(0, "model", refit.model)
    out.insert(1, "config", refit.config)
    out.insert(2, "horizon_steps", refit.horizon_steps)
    out["explained_quantity"] = "incremento de CPU (y − cpu_t)" if refit.model == "xgboost_delta" else ("correção do resíduo do Kalman" if refit.model == "hybrid" else "CPU no horizonte")
    out["bias_term_pp"] = float(contributions[:, -1].mean())
    out["additivity_max_abs_diff"] = float(np.max(np.abs(contributions.sum(axis=1) - refit.booster.predict(xgb.DMatrix(refit.X_test), output_margin=True))))
    return out


def reproduces(refit: Refit, tolerance: float) -> bool:
    """Um reajuste só é usado se reproduz as previsões de teste gravadas."""
    return bool(np.all(np.isfinite(refit.prediction)) and refit.max_abs_diff <= tolerance)


def run_refits(ctx, art, *, horizons: list[int], tolerance: float, repeats: int) -> tuple[list[Refit], list[dict]]:
    refitter = Refitter(ctx, art, repeats=repeats)
    plan = [("xgboost_v2", c) for c in ("A", "B", "C")] + [("xgboost_delta", c) for c in ("A", "C")] + [("ridge_v2", "A"), ("statespace", "A"), ("hybrid", "A")]
    refits, records = [], []
    for model, config in plan:
        for h in horizons:
            refit = getattr(refitter, model)(config, h)
            ok = reproduces(refit, tolerance)
            records.append({"model": model, "config": config, "horizon_steps": h, "max_abs_diff_pp": refit.max_abs_diff, "tolerance_pp": tolerance, "reproduced": ok})
            if ok:
                refits.append(refit)
    return refits, records
