"""Família de energia: modelo de Fan, trajetórias de seis passos, energia de referência e projetada.

Grandeza: energia de TI modelada de um servidor substituto à utilização agregada da frota
(estimativa paramétrica, não consumo medido). A referência integra a trajetória de CPU
observada; a projetada, a trajetória prevista, sempre com o extremo esquerdo na CPU observada na
origem. Os três valores de r saem sempre; nenhum é escolhido pelo erro no teste. PUE não é
calculado.

Passos que a V2 não previu (2, 4 e 5): o passo 2 herda o hiperparâmetro do 3, e os passos 4 e
5 herdam o do 6 (`inherited_params`). O herdeiro é ajustado uma vez na região de desenvolvimento.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.energy.aggregate_effect import aggregate_effect
from energy_prediction.energy.fan import FanModel, utilization
from energy_prediction.energy.integrate import JOULES_PER_WH, horizon_energies
from energy_prediction.energy.propagate import STATEMENT as PROPAGATION_STATEMENT
from energy_prediction.energy.propagate import linearized_energy_difference, linearized_power_difference
from energy_prediction.energy.substitute import load_substitute
from energy_prediction.experiments.v3_context import V3Context, resolve_path
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT, target_column
from energy_prediction.models.delta import DeltaXGBoost
from energy_prediction.models.hybrid import HybridCorrector
from energy_prediction.models.registry import RidgeModel, XGBoostModel
from energy_prediction.models.statespace import StateSpaceForecaster
from energy_prediction.models.v3_registry import validate_family_models
from energy_prediction.runtime.output import dump_json

FAMILY = "energy"
QUANTITY = "energia de TI modelada (estimativa paramétrica, servidor substituto SPECpower_ssj2008)"
NEURAL = ("dlinear", "tcn", "nbeatsx")


def inheritance_map(path_steps: list[int], scored: list[int]) -> dict[int, int]:
    """Cada passo usa o hiperparâmetro do menor horizonte pontuado que o cobre (2->3, 4,5->6)."""
    mapping = {}
    for step in path_steps:
        candidates = [h for h in sorted(scored) if h >= step]
        if not candidates:
            raise ValueError(f"Passo {step} além do maior horizonte pontuado.")
        mapping[int(step)] = int(candidates[0])
    return mapping


class TrajectoryUnavailable(Exception):
    """A trajetória do modelo não pode ser construída nesta corrida (motivo na mensagem)."""


def _read_json(path: Path):
    if not path.is_file():
        raise TrajectoryUnavailable(f"{path.parent.name}/{path.name} ausente nesta corrida")
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_params(selections: list[dict], model: str, config: str = "A") -> dict[int, dict]:
    out = {}
    for record in selections:
        if record["model"] == model and record["config"] == config and record.get("contrast") is None and record.get("mode", "direct") == "direct":
            out[int(record["horizon_steps"])] = record["best_params"]
    if not out:
        raise TrajectoryUnavailable(f"sem seleção de {model}/{config}")
    return out


class TrajectoryBuilder:
    def __init__(self, ctx: V3Context, run_dir):
        self.ctx = ctx
        self.run_dir = run_dir
        self.table = ctx.table
        self.dev = ctx.splits.development
        self.test = ctx.splits.test
        self.columns = ctx.columns["A"]
        self.mapping = inheritance_map(ctx.path_steps, ctx.steps)
        self.origins = self.table["origin_index"].to_numpy()
        self.dev_end = int(self.origins[self.dev].max())
        self.checks: dict[str, dict] = {}
        self._forecaster = None

    @property
    def forecaster(self) -> StateSpaceForecaster:
        if self._forecaster is None:
            maxiter = int(self.ctx.cfg["families"]["residual"]["statespace"]["maxiter"])
            self._forecaster = StateSpaceForecaster(self.ctx.series["cpu"].to_numpy(), horizons=self.ctx.path_steps, maxiter=maxiter)
        return self._forecaster

    def _fit_steps(self, make_and_predict) -> np.ndarray:
        out = np.full((len(self.test), len(self.ctx.path_steps)), np.nan)
        for j, step in enumerate(self.ctx.path_steps):
            y = self.table[target_column(step)].to_numpy(dtype=float)
            train = self.dev[np.isfinite(y[self.dev])]
            out[:, j] = make_and_predict(step, self.mapping[step], train, y)
        return out

    def persistence(self) -> np.ndarray:
        cpu = self.table[PERSISTENCE_INPUT].to_numpy(dtype=float)[self.test]
        return np.repeat(cpu[:, None], len(self.ctx.path_steps), axis=1)

    def _residual_info(self) -> dict:
        return _read_json(self.run_dir.family_path("residual") / "family_info.json")

    def statespace(self) -> np.ndarray:
        specs = {int(k): v for k, v in self._residual_info()["statespace_specification_by_horizon"].items()}

        def predict(step, scored, train, y):
            return self.forecaster.forecasts(specs[scored], self.dev_end).by_horizon[step][self.origins[self.test]]

        return self._fit_steps(predict)

    def _v2_params(self, model: str) -> dict[int, dict]:
        if self.ctx.v2_dir is None:
            raise TrajectoryUnavailable("pasta da V2 não configurada")
        return _selection_params(_read_json(self.ctx.v2_dir / "selections.json"), model)

    def ridge_v2(self) -> np.ndarray:
        params = self._v2_params("ridge")
        X = self.table[self.columns].to_numpy(dtype=float)

        def predict(step, scored, train, y):
            return RidgeModel(params[scored]["alpha"]).fit(X[train], y[train]).predict(X[self.test])

        return self._fit_steps(predict)

    def xgboost_v2(self) -> np.ndarray:
        params = self._v2_params("xgboost")
        xgb = self.ctx.cfg["families"]["energy"]["xgboost_v2"]
        X = self.table[self.columns].to_numpy(dtype=float)

        def predict(step, scored, train, y):
            model = XGBoostModel(params[scored], seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
            return model.fit(X[train], y[train]).predict(X[self.test])

        return self._fit_steps(predict)

    def xgboost_delta(self) -> np.ndarray:
        params = _selection_params(_read_json(self.run_dir.family_path("residual") / "selections.json"), "xgboost_delta")
        xgb = self.ctx.cfg["families"]["residual"]["xgboost"]
        X = self.table[self.columns].to_numpy(dtype=float)

        def predict(step, scored, train, y):
            model = DeltaXGBoost(params[scored], feature_names=self.columns, seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
            return model.fit(X[train], y[train]).predict(X[self.test])

        return self._fit_steps(predict)

    def hybrid(self) -> np.ndarray:
        params = _selection_params(_read_json(self.run_dir.family_path("residual") / "selections.json"), "hybrid")
        specs = {int(k): v for k, v in self._residual_info()["statespace_specification_by_horizon"].items()}
        xgb = self.ctx.cfg["families"]["residual"]["xgboost"]
        X = self.table[self.columns].to_numpy(dtype=float)

        def predict(step, scored, train, y):
            forecasts = self.forecaster.forecasts(specs[scored], self.dev_end)
            base = forecasts.by_horizon[step][self.origins]
            corrector = HybridCorrector(params[scored], seed=self.ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
            corrector.fit(X[train], y[train], base[train], forecasts)
            return corrector.predict(X[self.test], base[self.test])

        return self._fit_steps(predict)

    def neural(self, model: str) -> np.ndarray:
        path = self.run_dir.family_path("neural") / "predictions.csv"
        if not path.is_file():
            raise TrajectoryUnavailable("família neural não concluída nesta corrida")
        frame = pd.read_csv(path)
        part = frame[(frame["model"] == model) & (frame["config"] == "A") & (frame["mode"] == "path") & (frame["split"] == "test_final")]
        if part.empty:
            raise TrajectoryUnavailable(f"trajetória A de {model} ausente (sem preenchimento pelo valor terminal)")
        wide = part.pivot(index="origin_index", columns="horizon_steps", values="y_pred")
        wide = wide.reindex(index=self.origins[self.test], columns=self.ctx.path_steps)
        return wide.to_numpy(dtype=float)

    def build(self, model: str) -> np.ndarray:
        if model in NEURAL:
            return self.neural(model)
        return getattr(self, model)()

    def consistency(self, model: str, trajectory: np.ndarray) -> dict | None:
        """Passos 1, 3 e 6 reajustados contra as previsões já gravadas no teste (mesmos parâmetros)."""
        sources = {
            "ridge_v2": (self.ctx.v2_dir / "predictions.csv" if self.ctx.v2_dir else None, "ridge"),
            "xgboost_v2": (self.ctx.v2_dir / "predictions.csv" if self.ctx.v2_dir else None, "xgboost"),
            "xgboost_delta": (self.run_dir.family_path("residual") / "predictions.csv", "xgboost_delta"),
            "hybrid": (self.run_dir.family_path("residual") / "predictions.csv", "hybrid"),
            "statespace": (self.run_dir.family_path("residual") / "predictions.csv", "statespace"),
        }
        if model not in sources or sources[model][0] is None or not sources[model][0].is_file():
            return None
        frame = pd.read_csv(sources[model][0])
        frame = frame[(frame["model"] == sources[model][1]) & (frame["config"] == "A") & (frame["split"] == "test_final")]
        out = {}
        for step in self.ctx.steps:
            stored = frame[frame["horizon_steps"] == step].set_index("origin_index")["y_pred"].reindex(self.origins[self.test])
            out[str(step)] = float(np.nanmax(np.abs(stored.to_numpy() - trajectory[:, step - 1])))
        return {"max_abs_difference_vs_stored_test_predictions": out}


def _point_model_for_figure(run_dir, available: list[str], horizon: int) -> dict:
    """Modelo pontual do gráfico de energia: menor MAE interno em A no maior horizonte (não o do teste)."""
    candidates = []
    for family in ("residual", "neural"):
        path = run_dir.family_path(family) / "selections.json"
        if not path.is_file():
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            if (
                record["model"] in available
                and record["config"] == "A"
                and int(record["horizon_steps"]) == horizon
                and record.get("contrast") is None
                and record.get("mode", "direct") == "direct"
                and record.get("rule") == "inner_validation"
            ):
                candidates.append((float(record["selection_score"]), record["model"]))
    if not candidates:
        return {"model": None, "rule": "sem seleção interna disponível"}
    score, model = min(candidates)
    return {"model": model, "inner_mae": score, "rule": f"menor MAE interno na configuração A, horizonte {horizon} passos"}


def run(ctx: V3Context, out: Path, *, run_dir, **_runtime) -> dict:
    fam = ctx.cfg["families"]["energy"]
    substitute, fan = load_substitute(resolve_path(ctx.root, fam["substitute"]))  # falha antes de qualquer energia
    models = validate_family_models(FAMILY, list(fam["trajectory_models"]))
    exponents = [float(r) for r in fam["exponents"]]

    effect, effect_summary = aggregate_effect(
        ctx.grid, ctx.machine_info, fan, exponents, unstable_below=float(fam["unstable_mean_utilization"]), population=ctx.population
    )
    if not fam["fleet_sum"]:
        effect = effect.drop(columns=["sum_of_p_w"])

    builder = TrajectoryBuilder(ctx, run_dir)
    test = ctx.splits.test
    origins = ctx.table["origin_index"].to_numpy()[test]
    timestamps = ctx.table["timestamp"].to_numpy()[test]
    cpu_origin = ctx.table[PERSISTENCE_INPUT].to_numpy(dtype=float)[test]
    observed = np.column_stack([ctx.table[target_column(s)].to_numpy(dtype=float)[test] for s in ctx.path_steps])
    inherited = [s for s, h in builder.mapping.items() if s != h]

    trajectories, availability, excluded = {}, {}, []
    for model in models:
        try:
            trajectories[model] = builder.build(model)
            availability[model] = {"available": True, "consistency": builder.consistency(model, trajectories[model])}
        except TrajectoryUnavailable as reason:
            availability[model] = {"available": False, "reason": str(reason)}
            excluded.append({"item": f"energia de {model}", "reason": str(reason)})

    trajectory_rows = []
    for model, path in trajectories.items():
        for j, step in enumerate(ctx.path_steps):
            trajectory_rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "origin_index": origins,
                        "step": step,
                        "y_true": observed[:, j],
                        "y_pred": path[:, j],
                        "inherited_params": bool(step in inherited and model not in ("persistence",) + NEURAL),
                    }
                )
            )

    energy_rows, power_rows, summary_rows = [], [], []
    invalid_counts = {}
    for r in exponents:
        p_origin, invalid_origin = fan.power(cpu_origin, r)
        p_observed, invalid_observed = fan.power(observed, r)
        reference = horizon_energies(p_origin, p_observed, ctx.steps, ctx.step_seconds)
        u_observed, _ = utilization(observed)
        for model, path in trajectories.items():
            p_predicted, invalid_predicted = fan.power(path, r)
            invalid_counts[f"{model}/r={r:g}"] = invalid_predicted
            projected = horizon_energies(p_origin, p_predicted, ctx.steps, ctx.step_seconds)
            u_predicted, _ = utilization(path)
            linear_power = linearized_power_difference(fan, u_observed, u_predicted, r)
            linear_energy = linearized_energy_difference(linear_power, ctx.steps, ctx.step_seconds)
            for h in ctx.steps:
                e_ref, e_proj = reference[h], projected[h]
                frame = pd.DataFrame(
                    {
                        "model": model,
                        "r": r,
                        "horizon_steps": h,
                        "origin_index": origins,
                        "timestamp": timestamps,
                        "reference_energy_j": e_ref,
                        "projected_energy_j": e_proj,
                        "reference_energy_wh": e_ref / JOULES_PER_WH,
                        "projected_energy_wh": e_proj / JOULES_PER_WH,
                        "energy_difference_wh": (e_proj - e_ref) / JOULES_PER_WH,
                        "absolute_error_wh": np.abs(e_proj - e_ref) / JOULES_PER_WH,
                        "relative_error": (e_proj - e_ref) / e_ref,
                        "linearized_difference_wh_diagnostic": linear_energy[h] / JOULES_PER_WH,
                        "complete_trajectory": np.isfinite(e_ref) & np.isfinite(e_proj),
                        "inherited_params": bool(any(s in inherited for s in range(1, h + 1)) and model not in ("persistence",) + NEURAL),
                    }
                )
                energy_rows.append(frame)
                done = frame[frame["complete_trajectory"]]
                summary_rows.append(
                    {
                        "quantity": QUANTITY,
                        "model": model,
                        "r": r,
                        "horizon_steps": h,
                        "n_complete": int(len(done)),
                        "n_incomplete": int((~frame["complete_trajectory"]).sum()),
                        "total_reference_wh": float(done["reference_energy_wh"].sum()),
                        "total_projected_wh": float(done["projected_energy_wh"].sum()),
                        "mean_absolute_error_wh": float(done["absolute_error_wh"].mean()) if len(done) else np.nan,
                        "mean_absolute_relative_error": float(done["relative_error"].abs().mean()) if len(done) else np.nan,
                        "mean_signed_difference_wh": float(done["energy_difference_wh"].mean()) if len(done) else np.nan,
                    }
                )
            power_rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "r": r,
                        "origin_index": origins,
                        **{f"power_difference_w_step_{s}": p_predicted[:, j] - p_observed[:, j] for j, s in enumerate(ctx.path_steps)},
                        **{f"linearized_power_difference_w_step_{s}_diagnostic": linear_power[:, j] for j, s in enumerate(ctx.path_steps)},
                    }
                )
            )
        invalid_counts[f"observado/r={r:g}"] = invalid_origin + invalid_observed

    pd.concat(trajectory_rows, ignore_index=True).to_csv(out / "trajectories.csv", index=False)
    effect.to_csv(out / "aggregate_effect.csv", index=False)
    pd.concat(energy_rows, ignore_index=True).to_csv(out / "energy_by_origin.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "energy_summary.csv", index=False)
    pd.concat(power_rows, ignore_index=True).to_csv(out / "power_propagation.csv", index=False)
    figure_model = _point_model_for_figure(run_dir, list(trajectories), max(ctx.steps))
    dump_json(
        out / "energy_info.json",
        {
            "quantity": QUANTITY,
            "statement": (
                "O trace não fornece potência nem energia medidas. A potência obtida da CPU agregada é a de um "
                "servidor substituto, não a potência total da frota; a soma das máquinas presentes está em "
                "aggregate_effect.csv como coluna separada."
            ),
            "propagation_statement": PROPAGATION_STATEMENT,
            "substitute": substitute,
            "model": "P(u) = P_idle + (P_busy - P_idle)(2u - u^r)",
            "exponents": exponents,
            "exponent_rule": "os três valores de r são reportados; nenhum é escolhido pelo erro no teste",
            "integration": {"rule": "trapézio", "step_seconds": ctx.step_seconds, "left_endpoint": "P(u(t)) observado"},
            "inheritance": {
                "map_step_to_scored_horizon": {str(k): v for k, v in builder.mapping.items()},
                "inherited_steps": inherited,
                "applies_to": ["ridge_v2", "xgboost_v2", "xgboost_delta", "hybrid", "statespace (especificação)"],
                "note": "Herdeiros ajustados uma vez na região de desenvolvimento; não é uma busca nova.",
            },
            "availability": availability,
            "invalid_utilization_counts": invalid_counts,
            "aggregate_effect": effect_summary,
            "figure_point_model": figure_model,
        },
    )
    return {
        "models_with_energy": sorted(trajectories),
        "exponents": exponents,
        "figure_point_model": figure_model.get("model"),
        "excluded": excluded,
    }
