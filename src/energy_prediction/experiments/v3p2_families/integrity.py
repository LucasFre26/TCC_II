"""Família de integridade da V3 Parte 2: blocos, painel, população e reprodução da V2.

Roda antes de qualquer outra. Grava os blocos resolvidos, a conferência do painel contra a série
agregada, a auditoria da população e a reprodução do XGBoost da V2 no teste herdado: com treino
12–1991 e os hiperparâmetros gravados em `results/v2-full/main/selections.json`, as previsões
precisam coincidir com as preservadas dentro da tolerância. Se não coincidirem, a família falha e
nenhuma comparação é gravada.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.experiments.v3_context import resolve_path
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.registry import XGBoostModel
from energy_prediction.preprocessing.population_audit import audit_population
from energy_prediction.runtime.output import dump_json

FAMILY = "integrity"


class ReproductionError(RuntimeError):
    """O XGBoost da V2 reajustado no teste herdado não reproduz as previsões preservadas."""


def reproduce_v2_xgboost(ctx, v2_dir: Path, tolerance: float) -> dict:
    block = next(b for b in ctx.blocks if b.role == "inherited_test")
    selections = json.loads((v2_dir / "selections.json").read_text(encoding="utf-8"))
    preserved = pd.read_csv(v2_dir / "predictions.csv")
    preserved = preserved[(preserved["split"] == "test_final") & (preserved["model"] == "xgboost") & (preserved["config"] == "A")]
    columns = ctx.columns["A"]
    X = ctx.table[columns].to_numpy(dtype=float)
    origins = ctx.table["origin_index"].to_numpy()
    xgb = ctx.cfg["families"]["baselines"]["xgboost"]
    report = {"block": block.label, "train_range": list(block.train_range), "tolerance": float(tolerance), "horizons": []}
    for steps in ctx.steps:
        record = next(r for r in selections if r["model"] == "xgboost" and r["config"] == "A" and int(r["horizon_steps"]) == steps)
        y = ctx.table[target_column(steps)].to_numpy(dtype=float)
        model = XGBoostModel(record["best_params"], seed=ctx.seed, n_jobs=int(xgb["n_jobs"]), fixed=xgb.get("fixed"))
        model.fit(X[block.train], y[block.train])
        pred = pd.Series(model.predict(X[block.evaluation]), index=origins[block.evaluation])
        ref = preserved[preserved["horizon_steps"] == steps].set_index("origin_index")["y_pred"]
        if not pred.index.equals(ref.sort_index().index):
            raise ReproductionError(f"Origens do teste herdado diferem das preservadas no horizonte {steps}.")
        diff = float(np.max(np.abs(pred.to_numpy() - ref.sort_index().to_numpy())))
        report["horizons"].append({"horizon_steps": int(steps), "best_params": record["best_params"], "max_abs_diff_pp": diff, "n": int(len(pred))})
    report["max_abs_diff_pp"] = max(h["max_abs_diff_pp"] for h in report["horizons"])
    report["passed"] = bool(report["max_abs_diff_pp"] <= float(tolerance))
    if not report["passed"]:
        raise ReproductionError(f"XGBoost da V2 não reproduzido no teste herdado: diferença máxima {report['max_abs_diff_pp']:.3g} p.p. > {tolerance}.")
    return report


def run(ctx, out: Path, **_runtime) -> dict:
    dump_json(out / "blocks.json", {"blocks": ctx.resolved["blocks"], "dropped_blocks": ctx.dropped_blocks, "inherited_note": "dia 7 = teste herdado da V3 (inspecionado na V2, na V3 e na análise expandida); nunca confirmação independente"})
    panel_check = ctx.check_panel()
    dump_json(out / "panel_check.json", panel_check)
    audit = audit_population(ctx.grid, ctx.meta, ctx.machine_info, step_seconds=ctx.step_seconds, cohort_cpu_num=None, cohort_min_machines=1)
    v3_run = resolve_path(ctx.root, ctx.cfg["reference"]["v3_run"])
    if v3_run is not None and (v3_run / "integrity/population_audit.json").is_file():
        v3_audit = json.loads((v3_run / "integrity/population_audit.json").read_text(encoding="utf-8"))
        keys = ("n_machines_main", "n_machines_cohort", "weight_main", "weight_cohort", "only_in_main")
        audit["matches_v3"] = {key: audit.get(key) == v3_audit.get(key) for key in keys if key in v3_audit}
        audit["matches_v3_all"] = bool(audit["matches_v3"]) and all(audit["matches_v3"].values())
    dump_json(out / "population_audit.json", audit)
    dump_json(out / "series_profile.json", {"aggregation": ctx.aggregation_info, "resolved": {k: v for k, v in ctx.resolved.items() if k != "blocks"}, "n_points": int(len(ctx.series))})
    v2_dir = resolve_path(ctx.root, ctx.cfg["reference"]["v2_dir"])
    excluded = []
    if v2_dir is None:
        if ctx.cfg["mode"] == "full":
            raise ReproductionError("O modo completo exige reference.v2_dir para a reprodução do teste herdado.")
        reproduction = {"skipped": True, "reason": "sem pasta da V2 (smoke sintético)"}
        excluded.append({"item": "reprodução do XGBoost da V2 no teste herdado", "reason": reproduction["reason"]})
    else:
        reproduction = reproduce_v2_xgboost(ctx, v2_dir, float(ctx.cfg["reference"]["reproduction_tolerance"]))
    dump_json(out / "reproduction.json", reproduction)
    return {
        "n_blocks": len(ctx.blocks),
        "panel_max_abs_diff_pp": panel_check["max_abs_diff_pp"],
        "reproduction_max_abs_diff_pp": reproduction.get("max_abs_diff_pp"),
        "excluded": excluded,
    }
