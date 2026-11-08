"""Família de integridade: auditoria da população, divisão interna e alinhamento com a V2.

Roda antes de qualquer ajuste. Se a auditoria não puder ser gravada, nenhuma família ajusta modelo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from energy_prediction.energy.fan import FanModel
from energy_prediction.energy.integrate import trapezoid_energy
from energy_prediction.evaluation.nested import check_alignment, eligible_for_window
from energy_prediction.experiments.v3_context import V3Context
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.neural.windows import channel_matrix, config_channels
from energy_prediction.preprocessing.population_audit import audit_population
from energy_prediction.runtime.output import dump_json
from energy_prediction.validation.rolling_origin import describe_splits

FAMILY = "integrity"


def _reference_folds(ctx: V3Context) -> list[dict] | None:
    if ctx.v2_dir is None:
        return None
    return json.loads((ctx.v2_dir / "folds.json").read_text(encoding="utf-8"))


def window_alignment(ctx: V3Context) -> list[dict]:
    """Para cada janela candidata: origens de treino que saem e confirmação de que a avaliação fica."""
    groups = ctx.cfg["features"]["ablation"]["C"]
    channels = config_channels(ctx.specs, groups)
    _, valid, _ = channel_matrix(ctx.series, channels)
    origins = ctx.table["origin_index"].to_numpy()
    evaluation = np.concatenate([*(f.validation for f in ctx.splits.folds), ctx.splits.test])
    rows = []
    for window in ctx.cfg["families"]["neural"]["window_candidates"]:
        eligible = eligible_for_window(origins, valid, int(window))
        rows.append(
            {
                "window_steps": int(window),
                "train_origins_without_history": int((~eligible[ctx.splits.development]).sum()),
                "evaluation_origins_without_history": int((~eligible[evaluation]).sum()),
                "test_range_unchanged": list(ctx.splits.test_range),
                "first_eligible_origin": int(origins[eligible].min()) if eligible.any() else None,
            }
        )
        if (~eligible[evaluation]).any():
            raise ValueError(f"Janela de {window} passos deixaria origens de avaliação sem histórico.")
    return rows


def math_checks() -> dict:
    """Casos fechados do modelo de Fan e da integral, sem previsão."""
    fan = FanModel(p_idle=100.0, p_busy=200.0)
    linear, _ = fan.power([0.0, 50.0, 100.0], 1.0)
    return {
        "fan_linear_P0_P05_P1": [float(v) for v in linear],
        "trapezoid_constant_150w_6_steps_joules": trapezoid_energy(np.full(7, 150.0), 300.0),
        "expected_constant_joules": 150.0 * 6 * 300.0,
    }


def run(ctx: V3Context, out: Path, **_runtime) -> dict:
    cohort = ctx.cfg["cohort"]
    audit = audit_population(
        ctx.grid,
        ctx.meta,
        ctx.machine_info,
        step_seconds=ctx.step_seconds,
        cohort_cpu_num=cohort["cpu_num"],
        cohort_min_machines=int(cohort["min_machines"]),
        expected_weights=ctx.cfg["reference"]["expected_weights"],
    )
    dump_json(out / "population_audit.json", audit)

    alignment = check_alignment(ctx.splits, _reference_folds(ctx), ctx.cfg["reference"]["expected_test_range"])
    alignment["neural_windows"] = window_alignment(ctx)
    dump_json(out / "alignment.json", alignment)
    dump_json(
        out / "nested_splits.json",
        {
            "enabled": ctx.nested,
            "parameters": ctx.resolved["nested"],
            "splits": [s.describe() for s in (ctx.inner or {}).values()],
            "dropped_folds": ctx.inner_dropped,
        },
    )
    dump_json(out / "folds.json", describe_splits(ctx.table, ctx.splits, target_column(ctx.steps[0])))
    dump_json(
        out / "series_profile.json",
        {
            "n_points": int(len(ctx.series)),
            "cpu_missing_points": int(ctx.series["cpu"].isna().sum()),
            "aggregation": ctx.aggregation_info,
            "resolved": ctx.resolved,
            "distribution_features": ctx.dist_info,
            "data_profile": {k: v for k, v in ctx.profile.items() if k not in ("missing_values",)},
        },
    )
    dump_json(out / "math_checks.json", math_checks())
    return {
        "weight_main": audit["weight_main"],
        "weight_cohort": audit["weight_cohort"],
        "only_in_main": [item["machine_id"] for item in audit["only_in_main"]],
        "test_range": list(ctx.splits.test_range),
    }
