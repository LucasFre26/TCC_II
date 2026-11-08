"""Auditoria da população: análise principal (sem filtro de status) contra a coorte (só ativas).

A principal usa a regra de `aggregate_weighted`: máquinas da grade com metadado, sem filtro de
status. A coorte usa `select_homogeneous_cohort`: ativas com o mesmo cpu_num. A auditoria não
presume que as contagens, os pesos ou as séries sejam iguais; ela os mede.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.preprocessing.cluster import aggregate_weighted, select_homogeneous_cohort


class PopulationAuditError(ValueError):
    """Os pesos divergem do esperado e a diferença não é explicada pelos identificadores listados."""


def audit_population(
    grid: pd.DataFrame,
    meta: pd.DataFrame,
    machine_info: pd.DataFrame,
    *,
    step_seconds: float,
    cohort_cpu_num: float | None,
    cohort_min_machines: int,
    expected_weights: dict | None = None,
) -> dict:
    grid_machines = set(grid["machine_id"].astype(str).unique())
    info = machine_info.set_index("machine_id")
    main = sorted(grid_machines & set(info.index))
    present = set(main)
    cohort, cohort_cpu = select_homogeneous_cohort(
        machine_info, cpu_num=cohort_cpu_num, min_machines=cohort_min_machines, present_machines=present
    )
    only_main = sorted(set(main) - set(cohort))
    only_cohort = sorted(set(cohort) - set(main))
    statuses = meta.assign(machine_id=meta["machine_id"].astype(str)).groupby("machine_id")["status"]
    detail = []
    for machine in only_main:
        values = statuses.get_group(machine).value_counts() if machine in statuses.groups else pd.Series(dtype=int)
        detail.append(
            {
                "machine_id": machine,
                "cpu_num": float(info.loc[machine, "cpu_num"]),
                "active": bool(info.loc[machine, "active"]),
                "status_counts": {str(k): int(v) for k, v in values.items()},
                "reason": "inativa (status fora dos ativos)"
                if not bool(info.loc[machine, "active"])
                else f"cpu_num diferente do da coorte ({cohort_cpu:g})",
            }
        )
    main_weight = float(info.loc[main, "cpu_num"].sum())
    cohort_weight = float(info.loc[cohort, "cpu_num"].sum())

    main_series, _ = aggregate_weighted(grid, machine_info, step_seconds=step_seconds)
    cohort_series, _ = aggregate_weighted(
        grid.loc[grid["machine_id"].isin(cohort)],
        machine_info.loc[machine_info["machine_id"].isin(cohort)],
        step_seconds=step_seconds,
    )
    joined = main_series[["bin", "cpu"]].merge(cohort_series[["bin", "cpu"]], on="bin", suffixes=("_main", "_cohort"))
    difference = (joined["cpu_main"] - joined["cpu_cohort"]).abs()
    audit = {
        "rule_main": "máquinas da grade com metadado, sem filtro de status (aggregate_weighted)",
        "rule_cohort": "máquinas ativas com o mesmo cpu_num (select_homogeneous_cohort)",
        "n_machines_metadata": int(machine_info["machine_id"].nunique()),
        "n_machines_grid": int(len(grid_machines)),
        "n_machines_grid_without_metadata": int(len(grid_machines - set(info.index))),
        "n_machines_main": int(len(main)),
        "n_machines_cohort": int(len(cohort)),
        "n_inactive_in_main": int((~info.loc[main, "active"]).sum()),
        "cohort_cpu_num": float(cohort_cpu),
        "weight_main": main_weight,
        "weight_cohort": cohort_weight,
        "weight_difference": main_weight - cohort_weight,
        "only_in_main": detail,
        "only_in_cohort": only_cohort,
        "series_comparison": {
            "n_common_bins": int(len(joined)),
            "max_abs_cpu_difference": float(difference.max()) if len(difference) else float("nan"),
            "mean_abs_cpu_difference": float(difference.mean()) if len(difference) else float("nan"),
            "series_identical": bool(len(difference) > 0 and np.nanmax(difference.to_numpy()) == 0.0),
        },
        "statement": "Principal e coorte são populações distintas; métricas são reportadas em separado.",
    }
    audit["expected_weights_check"] = _check_expected(audit, expected_weights)
    return audit


def _check_expected(audit: dict, expected: dict | None) -> dict | None:
    if not expected:
        return None
    main_ok = float(expected["main"]) == audit["weight_main"]
    cohort_ok = float(expected["cohort"]) == audit["weight_cohort"]
    listed = sum(item["cpu_num"] for item in audit["only_in_main"])
    expected_gap = float(expected["main"]) - float(expected["cohort"])
    # A diferença esperada (V2) e a observada precisam ser a soma dos cpu_num das máquinas listadas.
    explained = abs(expected_gap - listed) < 1e-9 and abs(audit["weight_difference"] - listed) < 1e-9
    check = {
        "expected_main": float(expected["main"]),
        "expected_cohort": float(expected["cohort"]),
        "main_matches": main_ok,
        "cohort_matches": cohort_ok,
        "expected_gap": expected_gap,
        "listed_cpu_num_sum": listed,
        "difference_explained_by_listed_ids": explained,
    }
    if not (main_ok and cohort_ok) and not explained:
        raise PopulationAuditError(
            f"Pesos da principal ({audit['weight_main']:g}) e da coorte ({audit['weight_cohort']:g}) divergem do "
            f"esperado ({expected['main']}, {expected['cohort']}) e a diferença não é explicada pelas máquinas listadas."
        )
    return check
