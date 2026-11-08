"""Ablação A/B/C/D só nas combinações executadas.

B e C são comparados com A do mesmo modelo nas mesmas origens. D é comparado com o A
reajustado nas origens do contraste (A_on_D). Combinação não executada não vira zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ablation_table(metrics: pd.DataFrame, comparisons: pd.DataFrame | None) -> pd.DataFrame:
    test = metrics[(metrics["split"] == "test_final") & (metrics["mode"] == "direct")]
    rows = []
    for population in test["population"].unique():
        pop = test[test["population"] == population]
        for (model, h), part in pop.groupby(["model", "horizon_steps"]):
            if model == "persistence":
                continue
            main = part[part["analysis_set"] == "main"].set_index("config")
            contrast = part[part["analysis_set"] == "d_contrast"].set_index("config")
            pairs = [(c, "A", main) for c in ("B", "C") if c in main.index and "A" in main.index]
            if "D" in contrast.index and "A_on_D" in contrast.index:
                pairs.append(("D", "A_on_D", contrast))
            for config, ref, frame in pairs:
                mae, ref_mae = frame.loc[config, "mae_pp"], frame.loc[ref, "mae_pp"]
                rows.append(
                    {
                        "population": population, "model": model, "horizon_steps": int(h), "horizon_min": part["horizon_min"].iloc[0], "config": config, "reference_config": ref,
                        "mae_pp": mae, "reference_mae_pp": ref_mae, "delta_mae_pp": mae - ref_mae, "delta_mae_pct": 100.0 * (mae / ref_mae - 1.0),
                        "rmse_pp": frame.loc[config, "rmse_pp"], "reference_rmse_pp": frame.loc[ref, "rmse_pp"], "n": int(frame.loc[config, "n"]), "executed": True,
                    }
                )
    out = pd.DataFrame(rows)
    if comparisons is not None and not out.empty:
        pre = comparisons[(comparisons["analysis_type"] == "pre_registrada") & (comparisons["comparison"].isin(["ablation_vs_A", "d_vs_a"])) & (comparisons["loss"] == "erro_absoluto")]
        pre = pre[["model_a", "config_a", "horizon_steps", "ci_low", "ci_high", "significant", "p_holm"]].rename(columns={"model_a": "model", "config_a": "config", "ci_low": "ci95_low_pp", "ci_high": "ci95_high_pp", "significant": "significant_block_1h"})
        out = out.merge(pre, on=["model", "config", "horizon_steps"], how="left")
    return out


def ablation_grid(table: pd.DataFrame, models: list[str], configs=("B", "C", "D"), horizons=(1, 3, 6)) -> pd.DataFrame:
    """Formato completo modelo × configuração × horizonte, com `executed` falso onde não houve ajuste."""
    rows = []
    main = table[table["population"] == "main"].set_index(["model", "config", "horizon_steps"])
    for m in models:
        for c in configs:
            for h in horizons:
                if (m, c, h) in main.index:
                    r = main.loc[(m, c, h)]
                    rows.append({"model": m, "config": c, "horizon_steps": h, "delta_mae_pp": r["delta_mae_pp"], "delta_mae_pct": r["delta_mae_pct"], "executed": True})
                else:
                    rows.append({"model": m, "config": c, "horizon_steps": h, "delta_mae_pp": np.nan, "delta_mae_pct": np.nan, "executed": False})
    return pd.DataFrame(rows)
