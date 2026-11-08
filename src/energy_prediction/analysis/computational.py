"""Complexidade e custo computacional, com a origem de cada medida explícita.

Redes: números da corrida original (CUDA, bf16), incluindo a seleção. Modelos clássicos:
reajuste de teste medido nesta análise em CPU (mediana de repetições). Os dois conjuntos só são
comparáveis em ordem de grandeza; o único tempo presente nas duas famílias é o do reajuste de teste.
"""

from __future__ import annotations

import platform

import numpy as np
import pandas as pd

MIB = 1024.0**2


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecido"


def neural_rows(profile: pd.DataFrame, device_name: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = profile.copy()
    p["horizon"] = p["horizon"].astype(str)
    complexity = pd.DataFrame(
        {
            "model": p["arch"], "config": p["config"], "mode": p["mode"], "horizon": p["horizon"], "complexity_kind": "parâmetros treináveis",
            "n_trainable_parameters": p["n_params"], "n_trees": np.nan, "n_leaves": np.nan, "n_coefficients": np.nan, "final_epochs": p["final_epochs"],
        }
    )
    performance = pd.DataFrame(
        {
            "model": p["arch"], "config": p["config"], "mode": p["mode"], "horizon": p["horizon"], "measurement_source": "corrida original (training_profile.csv)",
            "device": p["device"], "device_name": device_name, "precision": p["precisions"].astype(str).str.strip("[]'"), "n_fits": p["n_fits"],
            "train_seconds_total_with_selection": p["train_seconds_total"], "train_seconds_test_refit": p["train_seconds_test_refit"],
            "inference_seconds_test": p["inference_seconds_test"], "peak_cuda_memory_mib": p["peak_cuda_bytes_max"] / MIB,
        }
    )
    return complexity, performance


def classical_rows(refits, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = cpu_name()
    comp, perf = [], []
    for r in refits:
        c = r.complexity
        comp.append(
            {
                "model": r.model, "config": r.config, "mode": "direct", "horizon": str(r.horizon_steps),
                "complexity_kind": "árvores e folhas" if "n_trees" in c else ("coeficientes" if "n_coefficients" in c else "parâmetros do modelo de estados"),
                "n_trainable_parameters": c.get("n_coefficients", c.get("n_parameters", np.nan)), "n_trees": c.get("n_trees", np.nan), "n_leaves": c.get("n_leaves", np.nan),
                "n_coefficients": c.get("n_coefficients", np.nan), "final_epochs": np.nan,
            }
        )
        perf.append(
            {
                "model": r.model, "config": r.config, "mode": "direct", "horizon": str(r.horizon_steps), "measurement_source": "reajuste de teste medido na análise expandida",
                "device": "cpu", "device_name": name, "precision": "fp64", "n_fits": 1, "train_seconds_total_with_selection": np.nan,
                "train_seconds_test_refit": r.fit_seconds, "inference_seconds_test": r.predict_seconds, "peak_cuda_memory_mib": 0.0,
            }
        )
    perf.append({"model": "persistence", "config": "-", "mode": "direct", "horizon": "todos", "measurement_source": "sem ajuste", "device": "cpu", "device_name": name, "precision": "fp64", "n_fits": 0, "train_seconds_total_with_selection": 0.0, "train_seconds_test_refit": 0.0, "inference_seconds_test": np.nan, "peak_cuda_memory_mib": 0.0})
    comp.append({"model": "persistence", "config": "-", "mode": "direct", "horizon": "todos", "complexity_kind": "sem parâmetros", "n_trainable_parameters": 0, "n_trees": np.nan, "n_leaves": np.nan, "n_coefficients": np.nan, "final_epochs": np.nan})
    return pd.DataFrame(comp), pd.DataFrame(perf)


def tradeoff(complexity: pd.DataFrame, performance: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Custo e MAE de teste (A, direto) por modelo e horizonte, com marca de fronteira de Pareto."""
    test = metrics[(metrics["split"] == "test_final") & (metrics["population"] == "main") & (metrics["analysis_set"] == "main") & (metrics["mode"] == "direct") & (metrics["config"].isin(["A", "-"]))]
    rows = []
    for _, t in test.iterrows():
        h = str(int(t["horizon_steps"]))
        config = "-" if t["model"] == "persistence" else "A"
        horizon = "todos" if t["model"] == "persistence" else h
        c = complexity[(complexity["model"] == t["model"]) & (complexity["config"] == config) & (complexity["horizon"] == horizon)]
        p = performance[(performance["model"] == t["model"]) & (performance["config"] == config) & (performance["horizon"] == horizon)]
        if c.empty or p.empty:
            continue
        c, p = c.iloc[0], p.iloc[0]
        size = c["n_trainable_parameters"] if pd.notna(c["n_trainable_parameters"]) else c["n_leaves"]
        rows.append(
            {
                "model": t["model"], "horizon_steps": int(t["horizon_steps"]), "test_mae_pp": t["mae_pp"], "size_measure": size, "size_kind": c["complexity_kind"],
                "train_seconds_test_refit": p["train_seconds_test_refit"], "device": p["device"], "peak_cuda_memory_mib": p["peak_cuda_memory_mib"], "measurement_source": p["measurement_source"],
            }
        )
    out = pd.DataFrame(rows)
    out["pareto_time_mae"] = False
    for h, part in out.groupby("horizon_steps"):
        ordered = part.sort_values(["train_seconds_test_refit", "test_mae_pp"])
        best = np.inf
        for idx, r in ordered.iterrows():
            if r["test_mae_pp"] < best:
                out.loc[idx, "pareto_time_mae"] = True
                best = r["test_mae_pp"]
    return out
