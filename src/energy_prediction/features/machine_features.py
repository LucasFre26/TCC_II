"""Atributos por máquina para o modelo global (H2), vetorizados sobre o painel.

Para a máquina j na origem t, só valores até t: CPU em t, defasagens e janelas da configuração A
(sem preenchimento: exemplo com atributo ausente é descartado e contado), calendário, CPU relativa
à média de treino da máquina, média, desvio e volatilidade de treino, código de `failure_domain_2`
e a CPU agregada da frota em t com defasagens 1, 3 e 6. O alvo é a CPU da máquina em t + h.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_prediction.features.cpu_features import SECONDS_PER_DAY, GroupSpec
from energy_prediction.preprocessing.panel import MachinePanel

FLEET_LAGS = (1, 3, 6)


@dataclass
class MachineFeatures:
    names: list[str]
    dynamic: dict[str, np.ndarray]  # (intervalos, máquinas)
    static: dict[str, np.ndarray]  # (máquinas,)
    fleet: dict[str, np.ndarray]  # (intervalos,)
    cpu: np.ndarray  # (intervalos, máquinas)

    def design(self, origins: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Matriz (len(origins), atributos) para pares (origem, coluna da máquina)."""
        origins = np.asarray(origins, dtype=int)
        columns = np.asarray(columns, dtype=int)
        out = np.empty((len(origins), len(self.names)), dtype=np.float32)
        for k, name in enumerate(self.names):
            if name in self.dynamic:
                out[:, k] = self.dynamic[name][origins, columns]
            elif name in self.static:
                out[:, k] = self.static[name][columns]
            else:
                out[:, k] = self.fleet[name][origins]
        return out

    def target(self, origins: np.ndarray, columns: np.ndarray, steps: int) -> np.ndarray:
        index = np.asarray(origins, dtype=int) + int(steps)
        valid = index < self.cpu.shape[0]
        out = np.full(len(index), np.nan, dtype=np.float32)
        out[valid] = self.cpu[index[valid], np.asarray(columns, dtype=int)[valid]]
        return out


def build_machine_features(panel: MachinePanel, spec: GroupSpec, stats: pd.DataFrame, domains: pd.DataFrame, fleet_cpu: np.ndarray, *, step_seconds: float) -> MachineFeatures:
    cpu = panel.values["cpu"].astype(np.float32)
    frame = pd.DataFrame(cpu)
    dynamic: dict[str, np.ndarray] = {"cpu_t": cpu}
    for lag in spec.lags:
        dynamic[f"cpu_lag_{lag}"] = frame.shift(lag).to_numpy(dtype=np.float32)
    for window in spec.windows:
        rolling = frame.rolling(window, min_periods=window)
        dynamic[f"cpu_roll_mean_{window}"] = rolling.mean().to_numpy(dtype=np.float32)
        dynamic[f"cpu_roll_std_{window}"] = rolling.std(ddof=0).to_numpy(dtype=np.float32)
    s = stats.set_index("machine_id").reindex(panel.machines)
    mean = s["train_mean"].to_numpy(dtype=np.float32)
    dynamic["cpu_rel_train_mean"] = cpu - mean[None, :]
    codes = pd.Categorical(domains.set_index("machine_id")["failure_domain_2"].reindex(panel.machines).astype(str)).codes.astype(np.float32)
    static = {
        "train_mean": mean,
        "train_std": s["train_std"].to_numpy(dtype=np.float32),
        "train_volatility": s["train_volatility"].to_numpy(dtype=np.float32),
        "failure_domain_code": codes,
    }
    phase = 2.0 * math.pi * ((panel.bins * float(step_seconds)) % SECONDS_PER_DAY) / SECONDS_PER_DAY
    fleet_series = pd.Series(np.asarray(fleet_cpu, dtype=float))
    fleet = {"sin_day": np.sin(phase).astype(np.float32), "cos_day": np.cos(phase).astype(np.float32), "fleet_cpu_t": fleet_series.to_numpy(dtype=np.float32)}
    for lag in FLEET_LAGS:
        fleet[f"fleet_cpu_lag_{lag}"] = fleet_series.shift(lag).to_numpy(dtype=np.float32)
    names = [*dynamic, *static, *fleet]
    return MachineFeatures(names=names, dynamic=dynamic, static=static, fleet=fleet, cpu=cpu)


def sample_machines(stats: pd.DataFrame, n: int, *, seed: int, strata: int = 5) -> list[str]:
    """Amostra estratificada por quintil da média de treino (só máquinas com estatística), com semente."""
    has = stats[stats["has_statistics"]].copy()
    if n >= len(has):
        return sorted(has["machine_id"].tolist())
    rng = np.random.default_rng(int(seed))
    has["stratum"] = pd.qcut(has["train_mean"].rank(method="first"), q=int(strata), labels=False)
    chosen = []
    sizes = np.floor(has["stratum"].value_counts().sort_index() / len(has) * n).astype(int)
    remainder = int(n - sizes.sum())
    for stratum, size in sizes.items():
        size = int(size) + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        pool = sorted(has.loc[has["stratum"] == stratum, "machine_id"])
        chosen += list(rng.choice(pool, size=min(size, len(pool)), replace=False))
    return sorted(chosen)


def training_pairs(origins: np.ndarray, columns: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Produto (origens com passo `stride`) × máquinas; o mesmo corte temporal para todas as máquinas."""
    chosen = np.asarray(origins, dtype=int)[:: int(stride)]
    o = np.repeat(chosen, len(columns))
    c = np.tile(np.asarray(columns, dtype=int), len(chosen))
    return o, c


def design_effect(cpu_train: np.ndarray) -> dict:
    """Correlação média entre máquinas (séries padronizadas) e n efetivo por instante: M / (1 + (M − 1)·ρ̄)."""
    X = pd.DataFrame(cpu_train).ffill().bfill().to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 1e-9)
    Z = (X[:, keep] - X[:, keep].mean(axis=0)) / X[:, keep].std(axis=0)
    m = Z.shape[1]
    if m < 2:
        return {"n_machines": int(m), "mean_correlation": np.nan, "effective_per_instant": float(m)}
    rho = float((m * np.var(Z.mean(axis=1)) - 1.0) / (m - 1))
    return {"n_machines": int(m), "mean_correlation": rho, "effective_per_instant": float(m / (1.0 + (m - 1) * max(rho, 0.0)))}
