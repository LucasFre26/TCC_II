"""Frota sintética para o smoke da V3: grade por máquina e metadado, sem ler o Alibaba v2018.

Ciclo diário comum, ruído AR(1) compartilhado e deslocamento por máquina. As `inactive_machines`
últimas máquinas recebem um registro de status fora de USING (inativas para a coorte), com o
mesmo cpu_num das demais, reproduzindo a forma da diferença entre principal e coorte da V2.
Parâmetros opcionais da V3 Parte 2 (desligados por padrão, sem alterar os dados padrão): fator
comum adicional entre máquinas, salto de nível no último dia e domínio de falha no metadado.
Os números não são resultado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_fleet(
    *,
    n_machines: int,
    n_steps: int,
    seed: int,
    inactive_machines: int = 1,
    step_seconds: float = 300.0,
    common_factor: float = 0.0,
    level_shift_last_day: float = 0.0,
    failure_domains: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(n_steps))
    phase = 2 * np.pi * (t * step_seconds % 86400) / 86400
    shared = np.zeros(n_steps)
    for i in range(1, n_steps):
        shared[i] = 0.85 * shared[i - 1] + rng.normal(0, 1.2)
    if common_factor:
        # Gerador separado: o fator extra não muda os sorteios da frota padrão.
        extra_rng = np.random.default_rng(int(seed) + 1)
        factor = np.zeros(n_steps)
        for i in range(1, n_steps):
            factor[i] = 0.9 * factor[i - 1] + extra_rng.normal(0, float(common_factor))
        shared = shared + factor
    steps_per_day = int(round(86400 / step_seconds))
    shift = np.where(t >= int(n_steps) - steps_per_day, float(level_shift_last_day), 0.0)
    machines = [f"m_{i:03d}" for i in range(int(n_machines))]
    frames = []
    for position, machine in enumerate(machines):
        offset = rng.normal(0, 6)
        own = rng.normal(0, 3, n_steps)
        cpu = np.clip(38 + offset + 9 * np.sin(phase) + shared + own + shift, 0, 100)
        if position % 5 == 0:
            cpu = np.clip(cpu + 35, 0, 100)  # algumas máquinas quentes para a fração acima do limiar
        frames.append(
            pd.DataFrame(
                {
                    "machine_id": machine,
                    "bin": t,
                    "cpu": cpu,
                    "mem": np.clip(55 + 0.3 * (cpu - 40) + rng.normal(0, 1.5, n_steps), 0, 100),
                    "mem_gps": np.nan,
                    "net_in": np.clip(25 + 0.1 * (cpu - 40) + rng.normal(0, 2, n_steps), 0, 100),
                    "net_out": np.clip(20 + rng.normal(0, 2, n_steps), 0, 100),
                    "disk": np.clip(5 + rng.normal(0, 1, n_steps), 0, 100),
                }
            )
        )
    grid = pd.concat(frames, ignore_index=True)
    # Lacuna curta numa máquina: ausência por intervalo, como no trace.
    gap = (grid["machine_id"] == machines[1]) & grid["bin"].between(100, 104)
    grid.loc[gap, ["cpu", "mem", "net_in", "net_out", "disk"]] = np.nan
    inactive = set(machines[len(machines) - int(inactive_machines) :]) if inactive_machines else set()
    meta = pd.DataFrame(
        {
            "machine_id": machines,
            "time_stamp": 0.0,
            "cpu_num": 96,
            "status": ["USING"] * len(machines),
        }
    )
    extra = pd.DataFrame({"machine_id": sorted(inactive), "time_stamp": 1000.0, "cpu_num": 96, "status": "OFFLINE"})
    meta = pd.concat([meta, extra], ignore_index=True)
    if failure_domains:
        domain = {m: str(k % int(failure_domains)) for k, m in enumerate(machines)}
        meta["failure_domain_2"] = meta["machine_id"].map(domain)
    profile = {
        "source": "synthetic",
        "step_seconds": float(step_seconds),
        "n_machines": int(n_machines),
        "n_grid_points": int(len(grid)),
        "period_bins": [0, int(n_steps) - 1],
        "synthetic": {
            "n_machines": int(n_machines),
            "n_steps": int(n_steps),
            "seed": int(seed),
            "inactive_machines": sorted(inactive),
            "common_factor": float(common_factor),
            "level_shift_last_day": float(level_shift_last_day),
            "failure_domains": int(failure_domains),
        },
    }
    return grid, meta, profile
