"""Geradores sintéticos para os testes da V2. Não acessam o Alibaba v2018 nem a rede."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

USAGE_NAMES = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]
META_NAMES = ["machine_id", "time_stamp", "failure_domain_1", "failure_domain_2", "cpu_num", "mem_size", "status"]


def _write_csv_tar(path: Path, frame: pd.DataFrame, member: str, header: bool) -> None:
    payload = frame.to_csv(index=False, header=header).encode("utf-8")
    info = tarfile.TarInfo(name=member)
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))


def make_usage_frame(
    machines: dict[str, int],
    *,
    n_steps: int = 40,
    step: float = 10.0,
    seed: int = 0,
    async_offsets: bool = True,
    gap: tuple[str, int, int] | None = None,
) -> pd.DataFrame:
    """Uma linha por máquina e passo. `gap=(id, primeiro, ultimo)` remove registros dessa máquina."""
    rng = np.random.default_rng(seed)
    rows = []
    for position, machine in enumerate(machines):
        offset = float(rng.uniform(0, step * 0.5)) if async_offsets else 0.0
        for i in range(n_steps):
            if gap is not None and machine == gap[0] and gap[1] <= i <= gap[2]:
                continue
            rows.append(
                {
                    "machine_id": machine,
                    "time_stamp": i * step + offset,
                    "cpu_util_percent": float(np.clip(30 + 5 * position + rng.normal(0, 3), 1, 99)),
                    "mem_util_percent": float(np.clip(50 + rng.normal(0, 2), 1, 99)),
                    "mem_gps": float(np.clip(10 + rng.normal(0, 1), 0, 99)),
                    "mkpi": 3.0,
                    "net_in": float(np.clip(20 + rng.normal(0, 2), 0, 99)),
                    "net_out": float(np.clip(15 + rng.normal(0, 2), 0, 99)),
                    "disk_io_percent": float(np.clip(5 + rng.normal(0, 1), 0, 99)),
                }
            )
    return pd.DataFrame(rows, columns=USAGE_NAMES)


def make_meta_frame(machines: dict[str, int], status: dict[str, str] | None = None) -> pd.DataFrame:
    status = status or {}
    return pd.DataFrame(
        {
            "machine_id": list(machines),
            "time_stamp": 0,
            "failure_domain_1": 1,
            "failure_domain_2": "a",
            "cpu_num": list(machines.values()),
            "mem_size": 100,
            "status": [status.get(machine, "USING") for machine in machines],
        },
        columns=META_NAMES,
    )


def write_usage_archive(path: Path, usage: pd.DataFrame, *, header: bool = False) -> Path:
    _write_csv_tar(Path(path), usage, "machine_usage.csv", header)
    return Path(path)


def write_meta_archive(path: Path, meta: pd.DataFrame, *, header: bool = False) -> Path:
    _write_csv_tar(Path(path), meta, "machine_meta.csv", header)
    return Path(path)


def make_series(
    n: int = 600,
    *,
    step_seconds: float = 300.0,
    seed: int = 0,
    level_shift_at: float | None = None,
    random_walk: bool = False,
) -> pd.DataFrame:
    """Série agregada sintética: ciclo de 24 h, ruído AR(1) e exógenas correlacionadas com a CPU."""
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n) * step_seconds
    phase = 2 * np.pi * (timestamps % 86400) / 86400
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.5)
    cpu = 40 + 8 * np.sin(phase) + noise
    if random_walk:
        cpu = 50 + np.cumsum(rng.normal(0, 0.4, n))
    if level_shift_at is not None:
        cpu = cpu + np.where(np.arange(n) >= int(level_shift_at * n), -12.0, 0.0)
    cpu = np.clip(cpu, 0, 100)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu": cpu,
            "mem": np.clip(50 + 0.3 * (cpu - 40) + rng.normal(0, 1, n), 0, 100),
            "mem_gps": np.clip(10 + 0.1 * (cpu - 40) + rng.normal(0, 0.5, n), 0, 100),
            "net_in": np.clip(20 + rng.normal(0, 2, n), 0, 100),
            "net_out": np.clip(15 + rng.normal(0, 2, n), 0, 100),
            "disk": np.clip(5 + rng.normal(0, 1, n), 0, 100),
            "coverage": np.ones(n),
            "n_machines": np.full(n, 10),
        }
    )
