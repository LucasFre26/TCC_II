"""Painel denso intervalo × máquina a partir do cache de grade (V3 Parte 2).

Uma matriz float32 por variável, com forma (intervalos, máquinas), e uma máscara de presença.
Intervalo sem registro fica ausente (NaN) e nunca é preenchido aqui. Com fonte Alibaba, o
painel é gravado em `<cache_dir>/panel_<chave>/` (fora do git) e reaproveitado quando a chave
coincide; a chave combina a chave do cache de grade e a população. A média ponderada do painel
precisa reproduzir a série agregada usada pela V3 antes de qualquer família usá-lo.
"""

from __future__ import annotations

import hashlib
import json
import resource
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_VARIABLES = ("cpu", "mem", "net_in", "net_out", "disk")


class PanelCheckError(ValueError):
    """A média ponderada do painel não reproduz a série agregada."""


@dataclass
class MachinePanel:
    machines: list[str]
    bins: np.ndarray
    values: dict[str, np.ndarray]
    key: str
    info: dict

    @property
    def n_bins(self) -> int:
        return int(len(self.bins))

    @property
    def n_machines(self) -> int:
        return int(len(self.machines))

    def present(self, variable: str = "cpu") -> np.ndarray:
        return np.isfinite(self.values[variable])

    def column_of(self) -> dict[str, int]:
        return {m: j for j, m in enumerate(self.machines)}

    def subset(self, machines: list[str]) -> "MachinePanel":
        index = self.column_of()
        cols = [index[m] for m in machines]
        return MachinePanel(
            machines=list(machines),
            bins=self.bins,
            values={k: v[:, cols] for k, v in self.values.items()},
            key=f"{self.key}-subset{len(cols)}",
            info={**self.info, "subset_of": self.key},
        )


def panel_key(source_key: str, population: list[str]) -> str:
    digest = hashlib.sha256(("\n".join(sorted(population))).encode("utf-8")).hexdigest()[:12]
    return f"{source_key}-{digest}"


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def build_panel(grid: pd.DataFrame, population: list[str], bins: np.ndarray, *, key: str) -> MachinePanel:
    machines = sorted(str(m) for m in population)
    bins = np.asarray(bins, dtype=int)
    frame = grid.loc[grid["machine_id"].isin(set(machines))]
    col = pd.Categorical(frame["machine_id"].astype(str), categories=machines).codes
    row = frame["bin"].to_numpy(dtype=int) - int(bins[0])
    inside = (row >= 0) & (row < len(bins)) & (col >= 0)
    values = {}
    for name in PANEL_VARIABLES:
        matrix = np.full((len(bins), len(machines)), np.nan, dtype=np.float32)
        data = frame[name].to_numpy(dtype=float)
        matrix[row[inside], col[inside]] = data[inside]
        values[name] = matrix
    info = {
        "key": key,
        "shape": [int(len(bins)), int(len(machines))],
        "first_bin": int(bins[0]),
        "variables": list(PANEL_VARIABLES),
        "bytes": int(sum(v.nbytes for v in values.values())),
        "present_cells_cpu": int(np.isfinite(values["cpu"]).sum()),
        "grid_rows_used": int(inside.sum()),
        "peak_rss_mb_after_build": _peak_rss_mb(),
    }
    return MachinePanel(machines=machines, bins=bins, values=values, key=key, info=info)


def save_panel(panel: MachinePanel, directory: Path) -> None:
    directory = Path(directory)
    temporary = directory.with_name(f".{directory.name}.tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    for name, matrix in panel.values.items():
        np.save(temporary / f"{name}.npy", matrix)
    np.save(temporary / "bins.npy", panel.bins)
    (temporary / "machines.json").write_text(json.dumps(panel.machines), encoding="utf-8")
    (temporary / "panel.json").write_text(json.dumps(panel.info, ensure_ascii=False, indent=2), encoding="utf-8")
    if directory.exists():
        import shutil

        shutil.rmtree(directory)
    temporary.rename(directory)


def load_panel(directory: Path) -> MachinePanel:
    directory = Path(directory)
    info = json.loads((directory / "panel.json").read_text(encoding="utf-8"))
    values = {name: np.load(directory / f"{name}.npy") for name in info["variables"]}
    machines = json.loads((directory / "machines.json").read_text(encoding="utf-8"))
    return MachinePanel(machines=machines, bins=np.load(directory / "bins.npy"), values=values, key=info["key"], info=info)


def load_or_build_panel(grid: pd.DataFrame, population: list[str], bins: np.ndarray, *, source_key: str, cache_dir: Path | None) -> tuple[MachinePanel, dict]:
    """Reaproveita o painel gravado com a mesma chave; senão constrói (e grava, se houver `cache_dir`)."""
    key = panel_key(source_key, population)
    record = {"key": key, "source_key": source_key, "rebuilt": False, "persisted": cache_dir is not None}
    if cache_dir is not None:
        directory = Path(cache_dir) / f"panel_{key}"
        if (directory / "panel.json").is_file():
            panel = load_panel(directory)
            if panel.key == key and list(panel.info["shape"]) == [len(bins), len(population)]:
                record["path"] = str(directory)
                return panel, record
        stale = sorted(p.name for p in Path(cache_dir).glob("panel_*") if p.name != f"panel_{key}")
        record["other_panels_present"] = stale
    panel = build_panel(grid, population, bins, key=key)
    record["rebuilt"] = True
    if cache_dir is not None:
        directory = Path(cache_dir) / f"panel_{key}"
        save_panel(panel, directory)
        record["path"] = str(directory)
    return panel, record


def weighted_mean(panel: MachinePanel, weights: np.ndarray, variable: str = "cpu", columns: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Média ponderada por intervalo das máquinas presentes e o peso presente (regra da agregação principal)."""
    matrix = panel.values[variable] if columns is None else panel.values[variable][:, columns]
    w = np.asarray(weights, dtype=float) if columns is None else np.asarray(weights, dtype=float)[columns]
    present = np.isfinite(matrix)
    weight = (present * w[None, :]).sum(axis=1)
    numerator = np.where(present, matrix, 0.0).astype(float) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weight > 0, numerator / np.where(weight > 0, weight, 1.0), np.nan)
    return mean, weight


def check_against_series(panel: MachinePanel, weights: np.ndarray, series: pd.DataFrame, *, tolerance: float) -> dict:
    """Confere a média ponderada do painel contra `series['cpu']` em todos os intervalos."""
    mean, _ = weighted_mean(panel, weights)
    reference = series.set_index("bin")["cpu"].reindex(panel.bins).to_numpy(dtype=float)
    both = np.isfinite(mean) & np.isfinite(reference)
    mismatch_missing = int((np.isfinite(mean) != np.isfinite(reference)).sum())
    max_diff = float(np.max(np.abs(mean[both] - reference[both]))) if both.any() else 0.0
    report = {
        "tolerance_pp": float(tolerance),
        "max_abs_diff_pp": max_diff,
        "bins_compared": int(both.sum()),
        "bins_missing_mismatch": mismatch_missing,
        "passed": bool(max_diff <= float(tolerance) and mismatch_missing == 0),
    }
    if not report["passed"]:
        raise PanelCheckError(f"Painel não reproduz a série agregada: {report}")
    return report


def train_statistics(panel: MachinePanel, *, last_bin: int, step_seconds: float, min_coverage: float) -> pd.DataFrame:
    """Estatísticas por máquina só com intervalos até `last_bin` (inclusive).

    Cobertura, média, desvio, volatilidade (desvio das diferenças de um passo) e perfil horário
    médio (24 valores). Máquina com cobertura abaixo de `min_coverage` fica sem estatística.
    """
    stop = int(np.searchsorted(panel.bins, int(last_bin), side="right"))
    cpu = panel.values["cpu"][:stop].astype(float)
    coverage = np.isfinite(cpu).mean(axis=0) if stop else np.zeros(panel.n_machines)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(cpu, axis=0)
        std = np.nanstd(cpu, axis=0)
        volatility = np.nanstd(np.diff(cpu, axis=0), axis=0)
        hours = ((panel.bins[:stop] * float(step_seconds)) % 86400.0 // 3600.0).astype(int)
        profile = np.vstack([np.nanmean(cpu[hours == h], axis=0) if (hours == h).any() else np.full(panel.n_machines, np.nan) for h in range(24)]).T
    has = coverage >= float(min_coverage)
    frame = pd.DataFrame(
        {
            "machine_id": panel.machines,
            "train_coverage": coverage,
            "train_mean": np.where(has, mean, np.nan),
            "train_std": np.where(has, std, np.nan),
            "train_volatility": np.where(has, volatility, np.nan),
            "has_statistics": has,
        }
    )
    for h in range(24):
        frame[f"profile_h{h:02d}"] = np.where(has, profile[:, h], np.nan)
    return frame
