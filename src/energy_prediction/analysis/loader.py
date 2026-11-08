"""Leitura dos artefatos da V2 e de uma corrida da V3, com inventário e hashes.

Arquivos ausentes viram aviso; nenhuma tabela vazia é inventada no lugar deles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from energy_prediction.analysis.config import AnalysisPaths

V3_FILES = {
    "integrity": ["population_audit.json", "alignment.json", "folds.json", "nested_splits.json", "series_profile.json", "math_checks.json"],
    "residual": ["metrics.csv", "predictions.csv", "bootstrap_ci.csv", "selections.json", "family_info.json", "error_extremes.csv", "d_contrast_metrics.csv", "d_contrast_predictions.csv"],
    "neural": [
        "metrics.csv", "predictions.csv", "bootstrap_ci.csv", "selections.json", "family_info.json", "error_extremes.csv",
        "d_contrast_metrics.csv", "d_contrast_predictions.csv", "epoch_history.csv", "training_profile.csv",
        "window_selection.json", "optional_architectures.json",
    ],
    "quantile": ["metrics.csv", "predictions.csv", "bootstrap_ci.csv", "selections.json", "calibration.json", "interval_metrics.csv", "peak_metrics.csv"],
    "energy": ["energy_info.json", "energy_summary.csv", "energy_by_origin.csv", "trajectories.csv", "power_propagation.csv", "aggregate_effect.csv"],
}
V2_FILES = ["metrics.csv", "predictions.csv", "bootstrap_ci.csv", "selections.json", "error_extremes.csv", "series_profile.json", "exogenous_lead.json"]


class MissingArtifactError(FileNotFoundError):
    """Um artefato obrigatório da corrida não existe."""


REQUIRED = ["manifest.json", "residual/predictions.csv", "residual/metrics.csv", "neural/predictions.csv", "neural/metrics.csv"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(paths: list[Path]) -> dict[str, str]:
    """SHA-256 de todo arquivo sob cada caminho (pasta ou arquivo), por caminho absoluto."""
    out = {}
    for root in paths:
        if root.is_file():
            out[str(root)] = sha256(root)
        elif root.is_dir():
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                out[str(path)] = sha256(path)
    return out


@dataclass
class RunArtifacts:
    paths: AnalysisPaths
    manifest: dict
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    documents: dict[str, object] = field(default_factory=dict)
    inventory: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    def frame(self, key: str) -> pd.DataFrame | None:
        return self.frames.get(key)

    def doc(self, key: str):
        return self.documents.get(key)

    def require(self, key: str) -> pd.DataFrame:
        frame = self.frames.get(key)
        if frame is None:
            raise MissingArtifactError(f"Artefato obrigatório ausente: {key}.")
        return frame


def _read(path: Path):
    if path.suffix == ".csv":
        return pd.read_csv(path) if path.stat().st_size > 1 else None
    return json.loads(path.read_text(encoding="utf-8"))


def _register(art: RunArtifacts, key: str, path: Path, origin: str) -> None:
    if not path.is_file():
        art.warnings.append(f"Arquivo ausente: {path} ({origin}).")
        return
    value = _read(path)
    entry = {"key": key, "origin": origin, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if isinstance(value, pd.DataFrame):
        entry["rows"] = int(len(value))
        art.frames[key] = value
    elif value is None:
        art.warnings.append(f"CSV vazio ignorado: {path}.")
    else:
        art.documents[key] = value
    art.inventory.append(entry)


def load_run(paths: AnalysisPaths) -> RunArtifacts:
    for relative in REQUIRED:
        if not (paths.run_dir / relative).is_file():
            raise MissingArtifactError(f"Artefato obrigatório ausente: {paths.run_dir / relative}.")
    manifest = json.loads((paths.run_dir / "manifest.json").read_text(encoding="utf-8"))
    art = RunArtifacts(paths=paths, manifest=manifest)
    art.inventory.append({"key": "manifest", "origin": "v3", "path": str(paths.run_dir / "manifest.json"), "bytes": (paths.run_dir / "manifest.json").stat().st_size, "sha256": sha256(paths.run_dir / "manifest.json")})
    for family, names in V3_FILES.items():
        for name in names:
            key = f"{family}/{Path(name).stem}"
            _register(art, key, paths.run_dir / family / name, "v3")
    for population in ("main", "cohort"):
        for name in V2_FILES:
            _register(art, f"v2_{population}/{Path(name).stem}", paths.v2_dir / population / name, "v2")
    _register(art, "v2/run_summary", paths.v2_dir / "run_summary.json", "v2")
    _register(art, "v2/dataset_profile", paths.v2_dir / "dataset_profile.json", "v2")
    checkpoints = paths.run_dir / "neural" / "checkpoints"
    if checkpoints.is_dir():
        art.documents["neural/checkpoints"] = sorted(p.name for p in checkpoints.glob("*.pt"))
    return art
