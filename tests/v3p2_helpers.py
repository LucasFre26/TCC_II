"""Apoio dos testes da V3 Parte 2: configuração do smoke sintético e projeto temporário."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def v3p2_config(**overrides) -> dict:
    config = copy.deepcopy(yaml.safe_load((ROOT / "config/v3p2-smoke.yaml").read_text(encoding="utf-8")))
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config


def fast_config(**overrides) -> dict:
    """Smoke sintético reduzido para a suíte: uma rede, uma semente, poucos grupos."""
    base = {
        "families.robustness.networks": ["dlinear"],
        "families.robustness.seeds": [42],
        "families.robustness.recency_half_lives": [144],
        "families.hierarchical.criteria": ["mean", "failure_domain"],
        "families.hierarchical.k_values": [2],
        "families.energy.group_k_sensitivity": [2],
        "families.global_model.tcn.enabled": False,
        "families.calibration.spci.enabled": False,
        "statistics.n_boot": 50,
    }
    base.update(overrides)
    return v3p2_config(**base)


def write_project(tmp_path: Path, config: dict, *, preregistration: bool = False) -> Path:
    project = tmp_path / "proj"
    (project / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "config/specpower_substitutes.yaml", project / "config/specpower_substitutes.yaml")
    if preregistration:
        (project / "docs").mkdir(exist_ok=True)
        shutil.copy(ROOT / "docs/v3p2-preregistro.md", project / "docs/v3p2-preregistro.md")
    path = project / "config/v3p2.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path
