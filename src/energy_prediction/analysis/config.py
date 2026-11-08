"""Leitura e validação de config/v3-analysis.yaml e guarda das pastas protegidas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED = {
    "run_dir": None,
    "v2_dir": None,
    "output_subdir": None,
    "substitute": None,
    "dataset": ["enabled"],
    "tolerances": ["metric", "bootstrap", "energy_j", "refit_pp", "series_pp"],
    "models": ["order", "labels", "families", "highlight", "best_neural_rule", "neural"],
    "diagnostics": ["window_steps", "change_steps", "acf_max_lag", "cpu_bands", "recent_change_bins"],
    "peaks": ["quantiles", "merge_gap_steps", "tolerance_steps"],
    "statistics": ["n_boot", "seed", "blocks", "primary_block", "practical_relative_threshold", "exploratory_config"],
    "energy": ["exponents", "reference_exponent", "sensitivity"],
    "computational": ["timing_repeats"],
    "figures": ["dpi", "formats", "dark_variant"],
}

# Pastas da corrida que a análise nunca altera, além dos resultados da V1 e da V2.
RUN_FAMILIES = ("integrity", "residual", "neural", "quantile", "energy", "report")
PROTECTED_RESULTS = ("results/v1", "results/v2", "results/v2-full")


class AnalysisConfigError(ValueError):
    """Chave obrigatória ausente ou valor inválido em v3-analysis.yaml."""


class ProtectedPathError(ValueError):
    """A saída da análise coincide com um resultado preservado ou fica dentro dele."""


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    run_dir: Path
    v2_dir: Path
    output_dir: Path
    substitute: Path

    def protected(self) -> list[Path]:
        paths = [(self.project_root / p).resolve() for p in PROTECTED_RESULTS]
        paths += [(self.run_dir / family).resolve() for family in RUN_FAMILIES]
        paths.append((self.run_dir / "manifest.json").resolve())
        return paths


def validate(cfg: dict) -> dict:
    missing = []
    for key, children in REQUIRED.items():
        if key not in cfg:
            missing.append(key)
            continue
        for child in children or []:
            if not isinstance(cfg[key], dict) or child not in cfg[key]:
                missing.append(f"{key}.{child}")
    if missing:
        raise AnalysisConfigError(f"Chaves obrigatórias ausentes em v3-analysis.yaml: {', '.join(missing)}.")
    stats = cfg["statistics"]
    if int(stats["primary_block"]) not in [int(b) for b in stats["blocks"]]:
        raise AnalysisConfigError("statistics.primary_block precisa estar em statistics.blocks.")
    if float(cfg["energy"]["reference_exponent"]) not in [float(r) for r in cfg["energy"]["exponents"]]:
        raise AnalysisConfigError("energy.reference_exponent precisa estar em energy.exponents.")
    if not set(cfg["figures"]["formats"]) <= {"png", "svg", "pdf"}:
        raise AnalysisConfigError("figures.formats aceita só png, svg e pdf.")
    return cfg


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return validate(yaml.safe_load(handle))


def guard_output(paths: AnalysisPaths) -> None:
    """Falha antes de qualquer escrita se a saída tocar uma pasta protegida."""
    target = paths.output_dir.resolve()
    for protected in paths.protected():
        if target == protected or protected in target.parents or target in protected.parents:
            raise ProtectedPathError(f"A saída {target} coincide com {protected}, contém ou está dentro dessa pasta preservada.")


def resolve_paths(cfg: dict, project_root: Path) -> AnalysisPaths:
    root = Path(project_root).resolve()

    def full(value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    run_dir = full(cfg["run_dir"])
    paths = AnalysisPaths(
        project_root=root,
        run_dir=run_dir,
        v2_dir=full(cfg["v2_dir"]),
        output_dir=run_dir / cfg["output_subdir"],
        substitute=full(cfg["substitute"]),
    )
    guard_output(paths)
    return paths
