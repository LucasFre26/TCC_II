"""Esquema e leitura da configuração da V3.

Chaves desconhecidas são erro. `validation.nested` é opcional: ausente, a seleção é a da V2
(média da validação dos folds de desenvolvimento).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from energy_prediction.models.v3_registry import validate_family_models
from energy_prediction.runtime.device import DEVICE_CHOICES

FREE = "free"  # conteúdo livre, validado por quem consome


class Optional_:
    """Marca uma chave opcional; `schema` descreve o valor quando presente."""

    def __init__(self, schema=None):
        self.schema = schema


class V3ConfigError(ValueError):
    """Chave obrigatória ausente, desconhecida ou valor inválido na configuração da V3."""


XGB_SCHEMA = {"grid": FREE, "fixed": FREE, "n_jobs": None}

SCHEMA = {
    "seed": None,
    "run_root": None,
    "device": None,
    "data": {
        "source": None,
        "usage_archive": None,
        "meta_archive": None,
        "cache_dir": None,
        "chunk_rows": None,
        "window": {"start_seconds": None, "end_seconds": None},
        "machines": {"ids": None, "sample_size": None},
        "granularity": {"sample_rows": None, "min_conformity": None},
        "grid_step": None,
        "synthetic": FREE,
    },
    "aggregation": {"strategy": None, "active_statuses": None},
    "cohort": {"enabled": None, "cpu_num": None, "min_machines": None},
    "features": {
        "groups": FREE,
        "ablation": FREE,
        "distribution": {
            "enabled": None,
            "elevated_cpu_threshold": None,
            "dispersion_change_steps": None,
            "skewness": None,
            "skewness_max_undefined_fraction": None,
        },
    },
    "horizons": None,
    "trajectory_steps": None,
    "validation": {
        "strategy": None,
        "n_folds": None,
        "validation_size": None,
        "test_size": None,
        "min_train_size": None,
        "purge": None,
        "nested": Optional_(
            {
                "enabled": None,
                "validation_size": None,
                "purge": None,
                "min_train_size": None,
                "infeasible_policy": None,
            }
        ),
    },
    "reference": {
        "v2_dir": None,
        "expected_test_range": None,
        "expected_weights": FREE,
    },
    "comparisons": {
        "pairs": None,
        "d_contrast_models": None,
        "bootstrap": {"primary_block_steps": None, "sensitivity_blocks": None, "n_boot": None},
        "peak_quantile": None,
    },
    "families": {
        "residual": {
            "models": None,
            "configs": None,
            "xgboost": XGB_SCHEMA,
            "statespace": {"specifications": None, "maxiter": None},
        },
        "neural": {
            "models": None,
            "budget_hours": None,
            "window_candidates": None,
            "window_selection": {"linear": None, "learning_rate": None},
            "loop": {
                "max_epochs": None,
                "patience": None,
                "grad_clip_norm": None,
                "batch_size": None,
                "mixed_precision": None,
                "finite_check_epochs": None,
            },
            "dlinear": {"linear": None, "learning_rate": None, "moving_average": None},
            "tcn": {"channels": None, "dilations": None, "kernel_size": None, "dropout": None, "learning_rate": None},
            "nbeatsx": {"width": None, "layers": None, "trend_degree": None, "harmonics": None, "learning_rate": None},
            "d_carrier": None,
            "extra_path_configs": None,
            "optional": {
                "enabled": None,
                "order": None,
                "patchtst_min_remaining_hours": None,
                "nhits": FREE,
                "patchtst": FREE,
            },
        },
        "quantile": {
            "quantiles": None,
            "configs": None,
            "xgboost": XGB_SCHEMA,
            "conformal": {"upper_quantile": None, "auxiliary_quantile": None, "min_residuals": None},
        },
        "energy": {
            "substitute": None,
            "exponents": None,
            "unstable_mean_utilization": None,
            "trajectory_models": None,
            "fleet_sum": None,
            "ridge_v2": {"alphas": None},
            "xgboost_v2": XGB_SCHEMA,
        },
    },
}

KNOWN_PAIRS = ("new_vs_persistence_A", "new_vs_v2_xgboost_A", "ablation_vs_A", "d_vs_a", "hybrid_vs_statespace")


def _check_keys(config, schema, path: str = "") -> None:
    if not isinstance(config, dict):
        raise V3ConfigError(f"A chave {path or 'raiz'} deve ser um mapeamento.")
    required = [key for key, sub in schema.items() if not isinstance(sub, Optional_)]
    missing = [key for key in required if key not in config]
    if missing:
        raise V3ConfigError(f"Configuração sem chaves: {[f'{path}{key}' for key in missing]}")
    unknown = [key for key in config if key not in schema]
    if unknown:
        raise V3ConfigError(f"Configuração com chaves desconhecidas: {[f'{path}{key}' for key in unknown]}")
    for key, sub in schema.items():
        if key not in config:
            continue
        if isinstance(sub, Optional_):
            sub = sub.schema
        if isinstance(sub, dict):
            _check_keys(config[key], sub, f"{path}{key}.")


def _check_values(config: dict) -> None:
    if config["device"] not in DEVICE_CHOICES:
        raise V3ConfigError(f"device deve ser um de {list(DEVICE_CHOICES)}; recebido {config['device']!r}.")
    if config["data"]["source"] not in ("alibaba", "synthetic"):
        raise V3ConfigError("data.source deve ser 'alibaba' ou 'synthetic'.")
    nested = config["validation"].get("nested")
    if nested is not None and nested["infeasible_policy"] not in ("fail", "drop_fold"):
        raise V3ConfigError("validation.nested.infeasible_policy deve ser 'fail' ou 'drop_fold'.")
    unknown_pairs = [p for p in config["comparisons"]["pairs"] if p not in KNOWN_PAIRS]
    if unknown_pairs:
        raise V3ConfigError(f"Comparações desconhecidas: {unknown_pairs}. Conhecidas: {list(KNOWN_PAIRS)}.")
    families = config["families"]
    validate_family_models("residual", list(families["residual"]["models"]))
    validate_family_models("neural", list(families["neural"]["models"]))
    validate_family_models("neural_optional", list(families["neural"]["optional"]["order"]))
    validate_family_models("energy", list(families["energy"]["trajectory_models"]))
    carrier = families["neural"]["d_carrier"]
    if carrier not in families["neural"]["models"]:
        raise V3ConfigError(f"neural.d_carrier ({carrier!r}) precisa estar entre os modelos neurais obrigatórios.")
    for model in config["comparisons"]["d_contrast_models"]:
        if model not in ("xgboost_delta", carrier):
            raise V3ConfigError(f"Contraste D só é pré-registrado para xgboost_delta e {carrier}; recebido {model!r}.")
    if sorted(float(r) for r in families["energy"]["exponents"]) != sorted({float(r) for r in families["energy"]["exponents"]}):
        raise V3ConfigError("energy.exponents tem valores repetidos.")


def validate_config(config: dict) -> dict:
    _check_keys(config, SCHEMA)
    _check_values(config)
    return config


def load_v3_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def nested_enabled(config: dict) -> bool:
    nested = config["validation"].get("nested")
    return bool(nested and nested["enabled"])
