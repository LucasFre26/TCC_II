"""Esquema, leitura e pré-registro da configuração da V3 Parte 2.

Chaves desconhecidas são erro (mesmo verificador da V3). O modo `full` exige o documento de
pré-registro; os modos `smoke_synthetic` e `smoke_real` não exigem e marcam os números como não
resultado. A família de modelo pré-treinado vem desligada e só pode ser ligada com proveniência
declarada do corpus, sem o Alibaba Cluster Trace 2018.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from energy_prediction.experiments.v3_config import FREE, Optional_, V3ConfigError, _check_keys
from energy_prediction.runtime.device import DEVICE_CHOICES

MODES = ("full", "smoke_synthetic", "smoke_real")
FAMILIES = ("integrity", "baselines", "robustness", "hierarchical", "global_model", "regimes", "peaks", "calibration", "energy", "report")
DEPENDENCIES = {
    "integrity": (),
    "baselines": ("integrity",),
    "robustness": ("integrity", "baselines"),
    "hierarchical": ("integrity", "baselines"),
    "global_model": ("integrity", "baselines"),
    "regimes": ("baselines",),
    "peaks": ("baselines",),
    "calibration": ("integrity",),
    "energy": ("baselines", "hierarchical", "calibration"),
    "report": ("integrity",),
}
OPTIONAL_INPUTS = {"energy": ("global_model",)}
NETWORKS = ("dlinear", "tcn", "patchtst")
CRITERIA = ("mean", "volatility", "profile", "failure_domain")
RECONCILIATIONS = ("bottom_up", "ols", "mint_shrink")
CALIBRATION_METHODS = ("static", "rolling", "aci", "spci")
ENERGY_MODELS = ("persistence", "xgboost_v2", "hybrid", "hierarchical", "global")
FORBIDDEN_CORPUS_TERMS = ("alibaba_cluster_trace_2018", "alibaba cluster trace 2018", "alibaba 2018", "ali2018", "lotsa")


class V3P2ConfigError(V3ConfigError):
    """Configuração inválida da V3 Parte 2."""


XGB = {"grid": FREE, "fixed": FREE, "n_jobs": None}
LOOP = {"max_epochs": None, "patience": None, "grad_clip_norm": None, "batch_size": None, "mixed_precision": None, "finite_check_epochs": None}

SCHEMA = {
    "seed": None,
    "mode": None,
    "run_root": None,
    "device": None,
    "preregistration": None,
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
        "persist_panel": None,
    },
    "aggregation": {"strategy": None, "active_statuses": None},
    "features": {"groups": FREE, "ablation": FREE},
    "horizons": None,
    "trajectory_steps": None,
    "protocol": {
        "steps_per_day": None,
        "development_days": None,
        "inherited_test_day": None,
        "expected_inherited_range": None,
        "purge": None,
        "inner": {"validation_size": None, "purge": None, "min_train_size": None, "infeasible_policy": None},
        "min_favorable_days": None,
    },
    "reference": {"v2_dir": None, "v3_run": None, "reproduction_tolerance": None, "check_hashes": None},
    "statistics": {"n_boot": None, "primary_block_steps": None, "sensitivity_blocks": None, "practical_relevance": None, "alpha": None},
    "panel": {"tolerance_pp": None, "min_train_coverage": None},
    "families": {
        "baselines": {"xgboost": XGB, "statespace": {"specification": None, "maxiter": None}},
        "robustness": {
            "networks": None,
            "primary_network": None,
            "seeds": None,
            "window": None,
            "loop": LOOP,
            "dlinear": {"linear": None, "learning_rate": None, "moving_average": None},
            "tcn": {"channels": None, "dilations": None, "kernel_size": None, "dropout": None, "learning_rate": None},
            "patchtst": FREE,
            "recency_half_lives": None,
        },
        "hierarchical": {"criteria": None, "k_values": None, "reconciliation": None, "min_group_size": None, "profile_n_init": None, "shrinkage": None, "n_jobs": None},
        "global_model": {
            "enabled": None,
            "n_machines": None,
            "origin_stride": None,
            "scale_machines": None,
            "scale_day": None,
            "grid": FREE,
            "n_jobs": None,
            "device": None,
            "tcn": {
                "enabled": None,
                "gate_ratio": None,
                "window": None,
                "channels": None,
                "dilations": None,
                "kernel_size": None,
                "dropout": None,
                "learning_rate": None,
                "batch_size": None,
                "examples_per_epoch": None,
                "max_epochs": None,
                "patience": None,
                "seeds": None,
            },
        },
        "regimes": {
            "slope_steps": None,
            "volatility_steps": None,
            "volatility_quantile": None,
            "rise_quantile": None,
            "fall_quantile": None,
            "min_regime_origins": None,
            "experts": None,
        },
        "peaks": {"lambdas": None, "max_mae_increase": None, "peak_quantile": None, "merge_gap": None, "tolerance": None},
        "calibration": {
            "quantiles": None,
            "methods": None,
            "primary_method": None,
            "rolling_windows": None,
            "aci_gammas": None,
            "window_hours": None,
            "spci": {"enabled": None, "budget_minutes": None, "lags": None, "n_estimators": None},
        },
        "energy": {
            "substitutes": None,
            "exponents": None,
            "models": None,
            "group_k_sensitivity": None,
            "heterogeneous": None,
            "bootstrap_window_steps": None,
        },
    },
    "pretrained": {
        "enabled": None,
        "model": None,
        "version": None,
        "corpus_source": None,
        "excludes_alibaba_2018": None,
        "assessment": Optional_(FREE),
    },
}


def _check_pretrained(cfg: dict) -> None:
    pre = cfg["pretrained"]
    if not pre["enabled"]:
        return
    missing = [k for k in ("model", "version", "corpus_source") if not pre.get(k)]
    if missing or pre.get("excludes_alibaba_2018") is not True:
        raise V3P2ConfigError(
            "pretrained.enabled exige model, version, corpus_source e excludes_alibaba_2018: true; "
            f"ausentes ou inválidos: {missing or ['excludes_alibaba_2018']}."
        )
    corpus = str(pre["corpus_source"]).lower()
    if any(term in corpus for term in FORBIDDEN_CORPUS_TERMS):
        raise V3P2ConfigError("O corpus declarado do modelo pré-treinado contém o Alibaba Cluster Trace 2018; a família não pode rodar.")


def _check_values(cfg: dict) -> None:
    if cfg["mode"] not in MODES:
        raise V3P2ConfigError(f"mode deve ser um de {list(MODES)}; recebido {cfg['mode']!r}.")
    if cfg["device"] not in DEVICE_CHOICES:
        raise V3P2ConfigError(f"device deve ser um de {list(DEVICE_CHOICES)}.")
    if cfg["data"]["source"] not in ("alibaba", "synthetic"):
        raise V3P2ConfigError("data.source deve ser 'alibaba' ou 'synthetic'.")
    if cfg["mode"] == "full":
        if not cfg["preregistration"]:
            raise V3P2ConfigError("O modo completo exige o documento de pré-registro (preregistration).")
        if cfg["data"]["source"] != "alibaba":
            raise V3P2ConfigError("O modo completo exige a fonte alibaba.")
    if cfg["mode"] == "smoke_synthetic" and cfg["data"]["source"] != "synthetic":
        raise V3P2ConfigError("smoke_synthetic exige data.source: synthetic.")
    proto = cfg["protocol"]
    if proto["inner"]["infeasible_policy"] not in ("fail", "drop_block"):
        raise V3P2ConfigError("protocol.inner.infeasible_policy deve ser 'fail' ou 'drop_block'.")
    if not proto["development_days"] or len(set(proto["development_days"])) != len(proto["development_days"]):
        raise V3P2ConfigError("protocol.development_days precisa de dias distintos.")
    if proto["inherited_test_day"] is not None and proto["inherited_test_day"] in proto["development_days"]:
        raise V3P2ConfigError("O dia do teste herdado não pode ser dia de desenvolvimento.")
    fam = cfg["families"]
    rob = fam["robustness"]
    bad = [n for n in rob["networks"] if n not in NETWORKS]
    if bad or rob["primary_network"] not in rob["networks"]:
        raise V3P2ConfigError(f"robustness.networks deve estar em {list(NETWORKS)} e conter primary_network; inválidos: {bad}.")
    if len(rob["seeds"]) < 1:
        raise V3P2ConfigError("robustness.seeds precisa de ao menos uma semente.")
    if cfg["mode"] == "full" and len(rob["seeds"]) < 3:
        raise V3P2ConfigError("O modo completo exige pelo menos 3 sementes nas redes.")
    hier = fam["hierarchical"]
    if [c for c in hier["criteria"] if c not in CRITERIA] or [r for r in hier["reconciliation"] if r not in RECONCILIATIONS]:
        raise V3P2ConfigError(f"hierarchical.criteria ⊂ {list(CRITERIA)} e reconciliation ⊂ {list(RECONCILIATIONS)}.")
    cal = fam["calibration"]
    if [m for m in cal["methods"] if m not in CALIBRATION_METHODS] or cal["primary_method"] not in cal["methods"] or cal["primary_method"] == "static":
        raise V3P2ConfigError("calibration.methods inválido ou primary_method fora da lista (o primário não pode ser o estático).")
    if sorted(cal["quantiles"]) != list(cal["quantiles"]) or 0.5 not in cal["quantiles"]:
        raise V3P2ConfigError("calibration.quantiles deve ser crescente e conter 0,5.")
    if [m for m in fam["energy"]["models"] if m not in ENERGY_MODELS]:
        raise V3P2ConfigError(f"energy.models ⊂ {list(ENERGY_MODELS)}.")
    if 0 not in fam["peaks"]["lambdas"]:
        raise V3P2ConfigError("peaks.lambdas precisa conter 0 (a referência sem peso).")
    _check_pretrained(cfg)


def validate_config(cfg: dict) -> dict:
    try:
        _check_keys(cfg, SCHEMA)
    except V3ConfigError as error:
        raise V3P2ConfigError(str(error)) from error
    _check_values(cfg)
    return cfg


def load_v3p2_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))


def preregistration_hash(root: Path, cfg: dict) -> dict | None:
    """SHA-256 do documento de pré-registro; obrigatório no modo completo."""
    value = cfg.get("preregistration")
    if not value:
        if cfg["mode"] == "full":
            raise V3P2ConfigError("O modo completo exige o documento de pré-registro.")
        return None
    path = Path(value) if Path(value).is_absolute() else Path(root) / value
    if not path.is_file():
        raise V3P2ConfigError(f"Documento de pré-registro ausente: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
