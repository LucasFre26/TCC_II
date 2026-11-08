"""Conjunto fechado de modelos da V3, por família.

Separado de `models.registry.ALLOWED_MODELS`: a V2 continua aceitando só persistência, Ridge e
XGBoost, e nenhum nome abaixo passa a ser válido nela.
"""

from __future__ import annotations

FAMILY_MODELS: dict[str, tuple[str, ...]] = {
    "integrity": (),
    "residual": ("persistence", "xgboost_delta", "statespace", "hybrid"),
    "neural": ("dlinear", "tcn", "nbeatsx"),
    "neural_optional": ("nhits", "patchtst"),
    "quantile": ("xgboost_quantile",),
    "energy": (
        "persistence",
        "statespace",
        "ridge_v2",
        "xgboost_v2",
        "xgboost_delta",
        "hybrid",
        "dlinear",
        "tcn",
        "nbeatsx",
    ),
}
FAMILIES = ("integrity", "residual", "neural", "quantile", "energy")


class UnknownV3ModelError(ValueError):
    """Nome de modelo fora do conjunto permitido da família."""


def validate_family_models(family: str, names: list[str]) -> list[str]:
    if family not in FAMILY_MODELS:
        raise UnknownV3ModelError(f"Família desconhecida: {family!r}. Famílias: {list(FAMILY_MODELS)}.")
    allowed = FAMILY_MODELS[family]
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise UnknownV3ModelError(
            f"Modelo(s) fora do conjunto permitido da família {family!r}: {unknown}. Permitidos: {list(allowed)}."
        )
    if len(set(names)) != len(names):
        raise UnknownV3ModelError(f"Modelos repetidos na família {family!r}: {names}.")
    return list(names)
