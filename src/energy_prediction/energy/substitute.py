"""Leitura do servidor substituto (resultado publicado do SPECpower_ssj2008).

Sem identificador, fonte, sistema, data, as duas potências e a frase de limitação, a energia não
é calculada. Só os pontos de 0% e 100% de carga são usados.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from energy_prediction.energy.fan import FanModel

REQUIRED = ("result_id", "source", "system", "published", "p_idle_w", "p_busy_w", "limitation")


class SubstituteSourceError(ValueError):
    """Registro do servidor substituto incompleto: a família de energia não roda."""


def load_substitute(path: Path) -> tuple[dict, FanModel]:
    path = Path(path)
    if not path.is_file():
        raise SubstituteSourceError(f"Arquivo do servidor substituto ausente: {path}")
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in REQUIRED if record.get(key) in (None, "")]
    if missing:
        raise SubstituteSourceError(
            f"{path.name} sem {missing}: a energia exige um resultado SPECpower_ssj2008 com referência; "
            "nenhuma energia foi gravada."
        )
    limitation = str(record["limitation"]).lower()
    if "hardware" not in limitation or "carga" not in limitation:
        raise SubstituteSourceError("A frase de limitação precisa dizer que a carga e o hardware não são os do cluster.")
    return dict(record), FanModel(p_idle=float(record["p_idle_w"]), p_busy=float(record["p_busy_w"]))


SET_REQUIRED = ("result_id", "source", "system", "published", "p_idle_w", "p_busy_w", "threads", "role")


def load_substitutes(path: Path) -> tuple[dict, list[dict], dict[str, FanModel]]:
    """Conjunto de substitutos da V3 Parte 2. Qualquer entrada sem identificador ou fonte interrompe a energia."""
    path = Path(path)
    if not path.is_file():
        raise SubstituteSourceError(f"Arquivo de substitutos ausente: {path}")
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = record.get("substitutes") or []
    if not entries:
        raise SubstituteSourceError(f"{path.name} sem substitutos.")
    models = {}
    for entry in entries:
        missing = [key for key in SET_REQUIRED if entry.get(key) in (None, "")]
        if missing:
            raise SubstituteSourceError(f"Substituto {entry.get('result_id')!r} sem {missing}: nenhuma energia foi gravada.")
        models[str(entry["result_id"])] = FanModel(p_idle=float(entry["p_idle_w"]), p_busy=float(entry["p_busy_w"]))
    if record.get("primary") not in models:
        raise SubstituteSourceError("O substituto primário precisa estar no conjunto.")
    limitation = str(record.get("limitation", "")).lower()
    if "hardware" not in limitation or "carga" not in limitation:
        raise SubstituteSourceError("A frase de limitação precisa dizer que a carga e o hardware não são os do cluster.")
    return dict(record), list(entries), models
