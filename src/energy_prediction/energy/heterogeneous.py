"""Cenário sintético de hardware heterogêneo (H6): substituto atribuído por hash do identificador.

A posição de SHA-256(machine_id) módulo o número de substitutos escolhe o servidor de cada
máquina; as proporções resultantes são registradas. O cenário é rotulado como sintético em todas
as saídas e não descreve o hardware do cluster do Alibaba.
"""

from __future__ import annotations

import hashlib

SCENARIO = "heterogeneo_sintetico"


def assign_substitutes(machines: list[str], substitute_ids: list[str]) -> dict[str, str]:
    ordered = list(substitute_ids)
    return {m: ordered[int(hashlib.sha256(str(m).encode("utf-8")).hexdigest(), 16) % len(ordered)] for m in machines}
