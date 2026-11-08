"""Janelas causais das redes, só com NumPy (não importa PyTorch).

A janela da origem t são os passos t-W+1..t da série agregada; nenhum valor posterior a t entra.
Lacunas são preenchidas por repetição do último valor observado (só passado). Origem cuja janela
ainda tem ausência (início da série) é inelegível: sai do ajuste daquele modelo, com contagem,
e nunca sai da avaliação, que fica longe do início.
Canais por configuração: as colunas dos grupos da configuração (CPU primeiro) e o calendário.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from energy_prediction.features.cpu_features import SECONDS_PER_DAY, TARGET_SERIES, GroupSpec


def config_channels(specs: dict[str, GroupSpec], groups: list[str]) -> list[str]:
    channels: list[str] = []
    for group in groups:
        for column in specs[group].columns:
            if column not in channels:
                channels.append(column)
    if not channels or channels[0] != TARGET_SERIES:
        raise ValueError("A CPU precisa ser o primeiro canal da janela.")
    return channels + ["sin_day", "cos_day"]


def channel_matrix(series: pd.DataFrame, channels: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Matriz (n, C) com preenchimento causal e máscara de validade por instante."""
    frame = series.copy()
    phase = 2.0 * math.pi * (frame["timestamp"].to_numpy(dtype=float) % SECONDS_PER_DAY) / SECONDS_PER_DAY
    frame["sin_day"] = np.sin(phase)
    frame["cos_day"] = np.cos(phase)
    raw = frame[channels].astype(float)
    filled = raw.ffill()
    valid = filled.notna().all(axis=1).to_numpy()
    info = {
        "channels": list(channels),
        "forward_filled_values": {c: int(raw[c].isna().sum() - filled[c].isna().sum()) for c in channels},
        "leading_invalid_steps": int((~valid).sum()),
    }
    return filled.to_numpy(dtype=float), valid, info


def window_tensor(matrix: np.ndarray, origins: np.ndarray, window: int) -> np.ndarray:
    """(n_origens, C, W) com os passos t-W+1..t de cada origem; exige origens elegíveis."""
    origins = np.asarray(origins, dtype=int)
    if np.any(origins - window + 1 < 0):
        raise ValueError("Origem sem histórico suficiente para a janela.")
    offsets = np.arange(-int(window) + 1, 1)
    index = origins[:, None] + offsets[None, :]
    return np.transpose(matrix[index], (0, 2, 1)).copy()


def receptive_field(kernel_size: int, dilations: list[int]) -> int:
    """Campo receptivo de uma pilha com uma convolução causal por dilatação."""
    return 1 + (int(kernel_size) - 1) * int(sum(int(d) for d in dilations))
