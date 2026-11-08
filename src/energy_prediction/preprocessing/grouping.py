"""Agrupamento de máquinas só com dados do treino de cada bloco (H1).

Critérios:
- `mean`: média de CPU no treino, em K grupos de igual contagem (quantis);
- `volatility`: desvio das variações de 5 min no treino, em K grupos de igual contagem;
- `profile`: perfil horário médio (24 valores) padronizado por máquina, KMeans com semente;
  grupos com menos de `min_group_size` máquinas são fundidos ao centróide mais próximo;
- `failure_domain`: `failure_domain_2` do metadado (K natural).
Máquinas sem estatística de treino (cobertura abaixo do mínimo, ausentes do treino ou sem
domínio) formam o grupo residual `residual`. Toda máquina da população pertence a exatamente um
grupo. As séries por grupo usam a regra da agregação principal (média ponderada por `cpu_num`
das máquinas presentes); valores ausentes do grupo ficam ausentes e só os atributos usam o
preenchimento causal pelo último valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.preprocessing.panel import MachinePanel, weighted_mean

RESIDUAL = "residual"
CRITERIA = ("mean", "volatility", "profile", "failure_domain")


class GroupingError(ValueError):
    """Critério desconhecido ou partição incompleta."""


def _quantile_groups(values: pd.Series, k: int) -> pd.Series:
    ranks = values.rank(method="first")
    labels = pd.qcut(ranks, q=int(k), labels=False, duplicates="drop")
    return labels.map(lambda v: f"q{int(v):02d}")


def _profile_groups(stats: pd.DataFrame, k: int, *, min_group_size: int, n_init: int, seed: int) -> pd.Series:
    from sklearn.cluster import KMeans

    cols = [f"profile_h{h:02d}" for h in range(24)]
    profile = stats[cols].to_numpy(dtype=float)
    keep = np.isfinite(profile).all(axis=1)
    centered = profile[keep] - profile[keep].mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    z = centered / np.where(scale < 1e-9, 1.0, scale)
    k = min(int(k), int(keep.sum()))
    model = KMeans(n_clusters=k, n_init=int(n_init), random_state=int(seed)).fit(z)
    labels = model.labels_.copy()
    centers = model.cluster_centers_.copy()
    active = set(range(k))
    while True:
        counts = {c: int((labels == c).sum()) for c in active}
        small = [c for c in sorted(active, key=lambda c: (counts[c], c)) if counts[c] < int(min_group_size)]
        if not small or len(active) == 1:
            break
        victim = small[0]
        active.discard(victim)
        members = labels == victim
        others = sorted(active)
        distance = ((z[members][:, None, :] - centers[others][None, :, :]) ** 2).sum(axis=2)
        labels[members] = np.array(others)[distance.argmin(axis=1)]
        for c in others:
            if (labels == c).any():
                centers[c] = z[labels == c].mean(axis=0)
    ordered = {c: f"p{i:02d}" for i, c in enumerate(sorted(active))}
    out = pd.Series(RESIDUAL, index=stats.index, dtype=object)
    out.loc[stats.index[keep]] = [ordered[c] for c in labels]
    return out


def assign_groups(
    stats: pd.DataFrame,
    criterion: str,
    k: int | None,
    *,
    domains: pd.DataFrame | None = None,
    min_group_size: int = 20,
    n_init: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """machine_id → group para um critério; `stats` vem de `train_statistics` do bloco."""
    if criterion not in CRITERIA:
        raise GroupingError(f"Critério desconhecido: {criterion!r}. Use {list(CRITERIA)}.")
    frame = stats.set_index("machine_id")
    group = pd.Series(RESIDUAL, index=frame.index, dtype=object)
    has = frame["has_statistics"].astype(bool)
    if criterion == "mean":
        group.loc[has] = _quantile_groups(frame.loc[has, "train_mean"], int(k))
    elif criterion == "volatility":
        group.loc[has] = _quantile_groups(frame.loc[has, "train_volatility"], int(k))
    elif criterion == "profile":
        group.loc[has] = _profile_groups(frame.loc[has], int(k), min_group_size=min_group_size, n_init=n_init, seed=seed).to_numpy()
    else:
        if domains is None:
            raise GroupingError("O critério failure_domain exige a tabela de domínios.")
        domain = domains.set_index("machine_id")["failure_domain_2"].reindex(frame.index)
        usable = has & domain.notna()
        group.loc[usable] = "d" + domain.loc[usable].astype(str)
    out = group.rename("group").reset_index()
    if out["machine_id"].duplicated().any():
        raise GroupingError("Máquina em mais de um grupo.")
    return out


def group_series(panel: MachinePanel, weights: np.ndarray, assignment: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Séries (intervalos, grupos) e pesos presentes por grupo, na ordem das máquinas do painel."""
    mapping = assignment.set_index("machine_id")["group"].reindex(panel.machines)
    if mapping.isna().any():
        raise GroupingError(f"{int(mapping.isna().sum())} máquinas do painel sem grupo.")
    labels = sorted(mapping.unique())
    values = np.full((panel.n_bins, len(labels)), np.nan)
    present_weight = np.zeros((panel.n_bins, len(labels)))
    group_of = mapping.to_numpy()
    for j, label in enumerate(labels):
        columns = np.flatnonzero(group_of == label)
        values[:, j], present_weight[:, j] = weighted_mean(panel, weights, columns=columns)
    return labels, values, present_weight
