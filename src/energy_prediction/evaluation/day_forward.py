"""Protocolo cronológico por dia da V3 Parte 2.

O bloco do dia d contém as origens 288·d − H .. 288·d + 287 − H (H = maior horizonte em passos),
de modo que todos os alvos do bloco caem no dia d. Para cada bloco:

    |--- treino interno ---|- purga -|- validação interna -|- purga -|==== bloco do dia d ====|
    |<----------------------- treino do bloco ------------->|

Hiperparâmetros são escolhidos só na validação interna; o modelo é reajustado no treino do
bloco e prevê o bloco uma vez. O dia 7 é o teste herdado da V3 (origens 2010–2297), já
inspecionado na V2, na V3 e na análise expandida; ele nunca é confirmação independente.
Tudo é medido em passos de origem (`origin_index`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from energy_prediction.evaluation.nested import InfeasibleInnerSplitError, InnerSplit, inner_split, select_best

DEVELOPMENT = "development"
INHERITED = "inherited_test"
INHERITED_INSPECTION = 4  # V2, V3, análise expandida da V3 e agora a V3 Parte 2


class BlockAlignmentError(ValueError):
    """O bloco do teste herdado não coincide com o teste registrado da V3."""


@dataclass
class DayBlock:
    day: int
    role: str  # development | inherited_test
    evaluation: np.ndarray  # posições na tabela
    evaluation_range: tuple[int, int]  # origens
    train: np.ndarray
    train_range: tuple[int, int]
    inner: InnerSplit
    purge_steps: int
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"day_{self.day}"

    @property
    def inspection(self) -> int | None:
        return INHERITED_INSPECTION if self.role == INHERITED else None

    def describe(self) -> dict:
        return {
            "block": self.label,
            "day": int(self.day),
            "role": self.role,
            "inspection_number": self.inspection,
            "evaluation_range": list(self.evaluation_range),
            "n_evaluation": int(len(self.evaluation)),
            "train_range": list(self.train_range),
            "n_train": int(len(self.train)),
            "purge_steps": int(self.purge_steps),
            **{f"inner_{k}": v for k, v in self.inner.describe().items() if k != "label"},
        }


def block_origin_range(day: int, steps_per_day: int, max_horizon: int) -> tuple[int, int]:
    start = int(day) * int(steps_per_day) - int(max_horizon)
    return start, start + int(steps_per_day) - 1


def make_day_blocks(
    origin_index: np.ndarray,
    *,
    development_days: list[int],
    inherited_day: int | None,
    steps_per_day: int,
    max_horizon: int,
    purge_steps: int,
    inner_validation_steps: int,
    inner_purge_steps: int,
    min_inner_train_steps: int,
    infeasible_policy: str = "fail",
    expected_inherited_range: list[int] | None = None,
    eligible: np.ndarray | None = None,
) -> tuple[list[DayBlock], list[dict]]:
    """Blocos de desenvolvimento e, se pedido, o teste herdado. Devolve também os blocos descartados."""
    index = np.asarray(origin_index, dtype=int)
    positions = np.arange(len(index))
    usable = np.ones(len(index), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    plan = [(int(d), DEVELOPMENT) for d in development_days]
    if inherited_day is not None:
        plan.append((int(inherited_day), INHERITED))
    blocks, dropped = [], []
    for day, role in plan:
        low, high = block_origin_range(day, steps_per_day, max_horizon)
        evaluation = positions[(index >= low) & (index <= high)]
        if len(evaluation) == 0:
            raise InfeasibleInnerSplitError(f"day_{day}: bloco sem origens no intervalo {low}–{high}.")
        if role == INHERITED and expected_inherited_range is not None:
            got = [int(index[evaluation].min()), int(index[evaluation].max())]
            if got != [int(v) for v in expected_inherited_range] or len(evaluation) != got[1] - got[0] + 1:
                raise BlockAlignmentError(
                    f"Teste herdado em {got} com {len(evaluation)} origens; esperado {list(expected_inherited_range)} contíguo."
                )
        train_end = low - int(purge_steps)  # exclusivo
        train = positions[(index < train_end) & usable]
        try:
            inner = inner_split(
                index,
                train,
                train_end,
                validation_steps=inner_validation_steps,
                purge_steps=inner_purge_steps,
                min_train_steps=min_inner_train_steps,
                label=f"day_{day}",
            )
        except InfeasibleInnerSplitError as error:
            if infeasible_policy != "drop_block" or role == INHERITED:
                raise
            dropped.append({"block": f"day_{day}", "reason": str(error)})
            continue
        blocks.append(
            DayBlock(
                day=day,
                role=role,
                evaluation=evaluation,
                evaluation_range=(int(index[evaluation].min()), int(index[evaluation].max())),
                train=train,
                train_range=(int(index[train].min()), int(index[train].max())),
                inner=inner,
                purge_steps=int(purge_steps),
            )
        )
    if not [b for b in blocks if b.role == DEVELOPMENT]:
        raise InfeasibleInnerSplitError("Nenhum bloco de desenvolvimento comporta a divisão configurada.")
    return blocks, dropped


def restrict_block(block: DayBlock, eligible: np.ndarray) -> DayBlock:
    """O mesmo bloco com treino e treino interno restritos a origens elegíveis (ex.: janela da rede).

    As origens avaliadas não mudam; se alguma não for elegível, a função falha.
    """
    mask = np.asarray(eligible, dtype=bool)
    if not mask[block.evaluation].all():
        raise BlockAlignmentError(f"{block.label}: origem avaliada sem histórico para o modelo.")
    inner = block.inner
    restricted = InnerSplit(
        label=inner.label,
        train=inner.train[mask[inner.train]],
        validation=inner.validation[mask[inner.validation]],
        train_range=inner.train_range,
        validation_range=inner.validation_range,
        available_train_steps=inner.available_train_steps,
        required_train_steps=inner.required_train_steps,
    )
    return DayBlock(
        day=block.day,
        role=block.role,
        evaluation=block.evaluation,
        evaluation_range=block.evaluation_range,
        train=block.train[mask[block.train]],
        train_range=block.train_range,
        inner=restricted,
        purge_steps=block.purge_steps,
        notes=list(block.notes),
    )


@dataclass
class BlockResult:
    best: int
    grid_scores: list[float]
    inner_prediction: np.ndarray
    block_prediction: np.ndarray
    best_epoch: int | None
    fit_info: list[dict] = field(default_factory=list)


FitPredict = Callable[..., tuple[np.ndarray, dict]]


def evaluate_block(n_grid: int, fit_predict: FitPredict, y: np.ndarray, block: DayBlock, *, metric) -> BlockResult:
    """Seleção na validação interna do bloco, reajuste no treino do bloco e previsão única do bloco.

    `fit_predict(g, train_positions, predict_positions, early_stop_positions=None, epochs=None)` segue
    a assinatura da V3: redes usam a validação para parar e o número de épocas no reajuste.
    """
    y = np.asarray(y, dtype=float)
    scores, cache, epochs, log = [], {}, {}, []
    for g in range(int(n_grid)):
        pred, info = fit_predict(g, block.inner.train, block.inner.validation, early_stop_positions=block.inner.validation, epochs=None)
        scores.append(float(metric(y[block.inner.validation], pred)))
        cache[g] = np.asarray(pred, dtype=float)
        epochs[g] = info.get("best_epoch")
        log.append({"phase": "selection", "grid": g, "block": block.label, **info})
    best = select_best(scores)
    pred, info = fit_predict(best, block.train, block.evaluation, early_stop_positions=None, epochs=epochs[best])
    log.append({"phase": "block_refit", "grid": best, "block": block.label, **info})
    return BlockResult(
        best=best,
        grid_scores=scores,
        inner_prediction=cache[best],
        block_prediction=np.asarray(pred, dtype=float),
        best_epoch=epochs[best],
        fit_info=log,
    )
