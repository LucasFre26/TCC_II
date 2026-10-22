"""Folds rolling-origin expansivos com purga em todo limite e teste final isolado.

Tudo é medido em passos da grade, sobre o `origin_index` de cada origem (tempo), e não sobre
a posição na tabela. Assim a purga continua em tempo mesmo quando origens incompletas foram
removidas.

    |---------- desenvolvimento ----------|-- purga --|---- teste final ----|
    [treino_k][purga][validação_k]  (k = 0..K-1, blocos de validação contíguos no fim do desenvolvimento)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUPPORTED_STRATEGIES = ("expanding",)


class PurgeTooSmallError(ValueError):
    """A purga configurada é menor que o maior horizonte somado à maior janela."""


class InfeasibleFoldsError(ValueError):
    """A série não comporta a configuração de folds, purga e teste."""


def required_purge_steps(max_horizon_steps: int, max_window_steps: int) -> int:
    return int(max_horizon_steps) + int(max_window_steps)


@dataclass
class Fold:
    index: int
    train: np.ndarray
    validation: np.ndarray
    train_range: tuple[int, int]
    validation_range: tuple[int, int]


@dataclass
class Splits:
    folds: list[Fold]
    development: np.ndarray
    test: np.ndarray
    purge_steps: int
    required_purge_steps: int
    development_range: tuple[int, int]
    test_range: tuple[int, int]
    total_span_steps: int = 0
    notes: list[str] = field(default_factory=list)


def make_splits(
    origin_index: np.ndarray,
    *,
    n_folds: int,
    validation_steps: int,
    test_steps: int,
    min_train_steps: int,
    purge_steps: int,
    max_horizon_steps: int,
    max_window_steps: int,
    strategy: str = "expanding",
) -> Splits:
    """Devolve posições (linhas da tabela) de treino/validação de cada fold, desenvolvimento e teste."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Estratégia de folds não suportada: {strategy!r}. Use {SUPPORTED_STRATEGIES}.")
    required = required_purge_steps(max_horizon_steps, max_window_steps)
    if purge_steps < required:
        raise PurgeTooSmallError(
            f"Purga de {purge_steps} passos é menor que o mínimo {required} "
            f"(maior horizonte {max_horizon_steps} + maior janela {max_window_steps})."
        )
    if n_folds < 1 or validation_steps < 1 or test_steps < 1:
        raise ValueError("n_folds, validation_steps e test_steps devem ser positivos.")
    index = np.asarray(origin_index, dtype=int)
    if len(index) == 0 or np.any(np.diff(index) <= 0):
        raise ValueError("origin_index deve ser estritamente crescente e não vazio.")

    first, last = int(index[0]), int(index[-1])
    span = last - first + 1
    needed = min_train_steps + purge_steps + n_folds * validation_steps + purge_steps + test_steps
    if span < needed:
        raise InfeasibleFoldsError(
            f"Série com {span} passos de origem; a configuração exige pelo menos {needed} "
            f"(treino mínimo {min_train_steps} + purga {purge_steps} + {n_folds} folds x {validation_steps} "
            f"+ purga {purge_steps} + teste {test_steps})."
        )

    test_start = last - test_steps + 1
    dev_end = test_start - purge_steps  # exclusivo
    positions = np.arange(len(index))
    folds: list[Fold] = []
    for k in range(n_folds):
        val_start = dev_end - (n_folds - k) * validation_steps
        val_end = val_start + validation_steps
        train_end = val_start - purge_steps  # exclusivo
        train = positions[index < train_end]
        validation = positions[(index >= val_start) & (index < val_end)]
        if len(train) == 0 or len(validation) == 0:
            raise InfeasibleFoldsError(
                f"Fold {k} ficou sem origens (treino {len(train)}, validação {len(validation)}); "
                "há lacunas demais na série de origens."
            )
        folds.append(
            Fold(
                index=k,
                train=train,
                validation=validation,
                train_range=(first, train_end - 1),
                validation_range=(val_start, val_end - 1),
            )
        )
    development = positions[index < dev_end]
    test = positions[index >= test_start]
    if len(test) == 0:
        raise InfeasibleFoldsError("Bloco de teste final sem origens.")
    return Splits(
        folds=folds,
        development=development,
        test=test,
        purge_steps=int(purge_steps),
        required_purge_steps=required,
        development_range=(first, dev_end - 1),
        test_range=(test_start, last),
        total_span_steps=span,
    )


def _target_stats(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def describe_splits(table: pd.DataFrame, splits: Splits, target: str) -> list[dict]:
    """Limites, contagens e média/desvio do alvo em treino e avaliação, por fold e no teste final."""
    y = table[target].to_numpy(dtype=float)
    rows = []
    for fold in splits.folds:
        train_mean, train_std = _target_stats(y[fold.train])
        eval_mean, eval_std = _target_stats(y[fold.validation])
        rows.append(
            {
                "split": f"fold_{fold.index}",
                "train_range": list(fold.train_range),
                "evaluation_range": list(fold.validation_range),
                "n_train": int(len(fold.train)),
                "n_evaluation": int(len(fold.validation)),
                "train_target_mean": train_mean,
                "train_target_std": train_std,
                "evaluation_target_mean": eval_mean,
                "evaluation_target_std": eval_std,
            }
        )
    train_mean, train_std = _target_stats(y[splits.development])
    eval_mean, eval_std = _target_stats(y[splits.test])
    rows.append(
        {
            "split": "test_final",
            "train_range": list(splits.development_range),
            "evaluation_range": list(splits.test_range),
            "n_train": int(len(splits.development)),
            "n_evaluation": int(len(splits.test)),
            "train_target_mean": train_mean,
            "train_target_std": train_std,
            "evaluation_target_mean": eval_mean,
            "evaluation_target_std": eval_std,
        }
    )
    return rows
