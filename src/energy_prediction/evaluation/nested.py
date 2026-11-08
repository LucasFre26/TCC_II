"""Seleção aninhada da V3, sem alterar `make_splits` da V2.

Dentro do treino externo de cada fold (e da região de desenvolvimento):

    |------ treino interno ------|-- purga --|-- validação interna --|  (fim do treino externo)

O ponto da grade de menor média da métrica interna entre os folds vence (empate: primeiro da
grade). Depois, o modelo é reajustado no treino externo de cada fold e pontuado uma vez na
validação externa, e reajustado na região de desenvolvimento e pontuado uma vez no teste.
Com a seleção aninhada desligada, vale a regra da V2: média da validação externa dos folds.
Tudo é medido em passos de origem (`origin_index`), não em posição na tabela.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from energy_prediction.validation.rolling_origin import Fold, Splits

TIE_TOLERANCE = 1e-12
DEVELOPMENT = "development"


class InfeasibleInnerSplitError(ValueError):
    """O treino externo não comporta treino interno, purga e validação interna."""


class AlignmentError(ValueError):
    """As origens de avaliação da V3 não coincidem com as da análise principal preservada da V2."""


@dataclass
class InnerSplit:
    label: str
    train: np.ndarray
    validation: np.ndarray
    train_range: tuple[int, int]
    validation_range: tuple[int, int]
    available_train_steps: int
    required_train_steps: int

    def describe(self) -> dict:
        return {
            "label": self.label,
            "inner_train_range": list(self.train_range),
            "inner_validation_range": list(self.validation_range),
            "n_inner_train": int(len(self.train)),
            "n_inner_validation": int(len(self.validation)),
            "available_train_steps": int(self.available_train_steps),
            "required_train_steps": int(self.required_train_steps),
        }


def inner_split(
    origin_index: np.ndarray,
    region_positions: np.ndarray,
    region_end: int,
    *,
    validation_steps: int,
    purge_steps: int,
    min_train_steps: int,
    label: str,
    eligible: np.ndarray | None = None,
) -> InnerSplit:
    """Divide uma região de treino (origens < `region_end`) em treino interno, purga e validação interna.

    `eligible` (máscara sobre as linhas da tabela) tira do ajuste as origens sem histórico suficiente
    para um modelo; o tamanho disponível é contado só a partir da primeira origem elegível.
    """
    index = np.asarray(origin_index, dtype=int)
    region = np.asarray(region_positions, dtype=int)
    if eligible is not None:
        region = region[np.asarray(eligible, dtype=bool)[region]]
    validation_start = int(region_end) - int(validation_steps)
    train_end = validation_start - int(purge_steps)  # exclusivo
    origins = index[region]
    train = region[origins < train_end]
    validation = region[(origins >= validation_start) & (origins < int(region_end))]
    first = int(origins.min()) if len(origins) else int(region_end)
    available = max(train_end - first, 0)
    if available < min_train_steps or len(train) == 0 or len(validation) == 0:
        raise InfeasibleInnerSplitError(
            f"{label}: treino interno com {available} passos de origem disponíveis; exigidos {min_train_steps} "
            f"(região até a origem {int(region_end) - 1}, validação interna de {validation_steps} passos, "
            f"purga de {purge_steps} passos)."
        )
    return InnerSplit(
        label=label,
        train=train,
        validation=validation,
        train_range=(first, train_end - 1),
        validation_range=(validation_start, int(region_end) - 1),
        available_train_steps=available,
        required_train_steps=int(min_train_steps),
    )


def build_inner_splits(
    origin_index: np.ndarray,
    splits: Splits,
    *,
    validation_steps: int,
    purge_steps: int,
    min_train_steps: int,
    infeasible_policy: str = "fail",
    eligible: np.ndarray | None = None,
) -> tuple[dict[str, InnerSplit], list[dict]]:
    """Divisão interna de cada fold externo. `drop_fold` só vale se declarado antes da corrida."""
    inner: dict[str, InnerSplit] = {}
    dropped: list[dict] = []
    for fold in splits.folds:
        label = f"fold_{fold.index}"
        try:
            inner[label] = inner_split(
                origin_index,
                fold.train,
                fold.train_range[1] + 1,
                validation_steps=validation_steps,
                purge_steps=purge_steps,
                min_train_steps=min_train_steps,
                label=label,
                eligible=eligible,
            )
        except InfeasibleInnerSplitError as error:
            if infeasible_policy != "drop_fold":
                raise
            dropped.append({"fold": label, "reason": str(error)})
    if not inner:
        raise InfeasibleInnerSplitError("Nenhum fold externo comporta a divisão interna configurada.")
    return inner, dropped


def splits_on_subset(origin_index: np.ndarray, template: Splits) -> Splits:
    """Os mesmos limites de tempo de `template` aplicados a um subconjunto de origens (ex.: configuração D)."""
    index = np.asarray(origin_index, dtype=int)
    positions = np.arange(len(index))
    folds = []
    for fold in template.folds:
        train = positions[index <= fold.train_range[1]]
        validation = positions[(index >= fold.validation_range[0]) & (index <= fold.validation_range[1])]
        folds.append(Fold(fold.index, train, validation, fold.train_range, fold.validation_range))
    return Splits(
        folds=folds,
        development=positions[index <= template.development_range[1]],
        test=positions[(index >= template.test_range[0]) & (index <= template.test_range[1])],
        purge_steps=template.purge_steps,
        required_purge_steps=template.required_purge_steps,
        development_range=template.development_range,
        test_range=template.test_range,
        total_span_steps=template.total_span_steps,
    )


def check_alignment(splits: Splits, reference_folds: list[dict] | None, expected_test_range: list[int] | None) -> dict:
    """Compara limites de folds e do teste com os da V2 preservada; diverge => AlignmentError."""
    report = {
        "test_range": list(splits.test_range),
        "fold_validation_ranges": [list(f.validation_range) for f in splits.folds],
        "checked_against_v2": reference_folds is not None,
        "checked_expected_test_range": expected_test_range is not None,
    }
    if expected_test_range is not None and list(splits.test_range) != [int(v) for v in expected_test_range]:
        raise AlignmentError(f"Teste da V3 em {list(splits.test_range)}; esperado {list(expected_test_range)}.")
    if reference_folds is not None:
        mine = {f"fold_{f.index}": (list(f.train_range), list(f.validation_range)) for f in splits.folds}
        mine["test_final"] = (list(splits.development_range), list(splits.test_range))
        theirs = {r["split"]: (list(r["train_range"]), list(r["evaluation_range"])) for r in reference_folds}
        if mine != theirs:
            raise AlignmentError(f"Folds da V3 {mine} diferem dos da V2 preservada {theirs}.")
    return report


def eligible_for_window(origin_index: np.ndarray, valid_series: np.ndarray, window: int) -> np.ndarray:
    """Origem t é elegível se os `window` passos t-window+1..t existem e não têm ausência.

    A janela lê só para trás a partir da origem; não move os índices de validação nem de teste.
    """
    valid = np.asarray(valid_series, dtype=bool)
    cumulative = np.concatenate([[0], np.cumsum(~valid)])
    origins = np.asarray(origin_index, dtype=int)
    start = origins - int(window) + 1
    ok = start >= 0
    missing = np.where(ok, cumulative[origins + 1] - cumulative[np.clip(start, 0, None)], 1)
    return ok & (missing == 0)


@dataclass
class GridResult:
    best: int
    grid_scores: list[float]
    fold_scores: list[list[float]]
    rule: str
    outer_predictions: dict[int, np.ndarray]
    test_predictions: np.ndarray
    selection_predictions: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    fit_info: list[dict] = field(default_factory=list)
    best_epochs: dict[int, int | None] = field(default_factory=dict)
    final_epochs: int | None = None


def select_best(grid_scores: list[float]) -> int:
    """Menor média; empate (dentro da tolerância) fica com o primeiro ponto da grade."""
    best = 0
    for g in range(1, len(grid_scores)):
        if grid_scores[g] < grid_scores[best] - TIE_TOLERANCE:
            best = g
    return best


FitPredict = Callable[..., tuple[np.ndarray, dict]]


def _median_epochs(values) -> int | None:
    epochs = [int(v) for v in values if v is not None]
    return int(np.median(epochs)) if epochs else None


def evaluate_grid(
    n_grid: int,
    fit_predict: FitPredict,
    y: np.ndarray,
    splits: Splits,
    inner: dict[str, InnerSplit] | None,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> GridResult:
    """Seleção (interna ou da V2), validação externa uma vez e teste final uma vez.

    `fit_predict(g, train_positions, predict_positions, early_stop_positions=None, epochs=None)`
    devolve (previsões, info). Redes usam `early_stop_positions` na seleção e o número de épocas
    do melhor estado (`info['best_epoch']`) nos reajustes; modelos tabulares ignoram os dois.
    """
    y = np.asarray(y, dtype=float)
    fold_scores: list[list[float]] = [[] for _ in range(n_grid)]
    epochs: dict[int, dict[int, int | None]] = {g: {} for g in range(n_grid)}
    cache: dict[int, dict[int, np.ndarray]] = {g: {} for g in range(n_grid)}
    info_log: list[dict] = []
    nested = inner is not None
    for g in range(n_grid):
        for fold in splits.folds:
            if nested:
                split = inner.get(f"fold_{fold.index}")
                if split is None:
                    continue
                train, evaluate = split.train, split.validation
            else:
                train, evaluate = fold.train, fold.validation
            pred, info = fit_predict(g, train, evaluate, early_stop_positions=evaluate, epochs=None)
            fold_scores[g].append(float(metric(y[evaluate], pred)))
            epochs[g][fold.index] = info.get("best_epoch")
            cache[g][fold.index] = pred
            info_log.append({"phase": "selection", "grid": g, "fold": fold.index, **info})
    grid_scores = [float(np.mean(scores)) for scores in fold_scores]
    best = select_best(grid_scores)

    outer: dict[int, np.ndarray] = {}
    for fold in splits.folds:
        if nested:
            pred, info = fit_predict(best, fold.train, fold.validation, early_stop_positions=None, epochs=epochs[best].get(fold.index))
            info_log.append({"phase": "outer_refit", "grid": best, "fold": fold.index, **info})
        else:
            pred = cache[best][fold.index]
        outer[fold.index] = pred
    final_epochs = _median_epochs(epochs[best].values())
    test_pred, info = fit_predict(best, splits.development, splits.test, early_stop_positions=None, epochs=final_epochs)
    info_log.append({"phase": "test_refit", "grid": best, **info})
    return GridResult(
        best=best,
        grid_scores=grid_scores,
        fold_scores=fold_scores,
        rule="inner_validation" if nested else "outer_fold_validation_v2",
        outer_predictions=outer,
        test_predictions=np.asarray(test_pred, dtype=float),
        selection_predictions=cache,
        fit_info=info_log,
        best_epochs=epochs[best],
        final_epochs=final_epochs,
    )
