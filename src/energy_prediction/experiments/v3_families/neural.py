"""Família neural: DLinear, TCN causal e N-BEATSx, modo direto (tabela principal) e trajetória.

Ordem e orçamento pré-registrados (D7): escolha da janela no DLinear direto A; DLinear direto
A/B/C/D; trajetória A do DLinear; TCN direto A/B/C; trajetória A do TCN; N-BEATSx direto A/B/C;
trajetória A do N-BEATSx. O que não couber no orçamento vira inviabilidade nomeada. Depois do
núcleo: trajetórias extras (se houver folga) e o portão de N-HiTS/PatchTST (D8), decidido só
pela validação interna. A trajetória é secundária e nunca substitui o modo direto.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.comparisons import bootstrap_blocks, compare_losses, load_v2_reference, loss_series
from energy_prediction.evaluation.error_analysis import error_extremes_table
from energy_prediction.evaluation.metrics import mae
from energy_prediction.evaluation.nested import (
    AlignmentError,
    InfeasibleInnerSplitError,
    eligible_for_window,
    evaluate_grid,
)
from energy_prediction.experiments.v3_common import (
    NO_CONFIG,
    PERSISTENCE,
    ordered_metrics,
    params_text,
    persistence_inner_mae,
    persistence_rows,
    predictions_frame,
    scored_rows,
    selection_record,
)
from energy_prediction.experiments.v3_context import V3Context
from energy_prediction.features.cpu_features import target_column
from energy_prediction.features.distribution import DIST_GROUP
from energy_prediction.models.neural.windows import channel_matrix, config_channels, window_tensor
from energy_prediction.models.v3_registry import validate_family_models
from energy_prediction.runtime.device import torch_installed
from energy_prediction.runtime.output import dump_json

FAMILY = "neural"
A_ON_D = "A_on_D"


class NeuralUnavailableError(RuntimeError):
    """PyTorch ausente: a família neural não roda neste ambiente (instale o extra `v3`)."""


class Budget:
    def __init__(self, hours: float):
        self.started = time.monotonic()
        self.seconds = float(hours) * 3600.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def exhausted(self) -> bool:
        return self.remaining() <= 0


@dataclass
class Unit:
    arch: str
    mode: str  # direct | path
    config: str  # A | B | C | D | A_on_D
    role: str  # core | extra_path | optional

    @property
    def label(self) -> str:
        return f"{self.arch}/{self.mode}/{self.config}"


def core_units(models: list[str], carrier: str, with_d: bool) -> list[Unit]:
    units = []
    for arch in models:
        configs = ["A", "B", "C"]
        if arch == carrier and with_d:
            configs += ["D", A_ON_D]
        units += [Unit(arch, "direct", c, "core") for c in configs]
        units.append(Unit(arch, "path", "A", "core"))
    return units


class NeuralFamily:
    def __init__(self, ctx: V3Context, device: str):
        if not torch_installed():
            raise NeuralUnavailableError("PyTorch não está instalado; instale o extra `v3` para a família neural.")
        self.ctx = ctx
        self.cfg = ctx.cfg["families"]["neural"]
        self.device = device
        self.seed = ctx.seed
        self.models = validate_family_models(FAMILY, list(self.cfg["models"]))
        groups = {name: list(g) for name, g in ctx.cfg["features"]["ablation"].items()}
        if ctx.d_table is not None:
            groups["D"] = [*groups["C"], DIST_GROUP]
        groups[A_ON_D] = groups["A"]
        self.channels = {name: config_channels(ctx.specs, g) for name, g in groups.items()}
        self.matrices = {name: channel_matrix(ctx.series, channels) for name, channels in self.channels.items()}
        self._windows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self.histories: list[dict] = []
        self.profiles: list[dict] = []
        self.last_model = None

    def tables(self, config: str):
        if config in ("D", A_ON_D):
            return self.ctx.d_table, self.ctx.d_splits
        return self.ctx.table, self.ctx.splits

    def windows(self, config: str, window: int) -> tuple[np.ndarray, np.ndarray]:
        key = (config, int(window))
        if key not in self._windows:
            table, _ = self.tables(config)
            matrix, valid, _ = self.matrices[config]
            origins = table["origin_index"].to_numpy()
            eligible = eligible_for_window(origins, valid, window)
            X = np.zeros((len(origins), matrix.shape[1], int(window)), dtype=np.float32)
            X[eligible] = window_tensor(matrix, origins[eligible], window)
            self._windows[key] = (X, eligible)
        return self._windows[key]

    def inner(self, config: str, window: int):
        table, splits = self.tables(config)
        _, eligible = self.windows(config, window)
        return self.ctx.inner_for(table, splits, eligible)

    def targets(self, config: str, mode: str, steps: int | None) -> np.ndarray:
        table, _ = self.tables(config)
        if mode == "direct":
            return table[[target_column(steps)]].to_numpy(dtype=float)
        return table[[target_column(s) for s in self.ctx.path_steps]].to_numpy(dtype=float)

    def grid(self, arch: str, window: int) -> tuple[list[dict], dict]:
        info: dict = {}
        if arch == "dlinear":
            dl = self.cfg["dlinear"]
            grid = [{"linear": lin, "learning_rate": float(lr), "moving_average": int(dl["moving_average"])} for lin in dl["linear"] for lr in dl["learning_rate"]]
        elif arch == "tcn":
            from energy_prediction.models.neural.tcn import feasible_dilations

            tcn = self.cfg["tcn"]
            stacks, removed = feasible_dilations([list(d) for d in tcn["dilations"]], int(tcn["kernel_size"]), window)
            info["removed_dilation_stacks"] = removed
            grid = [
                {"channels": int(c), "dilations": d, "kernel_size": int(tcn["kernel_size"]), "dropout": float(tcn["dropout"]), "learning_rate": float(tcn["learning_rate"])}
                for c in tcn["channels"]
                for d in stacks
            ]
        elif arch == "nbeatsx":
            nb = self.cfg["nbeatsx"]
            grid = [{k: nb[k] for k in ("width", "layers", "trend_degree", "harmonics")} | {"learning_rate": float(nb["learning_rate"])}]
        elif arch in ("nhits", "patchtst"):
            grid = [dict(self.cfg["optional"][arch])]
        else:  # pragma: no cover
            raise ValueError(arch)
        return grid, info

    @staticmethod
    def build(arch: str, params: dict, n_channels: int, window: int, horizon: int):
        p = {k: v for k, v in params.items() if k != "learning_rate"}
        if arch == "dlinear":
            from energy_prediction.models.neural.dlinear import DLinear

            return DLinear(n_channels, window, horizon, **p)
        if arch == "tcn":
            from energy_prediction.models.neural.tcn import TCN

            return TCN(n_channels, window, horizon, **p)
        if arch == "nbeatsx":
            from energy_prediction.models.neural.nbeatsx import NBEATSx

            return NBEATSx(n_channels, window, horizon, **p)
        if arch == "nhits":
            from energy_prediction.models.neural.optional import NHiTS

            return NHiTS(n_channels, window, horizon, **p)
        if arch == "patchtst":
            from energy_prediction.models.neural.optional import PatchTST

            return PatchTST(n_channels, window, horizon, **p)
        raise ValueError(arch)

    def fit_predict(self, arch: str, grid: list[dict], config: str, window: int, Y: np.ndarray):
        from energy_prediction.models.neural.loop import LoopConfig, predict, train

        table, _ = self.tables(config)
        X, eligible = self.windows(config, window)
        matrix = self.matrices[config][0]
        origins = table["origin_index"].to_numpy()
        finite = np.isfinite(Y).all(axis=1)
        horizon = Y.shape[1]

        def fp(g, train_pos, predict_pos, early_stop_positions=None, epochs=None):
            params = grid[g]
            if not eligible[predict_pos].all():
                raise AlignmentError("Origem de avaliação sem histórico para a janela; os índices da V2 não podem mudar.")
            usable = train_pos[eligible[train_pos] & finite[train_pos]]
            stop = None if early_stop_positions is None else early_stop_positions[eligible[early_stop_positions] & finite[early_stop_positions]]
            last = int(origins[usable].max())
            mean = np.nanmean(matrix[: last + 1], axis=0)
            std = np.nanstd(matrix[: last + 1], axis=0)
            std = np.where(std < 1e-8, 1.0, std)

            def scale(positions):
                return ((X[positions] - mean[None, :, None]) / std[None, :, None]).astype(np.float32)

            loop_cfg = LoopConfig.from_config(self.cfg["loop"], params["learning_rate"])
            result = train(
                lambda: self.build(arch, params, X.shape[1], window, horizon),
                scale(usable),
                (Y[usable] - mean[0]) / std[0],
                None if stop is None else scale(stop),
                None if stop is None else (Y[stop] - mean[0]) / std[0],
                cfg=loop_cfg,
                device=self.device,
                seed=self.seed,
                epochs=epochs,
            )
            pred, inference = predict(result.model, scale(predict_pos), device=self.device)
            self.last_model = result.model
            pred = pred * std[0] + mean[0]
            info = {
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "precision": result.precision,
                "precision_note": result.precision_note,
                "train_seconds": result.train_seconds,
                "inference_seconds": inference,
                "n_params": result.n_params,
                "peak_cuda_bytes": result.peak_cuda_bytes,
                "n_train": int(len(usable)),
                "n_train_without_history": int((~eligible[train_pos]).sum()),
                "history": result.history,
            }
            return (pred[:, 0] if horizon == 1 else pred), info

        return fp

    def record(self, unit: Unit, steps_label, result) -> None:
        histories, fits = [], []
        for entry in result.fit_info:
            entry = dict(entry)
            for row in entry.pop("history", []):
                histories.append({"arch": unit.arch, "config": unit.config, "mode": unit.mode, "horizon": steps_label, "phase": entry["phase"], "grid": entry["grid"], "fold": entry.get("fold"), **row})
            fits.append(entry)
        self.histories.extend(histories)
        test_fit = next(f for f in fits if f["phase"] == "test_refit")
        self.profiles.append(
            {
                "arch": unit.arch,
                "config": unit.config,
                "mode": unit.mode,
                "horizon": steps_label,
                "device": self.device,
                "n_params": test_fit["n_params"],
                "n_fits": len(fits),
                "train_seconds_total": float(sum(f["train_seconds"] for f in fits)),
                "train_seconds_test_refit": float(test_fit["train_seconds"]),
                "inference_seconds_test": float(test_fit["inference_seconds"]),
                "peak_cuda_bytes_max": int(max(f["peak_cuda_bytes"] for f in fits)),
                "precisions": sorted({f["precision"] for f in fits}),
                "precision_notes": sorted({f["precision_note"] for f in fits if f["precision_note"]}),
                "final_epochs": result.final_epochs,
                "n_train_without_history_test_refit": test_fit["n_train_without_history"],
            }
        )


def select_window(family: NeuralFamily, budget: Budget) -> tuple[int | None, dict]:
    """Janela de menor soma do MAE interno do DLinear direto A nos horizontes (empate: a mais curta)."""
    ws = family.cfg["window_selection"]
    params = [{"linear": ws["linear"], "learning_rate": float(ws["learning_rate"]), "moving_average": int(family.cfg["dlinear"]["moving_average"])}]
    record = {"rule": "soma do MAE interno do DLinear direto, configuração A, nos horizontes diretos", "candidates": []}
    best_window, best_score = None, np.inf
    for window in sorted(int(w) for w in family.cfg["window_candidates"]):
        try:
            inner = family.inner("A", window)
        except InfeasibleInnerSplitError as error:
            record["candidates"].append({"window_steps": window, "feasible": False, "reason": str(error)})
            continue
        if inner is None:
            raise ValueError("A escolha da janela exige a seleção aninhada ligada.")
        total, per_horizon = 0.0, {}
        for steps in family.ctx.steps:
            Y = family.targets("A", "direct", steps)
            fp = family.fit_predict("dlinear", params, "A", window, Y)
            scores = []
            for split in inner.values():
                pred, _ = fp(0, split.train, split.validation, early_stop_positions=split.validation)
                scores.append(mae(Y[split.validation, 0], pred))
            per_horizon[str(steps)] = float(np.mean(scores))
            total += per_horizon[str(steps)]
        record["candidates"].append({"window_steps": window, "feasible": True, "inner_mae_by_horizon": per_horizon, "inner_mae_sum": total})
        if total < best_score - 1e-12:
            best_window, best_score = window, total
    record["chosen_window_steps"] = best_window
    record["elapsed_seconds"] = budget.elapsed()
    return best_window, record


def _path_metric(scored_columns):
    def metric(y_true, y_pred):
        return float(np.mean([mae(y_true[:, j], y_pred[:, j]) for j in scored_columns]))

    return metric


def run_unit(family: NeuralFamily, unit: Unit, window: int, out_rows: dict) -> None:
    ctx = family.ctx
    table, splits = family.tables(unit.config)
    inner = family.inner(unit.config, window)
    grid, grid_info = family.grid(unit.arch, window)
    if not grid:
        raise InfeasibleInnerSplitError(f"{unit.label}: nenhuma pilha de dilatação cabe na janela de {window} passos.")
    model_config = "A" if unit.config == A_ON_D else unit.config
    is_d = unit.config in ("D", A_ON_D)
    bucket = out_rows["d" if is_d else "main"]
    role = "principal" if unit.mode == "direct" else "secundaria"
    if unit.mode == "direct":
        for steps in ctx.steps:
            Y = family.targets(unit.config, "direct", steps)
            result = evaluate_grid(len(grid), family.fit_predict(unit.arch, grid, unit.config, window, Y), Y[:, 0], splits, inner, metric=mae)
            family.record(unit, steps, result)
            keys = {"model": unit.arch, "config": model_config, "mode": "direct", "best_params": params_text(grid[result.best])}
            rows, frames = scored_rows(table, splits, steps=steps, step_seconds=ctx.step_seconds, keys=keys, outer=result.outer_predictions, test=result.test_predictions, inner_score=result.grid_scores[result.best] if inner else None)
            bucket["rows"].extend({**r, "table_role": role, "window_steps": window} for r in rows)
            bucket["frames"].extend(frames)
            bucket["selections"].append(selection_record(result, grid, model=unit.arch, config=model_config, horizon_steps=int(steps), mode="direct", window_steps=window, role=unit.role, **grid_info))
            _save_checkpoint(family, out_rows["checkpoints"], f"{unit.arch}_{unit.config}_direct_h{steps}")
    else:
        Y = family.targets(unit.config, "path", None)
        scored = [s - 1 for s in ctx.steps]
        result = evaluate_grid(len(grid), family.fit_predict(unit.arch, grid, unit.config, window, Y), Y, splits, inner, metric=_path_metric(scored))
        family.record(unit, "path", result)
        keys = {"model": unit.arch, "config": model_config, "mode": "path", "best_params": params_text(grid[result.best])}
        for steps in ctx.steps:
            outer = {k: v[:, steps - 1] for k, v in result.outer_predictions.items()}
            rows, frames = scored_rows(table, splits, steps=steps, step_seconds=ctx.step_seconds, keys=keys, outer=outer, test=result.test_predictions[:, steps - 1])
            bucket["rows"].extend({**r, "table_role": role, "window_steps": window} for r in rows)
            bucket["frames"].extend(frames)
        for steps in ctx.path_steps:
            if steps not in ctx.steps:
                bucket["frames"].append(predictions_frame(table, splits.test, split="test_final", model=unit.arch, config=model_config, steps=steps, y_pred=result.test_predictions[:, steps - 1], mode="path"))
        record = selection_record(result, grid, model=unit.arch, config=model_config, horizon_steps=0, mode="path", window_steps=window, role=unit.role, selection_metric="média do MAE interno nos passos 1, 3 e 6", **grid_info)
        bucket["selections"].append(record)
        _save_checkpoint(family, out_rows["checkpoints"], f"{unit.arch}_{unit.config}_path")


def _save_checkpoint(family: NeuralFamily, directory: Path, name: str) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in family.last_model.state_dict().items()}, directory / f"{name}.pt")


def gate(family: NeuralFamily, completed: set[str], selections: list[dict], window: int) -> dict:
    """D8: as três redes com A direto concluído e ao menos uma com MAE interno (A, 5 min) abaixo da persistência."""
    first = min(family.ctx.steps)
    direct_a = {arch: f"{arch}/direct/A" in completed for arch in family.models}
    inner = family.inner("A", window)
    reference = persistence_inner_mae(family.ctx.table, inner, first)
    scores = {
        s["model"]: float(s["selection_score"])
        for s in selections
        if s["config"] == "A" and s["mode"] == "direct" and int(s["horizon_steps"]) == first and s["model"] in family.models
    }
    passed = all(direct_a.values()) and any(score < reference for score in scores.values())
    return {
        "criterion": "três redes obrigatórias com A direto concluído e ao menos uma com MAE interno (A, 5 min) menor que a persistência na mesma validação interna; o teste final não entra",
        "direct_A_completed": direct_a,
        "persistence_inner_mae_h1": reference,
        "network_inner_mae_h1": scores,
        "passed": bool(passed),
    }


def _pairs(predictions: pd.DataFrame, d_predictions: pd.DataFrame | None, ctx: V3Context, models: list[str], carrier: str) -> tuple[pd.DataFrame, list[dict]]:
    pairs = ctx.cfg["comparisons"]["pairs"]
    blocks, primary, n_boot = bootstrap_blocks(ctx.cfg)
    v2 = load_v2_reference(ctx.v2_dir)
    direct = predictions[predictions["mode"] == "direct"]
    available = set(zip(direct["model"], direct["config"]))
    rows, skipped = [], []
    for steps in ctx.steps:
        def add(label, a, b, la=None, lb=None, frame=direct):
            la = la if la is not None else loss_series(frame, model=a[0], config=a[1], horizon_steps=steps)
            lb = lb if lb is not None else loss_series(frame, model=b[0], config=b[1], horizon_steps=steps)
            meta = {"comparison": label, "horizon_steps": int(steps), "model_a": a[0], "config_a": a[1], "model_b": b[0], "config_b": b[1]}
            rows.extend(compare_losses(la, lb, blocks=blocks, primary_block=primary, n_boot=n_boot, seed=ctx.seed, meta=meta))

        for arch in models:
            if (arch, "A") not in available:
                continue
            if "new_vs_persistence_A" in pairs:
                add("new_vs_persistence_A", (arch, "A"), (PERSISTENCE, NO_CONFIG))
            if "new_vs_v2_xgboost_A" in pairs:
                if steps not in v2:
                    skipped.append({"item": f"new_vs_v2_xgboost_A {arch} h{steps}", "reason": "previsões da V2 não configuradas"})
                else:
                    reference = v2[steps]
                    lb = pd.Series((reference["y_true"] - reference["y_pred"]).abs().to_numpy(), index=reference["origin_index"].to_numpy()).sort_index()
                    la = loss_series(direct, model=arch, config="A", horizon_steps=steps)
                    if not la.index.equals(lb.index):
                        raise AlignmentError(f"Origens do teste da V3 ({arch}, h{steps}) diferem das da V2 preservada.")
                    add("new_vs_v2_xgboost_A", (arch, "A"), ("xgboost_v2", "A"), la=la, lb=lb)
            if "ablation_vs_A" in pairs:
                for config in ("B", "C"):
                    if (arch, config) in available:
                        add("ablation_vs_A", (arch, config), (arch, "A"))
        if "d_vs_a" in pairs and d_predictions is not None and carrier in ctx.cfg["comparisons"]["d_contrast_models"]:
            d_direct = d_predictions[d_predictions["mode"] == "direct"]
            if {(carrier, "D"), (carrier, "A")} <= set(zip(d_direct["model"], d_direct["config"])):
                add("d_vs_a", (carrier, "D"), (carrier, "A"), frame=d_direct)
    return pd.DataFrame(rows), skipped


def run(ctx: V3Context, out: Path, *, device, **_runtime) -> dict:
    family = NeuralFamily(ctx, device["effective"])
    cfg = family.cfg
    budget = Budget(float(cfg["budget_hours"]))
    carrier = cfg["d_carrier"]
    out_rows = {"main": {"rows": [], "frames": [], "selections": []}, "d": {"rows": [], "frames": [], "selections": []}, "checkpoints": out / "checkpoints"}
    infeasible: list[dict] = []
    completed: set[str] = set()

    window, window_record = (None, {"chosen_window_steps": None})
    if budget.exhausted():
        window_record["reason"] = "orçamento esgotado antes da escolha da janela"
    else:
        window, window_record = select_window(family, budget)
    units = core_units(family.models, carrier, ctx.d_table is not None and "dlinear" in family.models)

    def attempt(unit: Unit) -> None:
        if window is None:
            infeasible.append({"item": unit.label, "role": unit.role, "reason": "nenhuma janela candidata viável ou escolhida"})
            return
        if budget.exhausted():
            infeasible.append({"item": unit.label, "role": unit.role, "reason": f"orçamento de {cfg['budget_hours']} h esgotado antes da unidade"})
            return
        try:
            run_unit(family, unit, window, out_rows)
            completed.add(unit.label)
        except InfeasibleInnerSplitError as error:
            infeasible.append({"item": unit.label, "role": unit.role, "reason": str(error)})

    for unit in units:
        attempt(unit)
    core_done = all(u.label in completed for u in units)
    if core_done:
        for arch in family.models:
            for config in cfg["extra_path_configs"]:
                if config == "D" and (arch != carrier or ctx.d_table is None):
                    continue
                attempt(Unit(arch, "path", config, "extra_path"))
    elif cfg["extra_path_configs"]:
        infeasible.append({"item": "trajetórias extras " + ",".join(cfg["extra_path_configs"]), "role": "extra_path", "reason": "núcleo não terminou dentro do orçamento"})

    optional_record: dict = {"enabled": bool(cfg["optional"]["enabled"]), "fitted": []}
    if cfg["optional"]["enabled"] and window is not None:
        optional_record["gate"] = gate(family, completed, out_rows["main"]["selections"], window)
        if not optional_record["gate"]["passed"]:
            optional_record["reason"] = "portão D8 não cumprido; N-HiTS e PatchTST não foram ajustados"
        else:
            for index, arch in enumerate(cfg["optional"]["order"]):
                if index > 0:
                    previous = cfg["optional"]["order"][index - 1]
                    hours = budget.remaining() / 3600.0
                    if f"{previous}/direct/A" not in completed or hours <= float(cfg["optional"]["patchtst_min_remaining_hours"]):
                        optional_record[f"{arch}_skipped"] = f"{previous} não terminou ou folga de {hours:.2f} h não supera {cfg['optional']['patchtst_min_remaining_hours']} h"
                        break
                unit = Unit(arch, "direct", "A", "optional")
                attempt(unit)
                if unit.label in completed:
                    optional_record["fitted"].append(arch)
    elif window is None:
        optional_record["reason"] = "sem janela escolhida"
    else:
        optional_record["reason"] = "arquiteturas opcionais desligadas na configuração"

    main, d = out_rows["main"], out_rows["d"]
    if not main["frames"]:
        predictions = pd.DataFrame(columns=["split", "model", "config", "mode", "horizon_steps", "origin_index", "timestamp", "y_true", "y_pred"])
        metrics = ordered_metrics([{}]).iloc[0:0]
    else:
        persist_rows, persist_frames = [], []
        for steps in ctx.steps:
            r, f = persistence_rows(ctx.table, ctx.splits, steps=steps, step_seconds=ctx.step_seconds)
            persist_rows.extend({**x, "table_role": "principal"} for x in r)
            persist_frames.extend(f)
        predictions = pd.concat(persist_frames + main["frames"], ignore_index=True)
        metrics = ordered_metrics(persist_rows + main["rows"])
    d_predictions = pd.concat(d["frames"], ignore_index=True) if d["frames"] else None
    intervals, skipped = _pairs(predictions, d_predictions, ctx, family.models, carrier) if main["frames"] else (pd.DataFrame(), [])

    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    intervals.to_csv(out / "bootstrap_ci.csv", index=False)
    if main["frames"]:
        extremes = [error_extremes_table(part, peak_quantile=float(ctx.cfg["comparisons"]["peak_quantile"])).assign(mode=mode) for mode, part in predictions.groupby("mode")]
        pd.concat(extremes, ignore_index=True).to_csv(out / "error_extremes.csv", index=False)
    if d["rows"]:
        ordered_metrics(d["rows"]).to_csv(out / "d_contrast_metrics.csv", index=False)
        d_predictions.to_csv(out / "d_contrast_predictions.csv", index=False)
    pd.DataFrame(family.histories).to_csv(out / "epoch_history.csv", index=False)
    pd.DataFrame(family.profiles).to_csv(out / "training_profile.csv", index=False)
    dump_json(out / "selections.json", main["selections"] + [{**s, "contrast": "d_vs_a"} for s in d["selections"]])
    dump_json(out / "window_selection.json", window_record)
    dump_json(out / "optional_architectures.json", optional_record)
    dump_json(
        out / "family_info.json",
        {
            "models": family.models,
            "device": device,
            "budget_hours": float(cfg["budget_hours"]),
            "elapsed_seconds": budget.elapsed(),
            "window_steps": window,
            "channels": family.channels,
            "completed_units": sorted(completed),
            "infeasible": infeasible,
            "core_completed": core_done,
            "skipped_comparisons": skipped,
            "statement": (
                "Tabela principal: modo direto (um modelo por horizonte). A trajetória de seis passos é secundária, "
                "alimenta a energia e não substitui o modo direto, mesmo com MAE menor no teste."
            ),
            "early_stopping": "seleção: melhor época na validação interna; reajustes: número de épocas do melhor estado (mediana entre folds no teste)",
        },
    )
    peak = max((p["peak_cuda_bytes_max"] for p in family.profiles), default=0)
    return {
        "window_steps": window,
        "completed_units": sorted(completed),
        "core_completed": core_done,
        "optional_fitted": optional_record["fitted"],
        "gate_passed": optional_record.get("gate", {}).get("passed"),
        "peak_cuda_bytes": int(peak),
        "excluded": [{"item": i["item"], "reason": i["reason"]} for i in infeasible]
        + ([{"item": "N-HiTS/PatchTST", "reason": optional_record["reason"]}] if optional_record.get("reason") else [])
        + [{"item": s["item"], "reason": s["reason"]} for s in skipped],
    }
