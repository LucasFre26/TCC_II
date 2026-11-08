"""Redes da V3 (DLinear, TCN, PatchTST) no protocolo por dia, em duas formulações (H8).

- `v3`: padronização global pelo treino e alvo em nível, exatamente como na família neural da V3;
- `window`: canal de CPU normalizado por janela (envoltório `WindowNormalized`), alvo na escala
  do desvio global, sem centrar.

Configuração A (CPU e calendário), janela fixa, modo direto, mesma grade e laço de treino da V3.
Cada semente repete a seleção na validação interna e o reajuste no treino do bloco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.evaluation.day_forward import evaluate_block, restrict_block
from energy_prediction.evaluation.metrics import mae
from energy_prediction.evaluation.nested import eligible_for_window
from energy_prediction.experiments.v3p2_families.common import BLOCK, INNER, block_frame
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.neural.windows import channel_matrix, config_channels, window_tensor

FORMULATIONS = ("v3", "window")


def network_grid(arch: str, rob: dict, window: int) -> tuple[list[dict], dict]:
    info: dict = {}
    if arch == "dlinear":
        dl = rob["dlinear"]
        grid = [{"linear": lin, "learning_rate": float(lr), "moving_average": int(dl["moving_average"])} for lin in dl["linear"] for lr in dl["learning_rate"]]
    elif arch == "tcn":
        from energy_prediction.models.neural.tcn import feasible_dilations

        tcn = rob["tcn"]
        stacks, removed = feasible_dilations([list(d) for d in tcn["dilations"]], int(tcn["kernel_size"]), window)
        info["removed_dilation_stacks"] = removed
        grid = [{"channels": int(c), "dilations": d, "kernel_size": int(tcn["kernel_size"]), "dropout": float(tcn["dropout"]), "learning_rate": float(tcn["learning_rate"])} for c in tcn["channels"] for d in stacks]
    elif arch == "patchtst":
        grid = [dict(rob["patchtst"])]
    else:
        raise ValueError(arch)
    return grid, info


def build_network(arch: str, params: dict, n_channels: int, window: int, horizon: int, formulation: str):
    from energy_prediction.experiments.v3_families.neural import NeuralFamily
    from energy_prediction.models.neural.window_norm import WindowNormalized

    inner = NeuralFamily.build(arch, params, n_channels, window, horizon)
    return WindowNormalized(inner) if formulation == "window" else inner


class NetworkData:
    """Janelas causais da configuração A sobre a série agregada (mesmas regras da V3)."""

    def __init__(self, ctx, window: int):
        self.ctx = ctx
        self.window = int(window)
        self.channels = config_channels(ctx.specs, ["cpu"])
        self.matrix, valid, self.info = channel_matrix(ctx.series, self.channels)
        origins = ctx.table["origin_index"].to_numpy()
        self.eligible = eligible_for_window(origins, valid, self.window)
        self.X = np.zeros((len(origins), self.matrix.shape[1], self.window), dtype=np.float32)
        self.X[self.eligible] = window_tensor(self.matrix, origins[self.eligible], self.window)
        self.origins = origins

    def blocks(self):
        return [restrict_block(b, self.eligible) for b in self.ctx.blocks]


def network_fit_predict(data: NetworkData, arch: str, grid: list[dict], formulation: str, steps: int, *, loop_cfg: dict, device: str, seed: int):
    from energy_prediction.models.neural.loop import LoopConfig, predict, train

    if formulation not in FORMULATIONS:
        raise ValueError(f"Formulação desconhecida: {formulation!r}")
    Y = data.ctx.table[[target_column(steps)]].to_numpy(dtype=float)
    finite = np.isfinite(Y).all(axis=1)
    X, matrix = data.X, data.matrix

    def fp(g, train_pos, predict_pos, early_stop_positions=None, epochs=None):
        params = grid[g]
        usable = train_pos[finite[train_pos]]
        stop = None if early_stop_positions is None else early_stop_positions[finite[early_stop_positions]]
        last = int(data.origins[usable].max())
        mean = np.nanmean(matrix[: last + 1], axis=0)
        std = np.nanstd(matrix[: last + 1], axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        if formulation == "window":
            center = mean.copy()
            center[0] = 0.0  # o canal de CPU não é centrado: a janela faz isso
        else:
            center = mean

        def scale(positions):
            return ((X[positions] - center[None, :, None]) / std[None, :, None]).astype(np.float32)

        def target(positions):
            return (Y[positions] - center[0]) / std[0]

        cfg = LoopConfig.from_config(loop_cfg, params["learning_rate"])
        result = train(
            lambda: build_network(arch, {k: v for k, v in params.items()}, X.shape[1], data.window, 1, formulation),
            scale(usable),
            target(usable),
            None if stop is None else scale(stop),
            None if stop is None else target(stop),
            cfg=cfg,
            device=device,
            seed=seed,
            epochs=epochs,
        )
        pred, inference = predict(result.model, scale(predict_pos), device=device)
        pred = pred[:, 0] * std[0] + center[0]
        return pred, {
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "precision": result.precision,
            "train_seconds": result.train_seconds,
            "inference_seconds": inference,
            "n_params": result.n_params,
            "peak_cuda_bytes": result.peak_cuda_bytes,
            "n_train": int(len(usable)),
        }

    return fp


def run_network(ctx, data: NetworkData, *, arch: str, formulation: str, seeds: list[int], rob: dict, device: str) -> tuple[list[pd.DataFrame], list[dict], list[dict]]:
    grid, info = network_grid(arch, rob, data.window)
    if not grid:
        raise ValueError(f"{arch}: nenhum ponto de grade cabe na janela de {data.window} passos.")
    frames, selections, profile = [], [], []
    blocks = data.blocks()
    for seed in seeds:
        for steps in ctx.steps:
            fp = network_fit_predict(data, arch, grid, formulation, steps, loop_cfg=rob["loop"], device=device, seed=int(seed))
            y = ctx.table[target_column(steps)].to_numpy(dtype=float)
            for block in blocks:
                result = evaluate_block(len(grid), fp, y, block, metric=mae)
                extra = {"seed": int(seed)}
                frames.append(block_frame(ctx.table, block, block.inner.validation, phase=INNER, model=arch, variant=formulation, steps=steps, y_pred=result.inner_prediction, **extra))
                frames.append(block_frame(ctx.table, block, block.evaluation, phase=BLOCK, model=arch, variant=formulation, steps=steps, y_pred=result.block_prediction, **extra))
                refit = result.fit_info[-1]
                selections.append({"model": arch, "variant": formulation, "seed": int(seed), "horizon_steps": int(steps), "block": block.label, "best_index": result.best, "best_params": grid[result.best], "grid_scores": result.grid_scores, "best_epoch": result.best_epoch, **info})
                profile.append(
                    {
                        "model": arch, "variant": formulation, "seed": int(seed), "horizon_steps": int(steps), "block": block.label,
                        "n_fits": len(result.fit_info), "train_seconds_total": float(sum(f.get("train_seconds", 0.0) for f in result.fit_info)),
                        "n_params": refit.get("n_params"), "precision": refit.get("precision"), "peak_cuda_bytes": int(max(f.get("peak_cuda_bytes", 0) for f in result.fit_info)),
                        "final_epochs": refit.get("epochs_run"), "device": device,
                    }
                )
    return frames, selections, profile
