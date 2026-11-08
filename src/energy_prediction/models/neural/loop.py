"""Laço de treino comum das redes da V3.

Adam, perda MSE na escala padronizada, lotes de 64, recorte de gradiente em norma 1, no máximo
50 épocas e parada antecipada com paciência 5 na perda da validação interna, guardando o melhor
estado. Nos reajustes (sem validação), treina o número de épocas do melhor estado da seleção.
Precisão mista só em CUDA e só se a perda continuar finita nas primeiras épocas; senão FP32.
Acumulação de gradiente não é usada. Só lotes da série agregada vão para o dispositivo.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass
class LoopConfig:
    max_epochs: int = 50
    patience: int = 5
    grad_clip_norm: float = 1.0
    batch_size: int = 64
    mixed_precision: str = "auto"  # auto | off
    finite_check_epochs: int = 2
    learning_rate: float = 1e-3

    @classmethod
    def from_config(cls, loop: dict, learning_rate: float) -> "LoopConfig":
        return cls(
            max_epochs=int(loop["max_epochs"]),
            patience=int(loop["patience"]),
            grad_clip_norm=float(loop["grad_clip_norm"]),
            batch_size=int(loop["batch_size"]),
            mixed_precision=str(loop["mixed_precision"]),
            finite_check_epochs=int(loop["finite_check_epochs"]),
            learning_rate=float(learning_rate),
        )


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    epochs_run: int = 0
    precision: str = "fp32"
    precision_note: str | None = None
    train_seconds: float = 0.0
    n_params: int = 0
    peak_cuda_bytes: int = 0


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _autocast_dtype(device: torch.device, mode: str):
    if mode != "auto" or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _evaluate(model, X, Y, batch, dtype) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb, yb = X[start : start + batch], Y[start : start + batch]
            with torch.autocast(device_type=X.device.type, dtype=dtype, enabled=dtype is not None):
                out = model(xb)
            total += float(nn.functional.mse_loss(out.float(), yb, reduction="sum"))
            count += yb.numel()
    return total / max(count, 1)


def _run(factory, X, Y, Xv, Yv, cfg: LoopConfig, device, seed, epochs, dtype) -> TrainResult:
    torch.manual_seed(seed)
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    max_epochs = int(epochs) if epochs is not None else cfg.max_epochs
    use_validation = Xv is not None and epochs is None and len(Xv) > 0
    best_loss, best_state, best_epoch, waited = np.inf, None, None, 0
    result = TrainResult(model=model, n_params=count_parameters(model))
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator).to(device)
        running, seen = 0.0, 0
        for start in range(0, len(X), cfg.batch_size):
            index = order[start : start + cfg.batch_size]
            xb, yb = X[index], Y[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                out = model(xb)
            loss = nn.functional.mse_loss(out.float(), yb)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            running += float(loss.detach()) * len(index)
            seen += len(index)
        train_loss = running / max(seen, 1)
        val_loss = _evaluate(model, Xv, Yv, cfg.batch_size, dtype) if use_validation else None
        result.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if dtype is not None and epoch <= cfg.finite_check_epochs:
            if not np.isfinite(train_loss) or (val_loss is not None and not np.isfinite(val_loss)):
                raise FloatingPointError("perda não finita em precisão mista")
        if use_validation:
            if val_loss < best_loss - 1e-12:
                best_loss, best_epoch, waited = val_loss, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                waited += 1
                if waited >= cfg.patience:
                    break
    result.epochs_run = len(result.history)
    if use_validation and best_state is not None:
        model.load_state_dict(best_state)
        result.best_epoch = best_epoch
    else:
        result.best_epoch = result.epochs_run
    if not np.isfinite(result.history[-1]["train_loss"]):
        raise FloatingPointError("perda de treino não finita")
    return result


def train(
    factory: Callable[[], nn.Module],
    X: np.ndarray,
    Y: np.ndarray,
    X_val: np.ndarray | None,
    Y_val: np.ndarray | None,
    *,
    cfg: LoopConfig,
    device: str,
    seed: int,
    epochs: int | None = None,
) -> TrainResult:
    """Treina uma rede; `epochs` fixo desliga a validação (reajuste)."""
    dev = torch.device(device)
    to = lambda a: None if a is None else torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)  # noqa: E731
    Xt, Yt, Xv, Yv = to(X), to(Y), to(X_val), to(Y_val)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    dtype = _autocast_dtype(dev, cfg.mixed_precision)
    note = None
    try:
        result = _run(factory, Xt, Yt, Xv, Yv, cfg, dev, seed, epochs, dtype)
    except FloatingPointError:
        if dtype is None:
            raise
        note = "precisão mista com perda não finita; refeito em FP32"
        dtype = None
        result = _run(factory, Xt, Yt, Xv, Yv, cfg, dev, seed, epochs, dtype)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        result.peak_cuda_bytes = int(torch.cuda.max_memory_allocated(dev))
    result.train_seconds = time.perf_counter() - started
    result.precision = "fp32" if dtype is None else ("bf16-mixed" if dtype == torch.bfloat16 else "fp16-mixed")
    result.precision_note = note or (None if dev.type == "cuda" else "CPU: precisão simples")
    return result


def predict(model: nn.Module, X: np.ndarray, *, device: str, batch_size: int = 512) -> tuple[np.ndarray, float]:
    """Previsão em FP32 e tempo de uma passagem de inferência."""
    dev = torch.device(device)
    model.eval()
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    started = time.perf_counter()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(Xt), batch_size):
            outputs.append(model(Xt[start : start + batch_size]).float().cpu())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return torch.cat(outputs).numpy().astype(float), elapsed
