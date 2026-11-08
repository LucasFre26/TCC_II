"""TCN global compacta (H2, sob portão): uma rede compartilhada por máquinas, lotes do painel.

As janelas (CPU da máquina escalada pelo desvio global do treino, seno e cosseno de 24 h) são
montadas em CPU por índice (origem, máquina) sobre o painel; só cada lote vai para o dispositivo.
O canal de CPU passa pelo envoltório de normalização por janela, porque as máquinas têm níveis
muito diferentes. Adam, MSE, recorte de gradiente, parada antecipada pela perda da validação
interna e reajuste com o número de épocas do melhor estado, como no laço da V3.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from energy_prediction.features.cpu_features import SECONDS_PER_DAY


class WindowSource:
    """Monta janelas (n, 3, W) sob demanda; nunca materializa o painel inteiro em tensor."""

    def __init__(self, cpu: np.ndarray, bins: np.ndarray, step_seconds: float, window: int, scale: float):
        self.cpu = cpu
        self.window = int(window)
        self.scale = float(scale) if scale > 1e-9 else 1.0
        phase = 2.0 * math.pi * ((bins * float(step_seconds)) % SECONDS_PER_DAY) / SECONDS_PER_DAY
        self.sin = np.sin(phase).astype(np.float32)
        self.cos = np.cos(phase).astype(np.float32)
        offsets = np.arange(-self.window + 1, 1)
        self._offsets = offsets

    def eligible(self, origins: np.ndarray, columns: np.ndarray) -> np.ndarray:
        o = np.asarray(origins, dtype=int)
        ok = o - self.window + 1 >= 0
        out = np.zeros(len(o), dtype=bool)
        idx = np.flatnonzero(ok)
        if len(idx):
            rows = o[idx, None] + self._offsets[None, :]
            out[idx] = np.isfinite(self.cpu[rows, np.asarray(columns, dtype=int)[idx, None]]).all(axis=1)
        return out

    def batch(self, origins: np.ndarray, columns: np.ndarray) -> np.ndarray:
        o = np.asarray(origins, dtype=int)
        rows = o[:, None] + self._offsets[None, :]
        cpu = self.cpu[rows, np.asarray(columns, dtype=int)[:, None]] / self.scale
        return np.stack([cpu, self.sin[rows], self.cos[rows]], axis=1).astype(np.float32)


def build_global_tcn(params: dict, window: int):
    from energy_prediction.models.neural.tcn import TCN
    from energy_prediction.models.neural.window_norm import WindowNormalized

    inner = TCN(3, int(window), 1, channels=int(params["channels"]), dilations=list(params["dilations"]), kernel_size=int(params["kernel_size"]), dropout=float(params["dropout"]))
    return WindowNormalized(inner)


def _loss(model, source: WindowSource, o, c, y, device, batch: int) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(o), batch):
            xb = torch.as_tensor(source.batch(o[start : start + batch], c[start : start + batch]), device=device)
            yb = torch.as_tensor(y[start : start + batch, None] / source.scale, dtype=torch.float32, device=device)
            total += float(torch.nn.functional.mse_loss(model(xb), yb, reduction="sum"))
            count += len(yb)
    return total / max(count, 1)


def train_global_tcn(source: WindowSource, train: tuple[np.ndarray, np.ndarray, np.ndarray], validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None, params: dict, *, device: str, seed: int, epochs: int | None = None) -> dict:
    """Treina com `examples_per_epoch` pares sorteados por época; devolve modelo, histórico e pico de VRAM."""
    import torch

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    dev = torch.device(device)
    model = build_global_tcn(params, source.window).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    o, c, y = train
    batch = int(params["batch_size"])
    n_epoch = min(int(params["examples_per_epoch"]), len(o))
    max_epochs = int(epochs) if epochs is not None else int(params["max_epochs"])
    best, best_state, best_epoch, waited, history = np.inf, None, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        pick = rng.choice(len(o), size=n_epoch, replace=False)
        running = 0.0
        for start in range(0, n_epoch, batch):
            idx = pick[start : start + batch]
            xb = torch.as_tensor(source.batch(o[idx], c[idx]), device=dev)
            yb = torch.as_tensor(y[idx, None] / source.scale, dtype=torch.float32, device=dev)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(idx)
        val = None if validation is None or epochs is not None else _loss(model, source, *validation, dev, batch)
        history.append({"epoch": epoch, "train_loss": running / n_epoch, "val_loss": val})
        if val is not None:
            if val < best - 1e-12:
                best, best_epoch, waited = val, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                waited += 1
                if waited >= int(params["patience"]):
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    return {"model": model, "history": history, "best_epoch": best_epoch or len(history), "train_seconds": time.perf_counter() - started, "peak_cuda_bytes": peak}


def predict_global_tcn(model, source: WindowSource, origins: np.ndarray, columns: np.ndarray, *, device: str, batch: int = 4096) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(len(origins), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(origins), batch):
            xb = torch.as_tensor(source.batch(origins[start : start + batch], columns[start : start + batch]), device=torch.device(device))
            out[start : start + batch] = model(xb)[:, 0].float().cpu().numpy() * source.scale
    return out
