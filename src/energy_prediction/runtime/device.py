"""Escolha do dispositivo das redes: auto | cuda | cpu.

Só consulta `torch.cuda.is_available()` em tempo de execução. Não instala nem altera driver.
Modelos clássicos, filtro de estados, bootstrap, atributos e energia ficam sempre em CPU.
"""

from __future__ import annotations

import importlib.util
import os

DEVICE_CHOICES = ("auto", "cuda", "cpu")


class DeviceError(ValueError):
    """Dispositivo desconhecido ou CUDA exigida sem estar disponível."""


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_device(requested: str) -> dict:
    """Dispositivo efetivo e o que o ambiente informou. `auto` cai para CPU sem CUDA."""
    if requested not in DEVICE_CHOICES:
        raise DeviceError(f"Dispositivo {requested!r} inválido. Use {list(DEVICE_CHOICES)}.")
    record = {
        "requested": requested,
        "effective": "cpu",
        "torch_installed": torch_installed(),
        "cuda_available": False,
        "device_name": None,
        "total_memory_bytes": None,
        "note": None,
    }
    if not record["torch_installed"]:
        if requested == "cuda":
            raise DeviceError("Dispositivo 'cuda' pedido, mas o PyTorch não está instalado (extra `v3`).")
        record["note"] = "PyTorch ausente; redes não podem ser treinadas neste ambiente."
        return record
    import torch

    available = bool(torch.cuda.is_available())
    record["cuda_available"] = available
    if requested == "cuda" and not available:
        raise DeviceError("Dispositivo 'cuda' pedido, mas torch.cuda.is_available() é falso.")
    if requested in ("auto", "cuda") and available:
        record["effective"] = "cuda"
        properties = torch.cuda.get_device_properties(0)
        record["device_name"] = properties.name
        record["total_memory_bytes"] = int(properties.total_memory)
    elif requested == "auto":
        record["note"] = "CUDA indisponível; treino neural em CPU."
    return record


def configure_determinism(seed: int, effective: str) -> dict:
    """Sementes de NumPy e PyTorch, cuDNN determinístico e benchmark desligado."""
    import numpy as np

    np.random.seed(seed)
    notes = []
    if not torch_installed():
        return {"seed": seed, "torch": False, "notes": notes}
    if effective == "cuda":
        # Exigido pelo cuBLAS para operações determinísticas; precisa estar antes da 1ª chamada CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if effective == "cuda":
        notes.append("Resultados em CUDA podem diferir em bits dos da CPU; a referência dos testes é a CPU.")
    return {
        "seed": seed,
        "torch": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": "warn_only",
        "notes": notes,
    }
