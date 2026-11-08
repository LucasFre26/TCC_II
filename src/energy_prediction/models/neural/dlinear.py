"""DLinear: decomposição por média móvel em tendência e sazonal, uma camada linear em cada parte.

Entrada (lote, canais, janela); saída (lote, passos). Linear compartilhada: a mesma camada para
todos os canais, combinados por pesos aprendidos. Por variável: uma camada por canal, somadas.
"""

from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = int(kernel)
        self.pool = nn.AvgPool1d(self.kernel, stride=1)

    def forward(self, x):
        left = (self.kernel - 1) // 2
        right = self.kernel - 1 - left
        padded = torch.cat([x[:, :, :1].expand(-1, -1, left), x, x[:, :, -1:].expand(-1, -1, right)], dim=2)
        return self.pool(padded)


class DLinear(nn.Module):
    def __init__(self, n_channels: int, window: int, horizon: int, *, linear: str = "shared", moving_average: int = 25):
        super().__init__()
        if linear not in ("shared", "individual"):
            raise ValueError(f"linear deve ser 'shared' ou 'individual'; recebido {linear!r}")
        kernel = min(int(moving_average), int(window))
        kernel = kernel if kernel % 2 == 1 else kernel - 1
        self.decompose = MovingAverage(max(kernel, 1))
        self.individual = linear == "individual"
        if self.individual:
            self.seasonal = nn.ModuleList(nn.Linear(window, horizon) for _ in range(n_channels))
            self.trend = nn.ModuleList(nn.Linear(window, horizon) for _ in range(n_channels))
        else:
            self.seasonal = nn.Linear(window, horizon)
            self.trend = nn.Linear(window, horizon)
            self.mix = nn.Linear(n_channels, 1)

    def forward(self, x):
        trend = self.decompose(x)
        seasonal = x - trend
        if self.individual:
            parts = [self.seasonal[c](seasonal[:, c]) + self.trend[c](trend[:, c]) for c in range(x.shape[1])]
            return torch.stack(parts, dim=1).sum(dim=1)
        per_channel = self.seasonal(seasonal) + self.trend(trend)  # (lote, canais, passos)
        return self.mix(per_channel.transpose(1, 2)).squeeze(-1)
