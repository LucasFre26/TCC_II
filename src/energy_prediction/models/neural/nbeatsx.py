"""N-BEATSx com um bloco de tendência e um de sazonalidade, residual e exógenas até a origem.

O canal 0 (CPU) é a série que os blocos explicam por retroprojeção; os demais canais da janela
(exógenas e calendário, todos até a origem) entram achatados na entrada de cada bloco. Nenhuma
exógena futura entra.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _trend_basis(length: int, degree: int) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / max(length, 1)
    return torch.stack([t**i for i in range(degree + 1)])  # (graus, comprimento)


def _seasonal_basis(length: int, harmonics: int) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / max(length, 1)
    rows = [torch.ones(length)]
    for k in range(1, harmonics + 1):
        rows.append(torch.cos(2 * math.pi * k * t))
        rows.append(torch.sin(2 * math.pi * k * t))
    return torch.stack(rows)


class BasisBlock(nn.Module):
    def __init__(self, input_size: int, width: int, layers: int, backcast_basis: torch.Tensor, forecast_basis: torch.Tensor):
        super().__init__()
        stack, size = [], input_size
        for _ in range(layers):
            stack += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.fc = nn.Sequential(*stack)
        self.theta_backcast = nn.Linear(width, backcast_basis.shape[0], bias=False)
        self.theta_forecast = nn.Linear(width, forecast_basis.shape[0], bias=False)
        self.register_buffer("backcast_basis", backcast_basis)
        self.register_buffer("forecast_basis", forecast_basis)

    def forward(self, features):
        hidden = self.fc(features)
        return self.theta_backcast(hidden) @ self.backcast_basis, self.theta_forecast(hidden) @ self.forecast_basis


class NBEATSx(nn.Module):
    def __init__(self, n_channels: int, window: int, horizon: int, *, width: int = 64, layers: int = 4, trend_degree: int = 3, harmonics: int = 4):
        super().__init__()
        input_size = window * n_channels
        self.blocks = nn.ModuleList(
            [
                BasisBlock(input_size, width, layers, _trend_basis(window, trend_degree), _trend_basis(horizon, trend_degree)),
                BasisBlock(input_size, width, layers, _seasonal_basis(window, harmonics), _seasonal_basis(horizon, harmonics)),
            ]
        )

    def forward(self, x):
        residual = x[:, 0, :]
        exogenous = x[:, 1:, :].flatten(1)
        forecast = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(torch.cat([residual, exogenous], dim=1))
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast
