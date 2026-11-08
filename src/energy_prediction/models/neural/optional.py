"""Arquiteturas opcionais, só ajustadas se o portão pré-registrado for cumprido.

N-HiTS (compacto): blocos com agrupamento por máximo em taxas diferentes na entrada; como os
horizontes têm no máximo seis passos, os coeficientes de previsão já estão na resolução final
(sem interpolação). PatchTST: janela dividida em trechos por canal, codificador Transformer
compartilhado entre canais e cabeça linear sobre a concatenação.
"""

from __future__ import annotations

import torch
from torch import nn


class NHiTSBlock(nn.Module):
    def __init__(self, n_channels, window, horizon, pool, width, layers):
        super().__init__()
        self.pool = nn.MaxPool1d(pool, stride=pool, ceil_mode=True)
        pooled = -(-window // pool)
        stack, size = [], pooled * n_channels
        for _ in range(layers):
            stack += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.fc = nn.Sequential(*stack)
        self.backcast = nn.Linear(width, window)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, x):
        hidden = self.fc(self.pool(x).flatten(1))
        return self.backcast(hidden), self.forecast(hidden)


class NHiTS(nn.Module):
    def __init__(self, n_channels: int, window: int, horizon: int, *, pool_sizes=(4, 2, 1), width: int = 64, layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(NHiTSBlock(n_channels, window, horizon, int(p), width, layers) for p in pool_sizes)

    def forward(self, x):
        residual = x[:, :1, :]
        exogenous = x[:, 1:, :]
        forecast = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(torch.cat([residual, exogenous], dim=1))
            residual = residual - backcast.unsqueeze(1)
            forecast = forecast + block_forecast
        return forecast


class PatchTST(nn.Module):
    def __init__(self, n_channels: int, window: int, horizon: int, *, patch_length=12, stride=6, d_model=32, heads=2, layers=2, dropout=0.1):
        super().__init__()
        if window < patch_length:
            raise ValueError("Janela menor que o trecho do PatchTST.")
        self.patch_length, self.stride = int(patch_length), int(stride)
        n_patches = (window - self.patch_length) // self.stride + 1
        self.embed = nn.Linear(self.patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(n_channels * n_patches * d_model, horizon)

    def forward(self, x):
        batch, channels, _ = x.shape
        patches = x.unfold(2, self.patch_length, self.stride)  # (lote, canais, trechos, comprimento)
        tokens = self.embed(patches.reshape(batch * channels, -1, self.patch_length)) + self.position
        encoded = self.encoder(tokens).reshape(batch, -1)
        return self.head(encoded)
