"""Normalização por janela do canal de CPU, como envoltório de qualquer rede da V3 (H8).

Entrada (lote, canais, janela) com o canal 0 = CPU dividida pelo desvio global do treino (sem
centrar) e os demais canais já padronizados. Para cada janela:

    x0' = (x0 − x0[t]) / (s_w + ε),   ŷ = x0[t] + (s_w + ε) · rede(x')

em que x0[t] é o último valor observado e s_w o desvio da própria janela. Só valores até a origem
entram. Somar uma constante c à série soma c/σ ao canal 0 e, portanto, c/σ à saída: a previsão
desnormalizada pelo mesmo σ fica deslocada exatamente por c (invariância a deslocamento).
Formulação próxima de RevIN e de NLinear (referências a conferir antes da redação).
"""

from __future__ import annotations

import torch
from torch import nn


class WindowNormalized(nn.Module):
    def __init__(self, inner: nn.Module, eps: float = 1e-3):
        super().__init__()
        self.inner = inner
        self.eps = float(eps)

    def forward(self, x):
        cpu = x[:, :1, :]
        last = cpu[:, :, -1:]
        scale = cpu.std(dim=2, keepdim=True, unbiased=False) + self.eps
        normalized = torch.cat([(cpu - last) / scale, x[:, 1:, :]], dim=1)
        out = self.inner(normalized)
        return last[:, 0, :] + scale[:, 0, :] * out
