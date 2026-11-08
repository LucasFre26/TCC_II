"""Rede convolucional temporal causal: uma convolução dilatada por bloco residual.

Preenchimento só à esquerda, então a saída no instante t depende de entradas até t. Campo
receptivo = 1 + (kernel - 1) · soma das dilatações; pilhas com campo maior que a janela são
removidas antes do treino (ver `feasible_dilations`).
"""

from __future__ import annotations

from torch import nn

from energy_prediction.models.neural.windows import receptive_field


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(nn.functional.pad(x, (self.left, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return nn.functional.relu(self.dropout(nn.functional.relu(self.conv(x))) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, n_channels: int, window: int, horizon: int, *, channels: int, dilations: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if receptive_field(kernel_size, dilations) > window:
            raise ValueError("Campo receptivo maior que a janela; a pilha deveria ter sido removida.")
        blocks, previous = [], n_channels
        for dilation in dilations:
            blocks.append(TemporalBlock(previous, channels, kernel_size, int(dilation), dropout))
            previous = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        return self.head(self.blocks(x)[:, :, -1])


def feasible_dilations(stacks: list[list[int]], kernel_size: int, window: int) -> tuple[list[list[int]], list[dict]]:
    kept, removed = [], []
    for stack in stacks:
        field = receptive_field(kernel_size, stack)
        if field > window:
            removed.append({"dilations": list(stack), "receptive_field": field, "window": int(window)})
        else:
            kept.append(list(stack))
    return kept, removed
