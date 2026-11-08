"""Estilo único das figuras: tema claro para impressão, paleta por família e rampas validadas.

Cor de identidade por família de modelo (uma entidade mantém a cor em todas as figuras):
V2 = azul, V3 residual = laranja, V3 neural = verde-água, persistência = cinza. Dentro de uma
família, o modelo se distingue por marcador e rótulo direto, nunca por uma cor nova.
Sequencial: azul claro → escuro. Divergente: azul ↔ vermelho com ponto médio cinza.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
OBSERVED = "#0b0b0b"

FAMILY_COLORS = {"referência": "#898781", "V2": "#2a78d6", "V3 residual": "#eb6834", "V3 neural": "#1baf7a", "V3 quantílico": "#6250d6"}
MARKERS = {
    "persistence": "o", "ridge_v2": "s", "xgboost_v2": "D", "xgboost_delta": "^", "statespace": "v", "hybrid": "D",
    "dlinear": "o", "tcn": "s", "nbeatsx": "^", "nhits": "v", "patchtst": "P",
}
HORIZON_MARKERS = {1: "o", 3: "s", 6: "D"}
HORIZON_LABEL = {1: "5 min", 3: "15 min", 6: "30 min"}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#b52c2c"])


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "figure.titlesize": 10,
            "figure.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def color(model: str, families: dict) -> str:
    return FAMILY_COLORS.get(families.get(model, ""), MUTED)


def figure(width: float = 6.6, height: float = 4.0, **kwargs):
    apply()
    return plt.subplots(figsize=(width, height), constrained_layout=True, **kwargs)


_DECIMAL = __import__("re").compile(r"(?<![\d.,])(\d+)\.(\d+)(?![\d.])")


def _comma(text: str) -> str:
    """Ponto decimal → vírgula; mantém o ponto de milhar (1 a 3 dígitos seguidos de exatamente 3)."""

    def swap(match):
        whole, frac = match.group(1), match.group(2)
        if len(frac) == 3 and len(whole) <= 3 and not whole.startswith("0"):
            return match.group(0)
        return f"{whole},{frac}"

    return _DECIMAL.sub(swap, text)


def _tick(value, _pos=None) -> str:
    text = f"{round(float(value), 10):g}"
    return text.replace("-", "−").replace(".", ",")


def localize(fig) -> None:
    """Vírgula decimal nos eixos lineares e nos textos da figura (português)."""
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(_tick))
    fig.canvas.draw()
    for text in fig.findobj(Text):
        content = text.get_text()
        if content and "." in content:
            text.set_text(_comma(content))
