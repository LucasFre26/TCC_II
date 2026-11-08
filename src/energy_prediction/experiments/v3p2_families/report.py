"""Família de relatório: gera o relatório da corrida só a partir dos arquivos das demais famílias."""

from __future__ import annotations

from pathlib import Path

from energy_prediction.experiments.v3p2_report import generate

FAMILY = "report"


def run(ctx, out: Path, *, run_dir=None, manifest=None, **_runtime) -> dict:
    if manifest is not None:
        run_dir.write_manifest(manifest)  # o relatório lê portões e exclusões do manifesto gravado
    summary = generate(run_dir.path, out)
    excluded = [] if summary["html"] else [{"item": "relatório HTML", "reason": "plotly (extra analysis) não instalado"}]
    return {**summary, "excluded": excluded}
