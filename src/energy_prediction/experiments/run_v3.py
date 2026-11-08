"""Executa a V3 por família: integridade, residual, neural, quantílica e energia.

Cada família grava em results/v3/<run_id>/<família>/ por escrita atômica. A auditoria da
população (família de integridade) precede qualquer ajuste. results/v1, results/v2 e
results/v2-full nunca são escritos. Potência e energia são estimativas paramétricas; PUE não é
calculado. Uso:

    python -m energy_prediction.experiments.run_v3 --config config/v3.yaml [--family residual]
        [--run-id <id>] [--device auto|cuda|cpu]
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from energy_prediction.experiments.v3_config import V3ConfigError, load_v3_config
from energy_prediction.experiments.v3_context import build_context, resolve_path
from energy_prediction.models.v3_registry import FAMILIES
from energy_prediction.runtime.device import DEVICE_CHOICES, configure_determinism, resolve_device
from energy_prediction.runtime.output import RunDirectory, guard_output_root
from energy_prediction.runtime.run_record import base_record, config_hash, data_identifier, library_versions, new_run_id

STATEMENT = (
    "Previsão de CPU no Alibaba Cluster Trace v2018 e energia de TI modelada pelo modelo de Fan num "
    "servidor substituto. O trace não tem medição elétrica; PUE não é calculado."
)


def project_root_from_config(config_path: Path) -> Path:
    """config/v3.yaml vive em Tcc-sourcecode/config; a raiz do código é o pai."""
    return Path(config_path).resolve().parent.parent


def _runner(family: str):
    if family == "integrity":
        from energy_prediction.experiments.v3_families import integrity as module
    elif family == "residual":
        from energy_prediction.experiments.v3_families import residual as module
    elif family == "neural":
        from energy_prediction.experiments.v3_families import neural as module
    elif family == "quantile":
        from energy_prediction.experiments.v3_families import quantile as module
    elif family == "energy":
        from energy_prediction.experiments.v3_families import energy as module
    else:  # pragma: no cover - validado antes
        raise V3ConfigError(f"Família desconhecida: {family!r}")
    return module.run


def _run_family(family: str, ctx, run_dir: RunDirectory, manifest: dict, runtime: dict) -> None:
    started = time.perf_counter()
    with run_dir.family(family) as temporary:
        summary = _runner(family)(ctx, temporary, **runtime) or {}
    excluded = summary.pop("excluded", [])
    manifest["families"][family] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    manifest["excluded"] = [e for e in manifest.get("excluded", []) if e.get("family") != family]
    manifest["excluded"].extend({"family": family, **item} for item in excluded)
    run_dir.write_manifest(manifest)


def execute(config_path: Path, *, family: str | None = None, run_id: str | None = None, device: str | None = None) -> dict:
    config_path = Path(config_path)
    cfg = load_v3_config(config_path)
    if device is not None:
        if device not in DEVICE_CHOICES:
            raise V3ConfigError(f"--device deve ser um de {list(DEVICE_CHOICES)}.")
        cfg["device"] = device
    if family is not None and family not in FAMILIES:
        raise V3ConfigError(f"Família desconhecida: {family!r}. Famílias: {list(FAMILIES)}.")
    root = project_root_from_config(config_path)
    output_root = guard_output_root(root, resolve_path(root, cfg["run_root"]))
    device_record = resolve_device(cfg["device"])
    determinism = configure_determinism(int(cfg["seed"]), device_record["effective"])

    ctx = build_context(cfg, root)  # erro de dados aqui não cria pasta de resultado

    run_id = run_id or new_run_id(cfg)
    run_dir = RunDirectory(root, output_root, run_id)
    manifest = run_dir.read_manifest()
    if manifest:
        if manifest["config_sha256"] != config_hash(cfg):
            raise V3ConfigError(f"A corrida {run_id} foi criada com outra configuração; use outro --run-id.")
    else:
        manifest = base_record(run_id=run_id, seed=int(cfg["seed"]), config=cfg, config_path=config_path, project_root=root)
        manifest["statement"] = STATEMENT
        manifest["config"] = cfg
    manifest["data"] = data_identifier(ctx.profile)
    manifest["versions"] = library_versions()
    manifest["device"] = device_record
    manifest["determinism"] = determinism
    runtime = {"run_dir": run_dir, "device": device_record}

    requested = list(FAMILIES) if family is None else [family]
    if "integrity" not in requested and not run_dir.is_done("integrity"):
        requested = ["integrity", *requested]  # a auditoria precede qualquer ajuste
    for name in requested:
        if name == "integrity" and run_dir.is_done("integrity") and family is None:
            continue
        _run_family(name, ctx, run_dir, manifest, runtime)
    if family is None:
        from energy_prediction.experiments.v3_report import generate

        generate(run_dir.path)
    manifest["output_dir"] = str(run_dir.path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="V3: experimentos finais de previsão de CPU e energia de TI modelada.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family", choices=list(FAMILIES))
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=list(DEVICE_CHOICES))
    args = parser.parse_args()
    manifest = execute(args.config, family=args.family, run_id=args.run_id, device=args.device)
    print(
        "V3 concluída. "
        f"run_id={manifest['run_id']} familias={list(manifest['families'])} "
        f"dispositivo={manifest['device']['effective']} saida={manifest['output_dir']}"
    )


if __name__ == "__main__":
    main()
