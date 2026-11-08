"""Executa a V3 Parte 2 por família, no protocolo cronológico por dia.

Cada família grava em results/v3-part2/<run_id>/<família>/ por escrita atômica. A integridade
precede qualquer outra família. results/v1, results/v2, results/v2-full e results/v3 nunca são
escritos; os arquivos de referência lidos têm o SHA-256 conferido no início e no fim. No modo
completo, o pré-registro é obrigatório e seu hash é conferido antes de cada família. Potência e
energia são estimativas paramétricas; PUE não é calculado. Uso:

    python -m energy_prediction.experiments.run_v3p2 --config config/v3p2.yaml [--family hierarchical]
        [--run-id <id>] [--device auto|cuda|cpu]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import time
from datetime import datetime, timezone
from pathlib import Path

from energy_prediction.experiments.v3_context import resolve_path
from energy_prediction.experiments.v3p2_config import (
    DEPENDENCIES,
    FAMILIES,
    V3P2ConfigError,
    load_v3p2_config,
    preregistration_hash,
)
from energy_prediction.runtime.device import DEVICE_CHOICES, configure_determinism, resolve_device
from energy_prediction.runtime.output import PROTECTED_RESULTS, RunDirectory, guard_output_root
from energy_prediction.runtime.run_record import base_record, config_hash, data_identifier, library_versions, new_run_id

PROTECTED = (*PROTECTED_RESULTS, "results/v3")
STATEMENT = (
    "V3 Parte 2: previsão de CPU no Alibaba Cluster Trace v2018 no protocolo por dia (dias de desenvolvimento "
    "exploratórios; dia 7 como teste herdado, quarta inspeção) e energia de TI modelada num servidor substituto. "
    "O trace não tem medição elétrica; PUE não é calculado."
)
NOT_RESULT = "Corrida de smoke: os números não são resultado."


class MissingFamilyError(RuntimeError):
    """Uma família pedida depende de outra que não foi concluída nesta corrida."""


class ReferenceChangedError(RuntimeError):
    """Um arquivo de referência (V2 ou V3) mudou durante a corrida."""


class PreregistrationChangedError(RuntimeError):
    """O documento de pré-registro mudou depois do início da corrida."""


def project_root_from_config(config_path: Path) -> Path:
    return Path(config_path).resolve().parent.parent


def tree_hashes(directory: Path | None) -> dict[str, str]:
    if directory is None or not Path(directory).exists():
        return {}
    out = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            out[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def reference_hashes(root: Path, cfg: dict) -> dict:
    if not cfg["reference"]["check_hashes"]:
        return {}
    return {key: tree_hashes(resolve_path(root, cfg["reference"][key])) for key in ("v2_dir", "v3_run") if cfg["reference"][key]}


def _module(family: str):
    return importlib.import_module(f"energy_prediction.experiments.v3p2_families.{family}")


def _check_dependencies(family: str, run_dir: RunDirectory, scheduled: list[str]) -> None:
    missing = [dep for dep in DEPENDENCIES[family] if not run_dir.is_done(dep) and dep not in scheduled]
    if missing:
        raise MissingFamilyError(f"A família {family!r} depende de {missing}, que não foram concluídas na corrida {run_dir.run_id}.")


def _run_family(family: str, ctx, run_dir: RunDirectory, manifest: dict, runtime: dict) -> None:
    started = time.perf_counter()
    with run_dir.family(family) as temporary:
        summary = _module(family).run(ctx, temporary, **runtime) or {}
    excluded = summary.pop("excluded", [])
    gates = summary.pop("gates", [])
    manifest["families"][family] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    manifest["excluded"] = [e for e in manifest.get("excluded", []) if e.get("family") != family] + [{"family": family, **item} for item in excluded]
    manifest["gates"] = [g for g in manifest.get("gates", []) if g.get("family") != family] + [{"family": family, **g} for g in gates]
    run_dir.write_manifest(manifest)


def execute(config_path: Path, *, family: str | None = None, run_id: str | None = None, device: str | None = None, no_amp: bool = False) -> dict:
    from energy_prediction.experiments.v3p2_context import build_context

    config_path = Path(config_path)
    cfg = load_v3p2_config(config_path)
    if device is not None:
        if device not in DEVICE_CHOICES:
            raise V3P2ConfigError(f"--device deve ser um de {list(DEVICE_CHOICES)}.")
        cfg["device"] = device
    if family is not None and family not in FAMILIES:
        raise V3P2ConfigError(f"Família desconhecida: {family!r}. Famílias: {list(FAMILIES)}.")
    root = project_root_from_config(config_path)
    output_root = guard_output_root(root, resolve_path(root, cfg["run_root"]), PROTECTED)
    prereg = preregistration_hash(root, cfg)  # falha antes de qualquer ajuste no modo completo
    before = reference_hashes(root, cfg)
    device_record = resolve_device(cfg["device"])
    determinism = configure_determinism(int(cfg["seed"]), device_record["effective"])

    run_id = run_id or new_run_id(cfg)
    run_dir = RunDirectory(root, output_root, run_id, protected=PROTECTED)
    requested = [f for f in FAMILIES if f != "report"] + ["report"] if family is None else [family]
    if "integrity" not in requested and not run_dir.is_done("integrity"):
        requested = ["integrity", *requested]
    scheduled: list[str] = []
    for name in requested:
        if run_dir.is_done(name):
            continue
        _check_dependencies(name, run_dir, scheduled)
        scheduled.append(name)

    ctx = build_context(cfg, root)  # erro de dados aqui não cria pasta de resultado
    manifest = run_dir.read_manifest()
    if manifest:
        if manifest["config_sha256"] != config_hash(cfg):
            raise V3P2ConfigError(f"A corrida {run_id} foi criada com outra configuração; use outro --run-id.")
        if prereg and manifest.get("preregistration", {}).get("sha256") not in (None, prereg["sha256"]):
            raise PreregistrationChangedError("O pré-registro mudou desde o início desta corrida.")
    else:
        manifest = base_record(run_id=run_id, seed=int(cfg["seed"]), config=cfg, config_path=config_path, project_root=root)
        manifest["statement"] = STATEMENT
        manifest["mode"] = cfg["mode"]
        manifest["config"] = cfg
        manifest["preregistration"] = prereg
        manifest["reference_hashes"] = {"before": {k: len(v) for k, v in before.items()}, "digest_before": config_hash(before)}
        if cfg["mode"] != "full":
            manifest["not_result"] = NOT_RESULT
        if not cfg["pretrained"]["enabled"]:
            reason = (cfg["pretrained"].get("assessment") or {}).get("reason", "família de modelo pré-treinado desligada")
            manifest["excluded"].append({"family": "pretrained", "item": "modelo pré-treinado (H7)", "reason": " ".join(str(reason).split())})
    manifest["data"] = data_identifier(ctx.profile)
    manifest["versions"] = library_versions()
    manifest["device"] = device_record
    manifest["determinism"] = determinism
    manifest["protocol"] = ctx.resolved["blocks"]
    manifest["no_amp"] = bool(no_amp)
    runtime = {"run_dir": run_dir, "device": device_record, "manifest": manifest, "no_amp": bool(no_amp)}

    for name in scheduled:
        if prereg:
            now = preregistration_hash(root, cfg)
            if now["sha256"] != prereg["sha256"]:
                raise PreregistrationChangedError("O documento de pré-registro mudou durante a corrida; a corrida foi interrompida.")
        _run_family(name, ctx, run_dir, manifest, runtime)

    after = reference_hashes(root, cfg)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise ReferenceChangedError(f"Arquivos de referência mudaram durante a corrida: {changed}")
    manifest["reference_hashes"]["digest_after"] = config_hash(after)
    manifest["reference_hashes"]["unchanged"] = True
    manifest["panel"] = ctx._panel_record
    run_dir.write_manifest(manifest)
    manifest["output_dir"] = str(run_dir.path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 Parte 2: protocolo por dia, previsão hierárquica, modelo global e energia da frota.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family", choices=list(FAMILIES))
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=list(DEVICE_CHOICES))
    parser.add_argument("--no-amp", action="store_true", help="desliga a precisão mista das redes (conferência em FP32)")
    args = parser.parse_args()
    manifest = execute(args.config, family=args.family, run_id=args.run_id, device=args.device, no_amp=args.no_amp)
    print(f"V3 Parte 2 concluída. run_id={manifest['run_id']} familias={list(manifest['families'])} dispositivo={manifest['device']['effective']} saida={manifest['output_dir']}")


if __name__ == "__main__":
    main()
