"""Análise expandida de uma corrida da V3.

Uso: python -m energy_prediction.analysis.run --config config/v3-analysis.yaml

Grava em results/v3/<run_id>/expanded-analysis/ (csv, figures, interactive, latex, reports,
metadata). Escreve numa pasta temporária e só a renomeia no fim; as pastas da V1, da V2 e das
famílias da V3 são conferidas byte a byte antes e depois.
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from energy_prediction.analysis import export, facts, figures, interactive, latex, report_md
from energy_prediction.analysis.config import load_config, resolve_paths
from energy_prediction.analysis.loader import hash_tree, sha256
from energy_prediction.analysis.pipeline import AnalysisResult, compute
from energy_prediction.runtime.output import dump_json
from energy_prediction.runtime.run_record import config_hash, git_commit, library_versions

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUBDIRS = ["csv", "figures/png", "figures/svg", "figures/pdf", "figures/data", "interactive", "latex", "reports", "metadata"]


class ProtectedChangedError(RuntimeError):
    """Um arquivo preservado mudou durante a análise."""


def _versions() -> dict:
    versions = library_versions()
    for name in ("scipy", "plotly"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def publish(result: AnalysisResult, tmp: Path) -> dict:
    cfg, info = result.cfg, result.info
    for sub in SUBDIRS:
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    csv_paths = export.write_tables(result.tables, tmp / "csv")
    dictionary = export.dictionary(result.tables)
    export.write_csv(dictionary, tmp / "metadata" / "data_dictionary.csv")
    (tmp / "metadata" / "data_dictionary.md").write_text(export.dictionary_markdown(dictionary), encoding="utf-8")
    records = figures.render_all(result.tables, info, cfg, tmp / "figures", list(cfg["figures"]["formats"]), int(cfg["figures"]["dpi"]))
    dump_json(tmp / "metadata" / "figures.json", records)
    for r in records:
        if not r.get("generated"):
            result.warnings.append(f"Figura {r['name']} não gerada: {r['reason']}.")
    f = facts.compute(result.tables, info)
    markdown = report_md.build(result.tables, f, info, cfg, records, result.checks, result.warnings)
    (tmp / "reports" / "scientific-analysis.md").write_text(markdown, encoding="utf-8")
    page = interactive.build(result.tables, f, info, cfg, records, markdown, [p.stem for p in csv_paths])
    (tmp / "interactive" / "relatorio.html").write_text(page, encoding="utf-8")
    latex.write(result.tables, f, cfg, records, tmp / "latex")
    return {"figures": records, "n_csv": len(csv_paths)}


def manifest(result: AnalysisResult, tmp: Path, final: Path, started: str, elapsed: float, outputs: dict, protected_check: dict) -> dict:
    art = result.art
    files = []
    for path in sorted(p for p in tmp.rglob("*") if p.is_file()):
        files.append({"path": str(path.relative_to(tmp)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    figure_records = [r for r in outputs["figures"] if r.get("generated")]
    return {
        "analysis": "análise expandida da V3",
        "run_id": art.run_id,
        "run_dir": str(result.paths.run_dir),
        "output_dir": str(final),
        "run_config_sha256": art.manifest.get("config_sha256"),
        "run_git": art.manifest.get("git"),
        "analysis_git": git_commit(result.paths.project_root),
        "analysis_config": result.cfg,
        "analysis_config_sha256": config_hash(result.cfg),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "versions": _versions(),
        "data_source": art.manifest.get("data"),
        "inputs": art.inventory,
        "integrity_checks": result.checks + [protected_check],
        "refits": result.info.get("refits", []),
        "warnings": result.warnings,
        "counts": {"csv": outputs["n_csv"], "figures": len(figure_records), "figure_files": sum(len(r["files"]) for r in figure_records)},
        "outputs": files,
        "statement": "Potência e energia são estimativas paramétricas de um servidor substituto; o trace não tem medição elétrica e PUE não foi calculado. O teste final foi reutilizado da V2.",
    }


def execute(config_path: Path, project_root: Path = PROJECT_ROOT) -> dict:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg, project_root)
    protected = paths.protected()
    before = hash_tree(protected)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clock = time.perf_counter()
    result = compute(cfg, paths)
    final = paths.output_dir
    tmp = final.with_name(f".{final.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        outputs = publish(result, tmp)
        after = hash_tree(protected)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        protected_check = {"check": "arquivos_preservados_inalterados", "passed": not changed, "detail": f"{len(before)} arquivos de results/v1, results/v2, results/v2-full e das famílias da V3 com SHA-256 iguais antes e depois", "changed": changed}
        if changed:
            raise ProtectedChangedError(f"Arquivos preservados mudaram: {changed[:5]}")
        record = manifest(result, tmp, final, started, time.perf_counter() - clock, outputs, protected_check)
        dump_json(tmp / "metadata" / "analysis_manifest.json", record)
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Análise expandida de uma corrida da V3.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "v3-analysis.yaml")
    args = parser.parse_args(argv)
    record = execute(args.config)
    print(f"Análise gravada em {record['output_dir']}")
    print(f"CSVs: {record['counts']['csv']} · figuras: {record['counts']['figures']} ({record['counts']['figure_files']} arquivos)")
    for warning in record["warnings"]:
        print(f"Aviso: {warning}")


if __name__ == "__main__":
    main()
