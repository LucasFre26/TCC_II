"""Identificador da corrida, versões, hashes da configuração e dos dados, commit."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "pyyaml", "statsmodels", "torch")


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_run_id(config: dict, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash(config)[:8]}"


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(path: Path) -> dict:
    """Commit atual e se há alterações locais; None fora de um repositório git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def data_identifier(profile: dict) -> dict:
    """Hash dos brutos e chave do cache da grade, ou a semente da série sintética."""
    return {
        "source": profile.get("source", "alibaba"),
        "file_sha256": profile.get("file_sha256"),
        "cache": profile.get("cache"),
        "synthetic": profile.get("synthetic"),
    }


def base_record(*, run_id: str, seed: int, config: dict, config_path: Path, project_root: Path) -> dict:
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(seed),
        "config_path": str(config_path),
        "config_sha256": config_hash(config),
        "versions": library_versions(),
        "git": git_commit(project_root),
        "families": {},
        "excluded": [],
    }
