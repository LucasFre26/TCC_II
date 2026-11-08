"""Leitura somente dos CSV internos dos arquivos compactados."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "value", "container_id")


class MissingArchiveError(FileNotFoundError):
    """Um arquivo de entrada configurado não existe."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_archives(gpu_archive: Path, memory_archive: Path) -> None:
    missing = [path for path in (gpu_archive, memory_archive) if not path.is_file()]
    if missing:
        listed = ", ".join(str(path) for path in missing)
        raise MissingArchiveError(f"Arquivo de entrada ausente: {listed}")


def load_metric_archive(path: Path) -> pd.DataFrame:
    """Lê o único CSV de métricas dentro do tar.gz, sem extrair em disco."""
    path = Path(path)
    if not path.is_file():
        raise MissingArchiveError(f"Arquivo de entrada ausente: {path}")

    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith(".csv")
            and not Path(member.name).name.startswith("._")
        ]
        if len(members) != 1:
            names = [member.name for member in members]
            raise ValueError(f"Esperado um CSV em {path.name}, encontrado: {names}")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Não foi possível ler {members[0].name} em {path}")
        frame = pd.read_csv(extracted)

    renamed = frame.rename(
        columns={
            "timestamp_anon": "timestamp",
            "container_ip": "container_id",
        }
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing}")

    out = renamed.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="raise")
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if out["timestamp"].isna().any() or out["value"].isna().any():
        raise ValueError(f"Valor ausente em timestamp ou value de {path.name}")
    out["container_id"] = out["container_id"].astype("string")
    return out
