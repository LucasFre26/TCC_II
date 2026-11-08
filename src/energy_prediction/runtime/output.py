"""Guarda dos resultados preservados e escrita atômica por família em results/v3/<run_id>/.

Cada família grava num diretório temporário oculto e só é renomeada para o nome final quando
termina sem erro. Uma falha apaga apenas o temporário daquela família.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

PROTECTED_RESULTS = ("results/v1", "results/v2", "results/v2-full")


class ProtectedOutputError(ValueError):
    """O diretório de saída coincide com um resultado preservado, contém um ou está dentro dele."""


class FamilyAlreadyDoneError(ValueError):
    """A família já foi concluída nesta corrida; uma nova corrida usa outro identificador."""


def guard_output_root(project_root: Path, output_root: Path, protected: tuple[str, ...] = PROTECTED_RESULTS) -> Path:
    """Falha antes de qualquer escrita se a saída tocar um dos resultados protegidos.

    O padrão protege results/v1, results/v2 e results/v2-full (V3). A V3 Parte 2 acrescenta
    results/v3 à lista.
    """
    target = Path(output_root).resolve()
    for relative in protected:
        guarded = (Path(project_root) / relative).resolve()
        if target == guarded or guarded in target.parents or target in guarded.parents:
            raise ProtectedOutputError(
                f"Saída ({target}) não pode coincidir com {relative}, conter esse diretório ou estar dentro dele."
            )
    return target


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo não serializável: {type(value)}")


def _clean(value):
    """Troca NaN/inf por None para gravar JSON válido."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def dump_json(path: Path, payload) -> None:
    Path(path).write_text(
        json.dumps(_clean(payload), ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )


def atomic_write_json(path: Path, payload) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    dump_json(temporary, payload)
    os.replace(temporary, path)


class RunDirectory:
    """results/v3/<run_id>/ com uma pasta por família e um manifest.json."""

    def __init__(self, project_root: Path, output_root: Path, run_id: str, protected: tuple[str, ...] = PROTECTED_RESULTS):
        self.output_root = guard_output_root(project_root, output_root, protected)
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"Identificador de corrida inválido: {run_id!r}")
        self.run_id = run_id
        self.path = self.output_root / run_id

    def family_path(self, family: str) -> Path:
        return self.path / family

    def is_done(self, family: str) -> bool:
        return self.family_path(family).is_dir()

    @contextmanager
    def family(self, family: str) -> Iterator[Path]:
        """Diretório temporário da família; renomeado ao final, apagado em caso de exceção."""
        final = self.family_path(family)
        if final.exists():
            raise FamilyAlreadyDoneError(
                f"A família {family!r} já está concluída em {self.path}; use outro --run-id para repetir."
            )
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / f".{family}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            yield temporary
            temporary.rename(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            self._remove_if_empty()
            raise

    def _remove_if_empty(self) -> None:
        try:
            if self.path.is_dir() and not any(self.path.iterdir()):
                self.path.rmdir()
        except OSError:
            pass

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def read_manifest(self) -> dict:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def write_manifest(self, manifest: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, manifest)
