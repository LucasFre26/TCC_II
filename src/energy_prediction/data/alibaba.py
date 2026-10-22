"""Leitura, validação e reamostragem de machine_usage e machine_meta do Alibaba v2018.

A leitura é em streaming: o CSV interno do tar.gz nunca é extraído em disco nem
carregado inteiro em memória. O resultado é uma tabela (máquina, intervalo da
grade) com a média de cada variável no intervalo.
"""

from __future__ import annotations

import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from energy_prediction.data.load_genai import MissingArchiveError, file_sha256

USAGE_COLUMNS = (
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
)
META_COLUMNS = (
    "machine_id",
    "time_stamp",
    "failure_domain_1",
    "failure_domain_2",
    "cpu_num",
    "mem_size",
    "status",
)
REQUIRED_USAGE = tuple(column for column in USAGE_COLUMNS if column != "mkpi")
REQUIRED_META = ("machine_id", "cpu_num", "status")

# Nome curto usado nas séries de trabalho -> coluna do trace.
SERIES_COLUMNS = {
    "cpu": "cpu_util_percent",
    "mem": "mem_util_percent",
    "mem_gps": "mem_gps",
    "net_in": "net_in",
    "net_out": "net_out",
    "disk": "disk_io_percent",
}
VALUE_COLUMNS = tuple(SERIES_COLUMNS.values())
VALID_RANGE = (0.0, 100.0)


class SchemaError(ValueError):
    """Coluna exigida ausente ou arquivo com formato inesperado."""


class IrregularGranularityError(ValueError):
    """Não há intervalo de amostragem predominante."""


def require_files(*paths: Path) -> None:
    missing = [Path(p) for p in paths if not Path(p).is_file()]
    if missing:
        listed = ", ".join(str(p) for p in missing)
        raise MissingArchiveError(
            f"Arquivo de entrada ausente: {listed}. Baixe machine_usage.tar.gz e "
            "machine_meta.tar.gz com Tcc-sourcecode/data/clusterdata-master/"
            "cluster-trace-v2018/fetchData.sh e ajuste os caminhos em config/v2.yaml."
        )


@contextmanager
def open_csv(path: Path) -> Iterator:
    """Abre o único CSV do tar.gz (ou um .csv simples) em modo binário."""
    path = Path(path)
    if not path.is_file():
        raise MissingArchiveError(f"Arquivo de entrada ausente: {path}")
    if path.suffix == ".csv":
        with path.open("rb") as handle:
            yield handle
        return
    with tarfile.open(path, "r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith(".csv")
            and not Path(member.name).name.startswith("._")
        ]
        if len(members) != 1:
            names = [member.name for member in members]
            raise SchemaError(f"Esperado um CSV em {path.name}, encontrado: {names}")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise SchemaError(f"Não foi possível ler {members[0].name} em {path}")
        try:
            yield extracted
        finally:
            extracted.close()


def sniff_columns(path: Path, default: tuple[str, ...], required: tuple[str, ...]) -> list[str] | None:
    """Devolve os nomes do cabeçalho, ou None se o arquivo não tem cabeçalho.

    Os arquivos oficiais não trazem cabeçalho; nesse caso vale a ordem do schema.txt.
    """
    with open_csv(path) as handle:
        first = handle.readline().decode("utf-8", errors="replace").strip()
    fields = [field.strip() for field in first.split(",")]
    if fields and fields[0] == "machine_id":
        missing = [column for column in required if column not in fields]
        if missing:
            raise SchemaError(f"Colunas ausentes em {Path(path).name}: {missing}")
        return fields
    if len(fields) < len(default):
        raise SchemaError(
            f"{Path(path).name} tem {len(fields)} campos sem cabeçalho; esperado {len(default)}."
        )
    return None


def _read_csv(path: Path, default: tuple[str, ...], required: tuple[str, ...], chunksize: int | None):
    header = sniff_columns(path, default, required)
    kwargs = {"usecols": list(required), "chunksize": chunksize}
    if header is None:
        kwargs.update(header=None, names=list(default))
    handle_cm = open_csv(path)
    return handle_cm, kwargs


def read_machine_meta(path: Path) -> pd.DataFrame:
    header = sniff_columns(Path(path), META_COLUMNS, REQUIRED_META)
    wanted = list(REQUIRED_META)
    if header is None or "time_stamp" in header:
        wanted.append("time_stamp")
    kwargs = {"usecols": wanted}
    if header is None:
        kwargs.update(header=None, names=list(META_COLUMNS))
    with open_csv(Path(path)) as handle:
        frame = pd.read_csv(handle, dtype={"machine_id": "string", "status": "string"}, **kwargs)
    frame["cpu_num"] = pd.to_numeric(frame["cpu_num"], errors="raise")
    if "time_stamp" in frame.columns:
        frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    else:
        frame["time_stamp"] = np.nan
    frame["machine_id"] = frame["machine_id"].astype(str)
    return frame


def iter_usage_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Blocos de machine_usage com as colunas exigidas, tipos numéricos e ids como texto."""
    cm, kwargs = _read_csv(Path(path), USAGE_COLUMNS, REQUIRED_USAGE, chunk_rows)
    with cm as handle:
        for chunk in pd.read_csv(handle, dtype={"machine_id": "string"}, **kwargs):
            chunk = chunk.loc[chunk["machine_id"] != "machine_id"]
            chunk["time_stamp"] = pd.to_numeric(chunk["time_stamp"], errors="raise")
            for column in VALUE_COLUMNS:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            yield chunk.reset_index(drop=True)


def mask_invalid(chunk: pd.DataFrame) -> dict[str, int]:
    """Marca como ausente o que está fora de [0, 100] (inclui -1 e 101 em disco).

    Devolve, por coluna, quantos valores presentes foram invalidados.
    """
    low, high = VALID_RANGE
    counts: dict[str, int] = {}
    for column in VALUE_COLUMNS:
        values = chunk[column]
        invalid = values.notna() & ((values < low) | (values > high))
        counts[column] = int(invalid.sum())
        chunk.loc[invalid, column] = np.nan
    return counts


def detect_granularity(path: Path, sample_rows: int, min_conformity: float, chunk_rows: int = 500_000) -> dict:
    """Passo predominante por máquina nas primeiras `sample_rows` linhas."""
    seen = 0
    frames = []
    for chunk in iter_usage_chunks(path, chunk_rows):
        frames.append(chunk[["machine_id", "time_stamp"]])
        seen += len(chunk)
        if seen >= sample_rows:
            break
    if not frames:
        raise SchemaError(f"{Path(path).name} não tem registros.")
    sample = pd.concat(frames, ignore_index=True).sort_values(["machine_id", "time_stamp"], kind="mergesort")
    diffs = sample.groupby("machine_id")["time_stamp"].diff().dropna()
    diffs = diffs[diffs > 0].round(6)
    if diffs.empty:
        return {"step_seconds": None, "conformity": 0.0, "n_intervals": 0, "sample_rows": int(seen), "regular": False}
    counts = diffs.value_counts()
    step = float(counts.index[0])
    conformity = float(counts.iloc[0] / len(diffs))
    return {
        "step_seconds": step,
        "conformity": conformity,
        "n_intervals": int(len(diffs)),
        "sample_rows": int(seen),
        "regular": bool(conformity >= min_conformity),
    }


def resolve_grid_step(grid_step, detected: dict) -> tuple[float, list[str]]:
    """`auto` usa o passo detectado; um número explícito vale sempre, com aviso se a detecção falhou."""
    warnings: list[str] = []
    if grid_step == "auto":
        if not detected["regular"]:
            raise IrregularGranularityError(
                "Sem intervalo de amostragem predominante "
                f"(conformidade {detected['conformity']:.3f}); defina grid_step explícito."
            )
        return float(detected["step_seconds"]), warnings
    step = float(grid_step)
    if step <= 0:
        raise ValueError("grid_step deve ser positivo.")
    if not detected["regular"]:
        warnings.append(
            "Granularidade irregular ou não detectada; usado o passo configurado "
            f"{step} s (conformidade {detected['conformity']:.3f})."
        )
    return step, warnings


def select_machines(meta: pd.DataFrame, ids: list[str] | None, sample_size: int | None, seed: int) -> list[str] | None:
    """Lista explícita ou amostra reprodutível das máquinas de machine_meta; None = todas."""
    if ids:
        return sorted(str(item) for item in ids)
    if sample_size:
        universe = sorted(meta["machine_id"].astype(str).unique())
        if sample_size >= len(universe):
            return universe
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(universe), size=int(sample_size), replace=False)
        return sorted(universe[i] for i in chosen)
    return None


class _Stats:
    def __init__(self) -> None:
        self.rows_read = 0
        self.rows_in_window = 0
        self.rows_after_machine_filter = 0
        self.duplicates = 0
        self.invalid = {column: 0 for column in VALUE_COLUMNS}
        self.missing = {column: 0 for column in VALUE_COLUMNS}
        self.min_ts = np.inf
        self.max_ts = -np.inf
        self.sorted_chunks = True


def build_grid_table(
    usage_path: Path,
    *,
    step_seconds: float,
    chunk_rows: int,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    machines: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Média por (máquina, intervalo), acumulando somas e contagens entre blocos."""
    stats = _Stats()
    parts: list[pd.DataFrame] = []
    machine_filter = set(machines) if machines is not None else None

    def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = pd.concat(frames)
        return merged.groupby(level=["machine_id", "bin"], sort=True).sum()

    for chunk in iter_usage_chunks(usage_path, chunk_rows):
        stats.rows_read += len(chunk)
        if not chunk["time_stamp"].is_monotonic_increasing:
            stats.sorted_chunks = False
        if start_seconds is not None:
            chunk = chunk.loc[chunk["time_stamp"] >= start_seconds]
        if end_seconds is not None:
            chunk = chunk.loc[chunk["time_stamp"] < end_seconds]
        stats.rows_in_window += len(chunk)
        if machine_filter is not None:
            chunk = chunk.loc[chunk["machine_id"].isin(machine_filter)]
        stats.rows_after_machine_filter += len(chunk)
        if chunk.empty:
            continue
        duplicated = chunk.duplicated(["machine_id", "time_stamp"], keep="first")
        stats.duplicates += int(duplicated.sum())
        chunk = chunk.loc[~duplicated].copy()
        for column, count in mask_invalid(chunk).items():
            stats.invalid[column] += count
        for column in VALUE_COLUMNS:
            stats.missing[column] += int(chunk[column].isna().sum())
        stats.min_ts = min(stats.min_ts, float(chunk["time_stamp"].min()))
        stats.max_ts = max(stats.max_ts, float(chunk["time_stamp"].max()))
        chunk["bin"] = np.floor(chunk["time_stamp"] / step_seconds).astype("int64")
        grouped = chunk.groupby(["machine_id", "bin"], sort=False)[list(VALUE_COLUMNS)]
        sums = grouped.sum(min_count=1).fillna(0.0).add_suffix("__sum")
        counts = grouped.count().add_suffix("__count")
        parts.append(pd.concat([sums, counts], axis=1))
        if len(parts) >= 16:
            parts = [combine(parts)]

    if not parts:
        raise ValueError("Nenhum registro de machine_usage restou depois de janela e filtro de máquinas.")
    total = combine(parts).reset_index()
    table = pd.DataFrame({"machine_id": total["machine_id"].astype(str), "bin": total["bin"]})
    for series, column in SERIES_COLUMNS.items():
        count = total[f"{column}__count"]
        table[series] = np.where(count > 0, total[f"{column}__sum"] / count.where(count > 0, 1), np.nan)
    table = table.sort_values(["machine_id", "bin"], kind="mergesort").reset_index(drop=True)
    return table, _profile_from_stats(stats, table, step_seconds)


def _profile_from_stats(stats: _Stats, table: pd.DataFrame, step_seconds: float) -> dict:
    kept = max(stats.rows_after_machine_filter - stats.duplicates, 1)
    bins = table["bin"]
    expected_per_machine = int(bins.max() - bins.min() + 1)
    coverage = table.groupby("machine_id")["bin"].nunique() / expected_per_machine
    return {
        "step_seconds": step_seconds,
        "n_machines": int(table["machine_id"].nunique()),
        "n_records_read": int(stats.rows_read),
        "n_records_in_window": int(stats.rows_in_window),
        "n_records_used": int(stats.rows_after_machine_filter - stats.duplicates),
        "n_grid_points": int(len(table)),
        "period_seconds": [float(stats.min_ts), float(stats.max_ts)],
        "period_bins": [int(bins.min()), int(bins.max())],
        "duplicate_records": int(stats.duplicates),
        "invalid_values": dict(stats.invalid),
        "missing_values": dict(stats.missing),
        "missing_fraction": {column: stats.missing[column] / kept for column in VALUE_COLUMNS},
        "machine_coverage": {
            "min": float(coverage.min()),
            "mean": float(coverage.mean()),
            "max": float(coverage.max()),
        },
        "input_time_sorted_within_chunks": bool(stats.sorted_chunks),
    }


def cache_key(parameters: dict) -> str:
    import hashlib

    payload = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def load_or_build_grid(
    usage_path: Path,
    meta_path: Path,
    *,
    cache_dir: Path,
    grid_step,
    chunk_rows: int,
    granularity_sample_rows: int,
    granularity_min_conformity: float,
    start_seconds: float | None,
    end_seconds: float | None,
    machine_ids: list[str] | None,
    machine_sample_size: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tabela da grade, metadados das máquinas e perfil, com cache por hash e parâmetros."""
    require_files(usage_path, meta_path)
    hashes = {"machine_usage": file_sha256(usage_path), "machine_meta": file_sha256(meta_path)}
    meta = read_machine_meta(meta_path)
    machines = select_machines(meta, machine_ids, machine_sample_size, seed)
    detected = detect_granularity(usage_path, granularity_sample_rows, granularity_min_conformity)
    step, warnings = resolve_grid_step(grid_step, detected)
    key = cache_key(
        {
            "hashes": hashes,
            "step": step,
            "start": start_seconds,
            "end": end_seconds,
            "machines": machines,
        }
    )
    cache_dir = Path(cache_dir)
    table_path = cache_dir / f"grid_{key}.csv.gz"
    profile_path = cache_dir / f"grid_{key}.profile.json"
    if table_path.is_file() and profile_path.is_file():
        table = pd.read_csv(table_path, dtype={"machine_id": "string"})
        table["machine_id"] = table["machine_id"].astype(str)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["cache"] = {"hit": True, "key": key}
        return table, meta, profile

    table, profile = build_grid_table(
        usage_path,
        step_seconds=step,
        chunk_rows=chunk_rows,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        machines=machines,
    )
    profile.update(
        {
            "granularity": detected,
            "warnings": warnings,
            "file_sha256": hashes,
            "machine_selection": {
                "explicit_ids": bool(machine_ids),
                "sample_size": machine_sample_size,
                "seed": seed,
                "n_selected": None if machines is None else len(machines),
            },
            "window_seconds": [start_seconds, end_seconds],
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False, compression="gzip")
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    profile["cache"] = {"hit": False, "key": key}
    return table, meta, profile
