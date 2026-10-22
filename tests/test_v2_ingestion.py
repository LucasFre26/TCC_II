"""Ingestão do Alibaba v2018 com arquivos sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.data.alibaba import (
    IrregularGranularityError,
    SchemaError,
    build_grid_table,
    detect_granularity,
    iter_usage_chunks,
    load_or_build_grid,
    mask_invalid,
    read_machine_meta,
    require_files,
    resolve_grid_step,
    select_machines,
)
from energy_prediction.data.load_genai import MissingArchiveError, file_sha256
from synthetic import (
    make_meta_frame,
    make_series,
    make_usage_frame,
    write_meta_archive,
    write_usage_archive,
)

MACHINES = {"m_1": 32, "m_2": 96, "m_3": 96}


def test_make_series_is_reproducible_by_seed():
    left = make_series(50, seed=3)
    right = make_series(50, seed=3)
    other = make_series(50, seed=4)
    pd.testing.assert_frame_equal(left, right)
    assert not left["cpu"].equals(other["cpu"])


def test_generated_archives_are_readable(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=5)
    path = write_usage_archive(tmp_path / "machine_usage.tar.gz", usage)
    meta_path = write_meta_archive(tmp_path / "machine_meta.tar.gz", make_meta_frame(MACHINES))
    chunks = list(iter_usage_chunks(path, chunk_rows=4))
    assert sum(len(chunk) for chunk in chunks) == len(usage)
    meta = read_machine_meta(meta_path)
    assert dict(zip(meta["machine_id"], meta["cpu_num"])) == MACHINES


def test_missing_archive_error_mentions_download(tmp_path: Path):
    with pytest.raises(MissingArchiveError, match="fetchData.sh"):
        require_files(tmp_path / "ausente.tar.gz")


def test_missing_column_names_the_column_and_file(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=3).drop(columns=["net_in"])
    path = write_usage_archive(tmp_path / "machine_usage.tar.gz", usage, header=True)
    with pytest.raises(SchemaError, match="net_in"):
        list(iter_usage_chunks(path, 10))


def test_headerless_file_with_too_few_fields_is_rejected(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=3).iloc[:, :5]
    path = write_usage_archive(tmp_path / "machine_usage.tar.gz", usage, header=False)
    with pytest.raises(SchemaError):
        list(iter_usage_chunks(path, 10))


def test_raw_archive_hash_is_unchanged_by_reading(tmp_path: Path):
    usage_path = write_usage_archive(tmp_path / "u.tar.gz", make_usage_frame(MACHINES, n_steps=6))
    before = file_sha256(usage_path)
    build_grid_table(usage_path, step_seconds=10.0, chunk_rows=5)
    detect_granularity(usage_path, 100, 0.8)
    assert file_sha256(usage_path) == before


def test_invalid_values_become_missing_and_are_counted():
    frame = pd.DataFrame(
        {
            "cpu_util_percent": [0.0, 100.0, 50.0],
            "mem_util_percent": [1.0, 2.0, 3.0],
            "mem_gps": [1.0, 2.0, 3.0],
            "net_in": [1.0, 2.0, 3.0],
            "net_out": [1.0, 2.0, 3.0],
            "disk_io_percent": [-1.0, 101.0, 0.0],
        }
    )
    counts = mask_invalid(frame)
    assert counts["disk_io_percent"] == 2
    assert frame["disk_io_percent"].isna().tolist() == [True, True, False]
    assert frame["cpu_util_percent"].tolist() == [0.0, 100.0, 50.0]
    disk_boundaries = pd.DataFrame({"disk_io_percent": [0.0, 100.0]})
    for column in frame.columns.difference(["disk_io_percent"]):
        disk_boundaries[column] = 1.0
    assert mask_invalid(disk_boundaries)["disk_io_percent"] == 0


def test_out_of_order_input_gives_same_grid(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=12)
    shuffled = usage.sample(frac=1.0, random_state=1).reset_index(drop=True)
    ordered_path = write_usage_archive(tmp_path / "a.tar.gz", usage)
    shuffled_path = write_usage_archive(tmp_path / "b.tar.gz", shuffled)
    left, _ = build_grid_table(ordered_path, step_seconds=10.0, chunk_rows=7)
    right, _ = build_grid_table(shuffled_path, step_seconds=10.0, chunk_rows=7)
    pd.testing.assert_frame_equal(left, right)
    assert left.groupby("machine_id")["bin"].apply(lambda s: s.is_monotonic_increasing).all()


def test_duplicate_key_is_counted_and_first_row_kept(tmp_path: Path):
    usage = make_usage_frame({"m_1": 32}, n_steps=4, async_offsets=False)
    duplicate = usage.iloc[[1]].copy()
    duplicate["cpu_util_percent"] = 99.0
    with_duplicate = pd.concat([usage, duplicate], ignore_index=True)
    path = write_usage_archive(tmp_path / "u.tar.gz", with_duplicate)
    table, profile = build_grid_table(path, step_seconds=10.0, chunk_rows=100)
    assert profile["duplicate_records"] == 1
    assert table.loc[table["bin"] == 1, "cpu"].iloc[0] == pytest.approx(usage["cpu_util_percent"].iloc[1])


def test_granularity_regular_and_irregular(tmp_path: Path):
    regular = write_usage_archive(tmp_path / "r.tar.gz", make_usage_frame(MACHINES, n_steps=30, step=10.0))
    found = detect_granularity(regular, 10_000, 0.8)
    assert found["step_seconds"] == pytest.approx(10.0)
    assert found["conformity"] == pytest.approx(1.0)
    assert found["regular"]

    rng = np.random.default_rng(0)
    irregular_frame = make_usage_frame({"m_1": 32}, n_steps=60, async_offsets=False)
    irregular_frame["time_stamp"] = np.cumsum(rng.uniform(1, 50, len(irregular_frame)))
    irregular = write_usage_archive(tmp_path / "i.tar.gz", irregular_frame)
    found_irregular = detect_granularity(irregular, 10_000, 0.8)
    assert not found_irregular["regular"]
    with pytest.raises(IrregularGranularityError):
        resolve_grid_step("auto", found_irregular)
    step, warnings = resolve_grid_step(60, found_irregular)
    assert step == 60.0 and warnings


def test_detected_step_is_not_assumed_to_be_five_minutes(tmp_path: Path):
    path = write_usage_archive(tmp_path / "u.tar.gz", make_usage_frame(MACHINES, n_steps=20, step=7.0))
    assert detect_granularity(path, 1000, 0.8)["step_seconds"] == pytest.approx(7.0)


def test_missing_values_and_machine_coverage(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=20, gap=("m_2", 5, 9))
    usage.loc[usage["machine_id"] == "m_1", "net_in"] = np.nan
    path = write_usage_archive(tmp_path / "u.tar.gz", usage)
    table, profile = build_grid_table(path, step_seconds=10.0, chunk_rows=1000)
    assert profile["machine_coverage"]["min"] < 1.0
    assert profile["machine_coverage"]["max"] == pytest.approx(1.0)
    assert profile["missing_values"]["net_in"] == 20
    assert 0 < profile["missing_fraction"]["net_in"] < 1
    assert len(table) == 20 * 3 - 5


def test_time_window_and_machine_subset(tmp_path: Path):
    usage = make_usage_frame(MACHINES, n_steps=30)
    path = write_usage_archive(tmp_path / "u.tar.gz", usage)
    table, profile = build_grid_table(
        path, step_seconds=10.0, chunk_rows=1000, start_seconds=100.0, end_seconds=200.0, machines=["m_1", "m_3"]
    )
    assert set(table["machine_id"]) == {"m_1", "m_3"}
    assert table["bin"].min() >= 10 and table["bin"].max() < 20
    assert profile["n_records_read"] == len(usage)


def test_seeded_machine_sample_is_reproducible():
    meta = make_meta_frame({f"m_{i}": 32 for i in range(50)})
    first = select_machines(meta, None, 10, seed=7)
    second = select_machines(meta, None, 10, seed=7)
    other = select_machines(meta, None, 10, seed=8)
    assert first == second and len(first) == 10
    assert first != other
    assert select_machines(meta, ["m_2", "m_1"], 10, 7) == ["m_1", "m_2"]
    assert select_machines(meta, None, None, 7) is None


def _load(tmp_path: Path, cache: Path, **overrides):
    usage_path = tmp_path / "u.tar.gz"
    meta_path = tmp_path / "m.tar.gz"
    if not usage_path.exists():
        write_usage_archive(usage_path, make_usage_frame(MACHINES, n_steps=25))
        write_meta_archive(meta_path, make_meta_frame(MACHINES))
    params = dict(
        cache_dir=cache,
        grid_step="auto",
        chunk_rows=100,
        granularity_sample_rows=10_000,
        granularity_min_conformity=0.8,
        start_seconds=None,
        end_seconds=None,
        machine_ids=None,
        machine_sample_size=None,
        seed=1,
    )
    params.update(overrides)
    return load_or_build_grid(usage_path, meta_path, **params)


def test_cache_is_reused_and_invalidated_by_parameters(tmp_path: Path):
    cache = tmp_path / "processed"
    first_table, _, first_profile = _load(tmp_path, cache)
    second_table, _, second_profile = _load(tmp_path, cache)
    assert first_profile["cache"]["hit"] is False
    assert second_profile["cache"]["hit"] is True
    pd.testing.assert_frame_equal(first_table, second_table, check_exact=False)
    _, _, third_profile = _load(tmp_path, cache, start_seconds=50.0)
    assert third_profile["cache"]["hit"] is False
    assert third_profile["cache"]["key"] != first_profile["cache"]["key"]
    assert len(list(cache.glob("grid_*.csv.gz"))) == 2


def test_profile_has_required_fields(tmp_path: Path):
    _, _, profile = _load(tmp_path, tmp_path / "processed")
    for key in (
        "n_machines",
        "n_records_used",
        "period_seconds",
        "step_seconds",
        "granularity",
        "invalid_values",
        "missing_values",
        "duplicate_records",
        "machine_coverage",
        "file_sha256",
        "input_time_sorted_within_chunks",
    ):
        assert key in profile
    assert profile["n_machines"] == 3
    assert set(profile["file_sha256"]) == {"machine_usage", "machine_meta"}
    import json

    json.dumps(profile)
