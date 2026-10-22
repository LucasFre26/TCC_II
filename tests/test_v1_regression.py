"""A V1 continua reproduzindo o results/v1/metrics.csv preservado."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from energy_prediction.experiments.run_v1 import execute

ROOT = Path(__file__).resolve().parents[1]
PRESERVED = ROOT / "results/v1/metrics.csv"
GENAI = ROOT / "data/clusterdata-master/cluster-trace-v2026-GenAI"


@pytest.mark.skipif(
    not (GENAI / "pod_gpu_duty_cycle_anon.tar.gz").is_file()
    or not (GENAI / "pod_memory_util_anon.tar.gz").is_file()
    or not PRESERVED.is_file(),
    reason="Arquivos GenAI ou results/v1/metrics.csv ausentes; regressão da V1 não verificada.",
)
def test_v1_output_matches_preserved_metrics(tmp_path: Path):
    config = yaml.safe_load((ROOT / "config/v1.yaml").read_text(encoding="utf-8"))
    config["gpu_archive"] = str(ROOT / config["gpu_archive"])
    config["memory_archive"] = str(ROOT / config["memory_archive"])
    config["output_dir"] = str(tmp_path / "out_v1")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "v1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    before = PRESERVED.read_bytes()
    execute(config_path)

    regenerated = pd.read_csv(tmp_path / "out_v1/metrics.csv")
    preserved = pd.read_csv(PRESERVED)
    assert list(regenerated.columns) == list(preserved.columns)
    pd.testing.assert_frame_equal(regenerated, preserved, check_exact=False, rtol=1e-9, atol=1e-12)
    assert PRESERVED.read_bytes() == before
