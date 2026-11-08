"""V3, grupo 1: registro fechado, extra v3, configuração, proteção dos resultados e dispositivo."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from energy_prediction.experiments.run_v2 import load_config as load_v2_config
from energy_prediction.experiments.run_v3 import execute
from energy_prediction.experiments.v3_config import V3ConfigError, load_v3_config, nested_enabled, validate_config
from energy_prediction.models.registry import ALLOWED_MODELS, UnknownModelError, validate_model_names
from energy_prediction.models.v3_registry import FAMILY_MODELS, UnknownV3ModelError, validate_family_models
from energy_prediction.runtime import device as device_module
from energy_prediction.runtime.device import DeviceError, resolve_device
from energy_prediction.runtime.output import FamilyAlreadyDoneError, ProtectedOutputError, RunDirectory, guard_output_root
from v3_helpers import ROOT, smoke_config, tree_bytes, write_project

SRC = ROOT / "src/energy_prediction"
V3_NAMES = {name for names in FAMILY_MODELS.values() for name in names} - set(ALLOWED_MODELS)


def test_v2_registry_still_rejects_every_v3_model():
    assert ALLOWED_MODELS == ("persistence", "ridge", "xgboost")
    for name in sorted(V3_NAMES) + ["lstm"]:
        with pytest.raises(UnknownModelError):
            validate_model_names(["persistence", name])


def test_v3_family_sets_are_closed():
    with pytest.raises(UnknownV3ModelError, match="lstm"):
        validate_family_models("neural", ["dlinear", "lstm"])
    with pytest.raises(UnknownV3ModelError):
        validate_family_models("residual", ["dlinear"])
    assert validate_family_models("neural", ["dlinear", "tcn", "nbeatsx"]) == ["dlinear", "tcn", "nbeatsx"]


def test_v3_extra_is_pinned_and_outside_test_extra():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    v3 = re.search(r"^v3 = \[(.*)\]$", project, re.M).group(1)
    test = re.search(r"^test = \[(.*)\]$", project, re.M).group(1)
    assert re.search(r'"torch==[\d.]+"', v3) and re.search(r'"statsmodels==[\d.]+"', v3)
    assert "torch" not in test and "statsmodels" not in test


def test_v1_v2_and_light_v3_modules_do_not_import_torch():
    code = (
        "import sys\n"
        "import energy_prediction.experiments.run_v1, energy_prediction.experiments.run_v2\n"
        "import energy_prediction.experiments.v3_config, energy_prediction.evaluation.nested\n"
        "import energy_prediction.energy.fan, energy_prediction.energy.integrate, energy_prediction.models.delta\n"
        "import energy_prediction.models.neural.windows, energy_prediction.experiments.v3_report\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")})
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("name", ["v3.yaml", "v3-smoke.yaml", "v3-smoke-real.yaml"])
def test_shipped_v3_configs_validate(name):
    config = load_v3_config(ROOT / "config" / name)
    assert nested_enabled(config)
    assert list(config["features"]["ablation"]) == ["A", "B", "C"]


def test_unknown_keys_are_rejected_at_any_level():
    config = smoke_config()
    with pytest.raises(V3ConfigError, match="extra"):
        validate_config({**config, "extra": 1})
    nested = smoke_config()
    nested["validation"]["nested"]["nova"] = 1
    with pytest.raises(V3ConfigError, match="validation.nested.nova"):
        validate_config(nested)
    neural = smoke_config()
    neural["families"]["neural"]["models"] = ["dlinear", "lstm"]
    with pytest.raises(UnknownV3ModelError, match="lstm"):
        validate_config(neural)


def test_v2_config_without_nested_key_stays_valid():
    v2 = load_v2_config(ROOT / "config/v2-full.yaml")
    assert "nested" not in v2["validation"]
    config = smoke_config()
    del config["validation"]["nested"]
    assert not nested_enabled(validate_config(config))


def test_output_guard_protects_preserved_results(tmp_path: Path):
    root = tmp_path / "proj"
    for relative in ("results/v1", "results/v2", "results/v2-full", "results/v2-full/main", "results"):
        with pytest.raises(ProtectedOutputError):
            guard_output_root(root, root / relative)
    assert guard_output_root(root, root / "results/v3") == (root / "results/v3").resolve()


def test_run_into_v2_full_fails_without_touching_bytes(tmp_path: Path):
    config_path = write_project(tmp_path, smoke_config(run_root="results/v2-full"))
    preserved = tmp_path / "proj/results/v2-full/main"
    preserved.mkdir(parents=True)
    (preserved / "metrics.csv").write_text("model,mae\nxgboost,4.334\n")
    before = tree_bytes(tmp_path / "proj/results")
    with pytest.raises(ProtectedOutputError):
        execute(config_path, family="integrity", run_id="x")
    assert tree_bytes(tmp_path / "proj/results") == before


def test_family_write_is_atomic_and_keeps_other_families(tmp_path: Path):
    run = RunDirectory(tmp_path, tmp_path / "results/v3", "r1")
    with run.family("integrity") as out:
        (out / "population_audit.json").write_text("{}")
    with pytest.raises(RuntimeError):
        with run.family("neural") as out:
            (out / "metrics.csv").write_text("parcial")
            raise RuntimeError("falha no meio da família")
    assert run.is_done("integrity")
    assert not run.family_path("neural").exists()
    assert not any(p.name.startswith(".neural") for p in run.path.iterdir())
    with pytest.raises(FamilyAlreadyDoneError):
        with run.family("integrity"):
            pass


def test_auto_without_cuda_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = resolve_device("auto")
    assert record["effective"] == "cpu" and record["cuda_available"] is False
    assert "CUDA indisponível" in record["note"]
    assert resolve_device("cpu")["effective"] == "cpu"
    with pytest.raises(DeviceError):
        resolve_device("cuda")
    with pytest.raises(DeviceError):
        resolve_device("gpu")


def test_without_torch_auto_is_cpu(monkeypatch):
    monkeypatch.setattr(device_module, "torch_installed", lambda: False)
    assert resolve_device("auto")["effective"] == "cpu"
    with pytest.raises(DeviceError):
        resolve_device("cuda")


def test_no_code_path_installs_drivers_or_packages():
    pattern = re.compile(r"apt-get|apt install|pip install|nvidia-smi|pacman|dnf |yum |modprobe|cuda-toolkit")
    v3_files = [
        *SRC.joinpath("runtime").glob("*.py"),
        *SRC.joinpath("experiments/v3_families").glob("*.py"),
        SRC / "experiments/run_v3.py",
        *SRC.joinpath("models/neural").glob("*.py"),
    ]
    for path in v3_files:
        text = path.read_text(encoding="utf-8")
        assert not pattern.search(text), path
        for call in re.findall(r"subprocess\.run\(\s*\[([^\]]*)\]", text):
            assert call.strip().startswith('"git"'), (path, call)


def test_run_record_has_seed_hash_and_versions(tmp_path: Path):
    config_path = write_project(tmp_path, smoke_config())
    manifest = execute(config_path, family="integrity", run_id="rec")
    assert manifest["seed"] == 42 and len(manifest["config_sha256"]) == 64
    assert manifest["device"]["effective"] == "cpu"
    assert manifest["data"]["source"] == "synthetic"
    assert set(manifest["versions"]) >= {"python", "numpy", "xgboost", "torch", "statsmodels"}
    stored = yaml.safe_load((tmp_path / "proj/results/v3-smoke/rec/manifest.json").read_text())
    assert list(stored["families"]) == ["integrity"]
