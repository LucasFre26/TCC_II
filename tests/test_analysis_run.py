"""Análise expandida sobre a corrida real da V3 (pulada quando a pasta não existe).

Não treina modelo: a camada do cache de grade fica desligada, e a análise só lê artefatos.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.analysis import export, run
from energy_prediction.analysis.config import load_config, resolve_paths
from energy_prediction.analysis.pipeline import compute

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/v3/20261104T232620Z-5b659dad"
pytestmark = pytest.mark.skipif(not (RUN / "manifest.json").is_file() or not (ROOT / "results/v2-full/main/predictions.csv").is_file(), reason="corrida real da V3 ausente")


def _cfg(**changes):
    cfg = load_config(ROOT / "config/v3-analysis.yaml")
    cfg["dataset"]["enabled"] = False
    cfg.update(changes)
    return cfg


@pytest.fixture(scope="module")
def result():
    cfg = _cfg()
    return compute(cfg, resolve_paths(cfg, ROOT))


@pytest.fixture(scope="module")
def published(result, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis") / "expanded-analysis"
    return out, run.publish(result, out)


def test_every_integrity_check_passes(result):
    names = {c["check"] for c in result.checks}
    assert all(c["passed"] for c in result.checks)
    assert {"reproducao_bootstrap_pre_registrado", "reproducao_energia", "reproducao_pinball", "persistencia_identica_v2_v3"} <= names
    assert any("Camada do cache" in w for w in result.warnings)


def test_published_indicators_are_reproduced(result):
    m = result.tables["model_metrics"]
    test = m[(m["split"] == "test_final") & (m["population"] == "main") & (m["analysis_set"] == "main") & (m["mode"] == "direct")]
    v2 = pd.read_csv(ROOT / "results/v2-full/main/metrics.csv")
    stored = v2[(v2["model"] == "xgboost") & (v2["config"] == "A") & (v2["horizon_steps"] == 1) & (v2["split"] == "test_final")]["mae"].iloc[0]
    ours = test[(test["model"] == "xgboost_v2") & (test["config"] == "A") & (test["horizon_steps"] == 1)]["mae_pp"].iloc[0]
    assert ours == pytest.approx(stored, abs=1e-12)
    residual = pd.read_csv(RUN / "residual/metrics.csv")
    stored = residual[(residual["model"] == "hybrid") & (residual["config"] == "A") & (residual["horizon_steps"] == 6) & (residual["split"] == "test_final")]["mae"].iloc[0]
    assert test[(test["model"] == "hybrid") & (test["config"] == "A") & (test["horizon_steps"] == 6)]["mae_pp"].iloc[0] == pytest.approx(stored, abs=1e-12)
    boot = result.tables["bootstrap_intervals"]
    row = boot[(boot["analysis_type"] == "pre_registrada") & (boot["comparison"] == "new_vs_v2_xgboost_A") & (boot["model_a"] == "hybrid") & (boot["horizon_steps"] == 1) & (boot["block_steps"] == 12)].iloc[0]
    assert row["reproduction_max_abs_diff"] < 1e-12 and row["significant"] == row["stored_significant"]


def test_prediction_alignment_on_target_time(result):
    p = result.tables["predictions"]
    assert (p["target_index"] == p["origin_index"] + p["horizon_steps"]).all()
    pers = p[(p["model"] == "persistence") & (p["population"] == "main") & (p["horizon_steps"] == 1) & (p["split"] == "test_final")].set_index("origin_index")
    # o observado da origem t (alvo t+1) é a CPU na origem t+1, que a persistência usa como previsão
    shared = pers.index[:-1]
    assert (pers.loc[shared, "y_obs_cpu_pct"].to_numpy() == pers.loc[shared + 1, "y_pred_cpu_pct"].to_numpy()).all()
    assert p[(p["model"] == "persistence") & (p["population"] == "main") & (p["analysis_set"] == "main")].duplicated(["split", "horizon_steps", "origin_index"]).sum() == 0


def test_csv_exports_are_documented_and_not_empty(published, result):
    out, outputs = published
    files = sorted((out / "csv").glob("*.csv"))
    assert len(files) == outputs["n_csv"] == len(result.tables)
    dictionary = pd.read_csv(out / "metadata/data_dictionary.csv")
    for path in files:
        frame = pd.read_csv(path, nrows=5)
        assert len(frame) > 0, path.name
        documented = set(dictionary.loc[dictionary["csv_file"] == path.name, "column"])
        assert set(frame.columns) == documented, path.name
    assert dictionary["nature"].isin({"identificador", "observado", "previsto", "erro", "métrica", "inferência", "modelado", "registro", "parâmetro"}).all()
    assert not export.undocumented(result.tables)


def test_figures_have_all_formats_and_data(published):
    out, outputs = published
    generated = [r for r in outputs["figures"] if r.get("generated")]
    assert 20 <= len(generated) <= 35
    for r in generated:
        for fmt in ("png", "svg", "pdf"):
            assert (out / "figures" / fmt / f"{r['name']}.{fmt}").stat().st_size > 0
        assert (out / "figures/data" / f"{r['name']}.csv").is_file()
        assert r["question"] and r["caption"]
    skipped = [r for r in outputs["figures"] if not r.get("generated")]
    assert all("tabelas ausentes" in r["reason"] for r in skipped)


def test_html_report_is_offline_and_complete(published):
    out, _ = published
    page = (out / "interactive/relatorio.html").read_text(encoding="utf-8")
    # URLs dentro do código embutido do plotly.js (logotipo, atribuição de mapas) não são carregadas;
    # o que importa é nenhuma tag do documento buscar recurso externo.
    assert not re.search(r"<script[^>]+src=", page)
    markup = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", markup)
    assert not re.search(r"<link[^>]+http", markup)
    for anchor in ("visao-geral", "dataset", "comparacao", "temporal", "picos", "estatistica", "neural", "ablacao", "probabilistico", "energia", "limitacoes", "conclusoes"):
        assert f'id="{anchor}"' in page
    assert "Plotly" in page and "PUE não foi calculado" in page


def test_latex_tables_are_consistent_and_article_untouched(published):
    out, _ = published
    tables = sorted((out / "latex/tabelas").glob("*.tex"))
    assert len(tables) >= 8
    for path in tables:
        text = path.read_text(encoding="utf-8")
        assert "\\caption{" in text and "\\label{tab:" in text and "Fonte: Autoria própria." in text
        spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", text).group(1)
        n_cols = spec.count("c") + spec.count("l")
        body = [line for line in text.splitlines() if line.rstrip().endswith("\\\\") and "&" in line]
        assert body and all(line.count("&") == n_cols - 1 for line in body), path.name
    assert (out / "latex/proposta-secao-resultados.md").is_file()
    assert "booktabs" not in "".join(p.read_text() for p in tables)


def test_markdown_numbers_match_metrics(published, result):
    out, _ = published
    text = (out / "reports/scientific-analysis.md").read_text(encoding="utf-8")
    m = result.tables["model_metrics"]
    value = m[(m["split"] == "test_final") & (m["model"] == "xgboost_v2") & (m["config"] == "A") & (m["population"] == "main") & (m["horizon_steps"] == 6)]["mae_pp"].iloc[0]
    row = next(line for line in text.splitlines() if line.startswith("| XGBoost (V2) |"))
    assert f"{value:.2f}".replace(".", ",") == row.split("|")[-2].strip()
    for phrase in ("já tinha sido inspecionado na V2", "não demonstradas", "PUE não foi calculado", "sem garantia teórica"):
        assert phrase in text


def test_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    fake_run = tmp_path / "results/v3" / RUN.name
    fake_run.mkdir(parents=True)
    for item in RUN.iterdir():
        if item.name not in ("expanded-analysis", ".expanded-analysis.tmp"):
            (fake_run / item.name).symlink_to(item)
    cfg = _cfg(run_dir=str(fake_run), v2_dir=str(ROOT / "results/v2-full"), substitute=str(ROOT / "config/specpower_substitute.yaml"))
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    def broken(result, tmp):
        (tmp / "csv").mkdir(parents=True)
        (tmp / "csv/parcial.csv").write_text("a\n1\n")
        raise RuntimeError("falha simulada na publicação")

    monkeypatch.setattr(run, "publish", broken)
    with pytest.raises(RuntimeError, match="simulada"):
        run.execute(config_path, project_root=ROOT)
    assert not (fake_run / "expanded-analysis").exists() and not (fake_run / ".expanded-analysis.tmp").exists()


def test_missing_optional_artifact_becomes_warning(tmp_path):
    from energy_prediction.analysis.loader import load_run

    fake = tmp_path / "run"
    fake.mkdir()
    for item in RUN.iterdir():
        if item.name not in ("energy", "expanded-analysis"):
            (fake / item.name).symlink_to(item)
    (fake / "energy").mkdir()
    cfg = _cfg(run_dir=str(fake), v2_dir=str(ROOT / "results/v2-full"))
    art = load_run(resolve_paths(cfg, ROOT))
    assert any("aggregate_effect.csv" in w for w in art.warnings) and "energy/aggregate_effect" not in art.frames


def test_neural_history_ablation_grid_and_quantile_splits(result):
    h = result.tables["neural_training_history"]
    refits = h[h["phase"].isin(["outer_refit", "test_refit"])]
    assert len(refits) > 0 and refits["val_loss_mse_standardized"].isna().all() and not refits["has_validation_loss"].any()
    assert h.loc[h["phase"] == "selection", "val_loss_mse_standardized"].notna().all()
    grid = result.tables["ablation_grid"].set_index(["model", "config", "horizon_steps"])
    assert not grid.loc[("xgboost_v2", "D", 6), "executed"] and np.isnan(grid.loc[("xgboost_v2", "D", 6), "delta_mae_pct"])
    assert grid.loc[("xgboost_delta", "D", 6), "executed"]
    q = result.tables["quantile_calibration"]
    assert set(q.loc[q["variant"] == "ajustado", "split"]) == {"test_final"}


PUBLISHED = RUN / "expanded-analysis"


@pytest.mark.skipif(not (PUBLISHED / "metadata/analysis_manifest.json").is_file(), reason="análise completa ainda não gravada")
def test_full_analysis_output_with_grid_cache():
    manifest = json.loads((PUBLISHED / "metadata/analysis_manifest.json").read_text(encoding="utf-8"))
    checks = {c["check"]: c for c in manifest["integrity_checks"]}
    assert all(c["passed"] for c in checks.values())
    assert {"serie_reconstruida", "auditoria_populacao", "reproducao_reajustes", "arquivos_preservados_inalterados"} <= set(checks)
    assert all(r["reproduced"] for r in manifest["refits"])
    g = pd.read_csv(PUBLISHED / "csv/generalization.csv").set_index("model")
    assert np.allclose(g.loc["persistence", "normalized_generalization_ratio"], 1.0)
    shap = pd.read_csv(PUBLISHED / "csv/feature_importance.csv")
    assert shap["additivity_max_abs_diff"].max() < 1e-3
    listed = {o["path"] for o in manifest["outputs"]}
    on_disk = {str(p.relative_to(PUBLISHED)) for p in PUBLISHED.rglob("*") if p.is_file()} - {"metadata/analysis_manifest.json"}
    assert on_disk == listed
