"""Relatório da V3 gerado só a partir dos artefatos gravados em results/v3/<run_id>/.

Não ajusta modelo nem importa os módulos de modelagem: lê CSV e JSON e desenha as figuras.
Uso: python -m energy_prediction.experiments.v3_report --run results/v3/<run_id>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

POINT_ORDER = ["persistence", "xgboost_v2", "xgboost_delta", "statespace", "hybrid", "dlinear", "tcn", "nbeatsx", "nhits", "patchtst"]
COLORS = {
    "persistence": "#7f7f7f",
    "xgboost_v2": "#d62728",
    "xgboost_delta": "#ff7f0e",
    "statespace": "#2ca02c",
    "hybrid": "#17becf",
    "dlinear": "#1f77b4",
    "tcn": "#9467bd",
    "nbeatsx": "#8c564b",
    "nhits": "#e377c2",
    "patchtst": "#bcbd22",
    "ridge_v2": "#aec7e8",
}

LIMITATIONS = [
    "O bloco de teste final de 24 h é o mesmo já inspecionado na V2; reutilizá-lo não constitui um teste independente novo, e nenhum segundo teste foi recortado dentro dele.",
    "A análise principal agrega todas as máquinas da grade com metadado, sem filtro de status; a coorte usa só máquinas ativas com o mesmo cpu_num. As duas populações diferem (ver population_audit.json) e suas métricas são reportadas em separado.",
    "Potência e energia são estimativas paramétricas do modelo de Fan num servidor substituto do SPECpower_ssj2008: a carga do benchmark não é a utilização de CPU do trace, e o hardware não é o do cluster. O trace não fornece potência nem energia medidas.",
    "A potência calculada a partir da CPU agregada é a de um servidor substituto, não a potência total da frota; a soma das máquinas presentes aparece em coluna separada.",
    "Os passos 2, 4 e 5 das trajetórias de Ridge, XGBoost da V2, incremento e híbrido herdam o hiperparâmetro dos passos 3 e 6 (marca inherited_params); não houve busca nova para eles.",
    "A cobertura dos intervalos quantílicos é empírica; sob dependência temporal não há garantia teórica.",
    "As variâncias do modelo de espaço de estados são estimadas por máxima verossimilhança no segmento de treino; o estado na origem é sempre filtrado.",
    "A diferença entre energia projetada e de referência é entre duas grandezas modeladas, não erro contra medidor, e não é intervalo de confiança da energia elétrica do cluster.",
    "PUE não foi calculado.",
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() and path.stat().st_size > 1 else None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame is None or frame.empty:
        return "_sem dados_\n"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in shown.to_numpy())
    return "\n".join([header, rule, body]) + "\n"


class RunArtifacts:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.manifest = _read_json(self.run / "manifest.json") or {}
        config_path = Path(self.manifest.get("config_path", ""))
        self.root = config_path.resolve().parent.parent if config_path.name else self.run.parents[2]
        v2 = (self.manifest.get("config") or {}).get("reference", {}).get("v2_dir")
        self.v2_dir = None if v2 is None else (Path(v2) if Path(v2).is_absolute() else self.root / v2)

    def family(self, name: str) -> Path:
        return self.run / name

    def point_predictions(self) -> pd.DataFrame:
        frames = []
        for family in ("residual", "neural"):
            frame = _read_csv(self.family(family) / "predictions.csv")
            if frame is not None:
                if "mode" in frame:
                    frame = frame[frame["mode"] == "direct"]
                frames.append(frame)
        if self.v2_dir is not None and (self.v2_dir / "predictions.csv").is_file():
            v2 = pd.read_csv(self.v2_dir / "predictions.csv")
            v2 = v2[(v2["model"] == "xgboost") & (v2["config"] == "A")].assign(model="xgboost_v2", mode="direct")
            frames.append(v2)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["split", "model", "config", "horizon_steps", "origin_index"])

    def point_metrics(self) -> pd.DataFrame:
        frames = []
        for family in ("residual", "neural"):
            frame = _read_csv(self.family(family) / "metrics.csv")
            if frame is not None:
                frames.append(frame.assign(family=family))
        if self.v2_dir is not None and (self.v2_dir / "metrics.csv").is_file():
            v2 = pd.read_csv(self.v2_dir / "metrics.csv")
            v2 = v2[(v2["model"] == "xgboost")].assign(model="xgboost_v2", mode="direct", family="v2-full (sem reajuste)")
            frames.append(v2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _figures(art: RunArtifacts, directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    made = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=140)
        plt.close(fig)
        made.append(name)

    predictions = art.point_predictions()
    metrics = art.point_metrics()
    if not predictions.empty:
        test = predictions[(predictions["split"] == "test_final") & (predictions["horizon_steps"] == predictions["horizon_steps"].min())]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        reference = test[test["model"] == "persistence"].sort_values("origin_index")
        ax.plot(reference["timestamp"] / 3600, reference["y_true"], color="black", lw=1.4, label="observado")
        for model in POINT_ORDER:
            part = test[(test["model"] == model) & (test["config"].isin(["A", "-"]))].sort_values("origin_index")
            if not part.empty:
                ax.plot(part["timestamp"] / 3600, part["y_pred"], lw=0.9, color=COLORS.get(model), label=model)
        ax.set_xlabel("tempo do trace (h desde o início)")
        ax.set_ylabel("CPU (%)")
        ax.set_title("Teste final, 5 min, configuração A: observado e previsto")
        ax.legend(ncol=5, fontsize=7)
        finish(fig, "observed_vs_predicted_A.png")

    if not metrics.empty:
        test = metrics[(metrics["split"] == "test_final") & (metrics["config"].isin(["A", "-"])) & (metrics.get("mode", "direct").fillna("direct") == "direct")]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for model in POINT_ORDER:
            part = test[test["model"] == model].sort_values("horizon_seconds")
            if not part.empty:
                ax.plot(part["horizon_seconds"] / 60, part["mae"], marker="o", color=COLORS.get(model), label=model)
        ax.set_xlabel("horizonte (min)")
        ax.set_ylabel("MAE no teste final (p.p. de CPU)")
        ax.set_title("Erro por horizonte, configuração A")
        ax.legend(fontsize=7, ncol=2)
        finish(fig, "mae_by_horizon_A.png")

    neural_sel = _read_json(art.family("neural") / "selections.json")
    if not predictions.empty and neural_sel:
        horizons = sorted(predictions["horizon_steps"].unique())
        chosen = [horizons[0], horizons[-1]]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, steps in zip(axes, chosen):
            candidates = [s for s in neural_sel if s["config"] == "A" and s.get("mode") == "direct" and int(s["horizon_steps"]) == steps and s.get("contrast") is None and s.get("role") == "core"]
            test = predictions[(predictions["split"] == "test_final") & (predictions["horizon_steps"] == steps)]
            for model in ["xgboost_v2", min(candidates, key=lambda s: s["selection_score"])["model"] if candidates else None]:
                if model is None:
                    continue
                part = test[(test["model"] == model) & (test["config"] == "A")]
                if not part.empty:
                    ax.hist(part["y_true"] - part["y_pred"], bins=30, alpha=0.55, color=COLORS.get(model), label=model)
            ax.set_title(f"Resíduos no teste, {steps * 5} min, A")
            ax.set_xlabel("observado - previsto (p.p.)")
            ax.legend(fontsize=7)
        finish(fig, "residual_histograms.png")

    energy_info = _read_json(art.family("energy") / "energy_info.json")
    summary = _read_csv(art.family("energy") / "energy_summary.csv")
    by_origin = _read_csv(art.family("energy") / "energy_by_origin.csv")
    if energy_info:
        sub = energy_info["substitute"]
        p_idle, p_busy = float(sub["p_idle_w"]), float(sub["p_busy_w"])
        u = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for r in energy_info["exponents"]:
            ax.plot(u * 100, p_idle + (p_busy - p_idle) * (2 * u - u**r), label=f"r = {r:g}")
        ax.set_xlabel("utilização de CPU (%)")
        ax.set_ylabel("potência modelada (W)")
        ax.set_title("Modelo de Fan no servidor substituto")
        ax.legend(fontsize=8)
        finish(fig, "fan_power_curves.png")
    if by_origin is not None and energy_info:
        horizon = int(by_origin["horizon_steps"].max())
        models = ["persistence"] + ([energy_info["figure_point_model"]["model"]] if energy_info["figure_point_model"].get("model") else [])
        exponents = energy_info["exponents"]
        fig, axes = plt.subplots(1, len(exponents), figsize=(4 * len(exponents), 3.4), sharey=False, squeeze=False)
        for ax, r in zip(axes[0], exponents):
            part = by_origin[(by_origin["horizon_steps"] == horizon) & (by_origin["r"] == r)]
            ref = part[part["model"] == "persistence"].sort_values("origin_index")
            ax.plot(ref["timestamp"] / 3600, ref["reference_energy_wh"], color="black", lw=1.2, label="referência")
            for model in models:
                p = part[part["model"] == model].sort_values("origin_index")
                ax.plot(p["timestamp"] / 3600, p["projected_energy_wh"], lw=0.9, color=COLORS.get(model), label=f"projetada {model}")
            ax.set_title(f"r = {r:g}, horizonte {horizon * 5} min")
            ax.set_xlabel("tempo do trace (h)")
        axes[0][0].set_ylabel("energia de TI modelada (Wh)")
        axes[0][0].legend(fontsize=7)
        finish(fig, "energy_reference_vs_projected.png")
    if summary is not None:
        horizon = int(summary["horizon_steps"].max())
        part = summary[summary["horizon_steps"] == horizon]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for model, group in part.groupby("model"):
            group = group.sort_values("r")
            ax.plot(group["r"], group["mean_absolute_error_wh"], marker="o", color=COLORS.get(model), label=model)
        ax.set_xlabel("expoente r do modelo de Fan")
        ax.set_ylabel("erro absoluto médio da energia (Wh)")
        ax.set_title(f"Energia projetada vs. referência, {horizon * 5} min")
        ax.legend(fontsize=7, ncol=2)
        finish(fig, "energy_error_vs_r.png")
    return made


def _execution_report(art: RunArtifacts, figures: list[str]) -> str:
    m = art.manifest
    lines = [f"# Relatório de execução da V3 — {m.get('run_id', art.run.name)}", "", m.get("statement", ""), ""]
    device = m.get("device", {})
    lines += [
        "## Corrida",
        "",
        f"- semente: {m.get('seed')}",
        f"- hash da configuração: `{m.get('config_sha256')}`",
        f"- dados: `{json.dumps(m.get('data'), ensure_ascii=False)}`",
        f"- commit: `{(m.get('git') or {}).get('commit')}` (alterações locais: {(m.get('git') or {}).get('dirty')})",
        f"- dispositivo pedido/efetivo: {device.get('requested')}/{device.get('effective')} ({device.get('device_name') or 'CPU'})",
        f"- versões: `{json.dumps(m.get('versions'), ensure_ascii=False)}`",
        "",
        "## Famílias concluídas",
        "",
    ]
    for name, record in (m.get("families") or {}).items():
        lines.append(f"- {name}: {record['elapsed_seconds']:.1f} s ({record['completed_utc']})")
    excluded = m.get("excluded") or []
    lines += ["", "## O que não rodou (orçamento, portão ou ausência de dado)", ""]
    lines += [f"- [{e.get('family')}] {e['item']}: {e['reason']}" for e in excluded] or ["- nada"]

    audit = _read_json(art.family("integrity") / "population_audit.json")
    if audit:
        only_main = ", ".join(
            "{} (cpu_num {:g}, {}, status {})".format(i["machine_id"], i["cpu_num"], i["reason"], i["status_counts"]) for i in audit["only_in_main"]
        )
        lines += [
            "",
            "## Auditoria da população",
            "",
            f"- principal: {audit['n_machines_main']} máquinas, peso {audit['weight_main']:g}; coorte: {audit['n_machines_cohort']} máquinas, peso {audit['weight_cohort']:g}.",
            f"- só na principal: {only_main or 'nenhuma'}.",
            f"- diferença máxima de CPU agregada entre as séries: {audit['series_comparison']['max_abs_cpu_difference']:.4f} p.p.; séries idênticas: {audit['series_comparison']['series_identical']}.",
        ]

    metrics = art.point_metrics()
    if not metrics.empty:
        test = metrics[(metrics["split"] == "test_final") & (metrics["config"].isin(["A", "-"]))]
        if "mode" in test:
            test = test[test["mode"].fillna("direct") == "direct"]
        pivot = test.pivot_table(index="model", columns="horizon_steps", values="mae", aggfunc="first")
        pivot = pivot.reindex([m for m in POINT_ORDER if m in pivot.index])
        pivot.columns = [f"MAE h{c}" for c in pivot.columns]
        lines += ["", "## MAE no teste final, configuração A (modo direto)", "", "Ordem de leitura, não de seleção: o menor MAE do teste não promove modelo.", "", _table(pivot.reset_index())]
        abc = metrics[(metrics["split"] == "test_final") & (metrics["config"].isin(["B", "C"]))]
        if "mode" in abc:
            abc = abc[abc["mode"].fillna("direct") == "direct"]
        if not abc.empty:
            pivot = abc.pivot_table(index=["model", "config"], columns="horizon_steps", values="mae", aggfunc="first")
            pivot.columns = [f"MAE h{c}" for c in pivot.columns]
            lines += ["", "## Ablação B e C (teste final)", "", _table(pivot.reset_index())]
    neural_metrics = _read_csv(art.family("neural") / "metrics.csv")
    if neural_metrics is not None and "mode" in neural_metrics:
        path = neural_metrics[(neural_metrics["mode"] == "path") & (neural_metrics["split"] == "test_final")]
        if not path.empty:
            pivot = path.pivot_table(index=["model", "config"], columns="horizon_steps", values="mae", aggfunc="first")
            pivot.columns = [f"MAE h{c}" for c in pivot.columns]
            lines += ["", "## Trajetória de seis passos (secundária; não substitui o modo direto)", "", _table(pivot.reset_index())]

    claims = []
    for family in ("residual", "neural", "quantile"):
        frame = _read_csv(art.family(family) / "bootstrap_ci.csv")
        if frame is not None and "primary_block" in frame:
            claims.append(frame[frame["primary_block"]].assign(family=family))
    if claims:
        boot = pd.concat(claims, ignore_index=True)
        columns = [c for c in ("family", "comparison", "quantile", "horizon_steps", "model_a", "config_a", "model_b", "config_b", "mean_diff", "ci_low", "ci_high", "significant") if c in boot]
        lines += [
            "",
            "## Comparações pré-registradas (bootstrap por blocos de 1 h, 95%)",
            "",
            "Diferença = perda de A menos perda de B; só é distinguível de zero quando o intervalo não contém zero. Blocos de 30 min e 2 h estão em bootstrap_ci.csv como sensibilidade.",
            "",
            _table(boot[columns]),
        ]

    d_frames = [(_read_csv(art.family(f) / "d_contrast_metrics.csv"), f) for f in ("residual", "neural")]
    d_rows = [f[(f["split"] == "test_final")].assign(family=name) for f, name in d_frames if f is not None]
    if d_rows:
        frame = pd.concat(d_rows, ignore_index=True)
        pivot = frame.pivot_table(index=["family", "model", "config"], columns="horizon_steps", values="mae", aggfunc="first")
        pivot.columns = [f"MAE h{c}" for c in pivot.columns]
        lines += ["", "## Contraste D contra A (origens com D completa, A recalculado nelas)", "", _table(pivot.reset_index())]

    neural_info = _read_json(art.family("neural") / "family_info.json")
    optional = _read_json(art.family("neural") / "optional_architectures.json")
    profile = _read_csv(art.family("neural") / "training_profile.csv")
    if neural_info:
        lines += ["", "## Redes", "", f"- janela escolhida: {neural_info['window_steps']} passos; núcleo concluído: {neural_info['core_completed']}; tempo da família: {neural_info['elapsed_seconds'] / 60:.1f} min."]
        if optional:
            lines.append(f"- portão N-HiTS/PatchTST: {json.dumps(optional.get('gate', {}).get('passed'))}; ajustadas: {optional.get('fitted')}; {optional.get('reason', '')}")
        if profile is not None:
            summary = profile.groupby("arch").agg(params=("n_params", "max"), train_s=("train_seconds_total", "sum"), peak_cuda_mb=("peak_cuda_bytes_max", "max"))
            summary["peak_cuda_mb"] = summary["peak_cuda_mb"] / 2**20
            lines += ["", _table(summary.reset_index(), 2)]

    intervals = _read_csv(art.family("quantile") / "interval_metrics.csv")
    if intervals is not None:
        columns = [c for c in ("config", "horizon_steps", "upper", "conformal_applied", "mean_width", "interval_coverage", "upper_coverage") if c in intervals]
        lines += ["", "## Intervalos quantílicos no teste (cobertura empírica)", "", _table(intervals[columns])]

    summary = _read_csv(art.family("energy") / "energy_summary.csv")
    energy_info = _read_json(art.family("energy") / "energy_info.json")
    if summary is not None and energy_info:
        sub = energy_info["substitute"]
        lines += [
            "",
            "## Energia de TI modelada (servidor substituto)",
            "",
            f"- servidor: {sub['system']} ({sub['result_id']}, publicado em {sub['published']}); P_idle = {sub['p_idle_w']} W, P_busy = {sub['p_busy_w']} W.",
            f"- {energy_info['statement']}",
            f"- {energy_info['propagation_statement']}",
            "",
            _table(summary[["model", "r", "horizon_steps", "n_complete", "total_reference_wh", "total_projected_wh", "mean_absolute_error_wh", "mean_absolute_relative_error"]], 3),
        ]
        effect = energy_info.get("aggregate_effect", {}).get("per_exponent", [])
        if effect:
            lines += ["", "### P(média) contra média de P (efeito de agregação)", "", _table(pd.DataFrame(effect), 3)]
    if figures:
        lines += ["", "## Figuras", ""] + [f"- figures/{name}" for name in figures]
    return "\n".join(lines) + "\n"


def generate(run: Path) -> Path:
    run = Path(run)
    if not (run / "manifest.json").is_file():
        raise FileNotFoundError(f"{run} não tem manifest.json; não é uma corrida da V3.")
    art = RunArtifacts(run)
    temporary = run / ".report.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        figures = _figures(art, temporary / "figures")
        (temporary / "execution_report.md").write_text(_execution_report(art, figures), encoding="utf-8")
        limitations = ["# Limitações da V3", ""] + [f"- {item}" for item in LIMITATIONS]
        (temporary / "limitations.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")
        final = run / "report"
        if final.exists():
            shutil.rmtree(final)
        temporary.rename(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório da V3 a partir dos artefatos gravados (sem novo treino).")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(f"Relatório gravado em {generate(args.run)}")


if __name__ == "__main__":
    main()
