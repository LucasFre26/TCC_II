"""Picos de utilização: limiares, erro condicionado, eventos e antecipação com falso alarme.

Regras (declaradas na configuração e no relatório):
- limiar primário: quantil da CPU observada no desenvolvimento (antes da primeira origem do teste);
  o quantil da amostra de teste é reportado para reproduzir os artefatos da V2 e da V3;
- evento: sequência de instantes consecutivos com CPU >= limiar, unindo sequências separadas por
  até `merge_gap` passos; o início só conta se o instante anterior estava abaixo do limiar;
- alerta no horizonte h para o instante tau: previsão feita em tau − h >= limiar;
- evento antecipado: algum alerta mira um instante a até `tolerance` passos do início e foi emitido
  antes do início (origem <= início − 1);
- falso alarme: episódio de alertas consecutivos sem evento a até `tolerance` passos.
Um quantil alto que ultrapassa o limiar não é, por si, antecipação: as contagens de falso alarme
acompanham sempre a taxa de eventos antecipados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def thresholds(cpu: pd.Series, dev_end: int, test_targets: pd.Series, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    dev = cpu.loc[:dev_end].dropna()
    for q in quantiles:
        rows.append({"threshold_kind": f"desenvolvimento_p{int(q * 100)}", "source": "CPU observada antes da primeira origem do teste", "quantile": q, "threshold_cpu_pct": float(dev.quantile(q)), "n_reference": int(len(dev))})
        rows.append({"threshold_kind": f"teste_p{int(q * 100)}", "source": "y_true do teste em cada horizonte (como nos artefatos da V2 e V3)", "quantile": q, "threshold_cpu_pct": np.nan, "n_reference": int(len(test_targets))})
    return pd.DataFrame(rows)


def peak_metrics(test: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Erro nos instantes de pico por modelo, configuração e horizonte, para cada limiar."""
    rows = []
    for key, part in test.groupby(["model", "config", "mode", "horizon_steps", "horizon_min"], sort=False):
        y = part["y_obs_cpu_pct"].to_numpy()
        e = part["error_obs_minus_pred_pp"].to_numpy()
        for _, t in table.iterrows():
            thr = t["threshold_cpu_pct"] if np.isfinite(t["threshold_cpu_pct"]) else float(np.quantile(y, t["quantile"]))
            peak = y >= thr
            under = e[peak] > 0
            rows.append(
                {
                    "model": key[0], "config": key[1], "mode": key[2], "horizon_steps": int(key[3]), "horizon_min": key[4],
                    "threshold_kind": t["threshold_kind"], "threshold_cpu_pct": thr, "n_total": len(y), "n_peak_points": int(peak.sum()),
                    "peak_mae_pp": float(np.mean(np.abs(e[peak]))) if peak.any() else np.nan,
                    "peak_bias_obs_minus_pred_pp": float(np.mean(e[peak])) if peak.any() else np.nan,
                    "peak_underestimation_frequency": float(np.mean(under)) if peak.any() else np.nan,
                    "peak_mean_underestimation_pp": float(np.mean(e[peak][under])) if under.any() else np.nan,
                    "peak_max_underestimation_pp": float(np.max(e[peak])) if peak.any() else np.nan,
                    "nonpeak_mae_pp": float(np.mean(np.abs(e[~peak]))) if (~peak).any() else np.nan,
                    "overall_max_underestimation_pp": float(max(e.max(), 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _runs(mask: np.ndarray, index: np.ndarray) -> list[tuple[int, int]]:
    """Sequências de índices consecutivos marcados (um salto no índice encerra a sequência)."""
    runs: list[tuple[int, int]] = []
    start = prev = None
    for i, flag in zip(index, mask):
        if flag:
            if start is not None and i == prev + 1:
                prev = i
                continue
            if start is not None:
                runs.append((int(start), int(prev)))
            start = prev = i
        elif start is not None:
            runs.append((int(start), int(prev)))
            start = None
    if start is not None:
        runs.append((int(start), int(prev)))
    return runs


def events(cpu: pd.Series, low: int, high: int, threshold: float, merge_gap: int) -> pd.DataFrame:
    y = cpu.loc[low:high]
    runs = _runs((y >= threshold).to_numpy(), y.index.to_numpy())
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rows = []
    for k, (a, b) in enumerate(merged):
        before = cpu.get(a - 1, np.nan)
        rows.append(
            {
                "event_id": k, "start_index": a, "end_index": b, "duration_steps": b - a + 1, "peak_cpu_pct": float(cpu.loc[a:b].max()),
                "cpu_before_start_pct": float(before), "onset_observed": bool(np.isfinite(before) and before < threshold), "threshold_cpu_pct": threshold,
            }
        )
    return pd.DataFrame(rows)


def alert_scores(signal: pd.Series, horizon: int, event_table: pd.DataFrame, *, low: int, high: int, threshold: float, tolerance: int) -> dict:
    """`signal`: previsão indexada pelo instante previsto, para um horizonte."""
    s = signal.loc[low:high]
    alert_targets = s.index[s.to_numpy() >= threshold].to_numpy()
    episodes = _runs(np.ones(len(alert_targets), dtype=bool), alert_targets) if len(alert_targets) else []
    evaluable = event_table[event_table["onset_observed"]]
    anticipated, leads = 0, []
    for _, ev in evaluable.iterrows():
        start = int(ev["start_index"])
        ok = [t for t in alert_targets if start - tolerance <= t <= start + tolerance and t - horizon <= start - 1]
        if ok:
            anticipated += 1
            leads.append(max(start - (t - horizon) for t in ok))
    windows = [(int(ev["start_index"]) - tolerance, int(ev["end_index"]) + tolerance) for _, ev in event_table.iterrows()]
    false = sum(1 for a, b in episodes if not any(a <= w1 and b >= w0 for w0, w1 in windows))
    return {
        "n_events": int(len(event_table)), "n_events_evaluable": int(len(evaluable)), "n_anticipated": anticipated,
        "anticipation_rate": anticipated / len(evaluable) if len(evaluable) else np.nan,
        "n_alert_points": int(len(alert_targets)), "n_alert_episodes": len(episodes), "n_false_alarm_episodes": false,
        "false_alarm_episode_fraction": false / len(episodes) if episodes else np.nan,
        "mean_lead_steps": float(np.mean(leads)) if leads else np.nan,
    }


def alert_table(signals: dict[tuple[str, str, int], pd.Series], cpu: pd.Series, thresholds_table: pd.DataFrame, *, low: int, high: int, merge_gap: int, tolerance: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Antecipação de eventos por sinal (modelo pontual ou quantil) e horizonte, para os limiares do desenvolvimento."""
    rows, event_rows = [], []
    for _, t in thresholds_table[thresholds_table["threshold_kind"].str.startswith("desenvolvimento")].iterrows():
        thr = float(t["threshold_cpu_pct"])
        table = events(cpu, low, high, thr, merge_gap)
        event_rows.append(table.assign(threshold_kind=t["threshold_kind"]))
        for (signal, kind, h), series in signals.items():
            rows.append({"signal": signal, "signal_kind": kind, "horizon_steps": h, "horizon_min": h * 5.0, "threshold_kind": t["threshold_kind"], "threshold_cpu_pct": thr, "low_index": low, "high_index": high, **alert_scores(series, h, table, low=low, high=high, threshold=thr, tolerance=tolerance)})
    return pd.DataFrame(rows), pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
