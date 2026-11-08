# Dicionário de dados da V3 Parte 2

Unidade de CPU: pontos percentuais (p.p.). Energia: Wh de TI modelada num servidor substituto (estimativa paramétrica).

## .report.tmp/primary_comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família
- `p_holm_global`: ver descrição da família
- `holm_global_reject`: ver descrição da família
- `practically_relevant`: ver descrição da família
- `computed`: ver descrição da família

## baselines/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## baselines/paths.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `step`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)

## baselines/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `level_shift`: ver descrição da família

## calibration/base_quantiles.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `horizon_steps`: ver descrição da família
- `quantile`: ver descrição da família
- `pinball`: ver descrição da família
- `raw_coverage`: ver descrição da família

## calibration/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## calibration/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## calibration/coverage_by_regime.csv
- `method`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `regime`: ver descrição da família
- `n`: ver descrição da família
- `coverage_upper`: fração de alvos abaixo do limite superior calibrado
- `coverage_bilateral`: ver descrição da família
- `mean_width`: ver descrição da família
- `mean_interval_score`: ver descrição da família
- `pinball_q50`: ver descrição da família
- `coverage_upper_peaks`: ver descrição da família
- `n_peaks`: ver descrição da família
- `n_infinite`: ver descrição da família

## calibration/coverage_windows.csv
- `method`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `window_steps`: ver descrição da família
- `min_window_coverage_upper`: ver descrição da família
- `max_window_coverage_upper`: ver descrição da família
- `std_window_coverage_upper`: ver descrição da família
- `n_windows`: ver descrição da família

## calibration/crossing.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `crossed_inner`: ver descrição da família
- `crossed_block`: ver descrição da família
- `n_block`: ver descrição da família

## calibration/limits.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `horizon_steps`: ver descrição da família
- `method`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `y_true`: CPU agregada observada no alvo (p.p.)
- `q_low_raw`: ver descrição da família
- `q50`: ver descrição da família
- `q_high_raw`: ver descrição da família
- `lower`: ver descrição da família
- `upper`: ver descrição da família
- `covered_upper`: ver descrição da família
- `covered_bilateral`: ver descrição da família
- `width`: ver descrição da família
- `interval_score`: interval score (Winkler) do limite bilateral
- `infinite_bound`: ver descrição da família
- `aci_level_upper`: ver descrição da família
- `is_peak`: ver descrição da família
- `regime`: ver descrição da família

## calibration/paths.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `step`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `lower`: ver descrição da família
- `upper`: ver descrição da família
- `method`: ver descrição da família

## calibration/summary.csv
- `method`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `coverage_upper`: fração de alvos abaixo do limite superior calibrado
- `coverage_bilateral`: ver descrição da família
- `mean_width`: ver descrição da família
- `mean_interval_score`: ver descrição da família
- `pinball_q50`: ver descrição da família
- `coverage_upper_peaks`: ver descrição da família
- `n_peaks`: ver descrição da família
- `n_infinite`: ver descrição da família

## calibration/timing.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `method`: ver descrição da família
- `seconds`: ver descrição da família

## energy/aggregation_error.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `route`: ver descrição da família
- `group_candidate`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `n_machines`: ver descrição da família
- `reference_wh`: energia de TI modelada pela rota por máquina com CPU observada (Wh)
- `route_observed_wh`: ver descrição da família
- `aggregation_wh`: rota observada − referência (Wh)

## energy/aggregation_summary.csv
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `route`: ver descrição da família
- `mean_reference_wh`: ver descrição da família
- `mean_aggregation_wh`: ver descrição da família
- `mean_relative_aggregation`: ver descrição da família

## energy/clipped_predictions.csv
- `block`: bloco por dia (day_d)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `model`: ver descrição da família
- `route`: ver descrição da família
- `clipped_values`: ver descrição da família

## energy/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## energy/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## energy/cumulative_24h.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `model`: ver descrição da família
- `route`: ver descrição da família
- `n_windows`: ver descrição da família
- `reference_wh`: energia de TI modelada pela rota por máquina com CPU observada (Wh)
- `predicted_wh`: ver descrição da família
- `cumulative_total_wh`: ver descrição da família
- `cumulative_aggregation_wh`: ver descrição da família
- `cumulative_forecast_wh`: ver descrição da família
- `ci_low_wh`: ver descrição da família
- `ci_high_wh`: ver descrição da família
- `bootstrap_block_windows`: ver descrição da família

## energy/decomposition.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model`: ver descrição da família
- `route`: ver descrição da família
- `group_candidate`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `n_machines`: ver descrição da família
- `reference_wh`: energia de TI modelada pela rota por máquina com CPU observada (Wh)
- `route_observed_wh`: ver descrição da família
- `route_predicted_wh`: ver descrição da família
- `total_wh`: rota prevista − referência (Wh)
- `aggregation_wh`: rota observada − referência (Wh)
- `forecast_wh`: rota prevista − rota observada (Wh)
- `reference_wh_per_server`: ver descrição da família
- `total_wh_per_server`: ver descrição da família

## energy/decomposition_heterogeneo_sintetico.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model`: ver descrição da família
- `route`: ver descrição da família
- `group_candidate`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `n_machines`: ver descrição da família
- `reference_wh`: energia de TI modelada pela rota por máquina com CPU observada (Wh)
- `route_observed_wh`: ver descrição da família
- `route_predicted_wh`: ver descrição da família
- `total_wh`: rota prevista − referência (Wh)
- `aggregation_wh`: rota observada − referência (Wh)
- `forecast_wh`: rota prevista − rota observada (Wh)
- `closure_wh`: ver descrição da família
- `reference_wh_per_server`: ver descrição da família
- `total_wh_per_server`: ver descrição da família
- `hardware`: ver descrição da família

## energy/machine_fallback.csv
- `block`: bloco por dia (day_d)
- `machine_step_cells_fallback`: ver descrição da família
- `fraction`: ver descrição da família

## energy/population_term.csv
- `block`: bloco por dia (day_d)
- `origin_index`: índice da origem na grade de 300 s
- `n_fixed_set`: ver descrição da família
- `fraction_fixed_of_present`: ver descrição da família
- `mean_population_term_pp`: ver descrição da família

## energy/power_bounds.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `r`: ver descrição da família
- `step`: ver descrição da família
- `substitute`: ver descrição da família
- `n`: ver descrição da família
- `ordered`: ver descrição da família
- `coverage_power_of_mean`: ver descrição da família
- `coverage_reference_per_server`: ver descrição da família
- `mean_width_w`: ver descrição da família
- `note`: ver descrição da família

## energy/summary.csv
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `scenario`: ver descrição da família
- `r`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model`: ver descrição da família
- `route`: ver descrição da família
- `n`: ver descrição da família
- `mean_reference_wh`: ver descrição da família
- `mean_abs_total_wh`: ver descrição da família
- `mean_total_wh`: ver descrição da família
- `mean_aggregation_wh`: ver descrição da família
- `mean_abs_forecast_wh`: ver descrição da família
- `mean_forecast_wh`: ver descrição da família
- `mean_relative_abs_total`: ver descrição da família

## global_model/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## global_model/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## global_model/cost.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `n_machines_sampled`: ver descrição da família
- `n_examples_train`: ver descrição da família
- `n_examples_dropped_missing`: ver descrição da família
- `fit_seconds_total`: ver descrição da família
- `inference_seconds`: ver descrição da família
- `wall_seconds`: ver descrição da família
- `xgb_device`: ver descrição da família
- `peak_rss_mb`: ver descrição da família

## global_model/gate.csv
- `horizon_steps`: ver descrição da família
- `global_inner_mae`: ver descrição da família
- `direct_inner_mae`: ver descrição da família
- `ratio`: ver descrição da família

## global_model/inherited_test.csv
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae_a`: ver descrição da família
- `mae_b`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `inspection_number`: ver descrição da família
- `note`: ver descrição da família
- `mae_diff`: ver descrição da família

## global_model/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## global_model/population_term.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `official_target`: ver descrição da família
- `same_set_observed`: ver descrição da família
- `difference`: ver descrição da família

## global_model/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `machines_used`: ver descrição da família
- `machines_dropped`: ver descrição da família

## global_model/sample_summary.csv
- `block`: bloco por dia (day_d)
- `n_machines_sampled`: ver descrição da família
- `n_examples_train_h1`: ver descrição da família
- `n_distinct_origins_train`: ver descrição da família
- `origin_stride`: ver descrição da família
- `design_n_machines`: ver descrição da família
- `design_mean_correlation`: ver descrição da família
- `design_effective_per_instant`: ver descrição da família
- `effective_sample_estimate`: ver descrição da família
- `note`: ver descrição da família

## global_model/scale.csv
- `block`: bloco por dia (day_d)
- `n_machines`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `inner_mae`: ver descrição da família
- `fit_seconds_total`: ver descrição da família
- `n_examples_train`: ver descrição da família
- `peak_rss_mb`: ver descrição da família
- `note`: ver descrição da família

## global_model/tcn_profile.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `seed`: ver descrição da família
- `device`: ver descrição da família
- `best_epoch`: ver descrição da família
- `train_seconds`: ver descrição da família
- `peak_cuda_bytes`: ver descrição da família

## hierarchical/assignments.csv
- `machine_id`: ver descrição da família
- `group`: ver descrição da família
- `block`: bloco por dia (day_d)
- `criterion`: ver descrição da família
- `k`: ver descrição da família

## hierarchical/candidates.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `horizon_steps`: ver descrição da família
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `method`: ver descrição da família
- `inner_aggregate_mae`: ver descrição da família

## hierarchical/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## hierarchical/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## hierarchical/group_metrics.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `group`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `mean_weight_share`: ver descrição da família
- `n_machines_group`: ver descrição da família

## hierarchical/group_metrics_weighted.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `weighted_group_mae`: ver descrição da família
- `weighted_group_bias`: ver descrição da família
- `n_groups`: ver descrição da família

## hierarchical/group_paths.csv
- `block`: bloco por dia (day_d)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `group`: ver descrição da família
- `step`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `y_pred_group`: ver descrição da família
- `weight_share_origin`: ver descrição da família

## hierarchical/group_predictions.csv
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `group`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `y_true_group`: ver descrição da família
- `y_pred_group`: ver descrição da família
- `weight_share_origin`: ver descrição da família
- `n_machines_group`: ver descrição da família

## hierarchical/inherited_test.csv
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae_a`: ver descrição da família
- `mae_b`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `inspection_number`: ver descrição da família
- `note`: ver descrição da família
- `mae_diff`: ver descrição da família

## hierarchical/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## hierarchical/paths.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `step`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)

## hierarchical/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `method`: ver descrição da família
- `chosen`: ver descrição da família

## hierarchical/structure.csv
- `block`: bloco por dia (day_d)
- `pc1_variance_share_machines`: ver descrição da família
- `n_machines_pca`: ver descrição da família
- `mean_corr_between_groups`: ver descrição da família

## hierarchical/timing.csv
- `block`: bloco por dia (day_d)
- `criterion`: ver descrição da família
- `k`: ver descrição da família
- `n_groups`: ver descrição da família
- `seconds`: ver descrição da família

## hierarchical/weight_drift.csv
- `block`: bloco por dia (day_d)
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `origin_index`: índice da origem na grade de 300 s
- `bottom_up_origin_weights`: ver descrição da família
- `bottom_up_target_weights`: ver descrição da família
- `difference`: ver descrição da família

## peaks/alerts.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `horizon_steps`: ver descrição da família
- `lambda`: ver descrição da família
- `threshold`: ver descrição da família
- `n_events`: ver descrição da família
- `n_events_evaluable`: ver descrição da família
- `n_anticipated`: ver descrição da família
- `anticipation_rate`: ver descrição da família
- `n_alert_points`: ver descrição da família
- `n_alert_episodes`: ver descrição da família
- `n_false_alarm_episodes`: ver descrição da família
- `false_alarm_episode_fraction`: ver descrição da família
- `mean_lead_steps`: ver descrição da família

## peaks/choices.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `chosen_lambda`: ver descrição da família
- `sigma_h`: ver descrição da família
- `peak_threshold`: ver descrição da família
- `inner_mae_l0`: ver descrição da família
- `inner_mae_l1`: ver descrição da família
- `inner_mae_l3`: ver descrição da família
- `inner_peak_under_l0`: ver descrição da família
- `inner_peak_under_l1`: ver descrição da família
- `inner_peak_under_l3`: ver descrição da família
- `allowed_lambdas`: ver descrição da família

## peaks/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## peaks/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## peaks/inherited_test.csv
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae_a`: ver descrição da família
- `mae_b`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `inspection_number`: ver descrição da família
- `note`: ver descrição da família
- `mae_diff`: ver descrição da família

## peaks/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## peaks/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `lam`: ver descrição da família
- `peak_threshold`: ver descrição da família

## peaks/timing.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `lambda`: ver descrição da família
- `train_seconds`: ver descrição da família

## peaks/tradeoff.csv
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `horizon_steps`: ver descrição da família
- `lambda`: ver descrição da família
- `chosen`: ver descrição da família
- `train_seconds`: ver descrição da família
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `peak_threshold`: ver descrição da família
- `n_peak`: ver descrição da família
- `peak_bias_obs_minus_pred`: ver descrição da família
- `peak_underestimation_fraction`: ver descrição da família
- `peak_mae`: ver descrição da família

## regimes/choices.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `chosen`: ver descrição da família
- `half_split_mae_select`: ver descrição da família
- `half_split_mae_inverse_mae`: ver descrição da família
- `threshold_volatility`: ver descrição da família
- `threshold_rise`: ver descrição da família
- `threshold_fall`: ver descrição da família

## regimes/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## regimes/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## regimes/confusion_detected_vs_realized.csv
- `detected`: ver descrição da família
- `realized`: ver descrição da família
- `count`: ver descrição da família
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família

## regimes/inherited_test.csv
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae_a`: ver descrição da família
- `mae_b`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `inspection_number`: ver descrição da família
- `note`: ver descrição da família
- `mae_diff`: ver descrição da família

## regimes/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## regimes/metrics_by_regime.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `regime`: ver descrição da família
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família

## regimes/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `regime`: ver descrição da família

## regimes/regime_counts.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `regime`: ver descrição da família
- `n_inner`: ver descrição da família
- `n_block`: ver descrição da família

## regimes/weights.csv
- `block`: bloco por dia (day_d)
- `horizon_steps`: ver descrição da família
- `method`: ver descrição da família
- `regime`: ver descrição da família
- `fallback_to_global`: ver descrição da família
- `persistence`: ver descrição da família
- `xgboost_v2`: ver descrição da família
- `xgboost_delta`: ver descrição da família
- `hybrid`: ver descrição da família

## robustness/comparisons.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `block_steps`: ver descrição da família
- `primary_block`: ver descrição da família
- `n_boot`: ver descrição da família
- `n`: ver descrição da família
- `n_days`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência
- `relative_diff`: ver descrição da família
- `ci_low`: limite inferior do IC de 95% (bootstrap estratificado por dia)
- `ci_high`: limite superior do IC de 95%
- `p_value`: ver descrição da família
- `days_favorable`: ver descrição da família
- `days_unfavorable`: ver descrição da família
- `decision`: melhor | pior | não distinguível (regra pré-registrada)
- `p_holm`: ver descrição da família
- `holm_reject`: ver descrição da família

## robustness/comparisons_by_day.csv
- `family`: ver descrição da família
- `hypothesis`: ver descrição da família
- `scope`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `metric`: ver descrição da família
- `day`: ver descrição da família
- `n`: ver descrição da família
- `loss_a`: ver descrição da família
- `loss_b`: ver descrição da família
- `mean_diff`: média da perda do candidato menos a da referência

## robustness/inherited_test.csv
- `model_a`: ver descrição da família
- `variant_a`: ver descrição da família
- `model_b`: ver descrição da família
- `variant_b`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `mae_a`: ver descrição da família
- `mae_b`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `inspection_number`: ver descrição da família
- `note`: ver descrição da família
- `mae_diff`: ver descrição da família

## robustness/metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## robustness/predictions.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `phase`: inner (validação interna) ou block (bloco avaliado)
- `origin_index`: índice da origem na grade de 300 s
- `timestamp`: ver descrição da família
- `cpu_t`: ver descrição da família
- `y_true`: CPU agregada observada no alvo (p.p.)
- `y_pred`: CPU prevista (p.p.)
- `seed`: ver descrição da família

## robustness/seed_metrics.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `seed`: ver descrição da família
- `block`: bloco por dia (day_d)
- `day`: ver descrição da família
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `n`: ver descrição da família
- `mae`: erro absoluto médio (p.p.)
- `rmse`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `max_abs_error`: ver descrição da família
- `underestimation_fraction`: ver descrição da família
- `persistence_mae_same_origins`: ver descrição da família
- `mase_vs_persistence`: ver descrição da família
- `train_level_shift`: ver descrição da família

## robustness/seed_spread.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `mae_mean`: ver descrição da família
- `mae_std_between_seeds`: ver descrição da família
- `bias_mean`: ver descrição da família
- `n_seeds`: ver descrição da família

## robustness/shift_bias.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `role`: development (exploratório) ou inherited_test (dia 7, quarta inspeção)
- `level_shift_vs_train`: ver descrição da família
- `bias_obs_minus_pred`: viés, observado − previsto (positivo = subestimação)
- `corr_shift_bias_all_blocks`: ver descrição da família

## robustness/training_profile.csv
- `model`: ver descrição da família
- `variant`: ver descrição da família
- `seed`: ver descrição da família
- `horizon_steps`: ver descrição da família
- `block`: bloco por dia (day_d)
- `n_fits`: ver descrição da família
- `train_seconds_total`: ver descrição da família
- `n_params`: ver descrição da família
- `precision`: ver descrição da família
- `peak_cuda_bytes`: ver descrição da família
- `final_epochs`: ver descrição da família
- `device`: ver descrição da família
