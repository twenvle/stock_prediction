# 実験レポート: test-development

- run ID: `real-smoke`
- 状態: `complete`
- 開始（UTC）: `2026-10-13T11:04:05.035805+00:00`
- 終了（UTC）: `2026-10-13T11:04:25.592247+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\data\raw\price_snapshot`
- snapshot ID: `synthetic-price`
- manifest SHA256: `c17fb0b315a7611a5ab70bfc7d75ac69b6bebcbb17aae200caf6117780bb1e23`
- 宣言済みファイルhash数: `9`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\data\raw\aux_snapshot`
- snapshot ID: `synthetic-auxiliary`
- manifest SHA256: `abfef1c14e1b9bdb21c74971f0381507adf3ae26e3780c16ad1ecbd40b7b2a2f`
- 宣言済みファイルhash数: `0`

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | complete | 2026-10-13T11:04:05.058739+00:00 | 2026-10-13T11:04:06.231029+00:00 |
| topix_normalization | complete | 2026-10-13T11:04:06.234027+00:00 | 2026-10-13T11:04:06.251153+00:00 |
| descriptors | complete | 2026-10-13T11:04:06.255152+00:00 | 2026-10-13T11:04:07.257714+00:00 |
| weekly_schedule | complete | 2026-10-13T11:04:07.261710+00:00 | 2026-10-13T11:04:07.601297+00:00 |
| universe | complete | 2026-10-13T11:04:07.607444+00:00 | 2026-10-13T11:04:09.405762+00:00 |
| earnings_events | complete | 2026-10-13T11:04:09.409765+00:00 | 2026-10-13T11:04:10.213073+00:00 |
| weekly_dataset | complete | 2026-10-13T11:04:10.216098+00:00 | 2026-10-13T11:04:14.502734+00:00 |
| splits | complete | 2026-10-13T11:04:14.506735+00:00 | 2026-10-13T11:04:14.716843+00:00 |
| modeling | complete | 2026-10-13T11:04:14.719854+00:00 | 2026-10-13T11:04:16.603888+00:00 |
| evaluation | complete | 2026-10-13T11:04:16.608897+00:00 | 2026-10-13T11:04:22.779578+00:00 |
| backtest | complete | 2026-10-13T11:04:22.784817+00:00 | 2026-10-13T11:04:25.589246+00:00 |

## 開発期間の予測評価

| model | target_column | signal_dates | eligible_rows | target_coverage | prediction_coverage | scoring_coverage | spearman_ic_mean | top_n_mean_target_mean | top_bottom_spread_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | target_return | 260 | 1040 | 0.996154 | 1 | 0.996154 | — | 0.000646562 | 0 |
| linear | target_return | 260 | 1040 | 0.996154 | 1 | 0.996154 | 0.970656 | 0.000646562 | 0 |
| momentum | target_return | 260 | 1040 | 0.996154 | 1 | 0.996154 | 0.0200772 | 0.000646562 | 0 |

## 開発期間の売買シミュレーション

| initial_cash_yen | ending_equity_yen | cumulative_return | annualized_return | annualized_volatility | sharpe_ratio_zero_risk_free | maximum_drawdown | entry_fill_coverage | total_transaction_cost_yen | main_result_complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e+07 | 4.01987e+06 | -0.598013 | -0.162048 | 0.0179536 | -9.83491 | -0.598013 | 1 | 6.79158e+06 | True |

## 警告・制限

- This experiment evaluates development-period predictions only; it does not run or unlock the 2025-2026 final period.
- J-Quants acquisition is intentionally outside the experiment run; immutable snapshot manifests identify the inputs.
- No confirmed delisting settlement table is supplied; the backtest will stop if a confirmed event must be settled.
- 入力rawスナップショットは変更せず、派生データと成果物を別ディレクトリへ保存しています。
- 個別工程の件数、欠損、coverage、除外理由、入力SHA256は各工程のmanifestを確認してください。
- バックテストは板・注文優先順位を再現せず、配当と税金を含まない価格リターンの研究用シミュレーションです。

## 保存先

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\data\interim\experiments\real-smoke`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\data\processed\experiments\real-smoke`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\reports\experiments\real-smoke`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real_2\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\backtest`
