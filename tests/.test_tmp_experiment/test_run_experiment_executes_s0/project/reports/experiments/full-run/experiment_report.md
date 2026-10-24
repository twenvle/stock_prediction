# 実験レポート: test-development

- run ID: `full-run`
- 状態: `complete`
- 開始（UTC）: `2026-10-13T10:58:34.155318+00:00`
- 終了（UTC）: `2026-10-13T10:58:34.432260+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\data\raw\price_snapshot`
- snapshot ID: `price-001`
- manifest SHA256: `0c8af354b50712d6a78bbee4ceda5def2ab9029cfeac2260efd1b4098b9a8cb0`
- 宣言済みファイルhash数: `1`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\data\raw\aux_snapshot`
- snapshot ID: `aux-001`
- manifest SHA256: `4ffaa3472469b13f1bfe6d00e90c7856ed2ef94133ab1d8050b1a3660a3865cc`
- 宣言済みファイルhash数: `1`

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | complete | 2026-10-13T10:58:34.183314+00:00 | 2026-10-13T10:58:34.189578+00:00 |
| topix_normalization | complete | 2026-10-13T10:58:34.193582+00:00 | 2026-10-13T10:58:34.327606+00:00 |
| descriptors | complete | 2026-10-13T10:58:34.330606+00:00 | 2026-10-13T10:58:34.337613+00:00 |
| weekly_schedule | complete | 2026-10-13T10:58:34.342607+00:00 | 2026-10-13T10:58:34.352142+00:00 |
| universe | complete | 2026-10-13T10:58:34.356170+00:00 | 2026-10-13T10:58:34.360196+00:00 |
| earnings_events | complete | 2026-10-13T10:58:34.365204+00:00 | 2026-10-13T10:58:34.369261+00:00 |
| weekly_dataset | complete | 2026-10-13T10:58:34.374141+00:00 | 2026-10-13T10:58:34.378140+00:00 |
| splits | complete | 2026-10-13T10:58:34.382324+00:00 | 2026-10-13T10:58:34.387022+00:00 |
| modeling | complete | 2026-10-13T10:58:34.391844+00:00 | 2026-10-13T10:58:34.402089+00:00 |
| evaluation | complete | 2026-10-13T10:58:34.405642+00:00 | 2026-10-13T10:58:34.414688+00:00 |
| backtest | complete | 2026-10-13T10:58:34.418810+00:00 | 2026-10-13T10:58:34.427252+00:00 |

## 開発期間の予測評価

| model | target_column | signal_dates | spearman_ic_mean | top_n_mean_target_mean | top_bottom_spread_mean |
| --- | --- | --- | --- | --- | --- |
| linear | target_return | 1 | 0.2 | 0.01 | 0.015 |

## 開発期間の売買シミュレーション

| model | target_column | initial_cash_yen | ending_equity_yen | cumulative_return | main_result_complete |
| --- | --- | --- | --- | --- | --- |
| linear | target_return | 1e+07 | 1.01e+07 | 0.01 | True |

## 警告・制限

- This experiment evaluates development-period predictions only; it does not run or unlock the 2025-2026 final period.
- J-Quants acquisition is intentionally outside the experiment run; immutable snapshot manifests identify the inputs.
- No confirmed delisting settlement table is supplied; the backtest will stop if a confirmed event must be settled.
- 入力rawスナップショットは変更せず、派生データと成果物を別ディレクトリへ保存しています。
- 個別工程の件数、欠損、coverage、除外理由、入力SHA256は各工程のmanifestを確認してください。
- バックテストは板・注文優先順位を再現せず、配当と税金を含まない価格リターンの研究用シミュレーションです。

## 保存先

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\data\interim\experiments\full-run`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\data\processed\experiments\full-run`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\artifacts\experiments\full-run`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\reports\experiments\full-run`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\backtest`
