# 実験レポート: test-development

- run ID: `full-run`
- 状態: `complete`
- 開始（UTC）: `2026-10-13T11:09:19.192224+00:00`
- 終了（UTC）: `2026-10-13T11:09:19.382370+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\data\raw\price_snapshot`
- snapshot ID: `price-001`
- manifest SHA256: `0c8af354b50712d6a78bbee4ceda5def2ab9029cfeac2260efd1b4098b9a8cb0`
- 宣言済みファイルhash数: `1`
- 取得範囲・条件: `{"end_date": "2026-12-31", "start_date": "2016-01-01"}`
- manifest集計: `{"price_rows": 1}`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\data\raw\aux_snapshot`
- snapshot ID: `aux-001`
- manifest SHA256: `4ffaa3472469b13f1bfe6d00e90c7856ed2ef94133ab1d8050b1a3660a3865cc`
- 宣言済みファイルhash数: `1`
- 取得範囲・条件: `{"end_date": "2026-12-31", "start_date": "2016-01-01"}`
- manifest集計: `{"listed_info_rows": 1}`

## 設定識別子

| 工程 | 設定ファイル | SHA256 |
| --- | --- | --- |
| preprocessing | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\preprocessing.json` | `467f937d7f2c9aa439dec2a33eecc90cee37252822190f8842b639d8e462d2ec` |
| descriptors | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\descriptors.json` | `1423e6ddba8353b6054cdef7c460cb7a6ba66a21d0bffbe4c10ef8a10449c50e` |
| universe | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\universe.json` | `ca16c8a63b744864da9e618961c3a93dda58f35158c68ba59c2fae2528e79b8a` |
| earnings_events | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\earnings_events.json` | `db980e9972969f08977daae7ad7cfa4fa43cb20b2e51f8540c0c765d264b87a1` |
| weekly_dataset | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\weekly_dataset.json` | `1ca7bd35e9b65b3e3e688540cbcdd7f6504d0885e76847582100553a80e6780d` |
| splits | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\splits.json` | `bb03a13efe5798e3536863d77f04c8edefbd09ab804269be201acf563e5d1c22` |
| modeling | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\modeling.json` | `d7b067088542bc4fc71149bade039cf54e6590adc4d29739873c9e6e23845b34` |
| evaluation | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\evaluation.json` | `e2285543ebc8911f59b3b7853edb9ce0bf16a79b01c751c9fa28e2cbb50cb3b8` |
| backtest | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\configs\backtest.json` | `c331a56bf4bbb72615f26dfa666382413126c217c5acc557a3679c53155d2dd9` |

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | complete | 2026-10-13T11:09:19.215946+00:00 | 2026-10-13T11:09:19.224936+00:00 |
| topix_normalization | complete | 2026-10-13T11:09:19.228945+00:00 | 2026-10-13T11:09:19.247863+00:00 |
| descriptors | complete | 2026-10-13T11:09:19.252273+00:00 | 2026-10-13T11:09:19.261610+00:00 |
| weekly_schedule | complete | 2026-10-13T11:09:19.265835+00:00 | 2026-10-13T11:09:19.277745+00:00 |
| universe | complete | 2026-10-13T11:09:19.282394+00:00 | 2026-10-13T11:09:19.288063+00:00 |
| earnings_events | complete | 2026-10-13T11:09:19.292817+00:00 | 2026-10-13T11:09:19.300748+00:00 |
| weekly_dataset | complete | 2026-10-13T11:09:19.303744+00:00 | 2026-10-13T11:09:19.309003+00:00 |
| splits | complete | 2026-10-13T11:09:19.312512+00:00 | 2026-10-13T11:09:19.322917+00:00 |
| modeling | complete | 2026-10-13T11:09:19.327073+00:00 | 2026-10-13T11:09:19.340517+00:00 |
| evaluation | complete | 2026-10-13T11:09:19.346674+00:00 | 2026-10-13T11:09:19.361455+00:00 |
| backtest | complete | 2026-10-13T11:09:19.367922+00:00 | 2026-10-13T11:09:19.376758+00:00 |

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

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\data\interim\experiments\full-run`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\data\processed\experiments\full-run`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\artifacts\experiments\full-run`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\reports\experiments\full-run`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_all_experiment_final\test_run_experiment_executes_s0\project\artifacts\experiments\full-run\backtest`
