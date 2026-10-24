# 実験レポート: test-development

- run ID: `real-smoke`
- 状態: `complete`
- 開始（UTC）: `2026-10-21T03:54:36.216913+00:00`
- 終了（UTC）: `2026-10-21T03:54:57.050530+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\data\raw\price_snapshot`
- snapshot ID: `synthetic-price`
- manifest SHA256: `c17fb0b315a7611a5ab70bfc7d75ac69b6bebcbb17aae200caf6117780bb1e23`
- manifest内の宣言済みSHA256数: `9`
- 取得範囲・条件: `{"end_date": "2024-12-31", "start_date": "2016-01-01"}`
- manifest集計: `{"price_rows": 9392}`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\data\raw\aux_snapshot`
- snapshot ID: `synthetic-auxiliary`
- manifest SHA256: `abfef1c14e1b9bdb21c74971f0381507adf3ae26e3780c16ad1ecbd40b7b2a2f`
- manifest内の宣言済みSHA256数: `0`
- 取得範囲・条件: `{"end_date": "2024-12-31", "start_date": "2016-01-01"}`
- manifest集計: `{"listed_info_rows": 9392}`

## 設定識別子

| 工程 | 設定ファイル | SHA256 |
| --- | --- | --- |
| preprocessing | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\preprocessing.json` | `467f937d7f2c9aa439dec2a33eecc90cee37252822190f8842b639d8e462d2ec` |
| descriptors | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\descriptors.json` | `1423e6ddba8353b6054cdef7c460cb7a6ba66a21d0bffbe4c10ef8a10449c50e` |
| universe | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\universe.json` | `ca16c8a63b744864da9e618961c3a93dda58f35158c68ba59c2fae2528e79b8a` |
| earnings_events | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\earnings_events.json` | `36efedca04ef630ffeb5fb38ca2ad5a0636a1724dede76c7fb78c7fd4102106d` |
| weekly_dataset | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\weekly_dataset.json` | `ab86d2cbc00d25099a4a563ccb8f1d6575a66a4b7ffb7de49ca4e50e29fcf467` |
| splits | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\splits.json` | `bb03a13efe5798e3536863d77f04c8edefbd09ab804269be201acf563e5d1c22` |
| modeling | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\modeling.json` | `d7b067088542bc4fc71149bade039cf54e6590adc4d29739873c9e6e23845b34` |
| evaluation | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\evaluation.json` | `e2285543ebc8911f59b3b7853edb9ce0bf16a79b01c751c9fa28e2cbb50cb3b8` |
| backtest | `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\configs\backtest.json` | `c331a56bf4bbb72615f26dfa666382413126c217c5acc557a3679c53155d2dd9` |

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | complete | 2026-10-21T03:54:36.243475+00:00 | 2026-10-21T03:54:37.562420+00:00 |
| topix_normalization | complete | 2026-10-21T03:54:37.566416+00:00 | 2026-10-21T03:54:37.585237+00:00 |
| descriptors | complete | 2026-10-21T03:54:37.590508+00:00 | 2026-10-21T03:54:38.709279+00:00 |
| weekly_schedule | complete | 2026-10-21T03:54:38.712277+00:00 | 2026-10-21T03:54:39.090933+00:00 |
| universe | complete | 2026-10-21T03:54:39.096092+00:00 | 2026-10-21T03:54:40.778361+00:00 |
| earnings_events | complete | 2026-10-21T03:54:40.781652+00:00 | 2026-10-21T03:54:41.574440+00:00 |
| weekly_dataset | complete | 2026-10-21T03:54:41.578438+00:00 | 2026-10-21T03:54:45.766645+00:00 |
| splits | complete | 2026-10-21T03:54:45.769644+00:00 | 2026-10-21T03:54:45.969636+00:00 |
| modeling | complete | 2026-10-21T03:54:45.973636+00:00 | 2026-10-21T03:54:47.847157+00:00 |
| evaluation | complete | 2026-10-21T03:54:47.850156+00:00 | 2026-10-21T03:54:54.226909+00:00 |
| backtest | complete | 2026-10-21T03:54:54.230912+00:00 | 2026-10-21T03:54:57.046535+00:00 |

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

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\data\interim\experiments\real-smoke`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\data\processed\experiments\real-smoke`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\reports\experiments\real-smoke`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_earnings_retry\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\backtest`
