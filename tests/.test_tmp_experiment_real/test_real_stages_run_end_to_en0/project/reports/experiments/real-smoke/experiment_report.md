# 実験レポート: test-development

- run ID: `real-smoke`
- 状態: `failed`
- 開始（UTC）: `2026-10-13T11:03:37.667534+00:00`
- 終了（UTC）: `2026-10-13T11:03:41.984569+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\data\raw\price_snapshot`
- snapshot ID: `synthetic-price`
- manifest SHA256: `c17fb0b315a7611a5ab70bfc7d75ac69b6bebcbb17aae200caf6117780bb1e23`
- 宣言済みファイルhash数: `9`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\data\raw\aux_snapshot`
- snapshot ID: `synthetic-auxiliary`
- manifest SHA256: `abfef1c14e1b9bdb21c74971f0381507adf3ae26e3780c16ad1ecbd40b7b2a2f`
- 宣言済みファイルhash数: `0`

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | complete | 2026-10-13T11:03:37.687834+00:00 | 2026-10-13T11:03:38.989141+00:00 |
| topix_normalization | complete | 2026-10-13T11:03:38.993590+00:00 | 2026-10-13T11:03:39.017606+00:00 |
| descriptors | complete | 2026-10-13T11:03:39.022600+00:00 | 2026-10-13T11:03:40.021479+00:00 |
| weekly_schedule | complete | 2026-10-13T11:03:40.025479+00:00 | 2026-10-13T11:03:40.369080+00:00 |
| universe | failed | 2026-10-13T11:03:40.374011+00:00 | 2026-10-13T11:03:41.983568+00:00 |

## 停止理由

`ValueError: Missing price partition for execution year 2025`

## 開発期間の予測評価

工程が未完了のため、結果表はありません。

## 開発期間の売買シミュレーション

工程が未完了のため、結果表はありません。

## 警告・制限

- This experiment evaluates development-period predictions only; it does not run or unlock the 2025-2026 final period.
- J-Quants acquisition is intentionally outside the experiment run; immutable snapshot manifests identify the inputs.
- No confirmed delisting settlement table is supplied; the backtest will stop if a confirmed event must be settled.
- 入力rawスナップショットは変更せず、派生データと成果物を別ディレクトリへ保存しています。
- 個別工程の件数、欠損、coverage、除外理由、入力SHA256は各工程のmanifestを確認してください。
- バックテストは板・注文優先順位を再現せず、配当と税金を含まない価格リターンの研究用シミュレーションです。

## 保存先

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\data\interim\experiments\real-smoke`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\data\processed\experiments\real-smoke`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\reports\experiments\real-smoke`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment_real\test_real_stages_run_end_to_en0\project\artifacts\experiments\real-smoke\backtest`
