# 実験レポート: test-development

- run ID: `failed-run`
- 状態: `failed`
- 開始（UTC）: `2026-10-13T10:58:34.562626+00:00`
- 終了（UTC）: `2026-10-13T10:58:34.598381+00:00`
- 乱数seed: `1729`

> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。

## 入力識別子

### price_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\data\raw\price_snapshot`
- snapshot ID: `price-001`
- manifest SHA256: `0c8af354b50712d6a78bbee4ceda5def2ab9029cfeac2260efd1b4098b9a8cb0`
- 宣言済みファイルhash数: `1`

### auxiliary_snapshot

- path: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\data\raw\aux_snapshot`
- snapshot ID: `aux-001`
- manifest SHA256: `4ffaa3472469b13f1bfe6d00e90c7856ed2ef94133ab1d8050b1a3660a3865cc`
- 宣言済みファイルhash数: `1`

## 工程

| 工程 | 状態 | 開始（UTC） | 終了（UTC） |
| --- | --- | --- | --- |
| preprocessing | failed | 2026-10-13T10:58:34.594665+00:00 | 2026-10-13T10:58:34.598381+00:00 |

## 停止理由

`RuntimeError: synthetic failure`

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

- interim: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\data\interim\experiments\failed-run`
- processed: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\data\processed\experiments\failed-run`
- artifacts: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\artifacts\experiments\failed-run`
- reports: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\reports\experiments\failed-run`
- modeling: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\artifacts\experiments\failed-run\modeling`
- evaluation: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\artifacts\experiments\failed-run\evaluation`
- backtest: `C:\Users\kyoma\Desktop\hobby\stock_prediction\.test_tmp_experiment\test_failure_stops_downstream_0\project\artifacts\experiments\failed-run\backtest`
