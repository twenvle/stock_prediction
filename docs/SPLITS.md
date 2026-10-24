# 時系列分割と最終評価期間の隔離

## 1. 責務と固定境界

`stock_prediction.splits` は、週次学習表から拡大型ウォークフォワード分割を作り、最終評価期間を
開発foldから隔離する。特徴量や目的変数の選択、適格行の絞り込み、欠損補完、外れ値処理、標準化は
行わない。これらは各foldの学習データだけでfitする後続工程の責務である。

研究仕様として次を固定する。

| 区分 | 期間 |
|---|---|
| 開発 | `2016-01-01`〜`2024-12-31` |
| fold 1 | 学習2016〜2019、検証2020 |
| fold 2 | 学習2016〜2020、検証2021 |
| fold 3 | 学習2016〜2021、検証2022 |
| fold 4 | 学習2016〜2022、検証2023 |
| fold 5 | 学習2016〜2023、検証2024 |
| 最終評価候補 | 設定で選んだ2025〜2026年内の期間 |

開発行は `signal_date >= 2016-01-01` かつ `exit_date <= 2024-12-31` を満たす完全なラベル期間だけを
対象にする。2024年中にシグナルがあってもラベル終了が2025年へ入る行は開発へ入れない。

## 2. 入力契約

DataFrame APIは最低限、次の列を必要とする。追加列は変更せず、分割判定にも使わない。

| 列 | 条件 |
|---|---|
| `signal_date` | Nullなし、日本の取引日を表すtimezoneなしの午前0時日付 |
| `code` | Nullなし、5文字の大文字英数字 |
| `exit_date` | Nullなし、`signal_date` 以後の公式東証取引日 |

`(signal_date, code)` は一意でなければならず、同じ `signal_date` の全銘柄は同じ `exit_date` を
持たなければならない。公式取引カレンダーはJ-Quants V2の `Date` / `HolDiv` 2列とし、入力の
全シグナル日・評価日とパージ履歴を覆う連続した全暦日を必要とする。

## 3. パージ

`purge_trading_sessions` は公式カレンダー上の取引日数で指定する。既定値は4で、READMEが定義する
通常週の月曜終値から金曜終値までの4取引日差に対応する。入力開発期間で観測した
`exit_date - signal_date` の最大取引日差より小さい設定はエラーにする。

検証年の最初のシグナルを `v`、公式カレンダー上で `v` の `p` 取引日前を `c` とすると、
学習行は次をすべて満たすものだけである。

```text
2016-01-01 <= signal_date < v
signal_date <= c
exit_date < v
```

このため、設定幅を満たさない学習候補と、ラベルが検証開始へ達する候補はfoldごとに `purged` として
記録される。パージは銘柄行単位ではなくシグナル日単位で適用する。

## 4. 隔離期間と選択外行

`configs/splits.json` の設定は次のとおりである。

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `isolation_start_date` | `2025-01-01` | 最終評価候補の開始日（両端を含む） |
| `isolation_end_date` | `2026-12-31` | 最終評価候補の終了日（両端を含む） |
| `purge_trading_sessions` | `4` | 検証前に確保する最小取引日差 |

研究仕様を誤って変更しないよう、隔離境界は2025〜2026年内だけを選べる。隔離行には
`signal_date >= isolation_start_date` かつ `exit_date <= isolation_end_date` の完全なラベル期間だけを
含める。選択した隔離期間の前後、2016年より前、2024年末または隔離終了をラベルが跨ぐ行は
`out_of_scope` として理由付きで残し、開発foldへ流用しない。

隔離期間を狭める操作は既存の最終評価を未使用へ戻すものではない。一度でも最終評価結果を見て
設計を変更した場合、その期間を再び未使用の最終評価として扱わない。

## 5. APIと保存形式

- `build_walk_forward_splits(dataset, trading_calendar, config=...)`：Notebook向けDataFrame API。
  入力を変更せず、各foldの `train_positions` / `validation_positions` / `purged_positions`、
  `isolated_positions`、`out_of_scope_positions` を `iloc` 用の位置配列として返す。
- `run_splits(...)`：CLIと共通の保存API。年別週次Parquetから必要な3列だけを読み、日付単位の
  分割定義とmanifestを新規出力へ保存する。

```text
data/processed/splits_<実行ID>/
  folds.parquet
  isolated.parquet
  out_of_scope.parquet
reports/splits_<実行ID>/
  manifest.json
```

`folds.parquet` は `fold_id`, `validation_year`, `role`, `signal_date`, `exit_date` を持つ。
`role` は `train`, `validation`, `purged` のいずれかである。隔離ファイルは最終評価候補の
シグナル日定義だけを持ち、特徴量や目的変数を複製しない。後続工程は日付定義または返された位置配列で
元の週次表を選択する。manifestには入力ファイルSHA256、行数・列、日付範囲、銘柄数、fold別件数、
パージ件数、隔離・選択外件数、設定、コードSHA256、ライブラリ版を記録する。

既存出力は上書きしない。保存中の失敗または `KeyboardInterrupt` では、manifestの `status` と
`completed_outputs` に完了済み出力を残す。

## 6. 既知の制限

- 同一スナップショットから生成した実データ週次学習表がまだないため、実foldの行数、銘柄数、
  coverage、実際の年境界パージ件数は未検証である。
- 分割は `is_signal_eligible` や `is_target_*_sample` を参照しない。後続工程で対象行を選ぶ際も、
  日付割当を変えず、同じシグナル日の全銘柄を同じfoldに保つ必要がある。
- 出力は分割定義であり、最終評価データの暗号化やアクセス権分離ではない。開発中は
  `isolated.parquet` で指定された行の目的変数・評価結果を読み込まない運用が必要である。
