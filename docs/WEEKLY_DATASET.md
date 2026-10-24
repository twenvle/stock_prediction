# 週次学習表のデータ辞書

## 1. 責務と時点基準

`stock_prediction.weekly_dataset` は、公式取引カレンダーから週次日程を作り、監査済み調整済み価格、
日次特徴量、時点整合ユニバース、決算イベント判定を週次主キーで結合する。欠損補完、外れ値処理、
標準化、特徴量選択、目的変数の最終選択、データ分割は行わない。

既定設定は `configs/weekly_dataset.json` に保存する。

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `min_trading_sessions_per_week` | 4 | 4取引日以上の完全な暦週だけを採用 |
| `require_topix` | `true` | TOPIX同期間リターンの完全なcoverageを必須にする |
| `signal_start_date` | `2017-02-06` | 現在の10年スナップショットで最初に採用する分析シグナル日 |

日程は月曜から日曜までの暦週単位で作る。カレンダー入力の先頭・末尾で週全体を確認できない週、
3取引日以下の週、最初の取引日が月曜・火曜以外の週、最後の取引日が木曜・金曜以外の週は除外する。
`entry_date` は `signal_date` の直後の東証取引日で、`exit_date` がREADMEの評価日 `e(t)` に相当する。
`signal_start_date` より前の週は `signal_before_configured_start` としてauditへ残すが、元の日次データは
`R_60`、上場履歴、流動性、決算クールダウンのウォームアップに使う。現在の設定では
2016-11-01〜2017-02-03がウォームアップ専用で、最初の採用週は2017-02-06である。

## 2. DataFrame入力

### 2.1 監査済み日次株価

`preprocessing.py` の出力から次の列だけを使う。

| 列 | 条件・用途 |
|---|---|
| `date`, `code` | 一意な日次主キー。コードは5文字の大文字英数字 |
| `adjusted_close` | 正の有限な調整済み終値。目的変数だけに使用 |
| `has_price` | 真の行だけを価格観測として使用 |
| `calendar_verified` | 全行が真でなければ停止 |

価格は補完しない。シグナル日価格がない `is_eligible=True` 行は工程間の不整合として停止する。
評価日価格がない行は停止せず、目的変数欠損として件数と理由を残す。上場廃止直前の価格欠損を
通常欠損として扱わないためであり、後続工程で上場廃止時の損益規則を別途定義する必要がある。

### 2.2 日次特徴量

`descriptors.py` の `date`, `code` とREADME 7節の11特徴量を必要とする。シグナル日行だけを
`(signal_date, code)` へ写像する。ユニバース合格行の特徴量行そのものが欠落していれば停止するが、
個々の特徴量Nullは保持し、`feature_missing_count` と `has_complete_features` で監査する。
Nullを全期間統計で補完しない。

### 2.3 時点整合ユニバース

最低限 `signal_date`, `code`, `entry_date`, `exit_date`, `is_eligible`, `exclusion_reasons` を必要とし、
それ以外のユニバース監査列と事後約定proxyもそのまま出力へ残す。日程はこのモジュールが公式
カレンダーから再生成した値と完全一致しなければならない。`is_executable` は将来情報を含むため、
`is_signal_eligible*` には使用しない。

### 2.4 決算イベント判定

`earnings_events.py` の15追加列と週次主キーを必要とする。キー集合はユニバースと完全一致させる。
決算出力にユニバース列のコピーが含まれる場合は値も照合し、不一致を黙って採用しない。
主判定には `is_earnings_eligible`、クールダウンなし感度分析には
`is_earnings_eligible_without_cooldown` を使う。`is_earnings_eligible_ex_post` は監査専用である。

### 2.5 TOPIXと取引カレンダー

取引カレンダーは連続した全暦日の `Date` / `HolDiv` 2列を必要とする。TOPIXは正規化済みの
`date` / `close` 2列で、日付は一意な東証取引日、終値は正の有限値またはNullとする。
既定設定では、出力対象週のシグナル日または評価日にTOPIX終値がなければ停止する。

## 3. 目的変数と適格性

銘柄 `i`、シグナル日 `t`、評価日 `e(t)` の調整済み終値を `P` とする。

```text
target_return(i,t) = P(i,e(t)) / P(i,t) - 1
target_topix_return(t) = TOPIX(e(t)) / TOPIX(t) - 1
target_excess_return(i,t) = target_return(i,t) - target_topix_return(t)
```

将来観測の有無と、シグナル時点で選択可能だったかを混ぜない。

| 列 | 意味 |
|---|---|
| `is_signal_eligible_without_cooldown` | `is_eligible` かつ予定決算跨ぎなし。クールダウンなし感度分析 |
| `is_signal_eligible` | `is_eligible` かつ主決算判定合格。シグナル時点情報だけ |
| `has_target_return` | 銘柄のシグナル日・評価日価格から単純リターンを観測できる |
| `has_target_excess_return` | 単純リターンと同期間TOPIXリターンをともに観測できる |
| `is_target_return_sample` | `is_signal_eligible` かつ `has_target_return` |
| `is_target_excess_return_sample` | `is_signal_eligible` かつ `has_target_excess_return` |

`target_missing_reasons` はシグナル価格、評価日価格、TOPIXのどれが不足したかを残す。
回帰ターゲットを単純リターンと超過リターンのどちらにするかは本工程で選ばない。日付横断順位や
標準化リターンも、2025年以降を見ない開発期間内のモデル比較で決める。

## 4. API、保存形式、監査

- `build_weekly_schedule`：Notebook用の日程DataFrame API。採用日程、全週audit、reportを返す。
- `build_weekly_dataset`：Notebook用の結合DataFrame API。入力DataFrameを変更しない。
- `run_weekly_schedule`：ユニバース工程へ渡す3列ParquetとJSON reportを新規保存する。
- `run_weekly_dataset`：年別入力を読み、年別Parquet・JSON・manifestを新規保存する。

```text
data/interim/weekly_schedule_<実行ID>.parquet
reports/weekly_schedule_<実行ID>.json

data/processed/weekly_dataset_<実行ID>/
  YYYY.parquet
reports/weekly_dataset_<実行ID>/
  YYYY.json
  manifest.json
```

manifestには入力ファイルの行数・列・SHA256、設定、カレンダー、TOPIX、出力行数、シグナル適格数、
目的変数別サンプル数、コードSHA256、ライブラリ版を保存する。年別処理中に通常例外または
`KeyboardInterrupt` が発生しても、完了済み年と現在年をmanifestへ残す。既存出力は上書きしない。

## 5. 既知の制限

- 現在の実データには、この工程に必要な時点銘柄マスター、決算予定・実績、生成済み特徴量、
  週次ユニバースが揃っていないため、実データ学習表の件数・coverageは未検証である。
- 調整済み価格は後日のコーポレートアクションで遡及改定され得る。入力スナップショットの取得時点と
  SHA256を実験記録へ残し、当時利用可能だった絶対水準と誤認しない。
- 評価日価格欠損、上場廃止、コード変更、合併の経済損益をこの工程では推測しない。
- 出力には事後の `is_executable` と `is_earnings_eligible_ex_post` が残るが、特徴量や
  シグナル時点の銘柄選択へ使わない。
