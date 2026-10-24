# 決算イベント判定のデータ辞書

## 1. 責務と時点基準

`stock_prediction.earnings_events` は、週次候補に決算跨ぎと決算後クールダウンの判定を付ける。
モデル特徴量、目的変数、元のユニバース判定、売買可能性は変更しない。入力DataFrameは変更せず、
CLI出力は `data/processed/`、監査結果は `reports/` の新規ディレクトリへ保存する。

主判定で利用する情報は次に限定する。

- 予定：`PubDate <= signal_date` の公表行。同一 `Code` / `FQName` / `FYE` では、その時点で
  最新の公表行だけを使う。
- クールダウン：シグナル確定時刻までの実績決算短信。`DiscDate == signal_date` では
  `DiscTime` を東証大引け（2024-11-05より前は15:00、同日以後は15:30）と比較し、
  大引け以前だけを利用可能とする。大引け後または時刻欠損は事後イベントとして分離する。
- 取引日数：公式の `Date` / `HolDiv` カレンダー。`HolDiv` が `"1"` または `"2"` の日だけを
  東証取引日として数え、平日や個別銘柄の価格観測日から推測しない。

予定日は時刻を持たないため、主判定の跨ぎ区間は保守的に `[signal_date, exit_date]` とする。
大引けまでに発表済みのシグナル日を0日目、発表日の次の東証取引日を1日目とし、0〜25日目は
クールダウン、26日目から解除する。

## 2. 入力

### 2.1 週次候補

必須列は次の3列。他の列は変更せず出力へ引き継ぐ。

| 列 | 型 | 条件 |
|---|---|---|
| `signal_date` | 日付 | 暦週の最初の東証取引日。Null不可 |
| `code` | 文字列 | 5文字の大文字英数字。Null不可 |
| `exit_date` | 日付 | 同じ暦週の最後の東証取引日かつ `signal_date` より後 |

`(signal_date, code)` は一意でなければならない。任意列 `entry_date` がある場合は、
`signal_date` の直後の東証取引日であり、`exit_date` 以前であることも検査する。

### 2.2 決算発表予定

J-Quants V2 `earnings_schedule/YYYY.parquet` の厳密な7列を受け取る。

| 列 | 意味 |
|---|---|
| `PubDate` | 予定情報が公表され利用可能になった日 |
| `SchDate` | 予定決算発表日。Nullは日付不明として監査列へ残す |
| `FQName` / `FYE` | 同一決算イベントの予定変更を識別する四半期名・決算期末 |
| `Code` | 銘柄コード |
| `CoName` / `CoNameEn` | 会社名。判定には使わない |

同じ `Code` / `FQName` / `FYE` / `PubDate` に複数行があっても、公表時刻がなく同日中の順序を
確定できないため、一方を恣意的に最新とはみなさず、その公表日の候補状態の集合として保持する。
既知の `SchDate` のいずれかが保有期間へ入れば跨ぎとして除外するが、同じ決算イベントは候補日が
複数あっても1件として数え、最初の日付を `first_scheduled_earnings_date` に残す。集合にNullが
含まれる有効期間は `active_schedule_date_missing=True` として監査する。Nullだけでは日付が
不明なので機械的な跨ぎ除外は行わない。次の異なる `PubDate` からは新しい集合だけを有効にする。
全列が同じ完全重複行は、取得または結合の重複として早期エラーにする。

### 2.3 財務サマリー実績

判定に必要な列は `DiscDate` / `DiscTime` / `Code` / `DiscNo` / `DocType`。
取得モジュールが許容する既知の財務列は追加されていてよいが、未知列は拒否する。

既定設定では実入力で確認した `1QFinancialStatements` / `2QFinancialStatements` /
`3QFinancialStatements` / `FYFinancialStatements` / `OtherPeriodFinancialStatements` の
各接頭辞だけを決算短信とみなす。この接頭辞集合は `configs/earnings_events.json` に保存し、
業績・配当予想修正等をクールダウン開始へ
混ぜない。同じ銘柄・開示日に複数の決算短信行があっても、クールダウンの日数計算では1イベント日と
して数える。同日行では最も早い既知の `DiscTime` を利用可能性判定に使い、時刻がすべてNullなら
大引けまでに利用可能だったとは認定しない。
`FinancialStatements` を含む未知の `DocType` が設定に一致しない場合は、クールダウン0件として
黙って進めずschema errorで停止する。

### 2.4 取引カレンダー

列は正確に `Date` / `HolDiv`。対象範囲の閉場日も含む連続した全暦日が必要で、最初の
`signal_date` より前に少なくとも設定日数（既定25）の東証取引日を含まなければならない。

## 3. 追加出力列

| 列 | 型 | 意味 |
|---|---|---|
| `scheduled_earnings_during_holding` | bool | その時点で公表済みの最新予定が保有期間に入る |
| `scheduled_earnings_count` | int | 該当する予定決算イベント数 |
| `first_scheduled_earnings_date` | 日付/Null | 最初の該当予定日 |
| `active_schedule_date_missing` | bool | その時点で有効な予定にNull日付がある |
| `actual_earnings_during_holding` | bool | 実際の決算短信が保有期間にあった事後判定 |
| `actual_earnings_count` | int | 保有期間中の実績イベント日数 |
| `first_actual_earnings_date` | 日付/Null | 最初の実績イベント日 |
| `last_known_earnings_date` | 日付/Null | シグナル確定時刻までに判明した直近実績決算日 |
| `trading_sessions_since_last_earnings` | nullable int | 直近実績の後、シグナル日までに経過した東証取引日数 |
| `is_in_post_earnings_cooldown` | bool | 上記日数が0〜25。週末発表後の最初の取引日も1 |
| `is_earnings_eligible_without_cooldown` | bool | 予定跨ぎだけを除いた感度分析列 |
| `is_earnings_eligible` | bool | 予定跨ぎなし、かつクールダウン外の主判定 |
| `is_earnings_eligible_ex_post` | bool | 主判定に実績跨ぎ監査も加えた事後列。シグナル時選択には使わない |
| `earnings_exclusion_reasons` | string | 主判定の除外理由をセミコロン区切りで保持 |
| `earnings_ex_post_exclusion_reasons` | string | 事後実績も含む除外理由 |

`is_earnings_eligible*` は決算工程だけの判定であり、元候補の `is_eligible` を置き換えない。
後続の週次学習表で `(signal_date, code)` をキーに明示的に結合する。

## 4. CLI出力と監査

CLIは候補を年別に処理し、必要な決算イベント列だけを全イベント年から読み込む。各年の
Parquet、年別JSON、全体manifestを保存する。manifestには設定、入力SHA256、実日付範囲、候補行数、
銘柄数、予定跨ぎ件数、クールダウン件数、実績跨ぎ件数、各判定のcoverage、ライブラリ版、
コードSHA256を記録する。同日複数状態は `same_day_schedule_state_groups`、該当入力行数は
`same_day_schedule_state_rows`、そのうちNull予定日を含むグループ数は
`same_day_schedule_state_groups_with_missing_date` に記録する。既存出力は上書きしない。
失敗または中断時も、出力先作成後に完了した年と状態をmanifestへ残す。

イベントファイル自体には「0件だった日」の行がないため、Parquetの日付最小・最大だけでは取得期間の
完全性を証明できない。使用前に補助取得スナップショットの `manifest.json` で、予定・財務実績が
必要な全暦日に対して取得完了していることを確認する。
