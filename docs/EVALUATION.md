# 予測評価仕様

## 1. 責務と時点整合性

`stock_prediction.evaluation` は、モデルが既に生成したシグナル時点適格行の予測を評価する。
モデルのfit、前処理、特徴量選択、閾値選択は行わない。日付別の横断面で指標を計算してから、
全期間、年別、業種別、年別業種別に時間方向へ集約する。

業種はJ-Quantsの時点銘柄マスターを `(signal_date, code)` で完全一致結合する。欠損時に前後日の
分類や現在の分類を補完しない。既定は33業種の `S33` / `S33Nm` であり、17業種を使う場合は
`configs/evaluation.json` の2列を `S17` / `S17Nm` へ変更した実験用設定を保存する。2025年以降の
結果を見て分類粒度、N、評価対象モデルを変更し、その期間を未使用の最終評価として扱ってはならない。

## 2. 設定と入力契約

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `top_n` | 10 | 日付別・業種日付別の上位/下位選択数 |
| `industry_date_column` | `Date` | 時点銘柄マスターの日付列 |
| `industry_security_column` | `Code` | 時点銘柄マスターの銘柄コード列 |
| `industry_code_column` | `S33` | 集計に使う業種コード列 |
| `industry_name_column` | `S33Nm` | 集計に使う業種名列 |

予測入力は `modeling.py` の `predictions.parquet` と同じ次の列を必要とする。

| 列 | 条件・用途 |
|---|---|
| `fold_id`, `validation_year` | 検証foldと年。年は `signal_date` の年と一致 |
| `signal_date`, `code` | シグナル日と5文字銘柄コード |
| `model`, `target_column` | 比較するモデル名とtarget名 |
| `actual_target`, `target_observed` | 有限実現値またはNullと、その観測可否 |
| `prediction` | 有限予測値またはNull。Nullはcoverage低下として保持 |

主キーは `(fold_id, signal_date, code, model, target_column)` とする。同じtarget・fold・日付では
全モデルの適格銘柄集合が完全一致しなければ停止する。モデル間で都合のよい評価対象だけを変えない。
時点銘柄マスターは設定した4列を必要とし、日付・コード主キーの重複を拒否する。

CLIは予測に含まれる年の `YYYY.parquet` だけを選択してからschemaを確認し、必要4列だけを読む。
Notebook APIは渡されたDataFrameを変更しない。

## 3. 日付別指標

シグナル日 `t` の適格銘柄数を `E_t`、target観測数を `Y_t`、予測生成数を `P_t`、両方がある
評価可能数を `S_t` とする。

```text
target_coverage(t)     = Y_t / E_t
prediction_coverage(t) = P_t / E_t
scoring_coverage(t)    = S_t / E_t
```

ICは同日の `S_t` 行だけで計算する。Spearman順位相関を主IC、Pearson相関を診断列とする。
2行未満、予測定数、target定数なら相関はNullとし、0へ置換しない。

上位・下位はtargetの有無を見ず、有限予測の降順、同値ならコード昇順で固定する。有効予測がN未満なら
存在する行だけを選び、`selection_count` と `top_n_complete` で不足を示す。選択後にtarget欠損が
判明しても、次順位の銘柄で埋めない。選択銘柄のtargetが全件観測できた場合だけ平均実現リターンと
上下位差を出し、それ以外はNullとする。`top_n_target_coverage` と
`bottom_n_target_coverage` に欠損の影響を残す。

業種別指標は、完全一致で業種が得られた行だけを各日・各業種内で同じ方法により計算する。
未分類行は「不明業種」へまとめず、全体・年別の `industry_coverage` とmanifestに件数を残す。

## 4. 時間方向の集約

IC、上位N平均、下位N平均、上下位差は、各シグナル日を1観測として等重みで平均・中央値・標本標準偏差を
出す。coverageは期間内の件数合計から求める行加重比率である。`*_valid_dates`、N充足日数、
選択target完全観測日数、選択数の最小・最大も保存する。銘柄行を独立標本とみなした標準誤差や
信頼区間は作らない。

業種内の上位Nと下位Nは、業種の有効予測数が `2N` 未満なら重複し得る。その日を隠さず、
`selection_count` とN充足日数を確認して解釈する。

## 5. APIと成果物

- `evaluate_predictions(predictions, listed_info, config=...)`：Notebook向けDataFrame API。
- `run_evaluation(...)`：保存済み予測と年別listed-infoを読む保存API。
- `stock-evaluation` / `python -m stock_prediction.evaluation`：同じ保存APIを呼ぶCLI。

```text
artifacts/evaluation_<実行ID>/
  predictions_with_industry.parquet
  daily_metrics.parquet
  summary_metrics.parquet
  yearly_metrics.parquet
  industry_daily_metrics.parquet
  industry_metrics.parquet
  industry_yearly_metrics.parquet
reports/evaluation_<実行ID>/
  manifest.json
```

`predictions_with_industry.parquet` は完全一致結合結果と `industry_observed` を保持する。
日付別成果物が再集約可能な最小時間単位で、summary/yearly/industry成果物はその時間方向の要約である。
manifestには設定、入力SHA256、読込年、対象期間・件数、分類coverage、コードSHA256、環境、
完了済み成果物を保存する。既存出力は上書きしない。失敗または中断時は `status`, `error_type`,
`error_message`, `completed_outputs` を残し、新しい実行IDで再実行する。

## 6. 制限

- 本指標はシグナル日終値から同週評価日終値までの予測評価であり、約定可能価格、コスト、未約定、
  売買単位を含むバックテストではない。
- 実データ週次学習表・予測成果物がまだないため、現時点の検証値は合成データだけである。
- target欠損には上場廃止等が含まれ得る。coverage低下を性能値と分離し、欠損日の上位Nを良好な
  サンプルだけで再構成しない。
- 業種区分は予測時点の分類であり、業種重要度やICを因果効果として解釈しない。
