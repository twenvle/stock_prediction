# 基準モデルとfold内前処理

## 1. 責務と期間隔離

`stock_prediction.modeling` は、`splits.py` のパージ付き拡大型ウォークフォワード分割を使い、
2016〜2024年の開発期間だけで基準モデルをfit・評価する。2020〜2024年を各1年の検証期間とし、
同じシグナル日の全銘柄は同じfoldに留める。

CLIは入力ディレクトリに2025年・2026年のParquetが存在しても、ファイル名で2016〜2024年だけを
選んでからschemaと値を読む。DataFrame APIは最初に `build_walk_forward_splits` で日付だけを分割し、
返された `development_positions` 以外の特徴量、適格性、目的変数を前処理・fit・評価へ渡さない。
これは論理的隔離であり、OS権限による物理的隔離ではない。

## 2. 設定と入力契約

既定設定は `configs/modeling.json` に保存する。

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `target_column` | `target_return` | `target_return` または `target_excess_return` |
| `feature_columns` | README 7節の11列 | 線形モデルへ固定順で渡す特徴量 |
| `momentum_feature` | `R_20` | 素朴モメンタム予測に使う列 |
| `top_n` | 10 | 日付別の上位・下位集計数 |

`target_return` は実行既定値であり、最終ターゲットの研究上の選択ではない。単純リターンと
TOPIX超過リターンの比較・最終選択は、2025年以降を見ずに開発期間だけで行う。

最低限、`signal_date`, `code`, `exit_date`, `is_signal_eligible`、選択targetと対応する
`is_target_*_sample`、設定した全特徴量を必要とする。主キー、日付、コード、bool、数値型、
非有限値を検査し、`is_target_*_sample` が「シグナル時点適格かつtarget観測済み」と一致しなければ
停止する。Null特徴量は保持し、fold内でだけ補完する。

## 3. モデルとfold内fit

各foldで、学習は対応する `is_target_*_sample=True` 行だけ、予測は検証年の
`is_signal_eligible=True` 行へ行う。目的変数が後で欠損と判明した行にも予測を残し、評価可能行と
混同しない。

| モデル | fitと予測 |
|---|---|
| `constant` | 学習foldのtarget平均を全検証行へ予測 |
| `momentum` | `R_20` を直接予測値に使用。欠損だけを学習foldの `R_20` 中央値で補完 |
| `linear` | 学習foldだけで各特徴量の中央値補完、平均、母標準偏差をfitし、切片あり最小二乗をfit |

線形モデルの検証行には保存済み統計で `transform` だけを行う。学習foldで全件Nullの特徴量は、
推測値を作らず停止する。定数特徴量のscaleは1とする。外れ値clip、特徴量選択、正則化は本基準には
含めず、後続比較で導入する場合も各学習fold内だけでfitする。

NumPyの `linalg.lstsq` を使うため乱数は使わない。設計行列がrank deficientの場合は最小ノルム解を
用い、rankと特異値を成果物へ残す。係数は予測上の関連であり、因果効果ではない。

## 4. 評価と成果物

- 行単位：MAE、RMSE、方向一致率
- シグナル日単位：Pearson、Spearman、上位N平均target、下位N平均target、上下位差
- coverage：シグナル時点適格行に対するtarget観測率と予測生成率を別々に記録

相関は同じシグナル日の観測済み銘柄が2件以上あり、予測・実現値がともに非定数の場合だけ計算する。
上位・下位は予測降順、同値ならコード昇順で決める。日付別指標の平均は時間方向の要約であり、
銘柄行を独立標本とみなした信頼区間は報告しない。

```text
artifacts/modeling_<実行ID>/
  predictions.parquet
  fold_metrics.parquet
  daily_metrics.parquet
  model_parameters.json
reports/modeling_<実行ID>/
  manifest.json
```

`model_parameters.json` はfold別の定数、モメンタム補完中央値、線形前処理統計、係数、rank、特異値を
JSONで保存する。manifestは設定、分割設定、2016〜2024年入力SHA256、カレンダーSHA256、期間・件数、
欠損率、環境、コードSHA256、完了済み出力を持つ。既存出力は上書きしない。失敗または中断時は
`status`, `error_type`, `error_message`, `completed_outputs` を残し、新しい実行IDで再実行する。

## 5. 既知の制限

- 実データ週次学習表が未生成のため、現時点の検証は合成データであり、実際のcoverageや性能ではない。
- 3モデルは予測精度の基準であり、取引コストや約定制約を含む売買バックテストではない。
- `target_return` と `target_excess_return` のどちらを最終採用するかは未確定である。
- OLSの係数は共線性により不安定になり得る。Ridge等を比較する場合も開発期間内だけで選択する。
