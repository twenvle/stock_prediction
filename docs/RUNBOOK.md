# データ取得から予測評価・バックテストまでの実行手順

本書は次の12モジュールをCLIまたはNotebookから実行するための例をまとめる。

1. `stock_prediction.data_download`：J-Quants V2原データ取得
2. `stock_prediction.auxiliary_data_download`：TOPIX・時点銘柄マスター・決算予定/実績取得
3. `stock_prediction.preprocessing`：日次株価の品質検査と中間データ生成
4. `stock_prediction.descriptors`：日次テクニカル特徴量生成
5. `stock_prediction.universe`：時点整合ユニバースと売買可能性proxy生成
6. `stock_prediction.earnings_events`：決算跨ぎ除外と決算後25取引日クールダウン
7. `stock_prediction.weekly_dataset`：週次日程、目的変数、特徴量・ユニバース・決算判定の結合
8. `stock_prediction.splits`：パージ付きウォークフォワード分割と最終評価期間の隔離
9. `stock_prediction.modeling`：定数・モメンタム・線形モデルとfold内前処理
10. `stock_prediction.evaluation`：IC・上位N・coverageの年別/業種別評価
11. `stock_prediction.backtest`：約定制約・コスト・上場廃止を含む売買バックテスト
12. `stock_prediction.experiment`：取得済みスナップショットから全工程を再実行する実験統括

CLI例はWindowsの **Command Prompt（cmd.exe）** を既定とする。PowerShell固有の構文は使わない。
すべてプロジェクトルートから実行する。初回は、CLIとNotebookで使用する同じPython環境へ
パッケージをインストールする。

```cmd
python -m pip install -e ".[dev]"
```

各保存APIは既存出力を上書きしない。例にある実行IDまたは出力先は毎回新しい値を使う。
`data/raw/` の原本を手作業で編集しない。

## 最初から実行する場合の推奨ルート

通常は各工程を個別に手入力せず、次の順序で実行する。

| 順序 | 実行ファイルまたは設定                   | 役割                                                 | 主な出力                                |
| ---: | ---------------------------------------- | ---------------------------------------------------- | --------------------------------------- |
|    0 | `pyproject.toml` / `environment.yml` | 実行環境を作る                                       | インストール済みの `stock_prediction` |
|    1 | `data_download.py`                     | 日次株価と公式取引カレンダーを取得する               | 価格raw snapshot                        |
|    2 | `auxiliary_data_download.py`           | TOPIX、時点銘柄マスター、決算予定・実績を取得する    | 補助raw snapshot                        |
|    3 | `configs/experiment.example.json`      | 2つのsnapshotと工程設定を固定する                    | `configs/experiment.local.json`       |
|    4 | `experiment.py --validate-only`        | 入力、schema、期間、設定、未使用run IDを検査する     | 書き込みなしの検査結果                  |
|    5 | `experiment.py`                        | 前処理から開発期間バックテストまでを依存順に実行する | 中間データ、予測、評価、レポート        |
|    6 | `reports/experiments/<run-id>/`        | manifestと実験レポートを確認する                     | 監査可能な開発期間の結果                |

ステップ5の `experiment.py` は、後述する3〜12節の前処理、特徴量、週次日程、ユニバース、
決算除外、週次学習表、分割、モデル検証、評価、バックテストを内部で順に呼ぶ。通常の一括実行で
これらを別々に先行実行する必要はない。個別CLIは、小期間の監査、schema確認、失敗原因の切り分け、
Notebookでの探索に使う。

### 1. 価格raw snapshotを取得する

現在のStandard 10年枠では2015年側を取得できないため、2016-11-01から取得し、最初の分析シグナルを
2017-02-06に固定する。2016-11-01〜2017-02-03は60取引日の特徴量・上場履歴、流動性、
25取引日の決算クールダウンに使うウォームアップ専用期間である。より長い契約期間のsnapshotへ
切り替える場合も、開始日は実データのcoverageを監査してから別設定として確定する。次の
`END_DATE` と `RUN_TAG` は実行時点に合わせて変更する。

```cmd
set "START_DATE=2016-11-01"
set "END_DATE=2026-10-01"
set "RUN_TAG=20261014_120000"
set "RAW_NAME=jquants_v2_prices_%RUN_TAG%"
set "RAW_DIR=data\raw\%RAW_NAME%"

python -m stock_prediction.data_download ^
  --start-date %START_DATE% ^
  --end-date %END_DATE% ^
  --output-root data\raw ^
  --output-dir "%RAW_NAME%"

type "%RAW_DIR%\manifest.json"
```

APIキーはコマンドへ書かない。環境変数 `JQUANTS_API_KEY` がなければ非表示プロンプトで入力する。
`manifest.json` が `status=completed` で、要求した全取引日が完了していることを確認する。

### 2. 同じ期間の補助raw snapshotを取得する

価格snapshotの取引カレンダーを使い、同じ `START_DATE` / `END_DATE` を指定する。取得量と契約プランを
確認するため、初回は2節の例で数日間のsmoke取得を行ってから全期間を取得する。

```cmd
set "AUX_NAME=jquants_v2_auxiliary_%RUN_TAG%"
set "AUX_DIR=data\raw\%AUX_NAME%"

python -m stock_prediction.auxiliary_data_download ^
  --start-date %START_DATE% ^
  --end-date %END_DATE% ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --output-root data\raw ^
  --output-dir "%AUX_NAME%"

type "%AUX_DIR%\manifest.json"
```

`topix.parquet`、`listed_info/`、`earnings_schedule/`、`financial_results/` と完了manifestがすべて
存在することを確認する。失敗したsnapshot同士や異なる取得期間を手作業で結合しない。

### 3. 実験設定を作り、事前検査する

```powershell
Copy-Item configs\experiment.example.json configs\experiment.local.json
```

`configs/experiment.local.json` の `price_snapshot_dir` と `auxiliary_snapshot_dir` を、ステップ1と2で
作成したディレクトリへ変更する。確認済みの上場廃止清算表がある場合だけ `delistings_path` を設定する。
その後、まだ使っていないrun IDで書き込みなしの検査を行う。

```powershell
python -m stock_prediction.experiment `
  --config configs\experiment.local.json `
  --run-id baseline-dev-001 `
  --validate-only
```

`status=valid` になるまで本実行へ進まない。

### 4. 全工程を実行する

```powershell
python -m stock_prediction.experiment `
  --config configs\experiment.local.json `
  --run-id baseline-dev-001
```

完了後は、まず次を確認する。

```text
reports/experiments/baseline-dev-001/manifest.json
reports/experiments/baseline-dev-001/config.resolved.json
reports/experiments/baseline-dev-001/experiment_report.md
artifacts/experiments/baseline-dev-001/modeling/predictions.parquet
artifacts/experiments/baseline-dev-001/evaluation/
artifacts/experiments/baseline-dev-001/backtest/
```

### 現在の実装で到達できる範囲

現在の `modeling.py` と `experiment.py` が生成する予測は、2016〜2024年だけを使った
2020〜2024年のウォークフォワード検証予測である。これに対する評価と売買シミュレーションは
**開発期間のモデル比較**であり、2025〜2026年の最終バックテストではない。

次の2機能はまだ実装されていないため、現時点では対応する実行ファイルもない。

- 2016〜2024年へ固定仕様のモデルを再fitし、隔離済みの2025〜2026年へ一度だけ推論・評価する工程
- 目的変数がまだ存在しない直近シグナル日について、翌週の銘柄ランキングを生成する運用推論工程

したがって、現在の「最初から最後まで」は、データ取得から**開発期間内の予測検証・コスト込み
バックテストまで**を意味する。最終期間または実運用の予測を追加するときは、2025年以降の結果を
モデル選択へ戻さない独立したfit/predict工程、設定、テスト、manifestを先に実装する。

## 1. `data_download.py`

### CLI

APIキーをコマンドライン引数へ書かない。`JQUANTS_API_KEY` が環境に設定済みならそれを使い、
未設定なら実行時に非表示プロンプトが出る。次は2024年上期を新しいスナップショットへ取得する例。
開始日・終了日は両端を含む。

```cmd
set "RAW_NAME=jquants_v2_manual_20241011_120000"

python -m stock_prediction.data_download ^
  --start-date 2024-01-01 ^
  --end-date 2024-06-30 ^
  --output-root data\raw ^
  --output-dir "%RAW_NAME%"

set "RAW_DIR=data\raw\%RAW_NAME%"
type "%RAW_DIR%\manifest.json"
```

`RAW_NAME` の日時部分は実行ごとに変更する。Command Promptでは `set "名前=値"` で変数を設定し、
`%名前%` で参照する。行末の `^` はCommand Promptの行継続記号であり、その後ろに空白を置かない。

インストール済みのエントリーポイントを使う場合、先頭を次に置き換えられる。

```cmd
stock-download --start-date 2024-01-01 --end-date 2024-06-30 --output-dir "%RAW_NAME%"
```

既定の5リクエスト/分を変更する場合は契約プランを確認したうえで
`--requests-per-minute` を明示する。途中で失敗または中断した場合は、同じ出力先へ追記せず、
manifestと `partial_prices/` を確認してから新しいスナップショットへ再取得する。

### Notebook

APIキーは `getpass` で非表示入力する。返り値からスナップショット、カレンダー、年別株価の
保存先とmanifestを取得できる。

```python
from getpass import getpass
from pathlib import Path

from stock_prediction.data_download import DownloadConfig, download_jquants_raw

root = next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "README.md").is_file()
    and (path / "src" / "stock_prediction").is_dir()
)

config = DownloadConfig(
    start_date="2024-01-01",
    end_date="2024-06-30",
    output_root=root / "data" / "raw",
    requests_per_minute=5,
)
download = download_jquants_raw(
    config,
    api_key=getpass("J-Quants API key: "),
)

snapshot_dir = download.output_dir
calendar_path = download.calendar_path
price_paths = download.price_paths
download.manifest["status"], snapshot_dir, len(price_paths)
```

## 2. `auxiliary_data_download.py`

TOPIX日足、各取引日時点の上場銘柄一覧、各暦日に公表された決算予定、各暦日の財務情報開示を
別の不変スナップショットへ取得する。日次株価取得で保存した公式取引カレンダーを入力し、
営業日を平日や株価観測日から推測しない。

### CLI

まず数日で契約プラン、取得可能期間、現行schemaを確認する。APIキーの扱いは
`data_download.py` と同じで、引数・ログ・manifestには保存しない。

```cmd
set "AUX_NAME=jquants_v2_auxiliary_manual_20241011_120000"
set "AUX_DIR=data\raw\%AUX_NAME%"

python -m stock_prediction.auxiliary_data_download ^
  --start-date 2024-01-01 ^
  --end-date 2024-06-30 ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --output-root data\raw ^
  --output-dir "%AUX_NAME%"

type "%AUX_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-download-auxiliary` に置き換えられる。4種類すべてが既定で、
一部だけを独立確認するときは次のようにカンマ区切りで指定する。

```cmd
stock-download-auxiliary ^
  --start-date 2024-01-01 ^
  --end-date 2024-01-10 ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --datasets topix,listed_info ^
  --output-dir "jquants_v2_auxiliary_smoke_20241011_120000"
```

`listed_info` は東証営業日ごと、`earnings_schedule` と `financial_results` は休日の公表を
取りこぼさないよう全暦日ごとに問い合わせる。長期間では最低でも「営業日日数 + 全暦日数×2」回に
ページネーション分を加えたリクエストになる。既定5リクエスト/分では全研究期間に長時間を要するため、
契約上限を確認して `--requests-per-minute` を設定し、まず年単位またはデータセット単位で所要時間を
測る。別実行のスナップショットを後から手作業で混合しない。

J-Quants V2 `/fins/summary` は数値項目も文字列、未記載値も空文字列として返す。取得処理は
`financial_results` の財務値をraw文字列のまま保持し、空文字列をNullやゼロへ変換しない。
日付列だけは要求日・形式の検査用に日時型へ正規化する。財務値を特徴量へ使う場合は、rawを
上書きせず、列ごとの数値変換と変換不能件数を後続の中間データmanifestへ記録する。取得manifestの
欠損率は、raw値を変えずに空文字列も欠損として集計する。

出力は次の構成になる。決算予定の `PubDate` と予定日 `SchDate`、実績の `DiscDate` / `DiscTime` を
別列のまま保持する。

```text
data/raw/<AUX_NAME>/
  trading_calendar.parquet
  topix.parquet
  listed_info/YYYY.parquet
  earnings_schedule/YYYY.parquet
  financial_results/YYYY.parquet
  master_transitions/YYYY.parquet
  code_change_candidates/YYYY.parquet
  partial/<dataset>/YYYY.parquet  # 中断・失敗時だけ
  manifest.json
```

`master_transitions` は連続して取得した銘柄マスター間の `appeared` / `disappeared` /
`attributes_changed` という観測差分である。上場・上場廃止の法的な確定事実ではない。
`code_change_candidates` は同じ境界で消失・出現したコードの会社名が一意に完全一致した候補で、
J-Quants銘柄マスターに旧新コードを結ぶ安定IDがないため **確認済みコード変更ではない**。
最初の取得営業日は比較元がなくbaselineとなる。合併・名称変更を含む確定イベントへ利用する前に、
別の公式コーポレートアクション情報で確認する。

途中で失敗または `Ctrl+C` で中断した場合、完了済み年と `partial/`、manifestの
`current_partition` / `partial_partition` を確認する。同じ出力先への再開・追記はせず、
新しい実行IDで再取得する。APIが空結果を返し得る決算予定・実績は空日を正常として記録する一方、
営業日の銘柄マスターまたはTOPIX欠落は契約期間・schemaを確認すべきエラーとして停止する。

### Notebook

CLIと同じ保存関数へDataFrameの取引カレンダーを渡す。入力DataFrameは変更されない。

```python
from getpass import getpass

import pandas as pd

from stock_prediction.auxiliary_data_download import (
    AuxiliaryDownloadConfig,
    download_auxiliary_data,
)

calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")
aux_config = AuxiliaryDownloadConfig(
    start_date="2024-01-01",
    end_date="2024-06-30",
    output_root=root / "data" / "raw",
    requests_per_minute=5,
)
auxiliary = download_auxiliary_data(
    aux_config,
    trading_calendar=calendar,
    api_key=getpass("J-Quants API key: "),
    calendar_source_path=snapshot_dir / "trading_calendar.parquet",
)

auxiliary.manifest["status"], auxiliary.output_dir, auxiliary.data_paths
```

TOPIX原本はJ-Quants列名 `Date` / `O` / `H` / `L` / `C` のまま保存する。
`descriptors.py` のDataFrame APIへ渡す場合は、原本を変更せず次のように2列へ写像する。

```python
topix_raw = pd.read_parquet(auxiliary.output_dir / "topix.parquet")
topix = topix_raw.rename(columns={"Date": "date", "C": "close"})[["date", "close"]]
```

CLIの `stock-descriptors --topix` は現時点では正規化済み `date` / `close` ファイルを要求するため、
この2列を `data/interim/` の新規ファイルへ保存して渡す。原本の `topix.parquet` は上書きしない。

## 3. `preprocessing.py`

### CLI

取得スナップショットの `prices/` と同じスナップショットの取引カレンダーを組み合わせる。
特徴量工程へ渡すデータでは、`--allow-unverified-calendar` を使わない。

```cmd
set "PREPROCESS_ID=manual_20241011_120000"
set "INTERIM_DIR=data\interim\prices_%PREPROCESS_ID%"
set "PREPROCESS_REPORT_DIR=reports\preprocessing_%PREPROCESS_ID%"

python -m stock_prediction.preprocessing ^
  --raw-dir "%RAW_DIR%\prices" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\preprocessing.json ^
  --output-dir "%INTERIM_DIR%" ^
  --report-dir "%PREPROCESS_REPORT_DIR%"

type "%PREPROCESS_REPORT_DIR%\manifest.json"
```

インストール済みなら、同じ引数で `stock-preprocess` を使える。小期間だけ監査するときは
`--start-date` / `--end-date` を追加する。開始前の行は出力されないため、後続の60セッション
特徴量に必要なウォームアップ期間も入力範囲へ含める。

### Notebook：年別ファイルを保存する

CLIと同じ保存処理をNotebookから呼ぶ例。`output_dir` と `report_dir` は存在しない新規パスにする。

```python
from datetime import datetime, timezone

from stock_prediction.preprocessing import PreprocessingConfig, run_preprocessing

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
preprocess_config = PreprocessingConfig()
preprocess_manifest = run_preprocessing(
    raw_dir=snapshot_dir / "prices",
    calendar_path=snapshot_dir / "trading_calendar.parquet",
    output_dir=root / "data" / "interim" / f"prices_{run_id}",
    report_dir=root / "reports" / f"preprocessing_{run_id}",
    config=preprocess_config,
)

preprocess_manifest["status"], preprocess_manifest["output_rows"]
```

### Notebook：DataFrameだけを処理する

小さな範囲の調査や単体確認では、入力DataFrameを変更しない `preprocess_prices` を使える。

```python
import pandas as pd

from stock_prediction.preprocessing import PreprocessingConfig, preprocess_prices

raw_prices = pd.read_parquet(
    snapshot_dir / "prices" / "2024.parquet",
    filters=[
        ("Date", ">=", pd.Timestamp("2024-01-01")),
        ("Date", "<=", pd.Timestamp("2024-01-31")),
    ],
)
calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")

preprocessed = preprocess_prices(
    raw_prices,
    config=PreprocessingConfig(),
    trading_calendar=calendar,
)
clean_prices = preprocessed.prices
quarantine = preprocessed.quarantine
quality_report = preprocessed.report

quality_report["output_rows"], quality_report["quarantined_rows"]
```

## 4. `descriptors.py`

### CLI

入力は、公式カレンダー付きで前処理され、全行が `calendar_verified=True` の年別Parquetである。
TOPIXを指定しない例では10特徴量を通常計算し、`RelativeTOPIX_20` だけを欠損として残す。

```cmd
set "DESCRIPTOR_ID=manual_20241011_120000"
set "DESCRIPTOR_DIR=data\processed\descriptors_%DESCRIPTOR_ID%"
set "DESCRIPTOR_REPORT_DIR=reports\descriptors_%DESCRIPTOR_ID%"

python -m stock_prediction.descriptors ^
  --prices-dir "%INTERIM_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\descriptors.json ^
  --output-dir "%DESCRIPTOR_DIR%" ^
  --report-dir "%DESCRIPTOR_REPORT_DIR%"

type "%DESCRIPTOR_REPORT_DIR%\manifest.json"
```

正規化済みの `date` / `close` 2列を持つTOPIXファイルがある場合は、次を追加して
`RelativeTOPIX_20` も生成する。

```cmd
python -m stock_prediction.descriptors ^
  --prices-dir "%INTERIM_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --topix data\interim\topix.parquet ^
  --require-topix ^
  --output-dir "data\processed\descriptors_with_topix_%DESCRIPTOR_ID%" ^
  --report-dir "reports\descriptors_with_topix_%DESCRIPTOR_ID%"
```

インストール済みなら、先頭を `stock-descriptors` に置き換えられる。CLIは年別に読み、
銘柄ごとの直前60セッションを翌年へ渡すため、全年度を一括でメモリへ載せない。

### Notebook：年別ファイルを保存する

```python
from stock_prediction.descriptors import DescriptorConfig, run_descriptors

descriptor_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
descriptor_manifest = run_descriptors(
    prices_dir=root / "data" / "interim" / f"prices_{run_id}",
    calendar_path=snapshot_dir / "trading_calendar.parquet",
    output_dir=root / "data" / "processed" / f"descriptors_{descriptor_id}",
    report_dir=root / "reports" / f"descriptors_{descriptor_id}",
    config=DescriptorConfig(volatility_ddof=1, require_topix=False),
    # topix_path=root / "data" / "interim" / "topix.parquet",
)

descriptor_manifest["status"], descriptor_manifest["output_rows"]
```

### Notebook：DataFrameだけを処理する

2024年を計算し、2023年末の最大60行/銘柄をウォームアップに使う例。入力DataFrameは変更されない。

```python
import pandas as pd

from stock_prediction.descriptors import DescriptorConfig, compute_descriptors

interim_dir = root / "data" / "interim" / f"prices_{run_id}"
prices_2023 = pd.read_parquet(interim_dir / "2023.parquet")
history = (
    prices_2023.sort_values(["code", "date"], kind="stable")
    .groupby("code", sort=False)
    .tail(60)
    .reset_index(drop=True)
)
prices_2024 = pd.read_parquet(interim_dir / "2024.parquet")
calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")

descriptor_result = compute_descriptors(
    prices_2024,
    trading_calendar=calendar,
    previous_prices=history,
    config=DescriptorConfig(volatility_ddof=1),
    # topix=pd.read_parquet(root / "data" / "interim" / "topix.parquet"),
)
features_2024 = descriptor_result.descriptors
descriptor_report = descriptor_result.report

features_2024.head(), descriptor_report["feature_missing_rate"]
```

## 5. `weekly_dataset.py`：週次日程

ユニバース工程より先に、公式カレンダーから再利用可能な週次日程を作る。既定値は4取引日以上で、
月曜または火曜が週初、木曜または金曜が評価日となる完全な暦週だけを採用する。出力は
`signal_date` / `entry_date` / `exit_date` の3列なので、そのまま `universe.py --schedule` へ渡せる。
現在の設定は `signal_start_date=2017-02-06` であり、それより前の週は
`signal_before_configured_start` として日程auditだけに残る。

```cmd
set "WEEKLY_SCHEDULE_ID=manual_20261012_120000"
set "WEEKLY_SCHEDULE=data\interim\weekly_schedule_%WEEKLY_SCHEDULE_ID%.parquet"
set "WEEKLY_SCHEDULE_REPORT=reports\weekly_schedule_%WEEKLY_SCHEDULE_ID%.json"

python -m stock_prediction.weekly_dataset schedule ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\weekly_dataset.json ^
  --output "%WEEKLY_SCHEDULE%" ^
  --report "%WEEKLY_SCHEDULE_REPORT%"

type "%WEEKLY_SCHEDULE_REPORT%"
```

インストール済みなら先頭を `stock-weekly-dataset schedule` に置き換えられる。入力カレンダーの
先頭・末尾で月曜から日曜まで揃わない週は、将来の営業日を推測せず除外する。

Notebookでは同じ日程関数を使う。

```python
from stock_prediction.weekly_dataset import WeeklyDatasetConfig, build_weekly_schedule

schedule_result = build_weekly_schedule(
    calendar,
    config=WeeklyDatasetConfig(min_trading_sessions_per_week=4, require_topix=True),
)
schedule = schedule_result.schedule
schedule_audit = schedule_result.audit
schedule_report = schedule_result.report
```

## 6. `universe.py`

### CLI：日次のシグナル時点ユニバース

入力は、補助取得の取引日別 `listed_info/`、公式カレンダー付き前処理済み株価、同じ期間を覆う
公式取引カレンダーである。既定値は上場後60取引日、直近20日中15日の価格形成、売買代金中央値
1億円、1銘柄100万円、参加率1%である。

```cmd
set "UNIVERSE_ID=manual_20241011_120000"
set "UNIVERSE_DIR=data\processed\universe_%UNIVERSE_ID%"
set "UNIVERSE_REPORT_DIR=reports\universe_%UNIVERSE_ID%"

python -m stock_prediction.universe ^
  --listed-info-dir "%AUX_DIR%\listed_info" ^
  --prices-dir "%INTERIM_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\universe.json ^
  --output-dir "%UNIVERSE_DIR%" ^
  --report-dir "%UNIVERSE_REPORT_DIR%"

type "%UNIVERSE_REPORT_DIR%\manifest.json"
```

この実行は銘柄マスターの全取引日を出力する。スケジュールがないため、`is_eligible` までは計算し、
`entry_*` / `exit_*` / `is_executable` はNullとする。インストール済みなら先頭を
`stock-universe` に置き換えられる。CLIは年別に処理し、直前60取引日の履歴だけを次年へ渡す。

### CLI：週次の事後売買可能性も付ける

`--schedule` は `signal_date` / `entry_date` / `exit_date` の3列だけを持つCSVまたはParquetである。
公式カレンダーに照らして、週初取引日、直後の取引日、同週末取引日であることを検査する。

```cmd
python -m stock_prediction.universe ^
  --listed-info-dir "%AUX_DIR%\listed_info" ^
  --prices-dir "%INTERIM_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --schedule "%WEEKLY_SCHEDULE%" ^
  --config configs\universe.json ^
  --output-dir "data\processed\weekly_universe_%UNIVERSE_ID%" ^
  --report-dir "reports\weekly_universe_%UNIVERSE_ID%"
```

スケジュール指定時は、そのシグナル日だけを出力する。翌取引日始値と評価日終値、出来高、売買代金、
値幅制限フラグから `is_executable` を計算するが、これは事後proxyであり `is_eligible` には入らない。
未来の `is_executable` を予測対象銘柄の選択や特徴量に使用しない。

### Notebook：DataFrameだけを処理する

```python
import pandas as pd

from stock_prediction.universe import UniverseConfig, compute_universe

listed_2024 = pd.read_parquet(auxiliary.output_dir / "listed_info" / "2024.parquet")
prices_2024 = pd.read_parquet(interim_dir / "2024.parquet")
calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")
schedule = pd.read_parquet(root / "data" / "interim" / "weekly_schedule.parquet")

universe_result = compute_universe(
    listed_2024,
    prices_2024,
    trading_calendar=calendar,
    schedule=schedule.loc[schedule["signal_date"].dt.year.eq(2024)],
    config=UniverseConfig(),
    # 年初の正確な判定には直前60取引日のprevious_listed_infoとprevious_pricesを渡す。
    # 年末シグナルの翌年約定にはexecution_pricesへ翌年株価を渡す。
)
universe = universe_result.universe
universe_report = universe_result.report

universe[["signal_date", "code", "is_eligible", "is_executable", "exclusion_reasons"]].head()
```

列定義、上場日数の上限、売買代金中央値、事後proxyの限界は
[ユニバースデータ辞書](UNIVERSE.md)を参照する。Notebookから年別保存まで行う場合は、CLIと同じ
`run_universe` を使う。

## 7. `earnings_events.py`

入力候補は `signal_date` / `code` / `exit_date` を必須とする。`universe.py` を使う場合は、
日次ユニバースではなく `--schedule` を指定した週次出力を渡す。任意の `entry_date` がある場合も、
公式カレンダー上のシグナル直後の取引日であることを検査する。

主判定はシグナル日までに公表済みの最新予定と、シグナル日より前の実績決算だけを使う。
保有期間中の実績開示は `is_earnings_eligible_ex_post` の事後監査だけに反映し、
`is_earnings_eligible` には混ぜない。同日開示は歴史的な東証大引けまでならクールダウン日0とし、
既定設定は0〜25取引日を除外して26日目から解除する。
`DocType` は `1Q` / `2Q` / `3Q` / `FY` / `OtherPeriod` の各
`FinancialStatements` 接頭辞を明示的に許可する。manifestの `actual_earnings_event_rows` と
`cooldown_rows` が不自然に0でないことも確認し、未知の決算短信形式はschema errorとして調査する。

### CLI

次は補助データスナップショットと週次ユニバースを組み合わせる例。出力先は毎回新しくする。

```cmd
set "EARNINGS_ID=manual_20241011_120000"
set "EARNINGS_DIR=data\processed\earnings_events_%EARNINGS_ID%"
set "EARNINGS_REPORT_DIR=reports\earnings_events_%EARNINGS_ID%"

python -m stock_prediction.earnings_events ^
  --candidates-dir data\processed\weekly_universe_manual_20261021_approved_start ^
  --earnings-schedule-dir "%AUX_DIR%\earnings_schedule" ^
  --financial-results-dir "%AUX_DIR%\financial_results" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\earnings_events.json ^
  --output-dir "%EARNINGS_DIR%" ^
  --report-dir "%EARNINGS_REPORT_DIR%"

type "%EARNINGS_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-earnings-events` に置き換えられる。CLIは候補を年別に処理し、
決算イベントは判定に必要な列だけを読む。補助取得のmanifestで、決算予定・実績が候補期間、
クールダウン用の先行25取引日、予定公表履歴を十分に覆うことを先に確認する。

### Notebook：DataFrame API

小期間の調査では同じ中核関数へDataFrameを渡す。入力は変更されない。以下の `event_years` は
例であり、実際には補助取得manifestを基準に必要な公表履歴と先行25取引日を含める。

```python
import pandas as pd

from stock_prediction.earnings_events import (
    EarningsEventConfig,
    compute_earnings_exclusions,
)

candidate_year = 2024
event_years = (2023, 2024)
weekly_candidates = pd.read_parquet(
    root / "data" / "processed" / f"weekly_universe_{run_id}" / f"{candidate_year}.parquet"
)
earnings_schedule = pd.concat(
    [
        pd.read_parquet(auxiliary.output_dir / "earnings_schedule" / f"{year}.parquet")
        for year in event_years
    ],
    ignore_index=True,
)
financial_results = pd.concat(
    [
        pd.read_parquet(auxiliary.output_dir / "financial_results" / f"{year}.parquet")
        for year in event_years
    ],
    ignore_index=True,
)
calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")

earnings_result = compute_earnings_exclusions(
    weekly_candidates,
    earnings_schedule,
    financial_results,
    trading_calendar=calendar,
    config=EarningsEventConfig(cooldown_trading_sessions=25),
)
earnings_flags = earnings_result.exclusions
earnings_report = earnings_result.report

earnings_flags[[
    "signal_date",
    "code",
    "scheduled_earnings_during_holding",
    "is_in_post_earnings_cooldown",
    "is_earnings_eligible_without_cooldown",
    "is_earnings_eligible",
    "is_earnings_eligible_ex_post",
]].head()
```

予定に発表時刻がないため、予定日がシグナル日と同日なら保守的に跨ぎとする。
`active_schedule_date_missing=True` は予定情報に日付がない監査対象で、自動除外ではない。
同じ `Code` / `FQName` / `FYE` / `PubDate` に複数状態がある場合は、同日内の順序を推測せず
集合として保持する。既知の候補日のいずれかが保有期間へ入れば決算イベント1件として除外し、
Nullも含む有効集合は `active_schedule_date_missing=True` とする。manifestでは
`same_day_schedule_state_groups` / `same_day_schedule_state_rows` /
`same_day_schedule_state_groups_with_missing_date` を確認する。
全列、境界、理由列、事後列の利用制限は [決算イベントデータ辞書](EARNINGS_EVENTS.md) を参照する。

## 8. `weekly_dataset.py`：週次学習表

週次ユニバースと決算判定まで完了した後、年別の監査済み株価、日次特徴量、週次ユニバース、
決算判定を結合する。TOPIX原本の `Date` / `C` は、2節の例どおり原本を上書きせず
`date` / `close` へ写像した中間ファイルを使う。次の変数は先行工程の新規出力を指す。

```cmd
set "WEEKLY_DATASET_ID=manual_20261012_130000"
set "TOPIX_FILE=data\interim\topix_normalized.parquet"
set "WEEKLY_DATASET_DIR=data\processed\weekly_dataset_%WEEKLY_DATASET_ID%"
set "WEEKLY_DATASET_REPORT_DIR=reports\weekly_dataset_%WEEKLY_DATASET_ID%"

python -m stock_prediction.weekly_dataset build ^
  --prices-dir "%INTERIM_DIR%" ^
  --descriptors-dir "data\processed\descriptors_with_topix_%DESCRIPTOR_ID%" ^
  --universe-dir "data\processed\weekly_universe_%UNIVERSE_ID%" ^
  --earnings-dir "%EARNINGS_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --topix "%TOPIX_FILE%" ^
  --config configs\weekly_dataset.json ^
  --output-dir "%WEEKLY_DATASET_DIR%" ^
  --report-dir "%WEEKLY_DATASET_REPORT_DIR%"

type "%WEEKLY_DATASET_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-weekly-dataset build` に置き換えられる。出力はシグナル年ごとの
`YYYY.parquet`、年別JSON、manifestである。既存出力は上書きしない。失敗または `Ctrl+C` では、
manifestの `status` / `current_partition` / `completed_partitions` を確認し、新しい実行IDで再実行する。

Notebookからは同じ中核関数へDataFrameを渡す。

```python
import pandas as pd

from stock_prediction.weekly_dataset import WeeklyDatasetConfig, build_weekly_dataset

prices = pd.read_parquet(interim_dir / "2024.parquet")
descriptors = pd.read_parquet(descriptor_dir / "2024.parquet")
universe = pd.read_parquet(universe_dir / "2024.parquet")
earnings = pd.read_parquet(earnings_dir / "2024.parquet")
calendar = pd.read_parquet(snapshot_dir / "trading_calendar.parquet")
topix = pd.read_parquet(root / "data" / "interim" / "topix_normalized.parquet")

weekly_result = build_weekly_dataset(
    prices,
    descriptors,
    universe,
    earnings,
    trading_calendar=calendar,
    topix=topix,
    config=WeeklyDatasetConfig(min_trading_sessions_per_week=4, require_topix=True),
)
weekly = weekly_result.dataset
weekly_report = weekly_result.report

weekly[[
    "signal_date",
    "code",
    "target_return",
    "target_excess_return",
    "is_signal_eligible",
    "is_target_return_sample",
    "is_target_excess_return_sample",
]].head()
```

`is_signal_eligible` はシグナル時点情報だけで決まり、`is_executable` や
`is_earnings_eligible_ex_post` を含まない。評価日価格欠損は補完せず `target_missing_reasons` に残す。
単純リターンとTOPIX超過リターンのどちらを最終ターゲットにするかは本工程では決めない。
詳細は [週次学習表データ辞書](WEEKLY_DATASET.md) を参照する。

## 9. `splits.py`：時系列分割と最終評価期間の隔離

週次学習表の年別Parquetから必要な `signal_date` / `code` / `exit_date` だけを読み、
2016〜2024年の開発foldと、選択した2025〜2026年の最終評価候補を分ける。既定設定は
2020〜2024年を各1年ずつ検証し、公式カレンダー上で4取引日のパージを設ける。

### CLI

出力先は毎回新しくする。`ISOLATION_START` / `ISOLATION_END` を変える場合は
`configs/splits.json` を直接上書きせず、新しい実験用設定JSONを保存して `--config` へ渡す。

```cmd
set "SPLIT_ID=manual_20261012_150000"
set "SPLIT_DIR=data\processed\splits_%SPLIT_ID%"
set "SPLIT_REPORT_DIR=reports\splits_%SPLIT_ID%"

python -m stock_prediction.splits ^
  --input-dir "%WEEKLY_DATASET_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\splits.json ^
  --output-dir "%SPLIT_DIR%" ^
  --report-dir "%SPLIT_REPORT_DIR%"

type "%SPLIT_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-splits` に置き換えられる。`folds.parquet` はfoldごとの
`train` / `validation` / `purged` シグナル日、`isolated.parquet` は最終評価候補、
`out_of_scope.parquet` は選択外または境界を跨ぐ期間を持つ。特徴量や目的変数は複製しない。
manifestの `maximum_development_label_horizon_sessions` が設定したパージ幅以下であり、各foldで
学習の `exit_date_max` が検証開始より前、隔離行がfoldに含まれないことを確認する。

途中で失敗または `Ctrl+C` になった場合は、manifestの `status` / `completed_outputs` を確認し、
同じ出力へ追記せず新しい実行IDで再実行する。

### Notebook：DataFrame API

読み込み済みの週次表と、同じスナップショットの公式カレンダーを共通関数へ渡す。返される配列は
元DataFrameに対するゼロ始まりの `iloc` 位置で、入力DataFrame自体は変更されない。

```python
from stock_prediction.splits import SplitConfig, build_walk_forward_splits

split_result = build_walk_forward_splits(
    weekly,
    calendar,
    config=SplitConfig(
        isolation_start_date="2025-01-01",
        isolation_end_date="2026-12-31",
        purge_trading_sessions=4,
    ),
)

for fold in split_result.folds:
    train = weekly.iloc[fold.train_positions]
    validation = weekly.iloc[fold.validation_positions]
    purged = weekly.iloc[fold.purged_positions]
    # 補完器、標準化器、特徴量選択器はtrainだけでfitする。

# 最終設計を固定するまで目的変数や評価結果を調べない。
isolated_definition = split_result.isolated_definitions
```

分割前に `is_target_*_sample` で行を削らない。同じシグナル日の割当を維持したまま、各fold内で
学習・評価対象を選ぶ。隔離期間を一度評価した後に設定やモデルを変更した場合、その期間を再び
未使用の最終評価として扱わない。詳細は [時系列分割仕様](SPLITS.md) を参照する。

## 10. `modeling.py`：基準モデルとfold内前処理

2016〜2024年の年別週次Parquetだけを読み、2020〜2024年の各検証foldで定数、`R_20`
モメンタム、線形最小二乗を比較する。入力ディレクトリに2025年・2026年ファイルがあっても、
CLIはschemaを含めてその内容を読まない。

### CLI

`target_return` は実行既定値であり、最終ターゲットの選択ではない。TOPIX超過リターンを開発期間で
比較するときは `configs/modeling.json` を上書きせず、`target_column` を
`target_excess_return` にした別の設定JSONを保存して `--config` へ渡す。

```cmd
set "MODEL_ID=manual_20261012_170000"
set "MODEL_DIR=artifacts\modeling_%MODEL_ID%"
set "MODEL_REPORT_DIR=reports\modeling_%MODEL_ID%"

python -m stock_prediction.modeling ^
  --input-dir "%WEEKLY_DATASET_DIR%" ^
  --calendar "%RAW_DIR%\trading_calendar.parquet" ^
  --config configs\modeling.json ^
  --split-config configs\splits.json ^
  --output-dir "%MODEL_DIR%" ^
  --report-dir "%MODEL_REPORT_DIR%"

type "%MODEL_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-modeling` に置き換えられる。`manifest.json` の
`status=complete`、`final_period_partitions_loaded=false`、開発入力年が2016〜2024年だけであること、
各foldの学習・検証件数、特徴量欠損率、出力SHA256を確認する。

出力の役割は次のとおり。

| ファイル                  | 内容                                                        |
| ------------------------- | ----------------------------------------------------------- |
| `predictions.parquet`   | 検証年のシグナル時点適格行に対する3モデルの予測と観測target |
| `fold_metrics.parquet`  | fold・モデル別の誤差、日付別相関要約、上下位差、coverage    |
| `daily_metrics.parquet` | シグナル日・モデル別の指標。相関不能日はNull                |
| `model_parameters.json` | fold別の補完・標準化統計、定数、係数、rank、特異値          |

途中で失敗または `Ctrl+C` になった場合、manifestの `status`, `error_type`,
`completed_outputs` を確認する。同じ出力へ追記せず、新しい実行IDで再実行する。

### Notebook：DataFrame API

読み込み済みの週次表と同じスナップショットの公式カレンダーを渡す。返り値には予測、fold指標、
日付別指標、fold別のfit済みパラメータ、分割監査が含まれる。入力DataFrameは変更されない。

```python
from stock_prediction.modeling import ModelingConfig, build_walk_forward_models
from stock_prediction.splits import SplitConfig

modeling_result = build_walk_forward_models(
    weekly,
    calendar,
    config=ModelingConfig(
        target_column="target_return",
        momentum_feature="R_20",
        top_n=10,
    ),
    split_config=SplitConfig(
        isolation_start_date="2025-01-01",
        isolation_end_date="2026-12-31",
        purge_trading_sessions=4,
    ),
)

predictions = modeling_result.predictions
fold_metrics = modeling_result.fold_metrics
daily_metrics = modeling_result.daily_metrics
linear_2020 = modeling_result.fitted_models["validation_2020"].linear
linear_2020.preprocessor.medians, linear_2020.coefficients
```

Notebookへ全期間の週次表を渡しても、modeling処理は `development_positions` 外の特徴量・targetを
fitや評価へ使わない。ただし物理的な閲覧防止ではないため、最終設計確定までは隔離期間の目的変数を
表示・集計しない。詳細は [基準モデル仕様](MODELING.md) を参照する。

## 11. `evaluation.py`：IC・上位N・coverage・年別/業種別評価

`modeling.py` の予測と、同じ研究スナップショットの取引日別 `listed_info` を使う。
既定は33業種（`S33` / `S33Nm`）である。17業種を主分析にする場合は、最終評価を見る前に
`industry_code_column` / `industry_name_column` を `S17` / `S17Nm` にした別設定を保存する。

### CLI

出力先は毎回新しくする。次の `AUX_DIR` は2節で作成した、予測対象期間を完全に含む補助データの
不変スナップショットを指す。予測と異なる取得版の時点銘柄マスターを無記録で混在させない。

```cmd
set "EVALUATION_ID=manual_20261012_190000"
set "EVALUATION_DIR=artifacts\evaluation_%EVALUATION_ID%"
set "EVALUATION_REPORT_DIR=reports\evaluation_%EVALUATION_ID%"

python -m stock_prediction.evaluation ^
  --predictions "%MODEL_DIR%\predictions.parquet" ^
  --industry-dir "%AUX_DIR%\listed_info" ^
  --config configs\evaluation.json ^
  --output-dir "%EVALUATION_DIR%" ^
  --report-dir "%EVALUATION_REPORT_DIR%"

type "%EVALUATION_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-evaluation` に置き換えられる。CLIは予測に含まれる年の
`listed_info/YYYY.parquet` だけを選んでからschemaを確認し、4列だけを読む。成果物は次のとおり。

| ファイル                              | 内容                                             |
| ------------------------------------- | ------------------------------------------------ |
| `predictions_with_industry.parquet` | 予測と完全一致した時点業種、分類観測フラグ       |
| `daily_metrics.parquet`             | シグナル日・モデル別のIC、上位/下位、3種coverage |
| `summary_metrics.parquet`           | 全期間の時間方向要約                             |
| `yearly_metrics.parquet`            | 年・モデル別の時間方向要約                       |
| `industry_daily_metrics.parquet`    | シグナル日・業種・モデル別の横断指標             |
| `industry_metrics.parquet`          | 業種・モデル別の全期間要約                       |
| `industry_yearly_metrics.parquet`   | 年・業種・モデル別の要約                         |

`manifest.json` の `status=complete`、`industry_years_loaded`、入力SHA256、分類coverage、
`completed_outputs` を確認する。途中で失敗または `Ctrl+C` になった場合は状態と完了済み成果物を確認し、
同じ出力へ追記せず新しい実行IDで再実行する。

### Notebook：DataFrame API

```python
import pandas as pd

from stock_prediction.evaluation import EvaluationConfig, evaluate_predictions

predictions = pd.read_parquet(model_dir / "predictions.parquet")
listed_parts = [
    pd.read_parquet(aux_dir / "listed_info" / f"{year}.parquet")
    for year in sorted(predictions["validation_year"].unique())
]
listed_info = pd.concat(listed_parts, ignore_index=True)

evaluation = evaluate_predictions(
    predictions,
    listed_info,
    config=EvaluationConfig(
        top_n=10,
        industry_code_column="S33",
        industry_name_column="S33Nm",
    ),
)

daily_metrics = evaluation.daily_metrics
yearly_metrics = evaluation.yearly_metrics
industry_metrics = evaluation.industry_metrics
industry_yearly_metrics = evaluation.industry_yearly_metrics
evaluation.report["industry_join"]
```

Notebook例は説明を優先して年別ファイルを結合している。大きい実データでは、CLIのように必要列と
予測対象のシグナル日だけを選択する。業種は完全一致結合だけを使い、欠損を現在の分類で埋めない。

上位・下位は有限予測だけで固定し、target欠損銘柄を次順位で置き換えない。選択内targetが全件
観測できない日の上位/下位平均はNullになるため、`top_n_target_coverage`、
`bottom_n_target_coverage`、`*_valid_dates` を性能値と一緒に確認する。詳細は
[予測評価仕様](EVALUATION.md) を参照する。

## 12. `backtest.py`：約定制約とコストを含む売買バックテスト

主結果は、シグナル日の次取引日始値で買い、評価日終値で売る。買い注文はその始値で未約定なら
失効し、売り注文だけは保有を消さないため後続取引日の終値へ繰り越す。片道コストの初期値は
手数料5bps、半スプレッド5bps、スリッページ10bpsであり、`configs/backtest.json` で変更できる。
これらは一般的な市場標準ではなく、実行ごとに保存する研究仮定である。

入力は同じ研究スナップショットから作った予測、年別週次候補表、監査済み日次株価、公式カレンダーを
使う。上場廃止清算表は任意だが、保有中の確定上場廃止を正しく清算するには必要である。

### CLI

```cmd
set "BACKTEST_ID=manual_20261013_120000"
set "BACKTEST_DIR=artifacts\backtest_%BACKTEST_ID%"
set "BACKTEST_REPORT_DIR=reports\backtest_%BACKTEST_ID%"

python -m stock_prediction.backtest ^
  --predictions "%MODEL_DIR%\predictions.parquet" ^
  --candidates-dir "%WEEKLY_DIR%" ^
  --prices-dir "data\interim\preprocessing_20261004" ^
  --calendar "data\raw\jquants_v2_manual_20241011_120000\trading_calendar.parquet" ^
  --config configs\backtest.json ^
  --output-dir "%BACKTEST_DIR%" ^
  --report-dir "%BACKTEST_REPORT_DIR%"

type "%BACKTEST_REPORT_DIR%\manifest.json"
```

インストール済みなら先頭を `stock-backtest` に置き換えられる。確定上場廃止清算表がある場合は
`--delistings path\to\delistings.parquet` を追加する。表の必須列は `code`, `delisting_date`,
`settlement_date`, `settlement_value_per_share` である。MBO、合併、破綻などの理由だけから清算値を
推測せず、旧株1株当たりの実際の経済的清算価値を入力する。

保有中の上場廃止について清算値がNullなら、主設定は `UnresolvedDelistingError` で停止し、
manifestを `failed` にする。全損感度分析は主結果と混同しない新しい設定・出力先で実行する。

```python
import json
from pathlib import Path

base = json.loads(Path("configs/backtest.json").read_text(encoding="utf-8"))
base["unresolved_delisting_policy"] = "write_off"
Path("configs/backtest_writeoff_sensitivity.local.json").write_text(
    json.dumps(base, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
```

`configs/backtest_writeoff_sensitivity.local.json` を `--config` に指定し、別の実行IDへ保存する。
このローカル感度設定は実行記録と一緒に保存し、主設定へ上書きしない。

主な成果物は次のとおり。

| ファイル                      | 内容                                                 |
| ----------------------------- | ---------------------------------------------------- |
| `orders.parquet`            | 未約定を含む全注文、理由、要求・約定株数、参加率上限 |
| `trades.parquet`            | 実約定と上場廃止清算、コスト内訳                     |
| `positions.parquet`         | 保有ごとの入口、出口、状態、損益                     |
| `corporate_actions.parquet` | 調整係数による株数変更                               |
| `daily_portfolio.parquet`   | 現金、時価、総資産、収益、ドローダウン               |
| `weekly_metrics.parquet`    | 選択・約定coverage、コスト、ターンオーバー           |
| `yearly_metrics.parquet`    | 年別収益、リスク、コスト、ターンオーバー             |
| `summary_metrics.parquet`   | 全期間の運用指標と完了状態                           |

### Notebook：DataFrame API

```python
import pandas as pd

from stock_prediction.backtest import BacktestConfig, backtest_portfolio

predictions = pd.read_parquet(model_dir / "predictions.parquet")
weekly = pd.concat(
    [pd.read_parquet(weekly_dir / f"{year}.parquet") for year in range(2020, 2025)],
    ignore_index=True,
)
prices = pd.concat(
    [pd.read_parquet(price_dir / f"{year}.parquet") for year in range(2020, 2025)],
    ignore_index=True,
)
calendar = pd.read_parquet(calendar_path)

result = backtest_portfolio(
    predictions,
    weekly,
    prices,
    trading_calendar=calendar,
    delistings=None,  # 確定清算表があればDataFrameを渡す
    config=BacktestConfig(
        model_name="linear",
        target_column="target_return",
        top_n=10,
        commission_bps=5.0,
        half_spread_bps=5.0,
        slippage_bps=10.0,
        exit_price_mode="evaluation_close",
    ),
)

result.summary_metrics
result.yearly_metrics
result.weekly_metrics
result.orders.query("status != 'filled'")
result.report["main_result_complete"]
```

Notebook例は可読性を優先して結合している。実データは大きいため、通常はCLIで必要年・必要列・
予測対象銘柄だけを読む。決済を評価日翌取引日始値へ変える感度分析では
`exit_price_mode="next_session_open"` とし、結果を別ディレクトリへ保存する。詳細な列、数式、
未約定、上場廃止規則は [売買バックテスト仕様](BACKTEST.md) を参照する。

## 13. `experiment.py`：全工程の再実行・入力識別・レポート

一括実験は、取得済みの2つの不変rawスナップショットを開始点にする。

- 価格snapshot：`prices/YYYY.parquet`、`trading_calendar.parquet`、`manifest.json`
- 補助snapshot：`topix.parquet`、年別 `listed_info` / `earnings_schedule` /
  `financial_results`、`manifest.json`

API取得を同じ実験コマンドへ含めない。同じ期間を後日取得したデータは訂正後の別入力になり得るため、
1節と2節で新規snapshotを作成した後、そのsnapshotを設定で固定する。実験CLIはrawを変更しない。
最初に評価したいシグナル日より前に、特徴量・上場履歴用の60取引日以上と、決算クールダウン判定用の
25取引日以上をカレンダーと入力に含める。現在の10年スナップショットでは2016-11-01から履歴を持たせ、
2017-02-06を最初の分析シグナルとする。2016年側は出力候補にせず、ウォームアップとして扱う。

### 設定を作る

例を直接上書きせず、実験専用設定へコピーし、2つのplaceholderを実在するsnapshotへ変更する。
`delistings_path` は確認済みの旧株1株当たり清算値を持つ表がある場合だけ指定する。

```powershell
Copy-Item configs\experiment.example.json configs\experiment.local.json
# configs\experiment.local.json の inputs を編集する。
```

相対パスはプロジェクトルート基準で解決する。`stage_configs` は全9設定を必須とし、未知キーや不足を
拒否する。既定設定を変更する比較実験では、元の `configs/*.json` を書き換えず、別名の設定を作って
実験設定から参照する。解決後の全設定値と各設定ファイルのSHA256は実験レポート配下へ保存される。

### 書き込みなしの事前検査

```powershell
python -m stock_prediction.experiment `
  --config configs\experiment.local.json `
  --run-id baseline-dev-001 `
  --validate-only
```

インストール済みなら先頭を `stock-experiment` に置き換えられる。`status=valid` になるまで実行しない。
事前検査は、両snapshotの完了manifest、価格・カレンダー・TOPIX・銘柄マスター・決算入力、全工程設定、
modelingとbacktestのtarget整合、出力run IDの未使用を確認するが、ファイルは作成しない。

現在のワークスペースには価格snapshot
`data/raw/jquants_v2_20261008T120437054560Z` はある一方、必要な補助snapshotがない。この状態では
事前検査が明示的に停止する。2節の補助取得を、対象期間を完全に含む新しい出力先へ完了してから使う。

### 全工程を実行する

```powershell
python -m stock_prediction.experiment `
  --config configs\experiment.local.json `
  --run-id baseline-dev-001
```

run IDを省略すると実験名とUTC時刻から生成する。指定する場合は英数字、ピリオド、アンダースコア、
ハイフンの1〜64文字にする。同じrun IDへの上書き・追記・自動再開はしない。

工程順は固定される。

1. 日次株価の監査・前処理
2. raw TOPIX `Date` / `C` の `date` / `close` への検査付き正規化
3. 銘柄別の特徴量生成
4. 公式カレンダーによる週次日程
5. 時点整合ユニバース
6. 決算跨ぎ・25取引日クールダウン
7. 週次学習表と目的変数
8. パージ付きウォークフォワード分割
9. 開発期間の基準モデル
10. IC・上位N・coverage・年別／業種別評価
11. 開発期間予測を使うコスト込み売買シミュレーション

保存先はrun IDで対応する。

| ルート                                   | 内容                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `data/interim/experiments/<run-id>/`   | 監査済み年別価格、正規化TOPIX                                                         |
| `data/processed/experiments/<run-id>/` | 特徴量、日程、ユニバース、決算、週次表、分割                                          |
| `artifacts/experiments/<run-id>/`      | modeling、evaluation、backtest成果物                                                  |
| `reports/experiments/<run-id>/`        | `manifest.json`、`config.resolved.json`、工程別manifest、`experiment_report.md` |

最上位 `manifest.json` の `status=complete`、全工程の状態、`development_period_only=true`、
`final_period_used_for_model_selection=false`、`final_period_evaluated=false` を確認する。上流工程が
隔離候補を作った件数は `final_period_prepared_rows` で区別する。`incomplete` はバックテストに
未決済建玉が残った状態であり、
完了扱いにしない。`failed` / `interrupted` の場合は `error` と最後の工程を確認し、同じ出力へ追記せず、
原因を修正して新しいrun IDで全工程を再生成する。個別工程が既に保存した部分成果は監査用に残す。

`experiment_report.md` は入力snapshot、工程状態、開発期間の予測評価summary、コスト込みbacktest summary、
警告と保存先をまとめる。詳細な欠損率・除外理由・年別件数・入力SHA256は要約で省略せず、対応する
工程別manifestとParquetで確認する。

### Notebook：同じ保存APIを使う

```python
from pathlib import Path

from stock_prediction.experiment import load_config, plan_experiment, run_experiment

root = Path.cwd()
config = load_config(root / "configs" / "experiment.local.json")

# 副作用なし。入力と設定を検査し、保存予定パスを確認する。
plan = plan_experiment(config, run_id="baseline-dev-notebook-001", project_root=root)
plan.warnings
plan.input_identifiers

# CLIと同じ保存処理。DataFrameをNotebookへ全件保持しない。
result = run_experiment(config, run_id=plan.run_id, project_root=root)
result.manifest["status"]
result.report_path
```

DataFrame単位の探索には各工程のNotebook APIを使い、一括再現では `run_experiment` を使う。
どちらの場合も2025〜2026年の隔離期間を特徴量選択、閾値、モデル選択へ使用しない。

## 実行後に最低限確認する項目

- 各manifestの `status` が `completed` または `complete` であること。
- 要求期間と実際の `date_min` / `date_max`、行数、銘柄数が想定と一致すること。
- ダウンロードの主キー重複が0で、前処理の隔離件数・理由を確認したこと。
- 特徴量別欠損率が、60セッションの履歴不足、価格なし行、TOPIX未指定などで説明できること。
- 同じスナップショットの価格と取引カレンダーを組み合わせていること。
- 補助取得では銘柄マスターだけが営業日問い合わせ、決算予定・実績が全暦日問い合わせとして
  manifestの `requested_dates` / `completed_dates` に残っていること。
- `master_transitions` と `code_change_candidates` を上場廃止・コード変更の確定情報と扱っていないこと。
- ユニバースの `listed_sessions_elapsed` は60で上限を切り、入力開始から60取引日未満を除外していること。
- `is_eligible` がシグナル時点情報だけ、`is_executable` が事後情報であること。
- ユニバースの除外理由別件数と `master_rows_without_price` を年別レポートで確認したこと。
- 決算予定は各シグナル日時点の最新公表行だけが使われ、後日の予定変更が過去へ遡及していないこと。
- 決算後クールダウンは同日公表済みの0日目と公式カレンダー上の1〜25取引日を含み、
  26日目解除になっていること。
- `is_earnings_eligible_ex_post` をシグナル時点の銘柄選択や特徴量へ使っていないこと。
- `active_schedule_date_missing` と決算予定・実績の取得coverageを補助取得manifestで確認したこと。
- 週次日程が4取引日以上で、週初が月曜・火曜、評価日が木曜・金曜であること。
- 週次結合の決算キーcoverageが完全で、ユニバース合格行に特徴量行とシグナル日価格があること。
- 評価日価格欠損件数と `target_missing_reason_counts` を確認し、黙って除外していないこと。
- `is_signal_eligible` と目的変数別の `is_target_*_sample` を混同していないこと。
- 分割のパージ幅が入力内の最大ラベル期間以上で、各foldの学習ラベルが検証開始より前に終わること。
- 2025年以降の隔離行と `out_of_scope` 行が、いずれの開発foldにも入っていないこと。
- modeling CLIが2016〜2024年だけを読み、`final_period_partitions_loaded=false` であること。
- 各foldの補完中央値・平均・scaleが学習対象行だけからfitされ、検証にはtransformだけであること。
- 3モデルの検証候補集合が同じで、target coverageとprediction coverageを混同していないこと。
- `model_parameters.json` の特徴量順、前処理統計、係数、rankがmanifestの設定と対応すること。
- evaluationのモデル間適格銘柄集合が一致し、3種coverageを混同していないこと。
- experimentの最上位manifestで、2つのsnapshot ID・manifest SHA256、全工程設定、乱数seed、
  `development_period_only=true`、`final_period_used_for_model_selection=false`、
  `final_period_evaluated=false`、工程順と停止位置を確認したこと。
- `experiment_report.md` の評価・売買結果を最終バックテストまたは将来利益の保証として扱っていないこと。
- 時点業種が `(signal_date, code)` の完全一致で、分類欠損件数と `industry_coverage` を確認したこと。
- 上位/下位の選択内target coverageとN充足日数を確認し、欠損銘柄を次順位で補充していないこと。
- backtestの予測順位に `actual_target` が使われず、`entry_date` がシグナルの直後の公式取引日であること。
- `orders.parquet` の未約定理由、部分約定、売り繰越、参加率上限、売買単位を確認したこと。
- `commission_bps` / `half_spread_bps` / `slippage_bps` と決済方式が事前設定どおりであること。
- `main_result_complete=true` かつopen positionが0であること。未解決上場廃止や端株を黙って除外して
  いないこと。
- 全損上場廃止の結果を主結果ではなく、明示的なwrite-off感度分析として扱っていること。
- `data/raw/` のファイルを後続工程が上書きしていないこと。

2025〜2026年の最終バックテスト期間は、特徴量選択、閾値決定、モデル選択、
ハイパーパラメータ調整に使わない。全期間の特徴量を機械的に生成しても、開発時の可視化や
比較対象は2016〜2024年へ限定する。
