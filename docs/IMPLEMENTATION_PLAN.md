# 実装計画

AGENTS.md が参照する `IMPLEMENTATION_PLAN.md` を本作業の記録先とする。
2026-10-08 に旧誤綴りファイルを本ファイルへ統合・削除し、
以後は正しい綴りの本ファイルのみを正本として使用する。

## 全工程の実験オーケストレーション（2026-10-13、実装済み）

### 目的と対象

`src/stock_prediction/experiment.py` を追加し、既存の独立工程を同じPython APIから順に実行する。
CLIとNotebookは同じ `run_experiment` を使い、実行時設定、raw入力スナップショット識別子、
工程別manifest、成果物、失敗・中断位置、最終Markdownレポートを実験単位で保存する。
研究仕様や個別工程の判定ロジックは変更しない。

### 実装前に確認した入力と既存責務

- 価格rawスナップショットは `prices/YYYY.parquet`、`trading_calendar.parquet`、`manifest.json` を持つ。
  現在の実スナップショットは2016-04-08〜2026-04-08を含む。
- 補助rawスナップショットは `topix.parquet`、年別 `listed_info`、`earnings_schedule`、
  `financial_results` とmanifestを持つ契約だが、現在のワークスペースには実スナップショットがない。
- 既存の保存APIは前処理、特徴量、週次日程、ユニバース、決算除外、週次表、分割、基準モデル、
  予測評価、バックテストを個別に実行でき、各工程が入力SHA256と中断時manifestを保存する。
- 補助取得のTOPIX原本は `Date` / `C`、後続工程は厳格に `date` / `close` の2列を要求するため、
  実験内で検査付きの派生ファイルを作り、原本を変更しない必要がある。
- modelingは2016〜2024年だけを読み、2020〜2024年の検証予測を生成する。上流工程は2025年以降の
  隔離候補を準備できるが、現時点では隔離期間へ最終学習済みモデルを適用する工程がないため、
  experimentの評価とバックテストは開発期間だけを対象にする。これを最終バックテストと表示しない。
- J-Quantsからの再取得は同じ日付指定でも後日の訂正により同一入力を保証しない。このため一括実験の
  開始点は既存の不変rawスナップショットとし、取得CLIはRUNBOOK上の前段として分離する。
- 本番依存はpandas、NumPy、PyArrow、requestsで足り、新規依存は不要である。

### 実装方針

- 厳格なJSON設定で、実験名、乱数seed、価格・補助rawスナップショット、任意の確定上場廃止表、
  各工程設定を指定する。相対パスはプロジェクトルート基準とし、未知キーを拒否する。
- 実行前検査で必要ファイル・ディレクトリ、全工程設定、入力と出力の分離を確認し、raw配下へ
  書き込まない。入力manifestと必要入力ファイルのSHA256を最上位manifestへ保存する。
- 同じrun IDに対する上書き・自動再開は行わない。`data/interim`、`data/processed`、`artifacts`、
  `reports` の実験別ディレクトリへ責務に応じて出力する。
- 工程は依存順に逐次実行し、先行工程が失敗したら後続を実行しない。最上位manifestとレポートを
  `failed` または `interrupted` に更新し、個別工程が保存済みの部分成果を保持する。
- 最終レポートは投資成果を誇張せず、入力識別子、設定、工程状態、件数・coverage、開発期間の
  予測指標・バックテスト指標、警告・制限を、既存manifestから機械的に要約する。
- `plan_experiment` を副作用なしの事前検査API、`run_experiment` をNotebook/CLI共通保存APIとし、
  CLIには `--validate-only` を用意する。

### 進捗

- [x] README.md / AGENTS.md / 全既存CLI・保存API・設定・データ配置・テストの確認
- [x] rawスナップショットを開始点にする範囲、開発期間だけを評価する境界、TOPIX正規化責務の確定
- [x] 厳格な実験設定、入力識別子、出力計画、事前検査APIの実装
- [x] 全工程オーケストレーション、中断manifest、Markdownレポート生成の実装
- [x] CLIエントリーポイント、既定設定例、Notebook利用可能な公開APIの追加
- [x] schema、上書き拒否、工程順、失敗・中断、入力不変、レポートのテスト
- [x] README、RUNBOOK、本計画書の実装結果・制限・検証結果の同期
- [x] 関連テスト、全テスト、CLIヘルプ、合成end-to-end例の検証

### 実装内容

- `src/stock_prediction/experiment.py` に、厳格な `ExperimentConfig` / `ExperimentInputs`、
  副作用なしの `plan_experiment`、Notebook/CLI共通の `run_experiment`、raw TOPIXを変更しない
  `normalize_topix`、最上位レポートの `render_experiment_report`、CLI `main` を追加した。
- 実験設定はschema version、実験名、32-bit範囲の乱数seed、価格・補助snapshot、任意の確定上場廃止表、
  9工程すべての設定ファイルを必須とし、未知・不足キーを拒否する。各工程固有の既存loaderでも再検査し、
  modelingとbacktestのtarget、backtestモデルが生成対象に含まれることを実行前に確認する。
- 事前計画は両snapshotの完了manifest、必要ファイル・年別partition、全設定、未使用run ID、入力と出力の
  非重複を検査する。snapshot ID、manifest SHA256、manifest内の宣言済みSHA256集合から作るdigest、
  カレンダー・TOPIX・上場廃止表のSHA256を最上位入力識別子へ保存する。
- 保存先を `data/interim/experiments/<run-id>`、`data/processed/experiments/<run-id>`、
  `artifacts/experiments/<run-id>`、`reports/experiments/<run-id>` に分けた。run IDは安全な文字だけを許し、
  4ルートのどれかが存在すれば開始前に拒否する。rawまたは入力配下への出力も拒否する。
- 前処理、TOPIX正規化、特徴量、週次日程、ユニバース、決算判定、週次表、分割、modeling、evaluation、
  backtestの11工程を固定順で呼ぶ。個別ロジックを複製せず既存保存APIを利用し、先行失敗後の工程は
  実行しない。失敗・`KeyboardInterrupt` 時も停止工程、例外、保存済み個別manifestのhash、完了工程、
  最終時刻を最上位manifestとMarkdownへ残す。
- 実験開始時にPythonとNumPyの乱数seedを設定し、既存工程は現状決定論的なので
  `randomized_stages=[]` も明示する。Notebook側のグローバル乱数状態は成功・失敗にかかわらず復元する。
- `config.resolved.json` は解決済み入力・出力パス、全工程の設定値・設定SHA256、工程順を保存する。
  `experiment_report.md` は入力取得範囲・件数、設定識別子、工程状態、保存済みevaluation/backtestの
  summary、警告、保存先を要約する。詳細監査値は個別manifestとParquetに残す。
- 上流工程が2025年以降の隔離候補を準備しても、modeling/evaluation/backtestは開発予測だけを使う。
  最上位manifestに `final_period_prepared_rows`、`final_period_used_for_model_selection=false`、
  `final_period_evaluated=false` を分けて保存し、「未準備」と「未評価」を混同しない。
- `configs/experiment.example.json`、`stock-experiment` entry point、READMEの概要、RUNBOOKの取得前提、
  `--validate-only`、CLI、Notebook、保存先、失敗時確認を追加した。新規本番依存は追加していない。

### 検証結果

- `python -m pytest tests/test_experiment.py -q -p no:cacheprovider --basetemp=.test_tmp_experiment_final`：
  **7 passed（23.29秒）**。厳格設定、事前検査の無書込、target不整合、TOPIX入力不変、全工程順、
  設定・入力・レポート保存、先行失敗時停止、既存run拒否、実工程end-to-endを確認した。
- 実工程の合成例は2016〜2024年、4銘柄、日次9,392行を使用した。週次表1,876行、シグナル適格
  1,828行、2020〜2024年の260シグナル日について3モデル合計3,120予測行、独立評価3行を生成した。
  backtestは1,040入口と2,080約定を記録し、未決済0、`main_result_complete=true` だった。
  合成価格パターンと固定コストによる値であり、実データ性能または将来利益の根拠ではない。
- 合成例の11工程はすべて `complete`、`final_period_used_for_model_selection=false`、
  `final_period_evaluated=false` となり、raw TOPIXの実行前後SHA256が一致した。
- 全テストは **232 passed（45.93秒）**。既存225件を含み、失敗・skip・警告はなかった。
- `python -m py_compile src/stock_prediction/experiment.py` と
  `python -m stock_prediction.experiment --help` は成功した。

### 現時点の制限と未解決事項

- 現在のワークスペースには価格snapshotはあるが、TOPIX、取引日別銘柄マスター、決算予定・財務実績を
  同期間で持つ補助snapshotがない。このため実データの `--validate-only` と全工程実行は未完了であり、
  補助取得後に実件数・coverage・性能を確認する必要がある。
- 最初の分析シグナルより前に、特徴量・上場履歴用60取引日以上と決算クールダウン用25取引日以上の
  入力・カレンダーが必要である。2016年から分析する場合は2015年側をウォームアップとして取得する。
- 現行modelingは開発foldの比較までで、2016〜2024年へ再fitした固定モデルを2025〜2026年へ適用する
  最終推論工程は未実装である。experimentのbacktestを最終バックテストと解釈しない。
- 入力snapshot全ファイルの実hashは各個別工程が読み込み時に計算する。最上位の実行前識別はsnapshot
  manifestそのもののhashと、manifestに宣言されたファイルhash集合のdigestであり、提供元manifestを
  信頼しない独立全件再hashではない。

## 売買バックテスト（2026-10-13、実装済み）

### 目的と対象

`src/stock_prediction/backtest.py` を追加し、Notebook向けDataFrame APIとCLIから同じ中核処理を
利用できるようにする。シグナル日終値後に発注し、次取引日始値以降の約定、設定可能な決済価格、
手数料・スプレッド・スリッページ、売買単位、売買代金参加率、未約定、値幅制限、売買停止、
上場廃止、ターンオーバーを監査可能な形で扱う。原本 `data/raw/` は変更しない。

### 実装前に確認した入力と既存責務

- `README.md` は予測評価と売買バックテストを分離し、エントリーをシグナル日の次取引日始値、
  決済価格を「評価日終値」または「評価日翌取引日始値」の固定設定と定めているが、主結果の方式は
  未確定である。
- `preprocessing.py` の監査済み日次株価は `date`, `code`, 未調整OHLC、`upper_limit`,
  `lower_limit`, `volume`, `turnover_value`, 調整係数・調整済みOHLCV、`has_price`,
  `calendar_verified` を持つ。実際の資金・株数・約定可能性には未調整価格・数量を使える。
- `universe.py` は `(signal_date, code)` ごとの時点適格性、過去20取引日の流動性、
  `entry_date` / `exit_date`、翌取引日始値と評価日終値による事後proxyを持つ。ただし現行proxyは
  ポートフォリオ資金、売買単位、実注文額、コスト、部分約定、注文持越しを扱わない。
- `modeling.py` の予測成果物は `fold_id`, `validation_year`, `signal_date`, `code`, `model`,
  `target_column`, `actual_target`, `target_observed`, `prediction` を持つ。バックテストはtargetを
  銘柄選択や約定判定に使わず、有限な予測とシグナル時点適格性だけで注文を作る必要がある。
- 現在のワークスペースには監査済み日次株価がある一方、同一スナップショットから生成した週次表、
  予測成果物、上場廃止の確定イベントおよび現金対価データはない。実装検証は合成データで行い、
  実データの投資成績とは解釈しない。
- `pyproject.toml` の本番依存は pandas、NumPy、PyArrow、requestsであり、本実装に新規依存は
  不要な見込みである。作業ディレクトリにはGitメタデータがなく、`git status` は実行不能だった。

### 確定した研究判断

- ユーザー確認により、主結果の決済価格は評価日終値とする。評価日翌取引日始値も設定で切り替え
  可能にし、非主方式は感度分析に使用できるようにする。
- 買い注文はシグナル日の次取引日始値だけで執行し、値幅制限、停止、価格・出来高不足、参加率・
  資金・売買単位によって約定できない残注文は当日失効とする。
- 上場廃止は確定イベントと現金対価が入力されていれば実条件で清算する。保有中の上場廃止について
  条件が不足する場合、主結果を未確定としてエラーにし、全損仮定は明示的な感度分析としてのみ
  選べるようにする。最終売買価格への暗黙の置換は行わない。
- 片道コストの初期値は、手数料5bps、半スプレッド5bps、スリッページ10bpsとする。これは
  一般的な市場標準とはみなさず、すべて設定で変更可能にし、manifestへ保存する。

### 進捗

- [x] README.md / AGENTS.md / 日次株価・ユニバース・週次表・予測schema・期間・依存・既存テストの確認
- [x] 既存実装と不足入力（上場廃止確定イベント・現金対価、実予測成果物）の識別
- [x] 主決済価格、買い注文有効期間、未解決上場廃止、主コスト仮定についてユーザー確認
- [x] DataFrame API、設定読込、schema検査、イベント駆動バックテストの実装
- [x] CLI、成果物、manifest、中断時状態保存の実装
- [x] コスト、売買単位、参加率、部分・未約定、値幅制限、停止、上場廃止、ターンオーバーのテスト
- [x] README、RUNBOOK、バックテスト仕様、CLIエントリーポイントの同期
- [x] 関連テスト、全テスト、CLIヘルプ、最小合成例の検証と結果記録

### 実装内容

- `src/stock_prediction/backtest.py` に、厳格な `BacktestConfig`、Notebook用
  `backtest_portfolio`、保存API `run_backtest`、CLI `main` を追加した。入力DataFrameを変更せず、
  予測順位、注文、約定、保有、日次資産、週次指標を同じ中核処理で生成する。
- 予測は設定したモデル・targetの `signal_date`, `code`, `prediction` だけを使用し、実現targetを
  読まない。候補表とは完全一致結合し、シグナルごとの日程一意性と次取引日エントリーを公式
  カレンダーで検査する。同値順位はコード昇順で固定した。
- 等金額予算、利用可能現金、売買単位、`median_turnover_value_20 * max_participation_rate` の最小値で
  買付株数を丸める。当日始値注文のサイズ決定に当日売買代金を使わない。参加率による部分約定を
  記録し、残る買い注文は失効する。
- 買いは未調整始値、主決済は未調整評価日終値を使う。評価日翌取引日始値も設定可能である。
  価格なし、出来高・売買代金0、ストップ高買い、ストップ安売りを未約定理由として保存する。
  売り未約定は最大20取引日（設定可）まで繰り越し、残る建玉を削除せず主結果を未完了にする。
- 手数料、半スプレッド、スリッページを別々の片道bpsとして実効価格・現金・コスト台帳へ反映した。
  初期値5/5/10bps、資金1,000万円、上位10、売買単位100株、参加率1%を
  `configs/backtest.json` に保存した。すべての値をmanifestへ保存する。
- `adjustment_factor` を当日始値前に保有株数へ反映し、変更前後、整数性、売買単位整合を
  `corporate_actions.parquet` に残す。端株等で通常単位の売却が完了しなければ未完了として残す。
- 任意の確定上場廃止清算表を追加した。1株当たり清算価値があればその値で清算し、欠損なら主設定は
  `UnresolvedDelistingError` で停止する。`write_off` は理由を明記した全損感度分析としてのみ選べる。
  価格欠損や `master_transitions` から上場廃止を推測しない。
- 注文、約定、保有、コーポレートアクション、日次資産、週次指標、年別指標、全体指標の8 Parquetを新規
  ディレクトリへ保存する。入力・コードSHA256、設定、環境、完了成果物をmanifestへ逐次保存し、
  失敗と `KeyboardInterrupt` も状態・例外・完了済み一覧として残す。
- CLIは予測対象年の週次候補列と、必要期間・必要銘柄・必要列だけの日次株価を読む。
  `stock-backtest` を `pyproject.toml` へ追加し、`docs/BACKTEST.md` に入力契約・数式・出力・制限、
  READMEとRUNBOOKに概要とCLI/Notebook手順を同期した。新しい本番依存関係は追加していない。

### 検証結果

- `python -m pytest tests/test_backtest.py -q -p no:cacheprovider --basetemp=.test_tmp_backtest_yearly`：
  **15 passed（2.58秒）**。
- `python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_all_backtest_final_3`：
  **225 passed（28.39秒）**。既存210件を含み、失敗・skip・警告はなかった。
- 既知値例は2024-01-08シグナル、1銘柄、初期資金100,000円、始値100円、評価日終値110円、
  100株単位、片道5/5/10bpsで実行した。900株を売買し、最終資産108,622.00675円、往復コスト
  377.99325円、two-way turnover 1.89となり、手計算と一致した。
- ストップ高買いの当日失効、参加率による100株だけの部分約定、ストップ安売りと出来高0による
  翌取引日再試行、最大再試行後も建玉を残して `main_result_complete=false` とすることを確認した。
- 評価日翌取引日始値への設定切替、調整係数0.5による保有株数2倍、確定上場廃止の現金清算、
  未解決上場廃止の主設定エラーと全損感度分析を確認した。
- `actual_target` を極端値へ変更しても注文、日次資産、全体指標が完全一致した。CLIとDataFrame APIの
  約定・全体指標も一致し、8成果物、complete manifest、既存出力拒否、failed/中断manifestを確認した。
- `python -m py_compile src/stock_prediction/backtest.py`、
  `python -m stock_prediction.backtest --help`、`configs/backtest.json` の厳格読込は成功した。
- 実データ読込smokeでは2024年Parquetから必要11列、銘柄`13010`、2024-01-04〜2024-01-10だけを
  Arrow filterで読み、4行・1銘柄・指定期間だけになった。全列・全銘柄の無条件読込は行っていない。
- 実データ監査済み日次株価は11年別Parquet、2016-04-08〜2026-04-08、計10,058,123行だった。
  ただし週次候補表、予測成果物、確定上場廃止清算表がないため、実データバックテストは実行していない。

### 現時点の制限と未解決事項

- 日次OHLCVと値幅フラグは板・気配・注文優先順位を再現しない。値幅到達日は一日全体を売買不能と
  するため保守的になり得る。
- 時点別売買単位がない行は既定100株を使う。異なる単位を含むデータでは候補表へ `lot_size` を
  追加し、manifestのfallback件数を0にしてから主結果を確定する。
- 配当、税金、信用金利、貸株料は未実装であり、現状は価格リターンのロングオンリー現物戦略である。
- 株式交換後の後継株を動的に保有するのではなく、確定した旧株1株当たり経済的清算価値を入力する。
  実清算値がない上場廃止は主結果として計算しない。
- 初期コスト5/5/10bpsは一般的な標準ではない。開発期間でコスト、資金量、上位N、参加率の感度を
  事前に比較し、2025年以降を見ずに主設定を固定する必要がある。

## 予測評価の独立実装（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/evaluation.py` を追加し、モデルの検証予測に対して、シグナル日ごとの
Spearman IC、上位N・下位Nの平均実現リターン、上下位差、target / prediction / scoring coverageを
計算する。日付別の値を時間方向に集約して全期間・年別・業種別・年別業種別の安定性を確認できる
ようにする。Notebook向けDataFrame APIとParquet入力のCLIは同じ中核処理を使う。

### 実装前に確認した入力と既存責務

- `modeling.py` の `predictions.parquet` は `fold_id`, `validation_year`, `signal_date`, `code`,
  `model`, `target_column`, `actual_target`, `target_observed`, `prediction` を持つ。現在の基準モデルは
  全適格行へ有限予測を出すが、評価層では欠損予測を許容してcoverage低下として明示する。
- `modeling.py` には日付別指標の簡易計算があるが、全期間・年別・業種別の独立成果物、
  時点業種の結合監査、単独CLIはない。IC等の基本数式は揃え、欠損targetがある上位Nでは
  下位銘柄を補充しない、より保守的な選択coverageを評価層で明示する。
- J-Quants時点銘柄マスターは取引日別に `Date`, `Code`, `S17` / `S17Nm`,
  `S33` / `S33Nm` を持つ。現行の週次表・modeling予測には業種列が引き継がれないため、CLIは
  年別listed-info Parquetから予測対象年の必要4列だけを読み、シグナル日・銘柄で完全一致結合する。
- 現在の実データは監査済み日次株価までで、時点銘柄マスター、週次学習表、modeling成果物がない。
  実装検証は合成データで行い、実際のIC・リターン・coverageとは解釈しない。

### 確定した実装判断

- 主ICはREADMEに合わせて日付横断のSpearman順位相関とし、同日の有効な予測・targetが2件以上で
  両系列が非定数の場合だけ算出する。Pearson相関も診断列として併記する。
- 上位・下位は予測降順、同値はコード昇順で決める。有効行がN未満なら存在する全行を使い、
  `selection_count` とN充足日数を残して不足を隠さない。日付別リターンの集約は各シグナル日を
  等重みとし、銘柄行を独立な時間標本とみなす信頼区間は作らない。
- 上位・下位の銘柄はtarget観測可否を見ずに有限予測だけで固定する。選択後にtarget欠損があっても
  次順位で補充せず、その日の平均targetと上下位差をNullにして選択内coverageを別記する。
- coverageの分母は入力予測に含まれるシグナル時点適格行とする。`target_coverage`,
  `prediction_coverage`, `scoring_coverage` を別々に出し、業種分類の完全一致coverageも別列で残す。
- 業種別指標は各日・各業種内で横断計算してから時間方向へ集約する。既定は東証33業種
  (`S33` / `S33Nm`) とし、17業種は設定ファイルの列名変更で選べる。業種欠損を現在の分類や前後日の
  値で補完せず、業種別指標から除外した件数とcoverageを全体・年別成果物へ残す。
- CLIは予測に含まれる年のlisted-infoファイルだけを選んでからschemaと値を読む。入力SHA256、設定、
  期間、行数、欠損件数、コード・環境版、完了済み成果物をmanifestへ保存し、既存出力は上書きしない。

### 進捗

- [x] README.md / AGENTS.md / 予測schema・時点業種schema・期間・依存・既存テストの確認
- [x] IC、上位N、上下位差、3種coverage、33業種既定、完全一致結合の定義確定
- [x] DataFrame API、設定読込、schema検査、日付別・集約評価の実装
- [x] CLI、成果物、manifest、中断時状態保存の実装
- [x] 指標既知値、欠損、同値順位、業種時点整合、CLI/API一致、入力保護のテスト
- [x] README、RUNBOOK、評価仕様、CLIエントリーポイントの同期
- [x] 関連テスト、全テスト、CLIヘルプの検証と結果記録

### 実装内容

- `src/stock_prediction/evaluation.py`：厳格な `EvaluationConfig`、入力schema検査、
  `evaluate_predictions`、日付別・全期間・年別・業種別・年別業種別集約、保存API
  `run_evaluation`、CLI `main` を追加した。入力DataFrameは変更しない。
- 同じ日・targetで全モデルの適格銘柄集合と実現targetが一致することを検査する。主キー重複、日付、
  validation年、5文字コード、文字列、bool、数値、非有限値、target観測フラグ不一致を早期エラーにする。
- ICは評価可能行、上位/下位の固定は有限予測行から計算する。3種coverage、選択数、N充足、
  選択target coverage、有効日数を残し、欠損targetによる順位の詰め替えをしない。
- 業種はlisted-infoの予測対象年・予測対象日・設定4列だけを読み、完全一致で結合する。全体日付別、
  全期間、年別成果物には業種分類coverageを残し、分類0件でもschema付きの空業種成果物を生成する。
- CLIは業種付き予測、日付別、全期間、年別、業種日付別、業種別、年別業種別の7 Parquetを新規保存し、
  入力SHA256、読込年、期間・件数、分類coverage、環境、コードSHA256、完了出力をmanifestに保存する。
  失敗・KeyboardInterrupt時も作成後のmanifestを更新する。
- `configs/evaluation.json` と `stock-evaluation` エントリーポイントを追加した。新規本番依存はない。
  `docs/EVALUATION.md` に入力契約、数式、欠損規則、成果物、制限を定義し、READMEとRUNBOOKを同期した。
- `tests/test_evaluation.py` を13件追加した。既知IC・上下位値、時間等重み集約、3種coverage、
  選択内欠損、同値順位、未来業種非干渉、完全一致0件、schema、モデル間公平性、17業種設定、
  対象年だけのCLI読込、API/CLI一致、既存出力拒否、中断manifestを検証する。

### 検証結果

- `python -m pytest tests/test_evaluation.py -q -p no:cacheprovider --basetemp=.test_tmp_evaluation_final`：
  **13 passed（2.37秒）**。
- `python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_all_evaluation_final_3`：
  **210 passed（27.51秒）**。既存197件を含み、失敗・skip・警告はなかった。
- 既知値例は2020年・2021年、各4銘柄、1モデル、計8予測行。日付別Spearman ICは `1.0`, `-1.0`、
  上位1平均targetは `0.4`, `0.1`、上下位差は `0.3`, `-0.3`、時間平均ICと上下位差はともに0となった。
- target欠損1件、予測欠損1件、業種欠損1件の4行例で、target / prediction / scoring / industry
  coverageがそれぞれ75% / 75% / 50% / 75%となった。上位3は予測だけで固定され、選択target
  coverage 2/3、上位平均target Nullとなり、欠損銘柄が次順位で補充されないことを確認した。
- 未来日の業種を変更しても過去の業種付き予測・業種指標・reportは完全一致した。完全一致分類が
  0件でもindustry coverage 0%とschema付き空成果物になった。
- CLI合成例は2020年4予測行だけを評価し、意図的に不正schemaにした未使用2021年Parquetを開かず、
  DataFrame APIと日付別・年別・業種別成果物が全行一致した。7成果物とcomplete manifestを確認し、
  強制KeyboardInterruptではinterrupted manifestと空の完了一覧が残った。
- `python -m py_compile src/stock_prediction/evaluation.py`、
  `python -m stock_prediction.evaluation --help`、`configs/evaluation.json` の読込は成功した。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。

### 現時点の未解決事項

- 実データ週次学習表と予測成果物が未生成のため、実データの評価件数と数値は未検証である。
- 33業種と17業種のどちらを主報告にするかは、開発期間での研究目的に応じて設定を固定する。
  本実装の既定は詳細な差を確認できる33業種であり、最終評価期間を見て切り替えない。

## 基準モデルとfold内前処理（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/modeling.py` を追加し、2016〜2024年の開発foldだけで、定数、
20取引日モメンタム、線形最小二乗モデルを同じ検証行へ予測する。Notebook向けDataFrame APIと、
年別週次Parquetを扱うCLIを同じ中核処理から提供する。2025〜2026年の隔離期間はCLIで読み込まず、
DataFrame APIでも既存の `splits.py` が返す開発位置以外の特徴量・目的変数をfitや評価へ渡さない。

### 実装前に確認した入力と既存責務

- `weekly_dataset.py` の主キーは `(signal_date, code)` で、README 7節の11特徴量、
  `target_return` / `target_excess_return`、シグナル時点の `is_signal_eligible`、目的変数別の
  `is_target_*_sample` を保持する。特徴量Nullはfold内補完のため意図的に残される。
- `splits.py` は2016〜2024年を開発期間、2020〜2024年を年次検証foldとし、同一シグナル日の
  全銘柄を一括割当する。4取引日以上のパージを検査し、2025〜2026年を開発foldから隔離する。
- 現在は同一スナップショットから生成した実データ週次学習表がない。実装と数値検証は合成データで
  行い、実データのcoverage、モデル性能、投資成績としては扱わない。
- 現依存はpandas、numpy、pyarrowで、scikit-learnは入っていない。基準モデルだけのために新しい
  本番依存を増やさず、NumPyの決定論的な最小二乗を用いる。

### 確定した実装判断

- 最終ターゲットは本工程で選ばない。`target_return` と `target_excess_return` を設定で切替可能にし、
  既定の `target_return` は開発用の実行既定値であって研究上の最終選択とはみなさない。
- 定数モデルは各foldの学習target平均、モメンタムモデルはREADMEで定義済みの `R_20` を直接予測値
  とする。モメンタム欠損は同じfoldの学習行だけで求めた中央値で補完する。
- 線形モデルはREADME 7節の11特徴量を固定順で使い、各foldの学習対象行だけで特徴量中央値、平均、
  母標準偏差をfitする。検証行はtransformだけを行う。学習foldで全欠損の特徴量は推測せず停止し、
  定数列のscaleは1として保持する。外れ値clipや特徴量選択はこの単純基準には追加しない。
- 学習は目的変数別の `is_target_*_sample=True` 行だけ、予測は検証年の
  `is_signal_eligible=True` 行へ行う。目的変数欠損行にも予測を残し、評価可能行とは分離して
  target coverageとprediction coverageを報告する。
- MAE、RMSE、方向一致率に加え、シグナル日単位のPearson/Spearman、上位N実現リターン、
  上下位スプレッド、coverageを保存する。銘柄行を独立な時間標本と見なした信頼区間は作らない。
- 乱数を使わず、設定、特徴量順、fold別前処理統計、係数、学習/検証期間、入力SHA256、
  ライブラリ版を成果物とmanifestへ保存する。

### 進捗

- [x] README.md / AGENTS.md / 週次schema・分割API・期間・依存・既存テストの確認
- [x] 最終ターゲットを選択せず設定切替にすること、3モデルとfold内fitの境界を確定
- [x] DataFrame API、設定読込、schema検査、3モデル、評価指標の実装
- [x] 開発年だけを読むCLI、成果物、manifest、中断時状態保存の実装
- [x] fold内fit、同一検証行、公平比較、隔離期間非干渉、CLI/API一致のテスト
- [x] README、RUNBOOK、モデル仕様、CLIエントリーポイントの同期
- [x] 関連テスト、全テスト、CLIヘルプの検証と結果記録

### 実装内容

- `src/stock_prediction/modeling.py`：`ModelingConfig`、fold内fit済み前処理・3モデル、
  `evaluate_walk_forward_models`、`build_walk_forward_models`、`run_modeling`、`main` を追加した。
  DataFrame APIは入力を変更せず、分割、fit、予測、fold別・日付別評価を同じ中核処理で行う。
- 線形前処理は特徴量ごとの学習fold中央値でNullを補完し、補完後の学習fold平均と母標準偏差で
  標準化する。検証では保存済み統計によるtransformだけを行う。NumPy最小二乗は切片、係数、
  設計rank、特異値を保持する。モメンタム補完中央値と定数target平均もfold別に保持する。
- 入力の必須列、主キー、日付、銘柄コード、bool、数値型、非有限値、sample flag整合性を検査する。
  学習foldで全Nullの特徴量、学習targetなし、検証適格行なしは黙って続行せずエラーにする。
- CLIは2016〜2024年の `YYYY.parquet` だけを列選択して読み、2025〜2026年ファイルはschemaも
  調べない。予測、fold指標、日付別指標、全fitパラメータを `artifacts/`、設定・入力SHA256・
  期間・件数・欠損率・環境・完了状態をmanifestとして `reports/` へ新規保存する。
- `configs/modeling.json` にtarget、固定特徴量順、モメンタム列、上位件数を保存した。
  `pyproject.toml` に `stock-modeling` を追加した。新しい本番依存関係は追加していない。
- `docs/MODELING.md` に入力契約、fold内fit、モデル、指標、成果物、制限を定義した。READMEと
  RUNBOOKに概要、CLI/Notebook例、中断・再実行手順、実行後の監査項目を同期した。
- `tests/test_modeling.py` を13件追加した。fold内統計、公平な検証候補、検証外れ値非混入、
  最終期間非干渉、coverage分離、target切替、schema、全Null、設定、CLI/API一致、最終年未読、
  既存出力拒否、中断manifestを検証する。

### 検証結果

- `python -m pytest tests/test_modeling.py -q -p no:cacheprovider --basetemp=.test_tmp_modeling`：
  **13 passed（4.89秒）**。
- `python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_all_modeling`：
  **197 passed（24.11秒）**。既存184件を含み、失敗・skip・警告はなかった。
- 合成例は2016〜2026年、4銘柄、88行、22シグナル日。開発72行、隔離16行である。
  各検証年は2シグナル日、シグナル時点適格6行で、3モデル合計の検証予測は90行、fold指標15行、
  日付別指標30行となった。2020年のtarget欠損1行にも3モデルの予測を残し、prediction coverage
  100%とtarget coverage低下を別々に記録した。
- 2020年検証特徴量を極端値へ変更しても2020年foldの補完中央値・平均・scaleは不変だった。
  2025〜2026年の全特徴量とtargetを極端値へ変更しても、全開発予測、全指標、全fitパラメータ、
  reportが完全一致した。
- CLIテストでは、意図的に必須列を持たない2025年・2026年Parquetを入力ディレクトリへ置いても
  2016〜2024年だけで完走し、DataFrame APIと予測・指標が全行一致した。4成果物とcomplete manifest、
  `final_period_partitions_loaded=false` を確認した。強制 `KeyboardInterrupt` ではinterrupted manifestと
  完了済み出力一覧が残った。
- `python -m py_compile src/stock_prediction/modeling.py` と
  `python -m stock_prediction.modeling --help` は成功した。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。

### 現時点の未解決事項

- 実データ週次学習表が未生成のため、実foldの行数・欠損率・coverage・性能は未検証である。
- 単純リターンとTOPIX超過リターンの最終選択、正則化線形モデル、特徴量選択、外れ値境界の比較は、
  本基準実装後に2016〜2024年だけで行う。2025年以降は選択に使わない。

## 時系列分割と最終評価期間の隔離（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/splits.py` を追加し、週次学習表の同じ `signal_date` に属する全銘柄を
同じ役割へ割り当てる。開発期間はREADMEとAGENTS.mdどおり2016〜2024年に固定し、
2020〜2024年を各1年の検証期間とする拡大型ウォークフォワードを生成する。2025〜2026年の
選択した期間は最終評価専用として開発foldから隔離する。中核はNotebookから利用できる
DataFrame API、保存処理は年別Parquet入力を扱うCLIとして実装する。

### 実装前に確認した入力と既存責務

- `weekly_dataset.py` の週次主キーは `(signal_date, code)` で、同一週の `exit_date`、特徴量、
  目的変数、シグナル時点の適格性と事後監査列を保持する。分割、欠損補完、標準化は行わない。
- README 8.1節は2016〜2024年の開発、2020〜2024年の年次拡大型検証、2025〜2026年の
  最終バックテスト隔離、目的変数の予測期間に応じたパージを定めている。
- 現在は同一スナップショットから生成した実データ週次学習表がないため、分割実装は合成データで
  検証する。既存の週次表生成工程と同様、合成検証を実データcoverageや投資成績と解釈しない。
- 新しい本番依存関係は不要で、既存のpandas、numpy、pyarrowだけを使う。

### 確定した実装判断

- 開発期間は `2016-01-01`〜`2024-12-31`、検証年は2020〜2024年に固定する。各foldの学習は
  開発開始以降かつ検証開始より前のデータを拡大型で使い、ランダム分割は行わない。
- パージ幅は公式取引カレンダー上の取引日数で設定し、既定を4取引日とする。これは通常週の
  月曜終値から金曜終値までの4取引日差に対応する。入力内の最大 `signal_date`〜`exit_date`
  取引日差より短い設定は拒否し、ラベルが検証開始日へ達する学習候補も必ず除外する。
- 隔離開始日・終了日は設定可能にするが、確定済み研究仕様を守るため2025〜2026年内に限定する。
  選択期間は `signal_date` から `exit_date` までが両端を含めて完全に収まる行だけとする。
  選択外、開発範囲外、境界を跨ぐ行は開発へ流用せず `out_of_scope` として件数と日付を残す。
- 分割は適格性や目的変数の観測可否で行を削らない。それらのフィルタとfold内前処理は後続の
  学習工程の責務とし、本工程は全行へ日付単位の分割定義を与える。

### 進捗

- [x] README.md / AGENTS.md / 週次schema・期間・依存・既存テスト・CLI構成の確認
- [x] 分割境界、パージ単位、隔離期間の設定範囲、選択外行の扱いを確定
- [x] DataFrame API、設定読込、schema・時点境界検査の実装
- [x] 年別入力CLI、分割定義Parquet、manifest、中断時状態保存の実装
- [x] パージ、同一シグナル日、隔離、未来行非干渉、schema、CLI/API一致のテスト
- [x] README、RUNBOOK、分割仕様書、CLIエントリーポイントの同期
- [x] 関連テスト、全テスト、CLIヘルプの検証と結果記録

### 実装内容

- `src/stock_prediction/splits.py`：`SplitConfig`、`WalkForwardFold`、`SplitResult`、
  `build_walk_forward_splits`、`run_splits`、`main`。Notebookの `iloc` 用位置配列とCLIの
  日付単位Parquetを同じ中核ロジックから生成する。
- 入力の必須列・型・主キー、同一シグナル日の評価日一意性、公式取引日、カレンダーcoverage、
  ラベル順序を検査する。パージ設定が入力開発期間の最大ラベル幅未満なら早期エラーにする。
- `configs/splits.json`：2025〜2026年の既定隔離境界と4取引日のパージ幅を保存した。
- `tests/test_splits.py`：12件。同一シグナル日一括割当、5foldの拡大型境界、パージ、隔離期間変更、
  期間末を跨ぐラベル、2027年追加による過去fold非変化、パージ不足、schema、設定、CLI/API一致、
  既存出力拒否、中断manifestを検証する。
- `docs/SPLITS.md`：入力契約、固定期間、パージ式、隔離境界、選択外理由、API、保存形式、制限を
  定義した。README、RUNBOOK、`pyproject.toml` に概要、実行例、`stock-splits` を同期した。
- CLIは年別Parquetの全特徴量を一括読込せず、分割に必要な3列だけを読む。`folds.parquet`、
  `isolated.parquet`、`out_of_scope.parquet` と、入力SHA256・fold別件数・設定・実行環境を持つ
  manifestを新規出力へ保存する。失敗・中断時は完了済み出力名をmanifestへ残す。

### 検証結果

- `python -m pytest tests/test_splits.py -q -p no:cacheprovider --basetemp=.test_tmp_splits_final`：
  **12 passed（2.32秒）**。
- `python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_all_final`：
  **184 passed（22.78秒）**。警告、失敗、skipはなかった。
- 合成例は2016〜2026年、2銘柄、26行、13シグナル日。開発22行、隔離4行、選択外0行、
  最大ラベル幅4取引日である。2020年foldでは2019-12-31の2銘柄を4取引日パージで除外し、
  検証開始2020-01-03より前に全学習ラベルが終了することを確認した。
- 隔離を2025-04-01〜2025-12-31へ狭めた例では2025年2行だけを隔離し、2026年2行を
  `after_selected_isolation` として選択外へ置き、どちらも開発foldへ入らないことを確認した。
- 2027年2行と極端な特徴量値を末尾へ追加しても、既存5foldの定義と位置配列は変化しなかった。
- 合成年別Parquetを使ったCLIは3つの分割定義とcomplete manifestを生成し、DataFrame APIと
  全行・全列一致した。`python -m py_compile` と `python -m stock_prediction.splits --help` も成功した。
- 2024年末または選択隔離期間末をラベルが跨ぐ4行は理由付き `out_of_scope` となり、開発・隔離の
  どちらにも入らないことを確認した。2つ目の出力保存直前に `KeyboardInterrupt` を発生させた例では、
  `status=interrupted` と完了済み `folds.parquet` がmanifestへ保持された。
- 実カレンダー `2016-10-08〜2026-10-07` から週次日程496件を作り、分割境界だけを1仮想銘柄で
  追加監査した。開発409週、隔離87週、選択外0週、最大ラベル幅4取引日。検証週は2020年48、
  2021年49、2022〜2024年各50で、各foldの学習週は162、210、259、309、359だった。
  年境界ではラベル重複も4取引日以内の学習シグナルもなくパージ0週だった。これは公式日程の
  境界監査であり、実銘柄のサンプル数・coverage・投資成績ではない。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。

### 既知の制限と未解決事項

- 実データ週次学習表が未生成のため、各foldの実際の行数・銘柄数・coverage、年境界で実際に
  パージされる週数は未検証である。実データ生成後にmanifestで確認する必要がある。
- 本工程の隔離は分割定義による論理的な隔離であり、暗号化やOSアクセス権による物理的隔離ではない。
  最終設計確定前は `isolated.parquet` が指す行の目的変数・評価結果を読まない運用が必要である。

## 週次学習表（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/weekly_dataset.py` を追加し、公式取引カレンダーから週初シグナル日、
翌取引日のエントリー日、同じ暦週の週末評価日を作る。シグナル日と評価日の調整済み終値から
README 6節の単純リターンとTOPIX超過リターンを作り、日次特徴量、時点整合ユニバース、
決算イベント判定を `(signal_date, code)` で検査付き結合する。中核はNotebookから呼べる
DataFrame API、保存処理は年別Parquetを扱うCLIとして分離する。

### 実装前に確認した入力と既存責務

- `descriptors.py` の主キーは `(date, code)`、出力はREADME 7節の11特徴量で、週次選択、
  欠損補完、標準化は行わない。
- `universe.py` の週次主キーは `(signal_date, code)`。`is_eligible` はシグナル時点情報だけ、
  `is_executable` は翌取引日始値・評価日終値を使う事後proxyとして分離されている。
- `earnings_events.py` は同じ週次主キーに `is_earnings_eligible`（主判定）、
  クールダウンなし感度分析列、事後監査列を追加し、元の `is_eligible` を上書きしない。
- 監査済み株価中間データ `data/interim/preprocessing_20261004/` は2016-04-08〜
  2026-04-08だが、公式カレンダー未指定で生成され、manifestの
  `allow_unverified_calendar=true`、全行の `calendar_verified=false` である。
- 新しい不変スナップショット `data/raw/jquants_v2_20261008T120437054560Z/` は
  2016-10-11〜2026-10-07の株価10,126,708行と、2016-10-08〜2026-10-07の
  公式カレンダー（東証取引日2,440日）を持つ。一方、このスナップショットに対する前処理、
  特徴量、時点銘柄マスター、決算予定・実績、週次ユニバースの実出力はまだ存在しない。
  したがって今回の実装は合成データで検証し、実データ学習表の生成完了とは扱わない。

### 確定した実装判断

- README 3節は、取引日数が極端に少ない週を除外し、最低取引日数を設定ファイルで管理すると
  定めている。ユーザー確認により、4日を主設定（3取引日以下を除外）として確定した。
- 月曜から日曜までを確認できる完全な暦週だけを対象とし、最初の東証取引日が月曜または火曜、
  最後の東証取引日が木曜または金曜であることも必要とする。入力カレンダー境界の不完全週から
  将来の営業日を推測しない。
- `exit_date` をREADMEの評価日 `e(t)` とし、`entry_date` はその週のシグナル直後の東証取引日とする。
- 単純リターンとTOPIX超過リターンを両方生成し、この工程では最終ターゲットを選択しない。
  既定ではTOPIX入力と対象週の完全な終値coverageを必須にする。
- ユニバースを主表とし、決算出力のキー集合は完全一致を要求する。ユニバース合格行の特徴量行と
  調整済みシグナル日終値は必須だが、個々の特徴量Nullはfold内前処理のため保持する。
- `is_signal_eligible = is_eligible and is_earnings_eligible` はシグナル時点情報だけで作る。
  将来価格の観測可否は `is_target_return_sample` / `is_target_excess_return_sample` へ分離し、
  `is_executable` と `is_earnings_eligible_ex_post` を主判定へ混ぜない。
- 評価日価格欠損は補完せず、`target_missing_reasons` とreportへ残す。上場廃止時の経済損益規則は
  未確定なので、この工程で通常欠損として推測・補正しない。

### 進捗

- [x] README.md / AGENTS.md / 既存schema・期間・テスト・CLI構成の確認
- [x] 最低4取引日のユーザー確認と設定保存
- [x] 週次日程、目的変数、検査付き結合のDataFrame API
- [x] 日程生成と年別結合のCLI、年別report、manifest、中断時進捗保持
- [x] 祝日週、短縮週、不完全境界週、既知リターン、欠損、schema、結合coverage、未来分離のテスト
- [x] README、RUNBOOK、週次データ辞書、CLIエントリーポイントの同期
- [x] 関連テスト、全テスト、CLIヘルプ、実カレンダー日程の検証

### 実装内容

- `src/stock_prediction/weekly_dataset.py`：`WeeklyDatasetConfig`、`WeeklyScheduleResult`、
  `WeeklyDatasetResult`、`build_weekly_schedule`、`build_weekly_dataset`、`run_weekly_schedule`、
  `run_weekly_dataset`、`main`。NotebookのDataFrame APIとCLIを同じ中核ロジックへ接続した。
- `configs/weekly_dataset.json`：最低4取引日、TOPIX必須を保存した。
- `tests/test_weekly_dataset.py`：14件。4日祝日週、3日週、不完全境界週、既知の銘柄/TOPIX
  リターン、入力不変性、評価日欠損、将来価格・事後約定の主判定非混入、重複・キー欠落・日程不一致、
  非取引日入力、クールダウンあり/なしと事後決算の分離、設定、日程CLI、年別CLI/API一致、既存出力拒否、
  中断時の完了年保持を検証した。
- `docs/WEEKLY_DATASET.md`：入力契約、日程、数式、適格性、欠損、保存形式、既知の制限を定義した。
- `README.md` / `docs/RUNBOOK.md` / `pyproject.toml`：実装概要、工程順、CLI/Notebook例、
  `stock-weekly-dataset` エントリーポイントを同期した。新しい依存関係は追加していない。

### 検証結果

- `python -m pytest tests/test_weekly_dataset.py -q`：**14 passed（2.99秒）**。
- `python -m pytest -q`：**172 passed（17.74秒）**。管理環境が `.pytest_cache` を作成できない
  `PytestCacheWarning` が1件あったが、テスト失敗はない。
- `python -m py_compile src/stock_prediction/weekly_dataset.py` と、親CLI・`schedule`・`build` の
  `--help` が成功した。
- 実カレンダー `2016-10-08〜2026-10-07`（全暦日3,652行、東証取引日2,440日）では、
  523暦週のうち496週を採用し、27週を除外した。採用シグナル日は2016-10-11〜2026-09-28。
  除外理由は4取引日未満27週、週初曜日13週、評価曜日12週、カレンダー境界2週、
  次取引日エントリーなし5週（理由は重複計上）である。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。

### 既知の制限と未解決事項

- 現在のリポジトリには、新しい公式カレンダースナップショットに対応する前処理・特徴量、
  時点銘柄マスター、決算予定・実績、週次ユニバースの実出力が揃っていない。実データの週次学習表、
  結合coverage、評価日欠損件数、サンプル数は未検証であり、合成検証を投資成績と解釈しない。
- 上場廃止時の目的変数とコード変更・合併の連結規則は未確定である。評価日価格欠損を単純除外すると
  上方バイアスになり得るため、後続の学習・評価工程へ進む前に公式イベント情報と扱いを確定する。
- 調整済み価格は取得後に遡及改定され得る。manifestの入力SHA256とスナップショット取得時点だけでは、
  各過去時点で知られていた調整係数を完全には証明しない。
- 2025〜2026年は機械的な週次表生成対象にできるが、最終ターゲット選択、特徴量選択、閾値、
  モデル、ハイパーパラメータの決定へ使わない。

## 決算イベント判定（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/earnings_events.py` を追加し、週次候補行、J-Quantsの決算発表予定、
財務サマリーの実績開示、公式取引カレンダーから、保有予定期間中の決算跨ぎと
決算発表後25取引日のクールダウンを判定する。Notebook用DataFrame APIとCLIは同じ中核関数を
使用し、`data/raw/` は変更しない。

### 確定した時点基準と実装判断

- 主判定の予定決算跨ぎには、各シグナル日までに公表された行（`PubDate <= signal_date`）だけを
  使用する。同じ `Code` / `FQName` / `FYE` の予定変更は、その時点で最新の公表行だけを有効にし、
  後日の変更を過去へ遡及適用しない。
- 予定データに発表時刻がないため、予定日がシグナル日と同日の場合も保守的に跨ぎ対象とする。
  判定区間は日付粒度で `[signal_date, exit_date]` とし、この保守的な包含関係を出力文書へ明記する。
- 実績開示は実入力で確認した `1Q` / `2Q` / `3Q` / `FY` / `OtherPeriod` の各
  `FinancialStatements` 接頭辞に一致する決算短信だけを対象とし、業績・配当予想の修正等を
  クールダウン開始イベントへ混ぜない。接頭辞は設定ファイルで固定・記録する。
- 実績の `DiscTime` は日本時間として、シグナル日の東証大引け時刻と比較する。現物市場の大引けは
  2024-11-05より前を15:00、同日以後を15:30とする。同日で大引け以前なら利用可能な実績として
  クールダウン日0、大引け後または時刻欠損なら保有期間中の事後イベントとして扱う。
- 発表日の次の東証取引日を1日目とし、発表日0日目と1〜25日目を除外、26日目から再び候補とする。
  平日推測や個別銘柄の価格観測日は使わず、`Date` / `HolDiv` の公式カレンダーだけで数える。
- 主列 `is_earnings_eligible` は「その時点で既知の予定跨ぎなし、かつクールダウン外」であり、
  将来の実績開示を使わない。`is_earnings_eligible_without_cooldown` を感度分析用、
  `is_earnings_eligible_ex_post` を実績開示も反映した事後監査用として分離する。

### 進捗

- [x] README.md / AGENTS.md / 既存schema・期間・テスト・CLI構成の確認
- [x] J-Quants財務サマリーが決算短信と予想修正を含むこと、予定変更が追加行になることの確認
- [x] DataFrame API、年別CLI、設定、manifestの実装
- [x] 予定変更、25取引日境界、休日、銘柄境界、未来不変性、schema、CLIのテスト
- [x] README、RUNBOOK、決算イベント列定義の同期
- [x] 関連テスト、全テスト、CLIヘルプ、最小合成例の検証

### 実装内容

- `src/stock_prediction/earnings_events.py`：`EarningsEventConfig` / `EarningsEventResult` /
  `compute_earnings_exclusions` / `run_earnings_events` / `main`。入力を変更しないDataFrame APIと、
  候補を年別処理するCLIを同じ中核ロジックへ接続した。
- `configs/earnings_events.json`：クールダウン25取引日と、実績決算短信を識別する実データ準拠の
  5接頭辞を保存した。
- `tests/test_earnings_events.py`：予定変更のas-of、未来変更で過去が不変、日0/25/26境界、休日、
  銘柄境界、DocType分離、実績跨ぎの事後分離、歴史的な東証大引け時刻、Null時刻、Null予定日、
  schema、設定、CLI/API一致、既存出力拒否、中断時の完了年・manifest保持を検証した。
- `docs/EARNINGS_EVENTS.md`：入力・出力列、時点基準、境界、欠損、manifest、coverage確認を定義した。
- `README.md` / `docs/RUNBOOK.md` / `pyproject.toml`：実装概要、CLI/Notebook例、
  `stock-earnings-events` エントリーポイントを同期した。新しい依存関係は追加していない。

### 検証結果

- 決算イベント単体テストで **13 passed（1.78秒）**。
- 全テスト：同じ書込み可能な一時領域を指定して **158 passed（17.93秒）**。
- `python -m stock_prediction.earnings_events --help` が終了コード0で、必須4入力、設定、出力、
  レポートの各CLI引数を表示することを確認した。
- 最小合成例では、後日公表された予定変更が過去シグナルを変えないこと、発表後25取引日目は除外、
  26日目は解除、2024-11-05以後の15:20同日開示は日0、15:40または時刻欠損は事後イベント、
  予定跨ぎ・クールダウンなし・事後実績の3判定が分離されることを確認した。
- 初回の全テストはWindowsサンドボックスの既定一時ディレクトリで既存モジュールの原子的
  `manifest.json` 置換が拒否され、141 passed / 15 failedだった。テストや実装を弱めず、
  リポジトリ内の新規一時ディレクトリを指定した再実行で全158件の通過を確認し、一時生成物は削除した。

### 既知の制限と未解決事項

- リポジトリ内に決算予定・財務実績の実スナップショットがないため、実データの行数、銘柄数、
  日付範囲、予定日Null率、DocType分布、除外件数は未検証である。補助データ取得後にmanifestの
  全暦日coverageを確認し、小期間から実行する。
- 予定データに発表時刻がないので、予定日がシグナル日と同日なら保守的に跨ぎとする。
  `SchDate` がNullの有効予定は監査列と件数へ残すが、日付不明のため自動除外しない。
- 実績同日判定の大引け時刻はJPXの制度変更に基づき2024-11-05を境に15:00から15:30へ切り替える。
  半日立会等で通常と異なる終了時刻を表す列は現行カレンダーにないため、追加データ取得時に再確認する。
- 実績跨ぎを含む `is_earnings_eligible_ex_post` は事後監査専用であり、特徴量、シグナル時点の
  銘柄選択、閾値調整には使用しない。

### 現時点の入力データ状況

- リポジトリ内の公式取引カレンダーは2016-10-08〜2026-10-07（全暦日3,652行、
  東証取引日2,440日）を収録している。
- 現在の `data/raw/` には決算予定・財務実績スナップショットがないため、実データ件数の検証は
  補助データ取得後の未解決事項とする。今回の動作検証は、実列名・型に合わせた合成データで行う。

## 時点整合ユニバース（2026-10-12、実装済み）

### 目的と対象

`src/stock_prediction/universe.py` を追加し、J-Quantsの取引日別銘柄マスター、監査済み日次株価、
公式取引カレンダーから、各シグナル日時点の内国普通株式、上場後60取引日、流動性、
売買可能性を判定する。Notebook用DataFrame APIとCLIは同じ中核関数を使用する。

### 確定した設定と時点基準

- 内国普通株式は、各取引日の銘柄マスターに存在し、商品区分 `ProdCat == "011"` の銘柄とする。
  現在の銘柄一覧を過去へ遡及適用しない。
- 上場後60取引日は、連続する日次銘柄マスターで60回の東証取引日が経過したことを確認する。
  入力開始日の既存銘柄も、入力内で60取引日の経過を確認するまで保守的に除外する。
- 流動性窓はシグナル日を含む直近20東証取引日とし、短い履歴で代替しない。価格形成15日以上、
  有効な価格形成日の未調整売買代金中央値1億円以上を主設定とする。
- 予定投資額は1銘柄100万円、参加率上限は売買代金の1%とする。参加率の分母には、シグナル時点で
  利用可能な直近20取引日の売買代金中央値を用いる。
- エントリーはシグナル翌取引日の未調整始値、決済は同週評価日の未調整終値とする。
  日次データだけでは実際の注文約定を確定できないため、正の有限価格、正の出来高、買い時の
  ストップ高なし、売り時のストップ安なしを保守的な売買可能性proxyとして明示する。
- 翌取引日・評価日の価格や値幅制限はシグナル時点では未知なので、`is_eligible` には含めない。
  事後的な `is_entry_executable` / `is_exit_executable` / `is_executable` として分離し、
  将来情報で銘柄選択を行わない。
- 監理・整理指定や注文板、日中の約定順序は現行入力にない。取得できない事実を推測せず、
  出力・manifest・文書に制限として残す。

### 進捗

- [x] README.md / AGENTS.md / 既存データschema・期間・テスト・補助データ仕様の確認
- [x] 流動性閾値、予定投資額、参加率、売買価格のユーザー確認
- [x] DataFrame API、年別CLI、設定、manifestの実装
- [x] 上場履歴、流動性、売買可能性、未来不変性、schema、年境界のテスト
- [x] README、RUNBOOK、データ辞書の同期
- [x] 関連テスト、全テスト、CLIヘルプ、最小合成例の検証

### 実装内容

- `src/stock_prediction/universe.py`：`UniverseConfig` / `UniverseResult` /
  `compute_universe` / `run_universe` / `main`。銘柄マスターを主表にして株価欠損日を保持し、
  上場区間ごとのrolling、判定理由、事後売買可能性proxyを生成する。DataFrame入力を変更せず、
  CLIは必要列だけを年別に読み、直前60取引日の状態を引き継ぎ、既存出力を上書きしない。
- `configs/universe.json`：ユーザー確認済みの60日、20日中15日、1億円、100万円、1%を保存した。
- `tests/test_universe.py`：内国普通株式、60日境界、消失・再出現、20日窓、欠損日、閾値境界、
  銘柄境界、祝日週、翌取引日/同週末、値幅制限、未来不変性、schema、年境界、CLI/manifestを検証。
- `pyproject.toml`：`stock-universe` エントリーポイントを追加。新しい依存関係は追加していない。
- README / `docs/RUNBOOK.md` / `docs/DATA_DICTIONARY.md` / `docs/UNIVERSE.md`：実行手順、
  入出力、単位、時点基準、除外理由、事後proxy、取得不能な監理・整理指定等の制限を同期した。

### 検証結果

- 新規テスト：
  `python -m pytest -q tests/test_universe.py --basetemp=.test_tmp_universe5 -p no:cacheprovider`
  は **17 passed（3.11秒）**。未来の銘柄区分・売買代金・エントリー/決済価格・値幅制限を
  変更しても、過去の流動性と `is_eligible` が変わらないことを含む。
- 全テスト最終実行：
  `python -m pytest -q --basetemp=.test_tmp_universe_release -p no:cacheprovider` は
  **145 passed（13.73秒）**。既存の日次取得、補助取得、前処理、特徴量の128件を含み回帰なし。
- 合成CLI例は2023-10-02から90東証セッション、2銘柄（普通株式1、対象外商品1）、180行を
  2年のParquetへ分割した。出力180行、普通株式は入力内60取引日経過後の30行だけが適格となり、
  年境界の先頭でも履歴が維持された。これは投資成績の評価ではない。
- `python -m stock_prediction.universe --help`、`py_compile`、`configs/universe.json` のJSON解析、
  RUNBOOKのPythonコードブロック8件のAST構文解析が成功した。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。

### データ状況と既知の制限

- リポジトリ内には株価・取引カレンダーの実データはあるが、補助取得の
  `listed_info/YYYY.parquet` はまだない。このため実データのユニバース件数・除外率は未計測で、
  原本を推測や現在銘柄で代用せず、合成データだけで最小実行を検証した。
- `ProdCat == "011"` はJ-Quantsの内国普通株式区分として厳格に要求する。古い/不完全なschemaで
  `ProdCat` がない場合は他列や銘柄名から推測せず停止する。
- 日次OHLCV、出来高、売買代金、値幅制限フラグでは注文板や実約定を確定できないため、
  `is_executable` は保守的な事後proxyである。`is_eligible` とは分離済み。
- 監理・整理指定の履歴は現行入力にない。長期停止は20日中の価格形成日数で検出できるが、
  指定そのものを検証済みとはしない。決算跨ぎと決算後25取引日は後続工程の責務である。

## 補助原データ取得（2026-10-12）

### 目的と対象

`src/stock_prediction/auxiliary_data_download.py` を追加し、J-Quants API V2からTOPIX、
各過去時点の上場銘柄マスター、決算発表予定、実際の財務開示を取得する。CLIとNotebookは
同じ保存関数を使い、`data/raw/` の新規スナップショット以外を変更しない。取得段階では
投資可能ユニバース、決算跨ぎ除外、25取引日クールダウン、特徴量、目的変数を生成しない。

### 確認済み仕様と判断

- 2026-10-12時点のJ-Quants公式仕様・公式Pythonクライアントで、V2 endpointと列を確認した。
  TOPIXは `/indices/bars/daily/topix`、銘柄マスターは `/equities/master`、決算予定は
  `/fins/earnings-date`、財務サマリーは `/fins/summary` を使う。認証は既存実装と同じ
  `x-api-key`、全endpointで `pagination_key` を追跡する。
- TOPIXは期間指定1回、銘柄マスターは各東証営業日を `date` 指定する。現在の銘柄一覧を過去へ
  遡及適用しない。日別応答の `Date` が要求日と一致しなければ停止する。
- 決算予定は `PubDate` が情報の公表/利用可能日、`SchDate` が予定対象日である。
  財務サマリーは `DiscDate` / `DiscTime` が実際の開示時点で、対象決算期間とは分離する。
  休日公表を取引日という仮定で落とさないため、この2種類は全暦日を問い合わせる。
- TOPIX、銘柄マスターは営業日の空結果を許容しない。決算予定・財務開示は0件の日が正当なので、
  空結果を安定した型・列を持つ0行DataFrameとして年別Parquetへ結合する。
- 現行銘柄マスターには旧コードと新コードを結ぶ安定した発行体IDやイベント種別がない。
  連続スナップショットから `appeared` / `disappeared` / `attributes_changed` を観測差分として
  保存するが、上場・上場廃止・合併とは断定しない。消失・出現コードの会社名が一意に完全一致
  した組だけを `code_change_candidates` に保存し、非確定候補であることをmanifestと文書に残す。
  確定イベントには別の公式コーポレートアクション情報との照合が必要である。
- 最初の取得営業日は比較元がないbaselineで、差分coverageは2営業日目から始まる。
- 長期取得は最低でも「営業日日数 + 全暦日数×2」回にページネーションを加えた通信となる。
  データセット選択を可能にし、年完了ごとに原本とmanifestを確定する。中断・通常例外時は
  取得済み当年を `partial/` に保存する。同一出力先への再開・追記は行わない。

### 進捗

- [x] README.md / AGENTS.md / 既存取得・前処理・特徴量・テスト・実データ期間の確認
- [x] 現行J-Quants V2 endpoint、クエリ、列、時点基準の確認
- [x] Notebook共通API / CLI / 年別保存 / schema検査 / manifest / 中断保存の実装
- [x] 銘柄マスター観測差分と非確定コード変更候補の実装
- [x] `pyproject.toml` エントリーポイント、README、データ辞書、RUNBOOKの同期
- [x] 新規テスト、全テスト、CLIヘルプ、最小合成実行の最終検証

### 実装内容

- `src/stock_prediction/auxiliary_data_download.py`：
  `AuxiliaryDownloadConfig` / `AuxiliaryDownloadResult` / `AuxiliaryJQuantsV2Client` /
  `load_trading_calendar` / 4種類のschema validator / `derive_master_transitions` /
  `download_auxiliary_data` / `main`。入力DataFrameを変更せず、既存出力とraw外出力を拒否する。
- `tests/test_auxiliary_data_download.py`：endpoint/クエリ、schema、日付・OHLC・コード・時刻、
  全暦日問い合わせ、空日の型、差分、非確定候補、年別保存、秘密非保存、中断、出力保護を検証。
- `pyproject.toml`：`stock-download-auxiliary` を追加。新しい依存関係は追加していない。
- README / `docs/RUNBOOK.md` / `docs/DATA_DICTIONARY.md`：取得手順、出力schema、時点基準、
  リクエスト量、中断時の扱い、確定できないイベントを同期した。

### 検証結果

- 新規テスト：
  `python -m pytest -q tests/test_auxiliary_data_download.py --basetemp=.test_tmp_aux_final -p no:cacheprovider`
  は **18 passed（1.70秒）**。4 endpointとクエリ、全暦日/営業日の使い分け、schema、
  TOPIX OHLCと営業日coverage、空日、差分、コード変更候補、中断部分保存、出力保護を含む。
- 全テスト最終実行：`python -m pytest -q --basetemp=.test_tmp_release -p no:cacheprovider` は
  **128 passed（12.65秒）**。既存の日次株価取得、前処理、特徴量の110件を含み、回帰なし。
- `python -m stock_prediction.auxiliary_data_download --help` と `py_compile` が成功し、
  必須の日付・カレンダー、任意のデータセット選択、出力、レート・再試行引数を確認した。
- 合成最小例は2025-01-01〜2025-01-03の3暦日（2営業日）、銘柄マスター4行、
  決算予定1行、財務開示1行、TOPIX2行。観測差分3行とコード変更候補1行を保存し、
  manifestの完了日数、SHA256、APIキー非保存を確認した。これは投資成績評価ではない。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、
  requests 2.34.2、pytest 9.1.1。許容依存バージョンの全組合せ検証ではない。
- APIキーと契約プランを必要とする実通信は行っていない。実データ取得前に短期間で契約期間、
  行数、schema、空日、manifestを確認する。

### 既知の制限・次工程

- 上場廃止・コード変更・合併を確定する公式イベントデータは、本モジュールの4 endpointにはない。
  `master_transitions` / `code_change_candidates` は探索・照合対象であり、バックテストの確定イベント
  として直接使わない。
- `/fins/earnings-date` とTOPIX等の利用可否・取得可能期間は契約プランに依存する。403や営業日の
  必須データ空結果を黙って欠損へ置換せず、manifestを失敗状態にして新規期間で再取得する。
- `financial_results` は決算短信以外の業績・配当予想修正等も含む財務サマリー原本である。
  保有期間中決算の実績日や25取引日クールダウンへ使う行種別は、後続の決算イベント工程で
  `DocType` 等を公式定義に基づいて決め、開示時刻とシグナル時刻を比較する。
- TOPIX原本は `Date/O/H/L/C` であり、既存特徴量DataFrame APIの `date/close` への写像は
  原本を上書きせず中間データで行う。この記録時点では未実装だったユニバース・決算除外・
  週次結合は2026-10-12に実装したが、対応する実データ出力はまだ生成していない。

## 財務サマリーraw文字列対応（2026-10-19）

### 発見事項と判断

- 実データ取得は `financial_results.NxFNCEPS must contain numbers or nulls` で停止した。
- J-Quants V2公式仕様では `/fins/summary` の全項目がJSON文字列で、数値項目も文字列、未記載値も
  Nullではなく空文字列である。従来の数値dtype要求は公式schemaと不一致だった。
- rawでは財務値を文字列のまま保持し、空文字列を欠損やゼロへ変換しない。数値化は列単位の単位・
  精度・欠損規則と変換不能件数を残せる後続前処理の責務とする。
- `DiscDate` と対象期間の日付列は、要求日一致と日付妥当性の検査のため従来どおり日時型へ正規化する。

### 変更と検証

- [x] 財務値列のraw schemaを文字列へ変更し、数値文字列と空文字列を保持する。
- [x] 合成fixtureを公式V2形式へ変更し、値・空文字列・空日の安定schema・非文字列拒否をテストする。
  rawの空文字列は保持したまま、manifest欠損率では欠損として数える。
- [x] 補助取得テスト18件と全テスト232件、および対象モジュールの構文検査を実行し、すべて成功した。
- [ ] 修正後の実API取得はユーザー環境で新規スナップショットとして実行し、manifestを監査する。

## 実行手順と残工程の整理（2026-10-11）

### 実行手順

- [x] `data_download.py`、`preprocessing.py`、`descriptors.py` の実際のCLI引数と公開関数を確認
- [x] `docs/RUNBOOK.md` にCommand Prompt CLI、Notebook保存API、Notebook DataFrame APIを記載
- [x] APIキー非表示、既存出力保護、公式カレンダー必須、60セッション履歴、TOPIX任意入力を明記
- [x] READMEから一元的な実行手順へリンク
- [x] 3モジュールの `--help` が例と一致することを確認

検証では3モジュールの `python -m ... --help` を実行し、`docs/RUNBOOK.md` 内の
Pythonコードブロック5件をAST構文解析した。Command Prompt例6件は、`set` / `%NAME%`、
行継続 `^`、パス引用を確認し、各CLI引数を実際の `--help` と照合した。文書のみの変更のため、
API通信・データ生成・全テスト再実行は行っていない。

### 現在の実装範囲とファイル境界

- 現在の `data_download.py` が取得するのは、定数で指定された `/markets/calendar` と
  `/equities/bars/daily` の2種類だけである。TOPIX、上場銘柄情報、決算予定・実績開示は
  取得していない。これらの取得関数とCLIオプションを同ファイルへ追加すること自体は可能である。
- 現在の `preprocessing.py` は日次株価16列と既知の任意列を監査する株価専用工程であり、
  docstringとレポートにも、ユニバース・決算除外・特徴量・目的変数は未生成と明記している。
  決算データ自体のschema正規化を同ファイルへ追加することは可能だが、保有期間中の決算除外と
  25取引日クールダウンはシグナル日・評価日が必要なため、現在の株価前処理の責務ではない。
- 下記の名前は「必ず別ファイルを作る」という指定ではなく、リークを独立テストするための
  **責務単位**である。小規模に保つなら既存ファイルの関数やCLIサブコマンドへ統合できる。
  一方、取得schema・時点基準・テストが異なる処理を1関数へ混在させない。

### `descriptors.py` 以降の責務と配置案（1〜7は2026-10-12までに実装済み）

以下は2026-10-11時点に作成した実装順である。1〜5は後続の各実装記録どおり完了し、
6〜7も後続の各実装記録どおり完了した。8以降が未実装である。実データ生成前には、
取得可能な履歴、列、開示時刻とcoverageを確認する。

1. 補助データ取得（`auxiliary_data_download.py`、実装済み）
   - TOPIX、各時点の上場銘柄情報、決算予定・実績開示、上場/上場廃止・コード変更など、
     株価とカレンダー以外の原データを変更せずスナップショット保存する。
   - 現在上場銘柄だけで過去を再構成せず、取得日時・公表日時・対象日を区別する。
2. 時点整合ユニバース（`universe.py`、実装済み）
   - 各シグナル日時点の東証上場内国普通株式、上場後60取引日、流動性、価格形成、
     監理・整理指定等を判定する。判定理由を列として残し、除外件数を報告する。
3. 決算イベント判定（`earnings_events.py`、実装済み）
   - 保有予定期間中の決算跨ぎ除外と、実際の東証取引カレンダーに基づく決算後25取引日の
     クールダウンを作る。予定日と実績開示日、利用可能時点を混同しない。
4. 週次学習表（`weekly_dataset.py`、実装済み）
   - 祝日を含む各週のシグナル日・評価日、最低取引日数、調整済み終値間の週次目的変数、
     TOPIX超過リターンを作り、特徴量・ユニバース・決算除外を `(signal_date, code)` で結合する。
   - この段階で初めて学習候補表を `data/processed/` に保存する。
5. 時系列分割（`splits.py`、実装済み）
   - 同じシグナル日の全銘柄を同じfoldへ入れ、2016〜2024年の拡大型ウォークフォワードと
     ラベル期間以上のパージを生成する。2025〜2026年を最終評価専用として隔離する。
6. 基準モデルと学習（`modeling.py`、実装済み）
   - 定数、単純モメンタム、線形最小二乗を実装し、補完器と標準化器を各学習foldだけでfitする。
     Ridge/Elastic Net、外れ値境界、特徴量選択は後続の開発期間内比較として未実装である。
7. `evaluation.py`（実装済み）
   - `modeling.py` のMAE/RMSE・方向一致率を補完し、日付単位のSpearman/Pearson IC、上位N、
     上下位差、coverage、年別・業種別安定性を同じ検証予測で評価する。
8. `backtest.py`
   - 次取引日始値以降の約定、決済価格設定、手数料・スプレッド・スリッページ、売買単位、
     参加率、未約定、値幅制限、停止、上場廃止、ターンオーバーを扱う。
   - 主結果の決済価格を確定する前にユーザー確認が必要。
9. 実験統括と報告（`experiment.py` と `reporting.py`、小規模なら `experiment.py` に統合）
   - 上記工程を同じ設定と入力識別子で再実行するCLI、実験manifest、基準比較、感度分析、
     失敗実験を含むレポートを作る。最終期間を見た後の仕様変更を記録する。

1〜7には対応する `configs/*.json` と `tests/test_*.py` を追加済みである。ただし7の時価総額別評価は、
週次評価入力へ時点時価総額を結合する契約が未確定のため未実装である。次の主要実装責務は8の
バックテストだが、その前に補助実データ、公式カレンダー確認済み前処理、特徴量、
週次ユニバース、決算判定を同じ入力スナップショットから生成し、結合coverage、評価日欠損、
実fold件数を監査する。ユニバースや決算除外を省略した表を最終学習データとして扱わない。

## 日次テクニカル特徴量（2026-10-11）

### 目的と対象

`src/stock_prediction/descriptors.py` に、README 7節で確定している11特徴量を生成する
共通関数APIとCLIを実装する。入力は `preprocessing.py` が生成した監査済みの日次観測行と、
J-Quants V2の取引カレンダーである。特徴量工程では銘柄ユニバース、週次シグナル日、目的変数、
決算除外、横断標準化、欠損補完を行わず、後続工程から独立して検証可能にする。

### 実装判断

- 価格モメンタム、移動平均、価格変動率、レンジ位置には提供元の調整済み価格を使う。
  出来高比には分割・併合の数量変化を調整した `adjusted_volume`、売買代金比には未調整の
  `turnover_value`（円）を使う。価格や数量は前方補完しない。
- 窓の単位は東証取引セッションとする。入力行の個数から営業日を推測せず、明示された
  `Date` / `HolDiv` カレンダーのセッション通番で、lagとrolling窓の連続性を検査する。
  `calendar_verified=False` の旧中間データは特徴量入力として受理しない。
- `R_5` / `R_20` / `R_60` は調整済み終値の単純リターン。`Volatility_5To20` と
  `Volatility_20` は日次対数リターンの標本標準偏差（`ddof=1`）を使う。各rolling集計は
  窓と同数の有限な観測を必須とし、履歴不足や窓内欠損を短い窓で代替しない。
- 分母が0または非有限、`High_20 == Low_20`、現在行に利用可能な価格がない場合は欠損とする。
  外れ値処理、欠損補完、標準化は学習fold内で行う後続工程の責務とし、本工程では行わない。
- TOPIX入力はこのリポジトリにまだ存在しないため、正規化済み `date` / `close` の任意入力とする。
  指定時はカレンダー整合性と主キーを検査し、`R_TOPIX_20` を同じ20セッションで計算する。
  未指定時は安定した列構成を保つため `RelativeTOPIX_20` を全欠損とし、manifest/reportへ
  `topix_supplied=false` を記録する。`require_topix=true` なら未指定を早期エラーにする。
- DataFrameを受け取る `compute_descriptors` をNotebook用の中核APIとし、CLIも同じ関数を使う。
  CLIは年別Parquetを順に処理し、銘柄ごとに最大60セッションの履歴を年境界へ引き継ぐ。
  全年度を無条件に一括読込せず、既存出力を上書きしない。
- 出力は `(date, code)` と11特徴量だけを年別Parquetへ保存する。行数、銘柄数、日付範囲、
  重複数、特徴量別欠損率、入力/出力SHA256、設定、ライブラリ版、コードSHA256、進捗状態を
  manifestへ保存する。これはまだ学習可能ユニバースや週次サンプルを表さない。

### 進捗

- [x] README.md / AGENTS.md / データ辞書 / 前処理API / 実データschema・期間の確認
- [x] 特徴量の利用可能時点、列、窓、欠損規則、TOPIX未取得時の扱いを決定
- [x] `descriptors.py`、設定、CLI、Notebook利用手順の実装
- [x] 銘柄境界、年境界、営業日連続性、既知入力、欠損、ゼロ除算、未来不変性のテスト
- [x] 関連テスト、CLIヘルプ、最小実データ例の検証
- [x] 検証結果、既知の制限、未解決事項の記録

### 実装内容

- `src/stock_prediction/descriptors.py`：`DescriptorConfig` / `DescriptorResult` /
  `compute_descriptors` / `run_descriptors` / `main`。DataFrame APIは入力を変更せず、CLIは
  年別処理、最大60セッションの履歴引継ぎ、既存出力保護、中断・失敗manifestを実装した。
- `configs/descriptors.json`：標準偏差の `ddof=1` とTOPIX任意入力を明示した。
- `tests/test_descriptors.py`：README数式の既知値、銘柄境界、価格なし行、欠落行、休日、
  定数/ゼロ系列、年境界、未来不変性、TOPIX任意/必須、schema、設定、CLI保存を検証した。
- `pyproject.toml`：`stock-descriptors` エントリーポイントを追加した。新しい依存は追加していない。
- README / `docs/DESCRIPTORS.md` / `docs/DATA_DICTIONARY.md`：CLI、Notebook共通API、
  入出力列、数式、単位、必要観測数、欠損規則、来歴、制限を同期した。

### 検証結果（2026-10-11）

- 新規テスト：
  `python -m pytest -q tests/test_descriptors.py --basetemp=.test_tmp_descriptors -p no:cacheprovider`
  は **13 passed（1.99秒）**。
- 全テスト：`python -m pytest -q --basetemp=.test_tmp_final -p no:cacheprovider` は
  **110 passed（16.83秒、最終再実行）**。取得・前処理の既存97件を含む。
- `python -m stock_prediction.descriptors --help`：必須の年別中間データ/カレンダー、任意TOPIX、
  新規出力、設定上書きの引数を確認した。
- 実データ最小例：取得済みスナップショットの銘柄 `99970`、2016-10-11〜2017-12-29、
  **303行**を公式カレンダーで前処理し、受理303行、隔離0行。`R_60` とTOPIX以外の
  全特徴量が利用可能な行は **243行**で、先頭60セッションを履歴不足として欠損にした。
  TOPIX未指定のため `RelativeTOPIX_20` は303行すべて欠損である。原本・既存中間データ・
  最終バックテスト結果は変更または参照していない。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、pytest 9.1.1。
  `pyproject.toml` の許容範囲すべてを検証したものではない。

### 現時点の未解決事項

- TOPIX日次データは未取得であり、`RelativeTOPIX_20` の実データ値は本作業だけでは検証できない。
- 歴史的な内国普通株式ユニバース、決算予定・実績、週次シグナル日と目的変数は別工程である。
- 提供元の調整済み価格は取得時点より後のコーポレートアクションで遡及改定され得る。
  入力スナップショットは識別するが、各過去日に実際に取得可能だった絶対水準とは認定しない。

## J-Quants API V2 原データ取得（2026-10-08）

### 目的と対象

APIキーを実行時に受け取り、指定期間の株価日足と取引カレンダーを
`data/raw/` 配下の新しいスナップショットへ保存する。CLI と Notebook は
`src/stock_prediction/` の同じ取得関数を利用する。既存原本は変更・上書きしない。

### 確認済み仕様と判断

- 現行の個人向け J-Quants API V2 を使用する。認証は `x-api-key`、ベースURLは
  `https://api.jquants.com/v2`。
- 株価は `/equities/bars/daily` を東証営業日ごとに取得し、カレンダーは
  `/markets/calendar` を期間指定で取得する。全ページを `pagination_key` で追跡する。
- 取引カレンダーは閉場日を含む全暦日を保存し、`HolDiv` の `1` / `2` の日だけを
  東証株価の取得対象とする。株価観測日から営業日を推定しない。
- 契約プランにより取得可能期間と毎分リクエスト数が異なるため、開始日・終了日は
  呼出側で必須指定し、保守的な既定値5回/分を設定可能にする。
- APIキーは引数、`JQUANTS_API_KEY`、CLIの非表示入力のいずれかで受け取る。
  コマンドライン引数、設定ファイル、Notebook、ログ、manifestへ保存しない。
- 実行ごとに一意なディレクトリを作り、カレンダー、年別株価Parquet、取得条件、
  行数、銘柄数、日付範囲、欠損率、重複数、ファイルSHA256をmanifestへ記録する。
- APIスキーマはV2の必須16列、公式の `MktCap` / `ExRT`、既知の前場・後場列だけを許容し、
  未知列、必須列欠落、主キー重複、要求日以外の行を早期エラーにする。`MktCap` は公式に
  削除予定のため任意列とし、`ExRT` も既知の任意列として扱う。原値の補完・補正は行わない。
- 途中失敗または `KeyboardInterrupt` 時は取得済みの当年行を `partial_prices/` に保存し、
  完了年、完了日、例外種別、状態をmanifestに残す。同じ出力先への自動再開や上書きはしない。

### 進捗

- [x] README.md / AGENTS.md / 既存データ・依存・テストの確認
- [x] J-Quants公式V2仕様（認証、エンドポイント、ページング、カレンダー区分）の確認
- [x] 共通クライアント、保存処理、CLIの実装
- [x] Notebook例、README、データ辞書、依存設定の更新
- [x] 合成レスポンスによる単体・統合テストとCLIヘルプ確認

### 未確定事項

- 修正後の全期間実API取得はユーザーのAPIキーと長時間の通信を必要とするため、コード側では
  秘密情報なしで再現可能な合成レスポンステストまでを必須検証とする。

### 実装内容

- `src/stock_prediction/data_download.py`：V2 HTTPクライアント、全ページ取得、リクエスト間隔、
  429/5xx再試行、カレンダー/株価スキーマ検査、年別保存、中断時部分保存、manifest、CLI。
- `notebooks/00_download_jquants_raw.ipynb`：`getpass` と共通関数を使うNotebook例。
- `tests/test_data_download.py`：ページング、レート制限、認証情報秘匿、スキーマ、保存、
  中断、既存出力保護、閉場日のみの期間、前処理接続を合成データで検証。
- `src/stock_prediction/preprocessing.py`：`MktCap` / `ExRT` とPremium等で付加される既知の
  前場・後場列を型・値検査して許容し、従来の基礎16列だけを中間データへ使う。
  未知列は引き続き拒否する。
- `pyproject.toml`：HTTPS用の最小依存 `requests>=2.31,<3` と `stock-download` を追加。
- README / DATA_DICTIONARY：秘密情報、出力、再実行、スキーマ、CLI/Notebook手順を同期。

### 検証結果（2026-10-08）

- `python -m pytest -q --basetemp=.test_tmp -p no:cacheprovider`：**89 passed（12.02秒）**。
  新規取得テスト17件と、既存前処理に追加したセッション列互換テスト2件を含む。
- 合成統合例：2024-12-30〜2025-01-03の全暦日5行、東証営業日3日、株価6行、
  2年分を保存。既知の前場列を原本に保持したまま、2025年分4行を後続前処理で受理した。
- 中断例：営業日2日のうち1日（2銘柄）完了後に `KeyboardInterrupt` を発生させ、
  `partial_prices/2025.parquet` の2行と中断状態・最終完了日をmanifestへ保存した。
- `python -m stock_prediction.data_download --help`：必須日付、保存先、頻度、再試行引数を確認。
- Notebook：JSON読込と全コードセルの構文コンパイル成功（6セル、出力・秘密情報なし）。
- 検証環境：Python 3.11.9、pandas 3.0.6、numpy 2.4.6、pyarrow 25.0.1、
  requests 2.34.2、pytest 9.1.1。他の許容バージョンすべての検証ではない。
- 初回実装時点では実API呼出しを行わず、既存 `data/raw/` は変更していない。

### `MktCap` / `ExRT` 追加列対応（2026-10-08）

- ユーザーの実API取得 `2016-10-08`〜`2026-10-07` で、カレンダー3,652行と
  取引予定日2,440日を確認した後、最初の株価応答に既存許可リスト外の `MktCap` / `ExRT` が
  あったため安全に停止した。完了取引日・株価行・部分価格ファイルはいずれも0で、既存の
  失敗スナップショットは変更していない。
- 公式V2仕様を再確認し、`MktCap` は時価総額（百万円）の数値またはNull、`ExRT` は
  権利落種類 `"1"` / `"2"` / `"3"` またはNullとして検査する。`MktCap` は日次株価APIから
  削除予定のため、両列がある現行応答と列がない応答の双方を受理する既知の任意列とした。
- 原本ダウンロードでは両列を仕様順で保持する。前処理ではParquet型と `ExRT` の値を検査するが、
  基礎16列だけを中間データへ写し、`MktCap` / `ExRT` を暗黙に特徴量へ追加しない。
- 追加テストは、両列の保持、全Null、列なし、型不正、`ExRT` 値不正、年別Parquet保存、
  後続前処理のDataFrame/Parquet入力を対象とする。
- 対象テスト：`tests/test_data_download.py tests/test_preprocessing.py` は
  **97 passed（11.85秒）**。
- 全テスト：`python -m pytest -q --basetemp=.test_tmp -p no:cacheprovider` は
  **97 passed（11.55秒）**。
- 修正後の実API再取得はAPIキーを使用し長時間実行となるため未実施。新規スナップショットで
  再実行し、manifestの完了日数・行数・列・欠損率を確認する必要がある。

### 残る制限

- 契約プラン外の期間はAPIの403または営業日の空レスポンスとして停止する。期間を縮めて
  新規スナップショットで再実行する。空レスポンスを休場日や欠損として推測しない。
- 通常のPython例外と `KeyboardInterrupt` は部分保存するが、プロセス強制終了、OS停止、
  電源断ではメモリ上の当年データを保存できない場合がある。
- 同一出力への再開・追記・上書きは原本不変性と訂正混在防止のため実装しない。

## 目的と範囲

`data/raw/YYYY.parquet` の日次株価を監査・整形し、再生成可能な中間データを
`data/interim/` に保存する。CLI と Notebook が同じ Python 関数を使用する。
原本は読み取り専用とし、特徴量・週次ラベル・ユニバース・学習・売買は後段の独立した工程とする。

## 確認済み事項（2026-10-04）

- README.md / AGENTS.md を読了。既存コードとテストはない。
- pyproject.toml / environment.yml / .gitignore / データ辞書 / 既存計画書は空。
- 年別 Parquet は2016～2026年の11本、ZIPも存在。ZIPは別スナップショットのため自動混合しない。
- 実スキーマを2016年のParquetで確認：Date timestamp[ns], Code string,
  O/H/L/C/Vo/Va/AdjFactor/AdjO/AdjH/AdjL/AdjC/AdjVo double, UL/LL string。
- 取引カレンダーは見当たらず、ユーザーへ取扱いを相談済み。回答未取得のため、
  実行確認は明示的な未検証モードに限定し、通常モードはカレンダー必須とした。
- 通常PATHにPythonはない。利用可能なPythonで不足するpyarrow/pytestをプロジェクト内
  `.runtime_dependencies/` に導入。これは本番依存設定とは別の検証用環境。

## 方針

1. 全年のスキーマ・実日付範囲・行数・銘柄数・欠損・異常を独立監査する。
2. 列名・型を正規化し、Date/Code主キー、OHLC関係、有限性・符号を検査する。
3. 不正行は理由付きで隔離し、重複はどちらかを恣意的に採用せず全件隔離する。
   無約定/停止を区別できない全OHLCV欠損行は保持し、補完しない。
4. 調整済み系列・未調整系列を別々に保持し、調整係数が1以外の日を明示する。
   提供調整済み価格は後続コーポレートアクションで遡及変更され得るため、
   当時利用可能な絶対価格として認定しない。調整係数から独自再構成しない。
5. カレンダー指定時はV2 Date/HolDivの型・全暦日coverage・営業日を確認する。
   未指定時は明示的な暫定モードだけ許容し、未検証フラグを保存する。
6. 銘柄ごとの直前観測行との調整済み終値リターンを品質監査にのみ使う。
   50%を超える絶対値を固定の監査閾値で記録し、補正・除外・最適化はしない。
   欠損を跨いだ補完はしない。年境界では前処理済み直前行を引き継ぐ。
7. 年別に必要列・指定期間だけ読み込み、入力SHA256、設定、コード識別子、
   ライブラリ版、実行日時、出力件数と欠損率を保存する。
8. テストと実データの最小CLI例で検証する。中断時も完成済みファイルのmanifestを残す。

## 進捗

- [x] リポジトリ仕様・実装状況・検証環境の確認
- [x] 全年原データ監査
- [x] Notebook共通API / CLI / 出力保護 / 品質レポート
- [x] 依存設定・データ辞書・READMEの使用例
- [x] 単体テスト / CLI・実データ検証
- [x] 結果・未解決事項の最終記録

## 実装したファイルとAPI

- `src/stock_prediction/preprocessing.py`：
  `PreprocessingConfig` / `PreprocessingResult` / `load_config` /
  `validate_trading_calendar` / `preprocess_prices` / `run_preprocessing` / `main`。
  DataFrame処理APIは入力を変更せず、保存APIは年別で処理する。
- `src/stock_prediction/__init__.py`：src構成のパッケージ。
- `configs/preprocessing.json`：ISO日付範囲、固定監査閾値、未検証モード許可。
  未検証許可の既定値はfalse。JSONの未知キー・不正な値は拒否。
- `notebooks/01_preprocessing.ipynb`：実データを期間・列選択して共通APIへ渡す実行例。
- `tests/test_preprocessing.py`：合成データで境界・異常・入力保護・CLIを検証。
- `pyproject.toml` / `environment.yml`：必要最小限のpandas/numpy/pyarrow、開発用pytest、
  editable install、`stock-preprocess` エントリーポイント。
- `.gitignore`：原本・派生物・検証用依存・キャッシュ・秘密情報をGit対象外にする。
- README.md / DATA_DICTIONARY.md / IMPLEMENTATION_PLAN.md：使用方法・制約と本記録へのリンク。

### 検査と保存の詳細

- 当初実査したV2原本は16列。2026-10-08の取得実装追加後は、必須16列に加えて
  公式仕様の `MktCap` / `ExRT` と既知の前場・後場列だけを型・値検査して許容し、
  未知列・不足列を早期拒否する。
  銘柄コードは5文字の大文字英数字として保持し、普通株式以外もこの工程では残す。
- 無効な主キー、重複全件、OHLC関係違反、非有限値、非正価格、不正調整係数、
  負数量、部分OHLCV欠損、既知閉場日の行を理由付きで隔離する。
- 全OHLCV欠損は保持し、`has_price=False` とする。数量ゼロは非負性を満たすため保持し件数監査。
- `is_adjustment_event` / `is_large_return` は監査用。分割・併合や異常リターンを自動補正しない。
  リターン演算のオーバーフローは欠損にし、`is_non_finite_return` と件数を残す。
- 監査リターンの比較元は、処理対象の同一銘柄の直前観測行。欠損・不正・重複の日を
  有効価格で埋めたり飛び越えたりしない。年末の不正/欠損行も次年への履歴に残す。
  銘柄の観測が丸1年ない場合も前回観測日を保持し、Notebook一括処理と一致させる。
- カレンダーの全暦日coverage、Date/HolDivスキーマ、営業日コードを検査する。
  カレンダーにあるのに全銘柄の行がない営業日もレポートに残す。
  個別銘柄の行欠落を上場・廃止・停止とは断定しない。
- 出力先の既存ディレクトリ、raw配下、入力の祖先、出力間の重なりを拒否する。
  解決済みパスで入力保護を確認する。
- 各年の株価/隔離Parquet、日付別/銘柄別欠損率Parquet、JSON監査結果、manifestを保存。
  manifestを年の完了ごとに更新し、失敗/KeyboardInterrupt時は完了年と状態を残す。
  同一出力先への自動再開は行わず、新しい出力先で再生成する。
- 年別ファイルは実日付がファイル年と一致することを事前検査する。
  この契約により年を跨ぐ主キー重複を防ぐ。日付無効や年不一致はファイル単位で停止する。

## 検証結果（2026-10-04）

### 単体・統合テスト

`python -m pytest -q` 相当を利用可能なPythonで実行し、**70 passed（4.72秒）**。
Python 3.12.14、pandas 3.0.1、numpy 2.3.5、pyarrow 25.0.1、pytest 9.1.1。
他の許容ライブラリ版すべての検証は行っていない。

対象は必須列/型、不正値/主キー全件隔離、欠損未補完、銘柄境界/年境界/未観測年、
未来変更で過去の行が変わらないこと、休日/営業日/coverage、Arrow date32、
CLIとNotebook関数の一致、raw SHA256不変、出力保護、失敗・中断時のmanifest保持、
演算オーバーフロー検出。週次ラベル・特徴量・決算除外・モデル・売買のテストは
それらの実装範囲に入っていないため今回の対象外。

### 最小実データ例とNotebook

- 対象：2016-04-08〜2016-04-15、**22,824行**。
- 出力22,824行、隔離0行、全OHLCV欠損1,164行、大リターンフラグ0行。
- CLI生成物：`data/interim/preprocessing_smoke_20160408_20160415/`。
- レポート：`reports/preprocessing_smoke_20160408_20160415/`。
- Notebookの5コードセルを同じ環境で実行し、CLI Parquetと全列・全行一致を確認。

### 全原本の監査と全期間処理

年別11ファイルの実日付範囲は **2016-04-08〜2026-04-08**。
**10,058,123行、5,344銘柄**。品質検査と固定ルールの整形だけを実行し、
2025年以降をモデル/特徴量/閾値選択には利用していない。

| 年 | 入力/出力行数 | 銘柄数 | 全OHLCV欠損行 | 調整イベント行 |
|---|---:|---:|---:|---:|
| 2016 | 683,149 | 3,893 | 28,719 | 239 |
| 2017 | 953,757 | 3,966 | 26,513 | 646 |
| 2018 | 964,473 | 4,052 | 29,624 | 373 |
| 2019 | 963,118 | 4,088 | 35,368 | 156 |
| 2020 | 981,070 | 4,154 | 31,695 | 128 |
| 2021 | 1,008,911 | 4,276 | 27,742 | 159 |
| 2022 | 1,023,708 | 4,332 | 34,973 | 103 |
| 2023 | 1,052,339 | 4,422 | 33,902 | 162 |
| 2024 | 1,070,937 | 4,522 | 39,965 | 235 |
| 2025 | 1,072,501 | 4,581 | 46,954 | 265 |
| 2026 | 284,160 | 4,495 | 12,792 | 94 |

- 全年スキーマ一致。主キー重複、負値、非有限値、ゼロ価格、OHLC不整合、
  部分OHLCV欠損、不正調整係数はいずれも0件。隔離行は0件。
- 全OHLCV欠損 **348,247行（3.4623%）**：補完・除外せず保持。
- 調整イベント **2,560行**：AdjFactorの範囲0.002〜1783、機械的な外れ値除去をしない。
- 監査リターンの絶対値50%超 **266行**：フラグを付けて保持。演算非有限0行。
  これは予測性能や投資成績の評価ではない。
- 英字銘柄コードも保持。UL/LLはすべて文字列0/1。
- 全期間処理完了：`data/interim/preprocessing_20261004/` と
  `reports/preprocessing_20261004/manifest.json`。開始05:30:51 UTC、終了05:31:37 UTC。
- 実行後、11原本すべてのSHA256が実行前と一致。manifestのコードSHA256も最終ソースと一致。
- 独立の初期監査資料：`reports/preprocessing_initial_audit/audit.json`。
  ZIP内月別株価は2016-04-05〜2026-05-25で列名も異なり、年別と同一スナップショットとは
  認定できない。今回の保存APIは年別原本だけを扱い、ZIPは使用しない。

## 未解決事項

- 取引カレンダー欠如：ユーザーへ相談済み、回答待ち。今回の実データ出力は全行
  `calendar_verified=False`。日付の営業日認定、週次日程、経過取引日数は未確定。
- 歴史的銘柄マスター・決算予定の利用可能日時・TOPIX等の結合は別工程。
- API取得日時/API版を原本から確定できなければ不明として記録する。
- 上場廃止・コード変更・合併・売買停止の理由は株価だけで断定しない。
- 提供調整済み絶対価格のPIT整合性は未認定。今回の出力は学習可能データではない。
- リポジトリはGit初期化されていないためGit版はnull。ソースSHA256で今回のコードを識別する。

## 再実行手順

通常のPython環境ではREADMEの `python -m pip install -e ".[dev]"` 後、次を実行する。
出力ディレクトリは実行ごとに新しく生成される。

```powershell
python -m stock_prediction.preprocessing --config configs/preprocessing.json --allow-unverified-calendar
python -m pytest -q
```

今回の環境では通常PATHにPythonがなかったため、利用可能なPythonへ
`PYTHONPATH=.runtime_dependencies;src` を設定して上記モジュールを実行した。
機械固有の実行ファイルパスはソース・設定・Notebookには埋め込まない。

## 決算予定の同日複数状態対応（2026-10-21、実装済み）

### 発見事項と採用方針

- 実スナップショットの決算予定154,546行を監査し、2021年に同じ
  `Code` / `FQName` / `FYE` / `PubDate` を持つ4行・2グループを確認した。
  各グループは既知の `SchDate` 1行と「予定日未定」を表すNull 1行から成る。
- `PubDate` は日付粒度で同日内の公表順を持たないため、非Null行またはNull行を恣意的に
  最新として選ばない。同じ公表日の行を、その時点で公表された候補状態の集合として扱う。
- 同日集合にNullがあれば、有効期間中の候補へ `active_schedule_date_missing=True` を付ける。
  既知の候補日のいずれかが保有期間へ入れば除外するが、同じ
  `Code` / `FQName` / `FYE` の決算イベントは候補日が複数あっても1件として数える。
- 次の異なる `PubDate` からは、その新しい公表日の集合だけを有効にする。未来の予定変更を
  過去へ適用しない既存の時点整合規則は維持する。
- 原本は変更しない。同日複数状態のグループ数・行数・Nullを含むグループ数を監査レポートへ残す。

### 進捗

- [x] README、既存実装・テスト・データschema、実データの該当行と影響範囲を確認
- [x] 同日集合の判定ロジックと監査項目を実装
- [x] 単体テスト、関連文書を更新
- [x] 関連テストと全テストを実行
- [x] 実スナップショットの該当銘柄・期間で同日集合の判定を検証
- [ ] 2016〜2026年の年別処理を完了（下記の独立した入力履歴不足で停止）

### 実装と検証結果

- `earnings-exclusions-v2` とし、同日の既知候補日を集合として評価する。複数候補日が同じ
  保有期間へ入っても `scheduled_earnings_count` は同一決算イベントにつき1だけ加算し、
  `first_scheduled_earnings_date` は最初の候補日とする。入力行順を逆にしても出力は不変である。
- 監査レポートとCLI manifestへ `same_day_schedule_state_groups`、
  `same_day_schedule_state_rows`、`same_day_schedule_state_groups_with_missing_date` を追加した。
- 決算イベント単体テストは **14 passed（2.65秒）**。全テストは
  **233 passed（52.53秒）**。既存 `.pytest_cache` への書込み権限警告が1件あったが、
  テスト用一時領域と全テスト本体は成功した。
- 実スナップショット全154,546予定行と、該当2銘柄の2021〜2022年候補198行を共通APIへ渡し、
  同日複数状態 **2グループ・4行**、うちNullを含むもの **2グループ**として監査できた。
  `active_schedule_date_missing=True` は88候補行。`79840` の2022-02-14週は候補日
  2022-02-15を保有期間内として除外し、同時に日付未定状態も監査した。

### 独立して判明した入力履歴不足

全期間CLIを新規出力先 `data/processed/earnings_events_manual_20261021_115544` /
`reports/earnings_events_manual_20261021_115544` で再実行したところ、同日複数状態のschema検査は
通過した。その後、最初の候補日2016-11-07より前の公式カレンダー取引日が3日しかなく、
25取引日のクールダウン履歴を保証できないため、既存の早期検査で停止した。2016年候補30,562行は
元ユニバースではすべて `is_eligible=False` だが、決算列だけを便宜的に補完せず、失敗manifestを
保持した。全期間出力を確定するには、2016-11-07より前の取引カレンダーと決算実績履歴を追加取得
するか、履歴不足候補の明示的な扱いを別途決定する必要がある。

## 取得可能期間に合わせた分析開始日の固定（2026-10-21、実装済み）

### ユーザー確認済み方針

- Standardプランの10年制約と現スナップショットの2016-11-01開始を前提に、2015年データの
  追加取得には依存しない。
- `R_60` と上場履歴60取引日のウォームアップを優先し、最初の分析シグナル日を
  **2017-02-06** とする。2016-11-01〜2017-02-03は特徴量・上場履歴・決算クールダウンの
  ウォームアップ専用で、週次候補、学習、評価へ含めない。
- 2025〜2026年の最終期間隔離と、2020〜2024年の開発検証foldは変更しない。最初のfoldの
  実学習サンプルは2017-02-06〜2019年となる。
- 手作業で年別Parquetを削らず、週次日程設定に開始日を保存し、ユニバース、決算、週次表へ
  同じ候補範囲を伝播させる。

### 進捗

- [x] 現カレンダー上の60取引日境界と実特徴量・ユニバースの最初の利用可能日を確認
- [x] 週次日程設定へ分析開始日を追加し、監査理由とmanifestへ保存
- [x] テストとREADME・仕様・実行手順を同期
- [x] 週次日程、ユニバース、決算除外を新規出力へ再生成して検証

### 実装と実データ検証結果

- `WeeklyDatasetConfig.signal_start_date` を追加し、`configs/weekly_dataset.json` で
  `2017-02-06` に固定した。日程reportは479週、最小2017-02-06、最大2026-09-14で、開始日前14週を
  `signal_before_configured_start` として監査した。
- `run_universe` は日程がない入力年を空Parquetとして出さず、`skipped_partitions` に
  `no_configured_signal_dates` として記録する一方、その年の銘柄マスターと株価を後年の履歴計算に残す。
  実出力 `data/processed/weekly_universe_input20161101-20261001_baseline-r001/` は2017〜2026年の
  1,992,542行、適格669,278行。2016年の156,642行は履歴専用で、2016年出力はない。
- 最初のシグナル2017-02-06は3,825候補、ユニバース適格1,234候補。事前監査では、この1,234候補の
  `R_60` がすべて非Nullであることを確認した。
- 決算除外の初回完了出力 `earnings_events_manual_20261021_approved_start` は、実入力の `DocType` が
  `3QFinancialStatements_...` のように四半期接頭辞を持つのに、旧設定が
  `FinancialStatements` 始まりを期待していたため、実績決算イベントとクールダウンが0行だった。
  この出力は後段で使用しない。実データで観測した5接頭辞へ修正し、未知の
  `FinancialStatements` 系を早期エラーにする検査を追加した。
- 修正版 `earnings-exclusions-v3` の実出力
  `data/processed/earnings_events_input20161101-20261001_baseline-r001/` は2017〜2026年の
  1,992,542行を完了した。予定跨ぎ141,142行、25取引日クールダウン782,221行、
  保有期間中の実績開示（事後監査）130,791行、決算条件合格1,083,459行。同日複数状態は
  2グループ・4行で、両グループにNull予定日を含むことをmanifestへ保存した。
- 関連テスト `test_earnings_events.py` / `test_experiment.py` / `test_universe.py` /
  `test_weekly_dataset.py` は **55 passed**。最終の全テストは
  `python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_all_approved_start` で
  **236 passed（47.18秒）**。
## 成果物ディレクトリ名の正規化（2026-10-24、実施済み）

### 方針

- `manual_YYYYMMDD_HHMMSS` は実際の実行日時と一致していないため、現行の入力系列を
  `input20161101-20261001_baseline-r001` で統一する。
- この日付は出力期間ではなく、固定したraw入力スナップショットのcoverageを表す。
- rawは価格と補助データを区別し、同期間の再取得を区別できるよう `snapshot01` を付ける。
- 現行成果物から参照されない旧版・失敗版・重複版は移行対象にしない。事前検査時点で既に存在しない
  data配下の旧成果物は再作成せず、残存するreportsだけを監査記録として保持する。
- `manifest.json` という標準ファイル名は維持し、保存済みの入出力パスだけを新しい場所へ同期する。
- Parquet等のデータ本体は変更せず、移行前後のSHA256一致を検証する。

### 進捗

- [x] README、実行手順、全dataディレクトリ、manifest依存関係を確認
- [x] 現行modeling／evaluation／backtestへつながる入力系列を特定
- [x] 移行対象の絶対パスと移行先の未使用を検査
- [x] ディレクトリ・ファイル名を変更
- [x] manifest、実行手順、保存済み参照パスを同期
- [x] データ本体のSHA256、旧パス残存、主要manifestの読込みを検証

### 実施結果

- 現行系列のdata、artifacts、reports計22パスを移動した。rawは
  `jquants_v2_{prices,auxiliary}_20161101-20261001_snapshot01`、派生系列は
  `input20161101-20261001_baseline-r001` に統一した。
- 移動前後で対象内の全ファイルSHA256が一致した。保存済み参照は101ファイル、以前から残っていた
  `snapshot01` なしのraw別名はSHA256を照合した163参照を確認後、58ファイルで正式名へ統合した。
- repository所有JSON 170ファイルを再読込みし、現行10工程manifestの参照欠落0件、
  `completed_outputs` のSHA256不一致0件を確認した。
- 主要Parquetを新パスから読み、週次2024年218,522行、予測501,657行、評価summary 3行、
  backtest summary 1行を確認した。backtestの既存状態 `incomplete_open_positions` は変更していない。
- 関連テスト `tests/test_experiment.py` は7件すべて成功した。
- 完全な旧名・新名対応は
  `reports/path_migrations/20261024_active_lineage_naming.json` に保存した。

