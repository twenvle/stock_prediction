from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_prediction import experiment
from stock_prediction.auxiliary_data_download import (
    EARNINGS_SCHEDULE_COLUMNS,
    FINANCIAL_RESULTS_COLUMNS,
)


STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "preprocessing": {
        "start_date": None,
        "end_date": None,
        "abnormal_return_threshold": 0.5,
        "allow_unverified_calendar": False,
    },
    "descriptors": {"volatility_ddof": 1, "require_topix": True},
    "universe": {
        "domestic_common_product_category": "011",
        "min_listing_sessions_elapsed": 60,
        "liquidity_window_sessions": 20,
        "min_price_formation_days": 15,
        "min_median_turnover_yen": 100000000.0,
        "planned_investment_yen": 1000000.0,
        "max_participation_rate": 0.01,
    },
    "earnings_events": {
        "cooldown_trading_sessions": 25,
        "earnings_doc_type_prefixes": [
            "1QFinancialStatements",
            "2QFinancialStatements",
            "3QFinancialStatements",
            "FYFinancialStatements",
            "OtherPeriodFinancialStatements",
        ],
    },
    "weekly_dataset": {
        "min_trading_sessions_per_week": 4,
        "require_topix": True,
        "signal_start_date": "2017-02-06",
    },
    "splits": {
        "isolation_start_date": "2025-01-01",
        "isolation_end_date": "2026-12-31",
        "purge_trading_sessions": 4,
    },
    "modeling": {
        "target_column": "target_return",
        "feature_columns": [
            "R_5",
            "R_20",
            "R_60",
            "PriceToSMA_20",
            "SMA_5To20",
            "Volatility_5To20",
            "Volatility_20",
            "RangePosition_20",
            "VolumeRatio_5To20",
            "TurnoverRatio_5To20",
            "RelativeTOPIX_20",
        ],
        "momentum_feature": "R_20",
        "top_n": 10,
    },
    "evaluation": {
        "top_n": 10,
        "industry_date_column": "Date",
        "industry_security_column": "Code",
        "industry_code_column": "S33",
        "industry_name_column": "S33Nm",
    },
    "backtest": {
        "model_name": "linear",
        "target_column": "target_return",
        "eligibility_column": "is_signal_eligible",
        "liquidity_column": "median_turnover_value_20",
        "lot_size_column": "lot_size",
        "initial_cash_yen": 10000000.0,
        "top_n": 10,
        "default_lot_size": 100,
        "max_participation_rate": 0.01,
        "commission_bps": 5.0,
        "half_spread_bps": 5.0,
        "slippage_bps": 10.0,
        "exit_price_mode": "evaluation_close",
        "max_exit_wait_sessions": 20,
        "unresolved_delisting_policy": "error",
        "annualization_sessions": 252,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    prices = root / "data" / "raw" / "price_snapshot"
    auxiliary = root / "data" / "raw" / "aux_snapshot"
    (prices / "prices").mkdir(parents=True)
    for name in ("listed_info", "earnings_schedule", "financial_results"):
        (auxiliary / name).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": ["2024-01-04"], "HolDiv": ["1"]}).to_parquet(
        prices / "trading_calendar.parquet", index=False
    )
    pd.DataFrame({"Date": ["2024-01-04"]}).to_parquet(
        prices / "prices" / "2024.parquet", index=False
    )
    pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "O": [100.0, 101.0],
            "H": [102.0, 103.0],
            "L": [99.0, 100.0],
            "C": [101.0, 102.0],
        }
    ).to_parquet(auxiliary / "topix.parquet", index=False)
    for name in ("listed_info", "earnings_schedule", "financial_results"):
        pd.DataFrame({"placeholder": [1]}).to_parquet(auxiliary / name / "2024.parquet", index=False)
    declared_hash = "a" * 64
    _write_json(
        prices / "manifest.json",
        {
            "status": "completed",
            "snapshot_id": "price-001",
            "provider": "J-Quants",
            "api_version": "v2",
            "request": {"start_date": "2016-01-01", "end_date": "2026-12-31"},
            "totals": {"price_rows": 1},
            "completed_partitions": [{"sha256": declared_hash}],
        },
    )
    _write_json(
        auxiliary / "manifest.json",
        {
            "status": "completed",
            "snapshot_id": "aux-001",
            "provider": "J-Quants",
            "api_version": "v2",
            "request": {"start_date": "2016-01-01", "end_date": "2026-12-31"},
            "totals": {"listed_info_rows": 1},
            "completed_outputs": [{"sha256": "b" * 64}],
        },
    )
    config_paths: dict[str, str] = {}
    for name, values in STAGE_CONFIGS.items():
        path = root / "configs" / f"{name}.json"
        _write_json(path, values)
        config_paths[name] = str(path.relative_to(root)).replace("\\", "/")
    config_path = root / "configs" / "experiment.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "experiment_name": "test-development",
            "random_seed": 1729,
            "inputs": {
                "price_snapshot_dir": "data/raw/price_snapshot",
                "auxiliary_snapshot_dir": "data/raw/aux_snapshot",
                "delistings_path": None,
            },
            "stage_configs": config_paths,
        },
    )
    return root, config_path


def _save_stage_manifest(report_dir: Path, status: str = "complete") -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=False)
    value = {"status": status, "output_rows": 2}
    _write_json(report_dir / "manifest.json", value)
    return value


def test_load_config_is_strict_and_plan_is_side_effect_free(tmp_path: Path) -> None:
    root, config_path = _project(tmp_path)
    config = experiment.load_config(config_path)
    plan = experiment.plan_experiment(config, run_id="planned-run", project_root=root)

    assert plan.run_id == "planned-run"
    assert plan.input_identifiers["price_snapshot"]["snapshot_id"] == "price-001"
    assert plan.input_identifiers["price_snapshot"]["declared_sha256_count"] == 1
    assert plan.stage_config_values["modeling"]["values"]["target_column"] == "target_return"
    assert not plan.paths.report_root.exists()
    assert not plan.paths.interim_root.exists()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    _write_json(config_path, payload)
    with pytest.raises(ValueError, match="unknown"):
        experiment.load_config(config_path)


def test_plan_rejects_nonmatching_model_and_backtest_targets(tmp_path: Path) -> None:
    root, config_path = _project(tmp_path)
    backtest_path = root / "configs" / "backtest.json"
    values = json.loads(backtest_path.read_text(encoding="utf-8"))
    values["target_column"] = "target_excess_return"
    _write_json(backtest_path, values)

    with pytest.raises(ValueError, match="target_column"):
        experiment.plan_experiment(
            experiment.load_config(config_path), run_id="mismatch", project_root=root
        )


def test_normalize_topix_rejects_duplicates_and_preserves_input() -> None:
    source = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "C": [100.0, 101.0],
            "O": [99.0, 100.0],
        }
    )
    original = source.copy(deep=True)
    normalized = experiment.normalize_topix(source)
    pd.testing.assert_frame_equal(source, original)
    assert list(normalized.columns) == ["date", "close"]
    assert normalized["close"].tolist() == [100.0, 101.0]

    duplicate = pd.concat([source.iloc[[0]], source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        experiment.normalize_topix(duplicate)


def test_run_experiment_executes_stages_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path = _project(tmp_path)
    calls: list[str] = []

    def stage_runner(name: str):
        def run(**kwargs: Any) -> dict[str, Any]:
            calls.append(name)
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=False)
            if name == "modeling":
                pd.DataFrame({"prediction": [0.1]}).to_parquet(
                    output_dir / "predictions.parquet", index=False
                )
            if name == "evaluation":
                pd.DataFrame(
                    {
                        "model": ["linear"],
                        "target_column": ["target_return"],
                        "signal_dates": [1],
                        "spearman_ic_mean": [0.2],
                        "top_n_mean_target_mean": [0.01],
                        "top_bottom_spread_mean": [0.015],
                    }
                ).to_parquet(output_dir / "summary_metrics.parquet", index=False)
            if name == "backtest":
                pd.DataFrame(
                    {
                        "model": ["linear"],
                        "target_column": ["target_return"],
                        "initial_cash_yen": [10_000_000.0],
                        "ending_equity_yen": [10_100_000.0],
                        "cumulative_return": [0.01],
                        "main_result_complete": [True],
                    }
                ).to_parquet(output_dir / "summary_metrics.parquet", index=False)
            return _save_stage_manifest(Path(kwargs["report_dir"]))

        return run

    def schedule_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append("weekly_schedule")
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"signal_date": pd.to_datetime(["2024-01-04"])}).to_parquet(
            output, index=False
        )
        report = {"included_weeks": 1}
        _write_json(Path(kwargs["report_path"]), report)
        return report

    monkeypatch.setattr(experiment, "run_preprocessing", stage_runner("preprocessing"))
    monkeypatch.setattr(experiment, "run_descriptors", stage_runner("descriptors"))
    monkeypatch.setattr(experiment, "run_weekly_schedule", schedule_runner)
    monkeypatch.setattr(experiment, "run_universe", stage_runner("universe"))
    monkeypatch.setattr(experiment, "run_earnings_events", stage_runner("earnings_events"))
    monkeypatch.setattr(experiment, "run_weekly_dataset", stage_runner("weekly_dataset"))
    monkeypatch.setattr(experiment, "run_splits", stage_runner("splits"))
    monkeypatch.setattr(experiment, "run_modeling", stage_runner("modeling"))
    monkeypatch.setattr(experiment, "run_evaluation", stage_runner("evaluation"))
    monkeypatch.setattr(experiment, "run_backtest", stage_runner("backtest"))

    topix_input = root / "data" / "raw" / "aux_snapshot" / "topix.parquet"
    before = hashlib.sha256(topix_input.read_bytes()).hexdigest()
    result = experiment.run_experiment(
        experiment.load_config(config_path), run_id="full-run", project_root=root
    )
    after = hashlib.sha256(topix_input.read_bytes()).hexdigest()

    assert result.manifest["status"] == "complete"
    assert result.manifest["final_period_evaluated"] is False
    assert result.manifest["final_period_used_for_model_selection"] is False
    assert [stage["name"] for stage in result.manifest["stages"]] == list(experiment.STAGE_ORDER)
    assert calls == [name for name in experiment.STAGE_ORDER if name != "topix_normalization"]
    assert before == after
    assert pd.read_parquet(result.paths.topix).columns.tolist() == ["date", "close"]
    assert result.manifest_path.is_file()
    assert (result.paths.report_root / "config.resolved.json").is_file()
    report = result.report_path.read_text(encoding="utf-8")
    assert "2025〜2026年の隔離期間" in report
    assert "0.2" in report
    assert "1.01e+07" in report


def test_failure_stops_downstream_and_keeps_top_level_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path = _project(tmp_path)

    def fail_preprocessing(**_: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(experiment, "run_preprocessing", fail_preprocessing)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        experiment.run_experiment(
            experiment.load_config(config_path), run_id="failed-run", project_root=root
        )

    report_root = root / "reports" / "experiments" / "failed-run"
    manifest = json.loads((report_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert len(manifest["stages"]) == 1
    assert manifest["stages"][0]["status"] == "failed"
    assert (report_root / "experiment_report.md").is_file()
    assert not (root / "data" / "processed" / "experiments" / "failed-run").exists()


def test_plan_rejects_existing_run_root(tmp_path: Path) -> None:
    root, config_path = _project(tmp_path)
    existing = root / "artifacts" / "experiments" / "existing-run"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exist"):
        experiment.plan_experiment(
            experiment.load_config(config_path), run_id="existing-run", project_root=root
        )


def test_real_stages_run_end_to_end_on_small_synthetic_history(tmp_path: Path) -> None:
    root, config_path = _project(tmp_path)
    price_snapshot = root / "data" / "raw" / "price_snapshot"
    auxiliary_snapshot = root / "data" / "raw" / "aux_snapshot"
    calendar_dates = pd.date_range("2015-11-01", "2025-01-31", freq="D")
    calendar = pd.DataFrame(
        {
            "Date": calendar_dates,
            "HolDiv": np.where(
                (calendar_dates.weekday < 5) & (calendar_dates <= pd.Timestamp("2024-12-31")),
                "1",
                "0",
            ),
        }
    )
    calendar.to_parquet(price_snapshot / "trading_calendar.parquet", index=False)
    sessions = pd.DatetimeIndex(calendar.loc[calendar["HolDiv"].eq("1"), "Date"])
    research_sessions = sessions[(sessions >= "2016-01-01") & (sessions <= "2024-12-31")]
    codes = ["10000", "10010", "10020", "10030"]

    all_price_parts: list[dict[str, Any]] = []
    for year in range(2016, 2025):
        year_sessions = research_sessions[research_sessions.year == year]
        records: list[pd.DataFrame] = []
        global_positions = sessions.get_indexer(year_sessions).astype("float64")
        for code_position, code in enumerate(codes):
            close = (
                100.0
                + 15.0 * code_position
                + 0.025 * global_positions
                + 2.0 * np.sin(global_positions / (7.0 + code_position))
                + 0.4 * np.cos(global_positions / (3.0 + code_position))
            )
            open_price = close * (1.0 + 0.001 * np.cos(global_positions / 5.0))
            high = np.maximum(open_price, close) * 1.01
            low = np.minimum(open_price, close) * 0.99
            volume = np.full(len(year_sessions), 2_000_000.0 + code_position * 100_000.0)
            records.append(
                pd.DataFrame(
                    {
                        "Date": year_sessions,
                        "Code": code,
                        "O": open_price,
                        "H": high,
                        "L": low,
                        "C": close,
                        "UL": "0",
                        "LL": "0",
                        "Vo": volume,
                        "Va": volume * close,
                        "AdjFactor": 1.0,
                        "AdjO": open_price,
                        "AdjH": high,
                        "AdjL": low,
                        "AdjC": close,
                        "AdjVo": volume,
                    }
                )
            )
        prices = pd.concat(records, ignore_index=True).sort_values(["Date", "Code"])
        price_path = price_snapshot / "prices" / f"{year}.parquet"
        prices.to_parquet(price_path, index=False)
        all_price_parts.append(
            {
                "year": year,
                "path": f"prices/{year}.parquet",
                "rows": len(prices),
                "sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
            }
        )

        listed = pd.MultiIndex.from_product(
            [year_sessions, codes], names=["Date", "Code"]
        ).to_frame(index=False)
        listed["CoName"] = listed["Code"].map(lambda value: f"Company {value}")
        listed["CoNameEn"] = listed["CoName"]
        listed["S17"] = "1"
        listed["S17Nm"] = "Sector17"
        listed["S33"] = listed["Code"].map(
            {"10000": "10", "10010": "10", "10020": "20", "10030": "20"}
        )
        listed["S33Nm"] = listed["S33"].map({"10": "Sector A", "20": "Sector B"})
        listed["ScaleCat"] = "TOPIX Small 1"
        listed["Mkt"] = "0111"
        listed["MktNm"] = "Prime"
        listed["Mrgn"] = "1"
        listed["MrgnNm"] = "Margin"
        listed["ProdCat"] = "011"
        listed.to_parquet(auxiliary_snapshot / "listed_info" / f"{year}.parquet", index=False)

        pd.DataFrame({column: pd.Series(dtype="object") for column in EARNINGS_SCHEDULE_COLUMNS}).to_parquet(
            auxiliary_snapshot / "earnings_schedule" / f"{year}.parquet", index=False
        )
        pd.DataFrame({column: pd.Series(dtype="object") for column in FINANCIAL_RESULTS_COLUMNS}).to_parquet(
            auxiliary_snapshot / "financial_results" / f"{year}.parquet", index=False
        )

    topix_position = np.arange(len(sessions), dtype="float64")
    topix_close = 1800.0 + 0.12 * topix_position + 8.0 * np.sin(topix_position / 11.0)
    pd.DataFrame(
        {
            "Date": sessions,
            "O": topix_close * 0.999,
            "H": topix_close * 1.005,
            "L": topix_close * 0.995,
            "C": topix_close,
        }
    ).to_parquet(auxiliary_snapshot / "topix.parquet", index=False)
    _write_json(
        price_snapshot / "manifest.json",
        {
            "status": "completed",
            "snapshot_id": "synthetic-price",
            "provider": "synthetic-test",
            "api_version": "v2-shape",
            "request": {"start_date": "2016-01-01", "end_date": "2024-12-31"},
            "totals": {"price_rows": sum(part["rows"] for part in all_price_parts)},
            "completed_partitions": all_price_parts,
        },
    )
    _write_json(
        auxiliary_snapshot / "manifest.json",
        {
            "status": "completed",
            "snapshot_id": "synthetic-auxiliary",
            "provider": "synthetic-test",
            "api_version": "v2-shape",
            "request": {"start_date": "2016-01-01", "end_date": "2024-12-31"},
            "totals": {"listed_info_rows": len(research_sessions) * len(codes)},
        },
    )

    result = experiment.run_experiment(
        experiment.load_config(config_path), run_id="real-smoke", project_root=root
    )

    assert result.manifest["status"] == "complete"
    assert result.manifest["final_period_evaluated"] is False
    assert result.manifest["final_period_used_for_model_selection"] is False
    weekly = pd.concat(
        [pd.read_parquet(path) for path in sorted(result.paths.weekly_dataset.glob("*.parquet"))],
        ignore_index=True,
    )
    predictions = pd.read_parquet(result.paths.modeling / "predictions.parquet")
    evaluation = pd.read_parquet(result.paths.evaluation / "summary_metrics.parquet")
    backtest = pd.read_parquet(result.paths.backtest / "summary_metrics.parquet")
    assert weekly["signal_date"].min() == pd.Timestamp("2017-02-06")
    assert weekly["signal_date"].max().year == 2024
    assert set(predictions["validation_year"]) == set(range(2020, 2025))
    assert len(predictions) > 0
    assert set(evaluation["model"]) == {"constant", "momentum", "linear"}
    assert bool(backtest.iloc[0]["main_result_complete"])
    assert result.report_path.is_file()
