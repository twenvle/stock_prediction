"""Calendar, target, leakage, schema, and CLI tests for weekly datasets."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.descriptors import FEATURE_COLUMNS
from stock_prediction.earnings_events import APPENDED_COLUMNS as EARNINGS_COLUMNS
from stock_prediction.preprocessing import SchemaError
from stock_prediction.weekly_dataset import (
    WeeklyDatasetConfig,
    build_weekly_dataset,
    build_weekly_schedule,
    load_config,
    main,
    run_weekly_dataset,
)


def make_calendar(start: str, end: str, sessions: list[str] | pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    open_dates = set(pd.to_datetime(sessions))
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(["1" if value in open_dates else "0" for value in dates], dtype="string"),
        }
    )


def make_descriptors(rows: list[tuple[str, str]], *, missing: tuple[int, str] | None = None) -> pd.DataFrame:
    records = []
    for index, (date, code) in enumerate(rows):
        record: dict[str, object] = {"date": date, "code": code}
        record.update({name: float(index + position + 1) / 100 for position, name in enumerate(FEATURE_COLUMNS)})
        records.append(record)
    output = pd.DataFrame.from_records(records, columns=["date", "code", *FEATURE_COLUMNS])
    if missing is not None:
        output.loc[missing[0], missing[1]] = np.nan
    return output


def make_universe(
    signal: str,
    entry: str,
    exit_date: str,
    codes: tuple[str, ...] = ("10000", "20000"),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [signal] * len(codes),
            "code": list(codes),
            "entry_date": [entry] * len(codes),
            "exit_date": [exit_date] * len(codes),
            "is_eligible": [True, *([False] * (len(codes) - 1))],
            "exclusion_reasons": ["", *(["not_domestic_common_stock"] * (len(codes) - 1))],
            "is_executable": pd.array([True] * len(codes), dtype="boolean"),
        }
    )


def make_earnings(universe: pd.DataFrame, *, first_eligible: bool = True) -> pd.DataFrame:
    size = len(universe)
    output = universe[["signal_date", "code"]].copy()
    output["scheduled_earnings_during_holding"] = False
    output["scheduled_earnings_count"] = 0
    output["first_scheduled_earnings_date"] = pd.NaT
    output["active_schedule_date_missing"] = False
    output["actual_earnings_during_holding"] = False
    output["actual_earnings_count"] = 0
    output["first_actual_earnings_date"] = pd.NaT
    output["last_known_earnings_date"] = pd.NaT
    output["trading_sessions_since_last_earnings"] = pd.array([pd.NA] * size, dtype="Int64")
    output["is_in_post_earnings_cooldown"] = False
    output["is_earnings_eligible_without_cooldown"] = True
    output["is_earnings_eligible"] = True
    output["is_earnings_eligible_ex_post"] = True
    output["earnings_exclusion_reasons"] = ""
    output["earnings_ex_post_exclusion_reasons"] = ""
    if not first_eligible:
        output.loc[0, "is_in_post_earnings_cooldown"] = True
        output.loc[0, "is_earnings_eligible"] = False
        output.loc[0, "is_earnings_eligible_ex_post"] = False
        output.loc[0, "earnings_exclusion_reasons"] = "post_earnings_cooldown"
        output.loc[0, "earnings_ex_post_exclusion_reasons"] = "post_earnings_cooldown"
    assert set(EARNINGS_COLUMNS).issubset(output.columns)
    return output


def make_prices(
    signal: str,
    exit_date: str,
    values: dict[str, tuple[float, float | None]],
) -> pd.DataFrame:
    records = []
    for code, (signal_close, evaluation_close) in values.items():
        records.append(
            {
                "date": signal,
                "code": code,
                "adjusted_close": signal_close,
                "has_price": True,
                "calendar_verified": True,
            }
        )
        records.append(
            {
                "date": exit_date,
                "code": code,
                "adjusted_close": evaluation_close,
                "has_price": evaluation_close is not None,
                "calendar_verified": True,
            }
        )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def one_week_inputs() -> dict[str, pd.DataFrame]:
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    calendar = make_calendar("2024-01-01", "2024-01-07", sessions)
    universe = make_universe("2024-01-02", "2024-01-03", "2024-01-05")
    return {
        "calendar": calendar,
        "universe": universe,
        "earnings": make_earnings(universe),
        "descriptors": make_descriptors(
            [("2024-01-02", "10000"), ("2024-01-02", "20000")],
            missing=(1, "R_60"),
        ),
        "prices": make_prices(
            "2024-01-02",
            "2024-01-05",
            {"10000": (100.0, 110.0), "20000": (50.0, 45.0)},
        ),
        "topix": pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-02", "2024-01-05"]), "close": [100.0, 102.0]}
        ),
    }


def test_schedule_includes_four_day_holiday_week_and_excludes_short_week() -> None:
    sessions = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-15",
            "2024-01-16",
            "2024-01-17",
        ]
    )
    result = build_weekly_schedule(make_calendar("2024-01-01", "2024-01-21", sessions))

    assert list(result.schedule["signal_date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-08")]
    assert list(result.schedule["entry_date"]) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-09")]
    assert list(result.schedule["exit_date"]) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert result.report["included_weeks"] == 2
    assert result.report["exclusion_reason_counts"]["fewer_than_minimum_sessions"] == 1
    assert result.report["exclusion_reason_counts"]["evaluation_weekday_not_thursday_or_friday"] == 1


def test_incomplete_calendar_boundary_week_is_not_invented() -> None:
    calendar = make_calendar(
        "2024-01-01",
        "2024-01-10",
        pd.bdate_range("2024-01-02", "2024-01-10"),
    )
    result = build_weekly_schedule(calendar)
    assert len(result.schedule) == 1
    assert result.schedule.iloc[0]["exit_date"] == pd.Timestamp("2024-01-05")
    assert result.report["exclusion_reason_counts"]["calendar_boundary_incomplete"] == 1


def test_schedule_excludes_signals_before_configured_start() -> None:
    sessions = pd.bdate_range("2024-01-01", "2024-01-19")
    result = build_weekly_schedule(
        make_calendar("2024-01-01", "2024-01-21", sessions),
        config=WeeklyDatasetConfig(signal_start_date="2024-01-08"),
    )

    assert list(result.schedule["signal_date"]) == [
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-15"),
    ]
    first_week = result.audit.loc[result.audit["signal_date"].eq(pd.Timestamp("2024-01-01"))].iloc[0]
    assert not first_week["is_included"]
    assert "signal_before_configured_start" in first_week["exclusion_reasons"]
    assert result.report["configured_signal_start_date"] == "2024-01-08"
    assert result.report["signal_date_min"] == "2024-01-08"
    assert result.report["exclusion_reason_counts"]["signal_before_configured_start"] == 1


def test_known_targets_join_all_components_without_mutating_inputs(one_week_inputs) -> None:
    originals = {name: value.copy(deep=True) for name, value in one_week_inputs.items()}
    result = build_weekly_dataset(
        one_week_inputs["prices"],
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        one_week_inputs["earnings"],
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    )
    for name, original in originals.items():
        assert_frame_equal(one_week_inputs[name], original)

    output = result.dataset.set_index("code")
    assert output.loc["10000", "target_return"] == pytest.approx(0.10)
    assert output.loc["10000", "target_topix_return"] == pytest.approx(0.02)
    assert output.loc["10000", "target_excess_return"] == pytest.approx(0.08)
    assert output.loc["10000", "is_signal_eligible"]
    assert output.loc["10000", "is_target_return_sample"]
    assert output.loc["10000", "is_target_excess_return_sample"]
    assert output.loc["10000", "trading_sessions_in_week"] == 4
    assert not output.loc["20000", "is_signal_eligible"]
    assert not output.loc["20000", "has_complete_features"]
    assert output.loc["20000", "feature_missing_count"] == 1
    assert result.report["signal_eligible_rows"] == 1
    assert result.report["target_return_sample_rows"] == 1


def test_missing_evaluation_price_is_reported_not_filled(one_week_inputs) -> None:
    prices = make_prices(
        "2024-01-02",
        "2024-01-05",
        {"10000": (100.0, None), "20000": (50.0, 45.0)},
    )
    result = build_weekly_dataset(
        prices,
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        one_week_inputs["earnings"],
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    )
    row = result.dataset.set_index("code").loc["10000"]
    assert row["is_signal_eligible"]
    assert not row["has_target_return"]
    assert not row["is_target_return_sample"]
    assert pd.isna(row["target_return"])
    assert row["target_missing_reasons"] == "evaluation_adjusted_close_unavailable"
    assert result.report["signal_eligible_missing_target_rows"] == 1


def test_future_price_and_execution_changes_do_not_change_signal_columns(one_week_inputs) -> None:
    baseline = build_weekly_dataset(
        one_week_inputs["prices"],
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        one_week_inputs["earnings"],
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    ).dataset
    changed_prices = one_week_inputs["prices"].copy()
    changed_prices.loc[
        changed_prices["date"].eq("2024-01-05") & changed_prices["code"].eq("10000"),
        "adjusted_close",
    ] = 70.0
    changed_universe = one_week_inputs["universe"].copy()
    changed_universe["is_executable"] = ~changed_universe["is_executable"]
    changed = build_weekly_dataset(
        changed_prices,
        one_week_inputs["descriptors"],
        changed_universe,
        one_week_inputs["earnings"],
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    ).dataset
    signal_columns = [
        "signal_date",
        "code",
        "is_eligible",
        "is_earnings_eligible",
        "is_signal_eligible",
        *FEATURE_COLUMNS,
    ]
    assert_frame_equal(baseline[signal_columns], changed[signal_columns])
    assert baseline.set_index("code").loc["10000", "target_return"] != changed.set_index("code").loc[
        "10000", "target_return"
    ]


def test_main_cooldown_flag_is_used_and_ex_post_flag_is_audit_only(one_week_inputs) -> None:
    cooldown = make_earnings(one_week_inputs["universe"], first_eligible=False)
    cooldown_output = build_weekly_dataset(
        one_week_inputs["prices"],
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        cooldown,
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    ).dataset.set_index("code")
    assert cooldown_output.loc["10000", "is_signal_eligible_without_cooldown"]
    assert not cooldown_output.loc["10000", "is_signal_eligible"]

    ex_post_only = make_earnings(one_week_inputs["universe"])
    ex_post_only.loc[0, "actual_earnings_during_holding"] = True
    ex_post_only.loc[0, "is_earnings_eligible_ex_post"] = False
    ex_post_only.loc[0, "earnings_ex_post_exclusion_reasons"] = "actual_earnings_during_holding"
    ex_post_output = build_weekly_dataset(
        one_week_inputs["prices"],
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        ex_post_only,
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    ).dataset.set_index("code")
    assert ex_post_output.loc["10000", "is_signal_eligible"]
    assert not ex_post_output.loc["10000", "is_earnings_eligible_ex_post"]


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_earnings", "key coverage must exactly match"),
        ("missing_descriptor", "descriptors are missing"),
        ("duplicate_prices", "duplicate date/code"),
        ("bad_schedule", "disagrees with the official weekly schedule"),
        ("non_session_price", "not TSE trading sessions"),
    ],
)
def test_schema_and_join_coverage_errors_fail_early(one_week_inputs, change: str, match: str) -> None:
    values = {name: value.copy(deep=True) for name, value in one_week_inputs.items()}
    if change == "missing_earnings":
        values["earnings"] = values["earnings"].iloc[:1]
    elif change == "missing_descriptor":
        values["descriptors"] = values["descriptors"].iloc[1:]
    elif change == "duplicate_prices":
        values["prices"] = pd.concat([values["prices"], values["prices"].iloc[[0]]], ignore_index=True)
    elif change == "bad_schedule":
        values["universe"]["exit_date"] = "2024-01-04"
    else:
        values["prices"].loc[0, "date"] = "2024-01-01"
    with pytest.raises(SchemaError, match=match):
        build_weekly_dataset(
            values["prices"],
            values["descriptors"],
            values["universe"],
            values["earnings"],
            trading_calendar=values["calendar"],
            topix=values["topix"],
        )


def test_config_rejects_unknown_and_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "weekly.json"
    path.write_text('{"min_trading_sessions_per_week": 4, "require_topix": true}', encoding="utf-8")
    assert load_config(path) == WeeklyDatasetConfig()
    path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(path)
    with pytest.raises(ValueError, match="2 through 5"):
        WeeklyDatasetConfig(min_trading_sessions_per_week=1)
    with pytest.raises(ValueError, match="ISO date string"):
        WeeklyDatasetConfig(signal_start_date="2024/01/08")
    with pytest.raises(ValueError, match="valid ISO date"):
        WeeklyDatasetConfig(signal_start_date="2024-02-30")


def _write_partition_inputs(base: Path, year: int, values: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("prices", "descriptors", "universe", "earnings"):
        directory = base / name
        directory.mkdir(exist_ok=True)
        values[name].to_parquet(directory / f"{year}.parquet", index=False)
        paths[name] = directory
    return paths


def test_cli_schedule_and_partitioned_build_match_dataframe_api(
    tmp_path: Path,
    capsys,
    one_week_inputs,
) -> None:
    paths = _write_partition_inputs(tmp_path, 2024, one_week_inputs)
    calendar_path = tmp_path / "calendar.parquet"
    topix_path = tmp_path / "topix.parquet"
    one_week_inputs["calendar"].to_parquet(calendar_path, index=False)
    one_week_inputs["topix"].to_parquet(topix_path, index=False)
    schedule_path = tmp_path / "weekly_schedule.parquet"
    schedule_report_path = tmp_path / "weekly_schedule.json"

    status = main(
        [
            "schedule",
            "--calendar",
            str(calendar_path),
            "--output",
            str(schedule_path),
            "--report",
            str(schedule_report_path),
        ]
    )
    assert status == 0
    assert len(pd.read_parquet(schedule_path)) == 1
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"
    status = main(
        [
            "build",
            "--prices-dir",
            str(paths["prices"]),
            "--descriptors-dir",
            str(paths["descriptors"]),
            "--universe-dir",
            str(paths["universe"]),
            "--earnings-dir",
            str(paths["earnings"]),
            "--calendar",
            str(calendar_path),
            "--topix",
            str(topix_path),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert status == 0
    assert '"status": "complete"' in capsys.readouterr().out
    output = pd.read_parquet(output_dir / "2024.parquet")
    expected = build_weekly_dataset(
        one_week_inputs["prices"],
        one_week_inputs["descriptors"],
        one_week_inputs["universe"],
        one_week_inputs["earnings"],
        trading_calendar=one_week_inputs["calendar"],
        topix=one_week_inputs["topix"],
    ).dataset
    assert_frame_equal(output, expected)
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["output_rows"] == 2
    assert manifest["target_return_sample_rows"] == 1

    with pytest.raises(FileExistsError):
        run_weekly_dataset(
            prices_dir=paths["prices"],
            descriptors_dir=paths["descriptors"],
            universe_dir=paths["universe"],
            earnings_dir=paths["earnings"],
            calendar_path=calendar_path,
            topix_path=topix_path,
            output_dir=output_dir,
            report_dir=tmp_path / "other-reports",
        )


def test_interruption_preserves_completed_partition_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("stock_prediction.weekly_dataset")
    sessions = pd.to_datetime(
        [
            "2023-12-25",
            "2023-12-26",
            "2023-12-27",
            "2023-12-28",
            "2023-12-29",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ]
    )
    calendar = make_calendar("2023-12-25", "2024-01-07", sessions)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    topix = pd.DataFrame({"date": sessions, "close": np.linspace(100.0, 108.0, len(sessions))})
    topix_path = tmp_path / "topix.parquet"
    topix.to_parquet(topix_path, index=False)
    directories = {name: tmp_path / name for name in ("prices", "descriptors", "universe", "earnings")}
    for directory in directories.values():
        directory.mkdir()
    for year, signal, entry, exit_date in (
        (2023, "2023-12-25", "2023-12-26", "2023-12-29"),
        (2024, "2024-01-02", "2024-01-03", "2024-01-05"),
    ):
        universe = make_universe(signal, entry, exit_date, codes=("10000",))
        make_prices(signal, exit_date, {"10000": (100.0, 101.0)}).to_parquet(
            directories["prices"] / f"{year}.parquet", index=False
        )
        make_descriptors([(signal, "10000")]).to_parquet(
            directories["descriptors"] / f"{year}.parquet", index=False
        )
        universe.to_parquet(directories["universe"] / f"{year}.parquet", index=False)
        make_earnings(universe).to_parquet(directories["earnings"] / f"{year}.parquet", index=False)

    original = module.build_weekly_dataset
    calls = 0

    def stop_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("test interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_weekly_dataset", stop_second)
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"
    with pytest.raises(KeyboardInterrupt):
        run_weekly_dataset(
            prices_dir=directories["prices"],
            descriptors_dir=directories["descriptors"],
            universe_dir=directories["universe"],
            earnings_dir=directories["earnings"],
            calendar_path=calendar_path,
            topix_path=topix_path,
            output_dir=output_dir,
            report_dir=report_dir,
        )
    assert (output_dir / "2023.parquet").is_file()
    assert not (output_dir / "2024.parquet").exists()
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["current_partition"] == 2024
    assert [item["year"] for item in manifest["completed_partitions"]] == [2023]
