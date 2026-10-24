"""Point-in-time and execution-safety tests for the universe builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.preprocessing import SchemaError
from stock_prediction.universe import (
    UniverseConfig,
    compute_universe,
    load_config,
    main,
    run_universe,
)


def make_calendar(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.date_range(sessions.min(), sessions.max(), freq="D")
    open_dates = set(sessions)
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(["1" if value in open_dates else "0" for value in dates], dtype="string"),
        }
    )


def make_master(
    sessions: pd.DatetimeIndex,
    codes: dict[str, str],
    *,
    starts: dict[str, int] | None = None,
    absent: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    starts = starts or {}
    absent = absent or {}
    records: list[dict[str, object]] = []
    for code, product_category in codes.items():
        for position, day in enumerate(sessions):
            if position < starts.get(code, 0) or position in absent.get(code, set()):
                continue
            records.append(
                {
                    "Date": day,
                    "Code": code,
                    "CoName": f"会社{code}",
                    "Mkt": "0111",
                    "MktNm": "プライム",
                    "ProdCat": product_category,
                }
            )
    frame = pd.DataFrame.from_records(records)
    for column in ("Code", "CoName", "Mkt", "MktNm", "ProdCat"):
        frame[column] = frame[column].astype("string")
    return frame


def make_prices(
    sessions: pd.DatetimeIndex,
    codes: tuple[str, ...],
    *,
    turnover: float = 100_000_000.0,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for code in codes:
        for position, day in enumerate(sessions):
            close = 100.0 + position
            records.append(
                {
                    "date": day,
                    "code": code,
                    "open": close - 1.0,
                    "close": close,
                    "upper_limit": "0",
                    "lower_limit": "0",
                    "volume": 1_000_000.0,
                    "turnover_value": turnover,
                    "has_price": True,
                    "calendar_verified": True,
                }
            )
    frame = pd.DataFrame.from_records(records)
    for column in ("code", "upper_limit", "lower_limit"):
        frame[column] = frame[column].astype("string")
    return frame


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-04", periods=75)


def test_domestic_common_and_sixty_elapsed_sessions_are_point_in_time(sessions: pd.DatetimeIndex) -> None:
    master = make_master(
        sessions,
        {"10000": "011", "20000": "120", "30000": "011"},
        starts={"30000": 10},
    )
    prices = make_prices(sessions, ("10000", "20000", "30000"))
    original_master = master.copy(deep=True)
    original_prices = prices.copy(deep=True)

    output = compute_universe(master, prices, trading_calendar=make_calendar(sessions)).universe

    assert_frame_equal(master, original_master)
    assert_frame_equal(prices, original_prices)
    before = output.query("signal_date == @sessions[59] and code == '10000'").iloc[0]
    boundary = output.query("signal_date == @sessions[60] and code == '10000'").iloc[0]
    etf = output.query("signal_date == @sessions[60] and code == '20000'").iloc[0]
    recent = output.query("signal_date == @sessions[60] and code == '30000'").iloc[0]
    assert before["listed_sessions_elapsed"] == 59
    assert not before["has_min_listing_history"]
    assert boundary["listed_sessions_elapsed"] == 60
    assert boundary["has_min_listing_history"]
    assert boundary["is_eligible"]
    assert not etf["is_domestic_common_stock"]
    assert "not_domestic_common_stock" in etf["exclusion_reasons"]
    assert recent["listed_sessions_elapsed"] == 50
    assert not recent["is_eligible"]


def test_listing_age_resets_after_master_disappearance(sessions: pd.DatetimeIndex) -> None:
    master = make_master(
        sessions,
        {"10000": "011", "99990": "120"},
        absent={"10000": {61}},
    )
    prices = make_prices(sessions, ("10000", "99990"))

    output = compute_universe(master, prices, trading_calendar=make_calendar(sessions)).universe
    reappeared = output.query("signal_date == @sessions[62] and code == '10000'").iloc[0]
    assert reappeared["listed_sessions_elapsed"] == 0
    assert not reappeared["has_min_listing_history"]


def test_liquidity_counts_missing_sessions_without_filling(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    final_window = set(sessions[-20:])
    five_missing = set(sorted(final_window)[:5])
    prices_15 = prices.loc[~prices["date"].isin(five_missing)].reset_index(drop=True)

    row_15 = compute_universe(master, prices_15, trading_calendar=make_calendar(sessions)).universe.iloc[-1]
    assert row_15["liquidity_window_observations"] == 20
    assert row_15["price_formation_days_20"] == 15
    assert row_15["median_turnover_value_20"] == pytest.approx(100_000_000.0)
    assert row_15["max_order_value_yen_20"] == pytest.approx(1_000_000.0)
    assert row_15["is_liquid"]

    six_missing = five_missing | {sorted(final_window)[5]}
    prices_14 = prices.loc[~prices["date"].isin(six_missing)].reset_index(drop=True)
    row_14 = compute_universe(master, prices_14, trading_calendar=make_calendar(sessions)).universe.iloc[-1]
    assert row_14["price_formation_days_20"] == 14
    assert not row_14["meets_price_formation_days"]
    assert not row_14["is_eligible"]
    assert "insufficient_price_formation_days" in row_14["exclusion_reasons"]


def test_turnover_and_participation_thresholds_are_inclusive(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011"})
    at_threshold = make_prices(sessions, ("10000",), turnover=100_000_000.0)
    below = make_prices(sessions, ("10000",), turnover=99_999_999.0)

    accepted = compute_universe(master, at_threshold, trading_calendar=make_calendar(sessions)).universe.iloc[-1]
    rejected = compute_universe(master, below, trading_calendar=make_calendar(sessions)).universe.iloc[-1]
    assert accepted["meets_median_turnover"] and accepted["meets_participation_limit"]
    assert not rejected["meets_median_turnover"]
    assert not rejected["meets_participation_limit"]
    assert not rejected["is_liquid"]


def test_liquidity_rolling_never_crosses_security_boundary(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011", "20000": "011"})
    prices = make_prices(sessions, ("10000", "20000"))
    prices.loc[prices["code"].eq("20000"), "turnover_value"] = 1_000_000.0

    final = (
        compute_universe(master, prices, trading_calendar=make_calendar(sessions))
        .universe.query("signal_date == @sessions[-1]")
        .set_index("code")
    )
    assert final.loc["10000", "median_turnover_value_20"] == pytest.approx(100_000_000.0)
    assert final.loc["10000", "is_liquid"]
    assert final.loc["20000", "median_turnover_value_20"] == pytest.approx(1_000_000.0)
    assert not final.loc["20000", "is_liquid"]


def test_future_execution_changes_never_change_signal_eligibility(sessions: pd.DatetimeIndex) -> None:
    signal_position = next(i for i, day in enumerate(sessions) if i >= 60 and day.weekday() == 0)
    signal_date = sessions[signal_position]
    entry_date = sessions[signal_position + 1]
    exit_date = sessions[signal_position + 4]
    current_sessions = sessions[: signal_position + 1]
    future_sessions = sessions[signal_position + 1 : signal_position + 5]
    master = make_master(current_sessions, {"10000": "011"})
    prices = make_prices(current_sessions, ("10000",))
    future = make_prices(future_sessions, ("10000",))
    schedule = pd.DataFrame(
        {"signal_date": [signal_date], "entry_date": [entry_date], "exit_date": [exit_date]}
    )

    baseline = compute_universe(
        master,
        prices,
        trading_calendar=make_calendar(sessions),
        schedule=schedule,
        execution_prices=future,
    ).universe
    changed_future = future.copy(deep=True)
    changed_future.loc[changed_future["date"].eq(entry_date), "upper_limit"] = "1"
    changed_future.loc[changed_future["date"].eq(exit_date), "close"] = 9999.0
    changed = compute_universe(
        master,
        prices,
        trading_calendar=make_calendar(sessions),
        schedule=schedule,
        execution_prices=changed_future,
    ).universe

    ex_ante_columns = [
        "signal_date",
        "code",
        "median_turnover_value_20",
        "price_formation_days_20",
        "is_liquid",
        "is_eligible",
        "exclusion_reasons",
    ]
    assert_frame_equal(baseline[ex_ante_columns], changed[ex_ante_columns])
    assert baseline.loc[0, "is_executable"]
    assert not changed.loc[0, "is_entry_executable"]
    assert not changed.loc[0, "is_executable"]
    assert changed.loc[0, "exit_price"] == pytest.approx(9999.0)


def test_future_master_and_price_changes_do_not_change_past_universe(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    cutoff = sessions[65]
    baseline = compute_universe(master, prices, trading_calendar=make_calendar(sessions)).universe

    changed_master = master.copy(deep=True)
    changed_prices = prices.copy(deep=True)
    changed_master.loc[changed_master["Date"].gt(cutoff), "ProdCat"] = "120"
    changed_prices.loc[changed_prices["date"].gt(cutoff), "turnover_value"] = 1.0
    changed = compute_universe(
        changed_master,
        changed_prices,
        trading_calendar=make_calendar(sessions),
    ).universe
    columns = [
        "signal_date",
        "code",
        "product_category",
        "listed_sessions_elapsed",
        "price_formation_days_20",
        "median_turnover_value_20",
        "is_liquid",
        "is_eligible",
        "exclusion_reasons",
    ]
    assert_frame_equal(
        baseline.loc[baseline["signal_date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["signal_date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_missing_or_limit_locked_execution_is_reported_not_used_for_selection(
    sessions: pd.DatetimeIndex,
) -> None:
    signal_position = next(i for i, day in enumerate(sessions) if i >= 60 and day.weekday() == 0)
    current_sessions = sessions[: signal_position + 1]
    master = make_master(current_sessions, {"10000": "011", "20000": "011"})
    prices = make_prices(current_sessions, ("10000", "20000"))
    future_sessions = sessions[signal_position + 1 : signal_position + 5]
    future = make_prices(future_sessions, ("10000", "20000"))
    future.loc[
        future["code"].eq("10000") & future["date"].eq(future_sessions[-1]), "lower_limit"
    ] = "1"
    future = future.loc[
        ~(future["code"].eq("20000") & future["date"].eq(future_sessions[0]))
    ].reset_index(drop=True)
    schedule = pd.DataFrame(
        {
            "signal_date": [current_sessions[-1]],
            "entry_date": [future_sessions[0]],
            "exit_date": [future_sessions[-1]],
        }
    )

    output = compute_universe(
        master,
        prices,
        trading_calendar=make_calendar(sessions),
        schedule=schedule,
        execution_prices=future,
    ).universe.set_index("code")
    assert output["is_eligible"].all()
    assert not output.loc["10000", "exit_not_limit_down"]
    assert not output.loc["10000", "is_exit_executable"]
    assert not output.loc["20000", "entry_price_available"]
    assert not output.loc["20000", "is_entry_executable"]


def test_previous_history_preserves_year_boundary_results() -> None:
    sessions = pd.bdate_range("2023-10-02", periods=90)
    calendar = make_calendar(sessions)
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    split_date = pd.Timestamp("2024-01-01")
    current_master = master.loc[master["Date"].ge(split_date)].reset_index(drop=True)
    current_prices = prices.loc[prices["date"].ge(split_date)].reset_index(drop=True)
    previous_master = master.loc[master["Date"].lt(split_date)].groupby("Code", sort=False).tail(60)
    previous_prices = prices.loc[prices["date"].lt(split_date)]

    full = compute_universe(master, prices, trading_calendar=calendar).universe
    expected = full.loc[full["signal_date"].ge(split_date)].reset_index(drop=True)
    split = compute_universe(
        current_master,
        current_prices,
        trading_calendar=calendar,
        previous_listed_info=previous_master,
        previous_prices=previous_prices,
    ).universe
    assert_frame_equal(split, expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda master, prices: (master.drop(columns="ProdCat"), prices), "missing required"),
        (lambda master, prices: (pd.concat([master, master.iloc[[0]]]), prices), "duplicate Date/Code"),
        (lambda master, prices: (master, prices.assign(calendar_verified=False)), "calendar_verified=False"),
    ],
)
def test_schema_errors_fail_early(sessions: pd.DatetimeIndex, mutation, match: str) -> None:
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    changed_master, changed_prices = mutation(master, prices)
    with pytest.raises(SchemaError, match=match):
        compute_universe(changed_master, changed_prices, trading_calendar=make_calendar(sessions))


def test_missing_master_snapshot_fails_instead_of_inventing_membership(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011"})
    master = master.loc[~master["Date"].eq(sessions[20])].reset_index(drop=True)
    prices = make_prices(sessions, ("10000",))
    with pytest.raises(SchemaError, match="missing TSE session snapshots"):
        compute_universe(master, prices, trading_calendar=make_calendar(sessions))


def test_schedule_requires_session_dates_and_strict_order(sessions: pd.DatetimeIndex) -> None:
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    monday = next(day for day in sessions if day.weekday() == 0 and day >= sessions[60])
    schedule = pd.DataFrame(
        {"signal_date": [monday], "entry_date": [monday], "exit_date": [monday + pd.offsets.BDay(4)]}
    )
    with pytest.raises(SchemaError, match="immediately after"):
        compute_universe(master, prices, trading_calendar=make_calendar(sessions), schedule=schedule)


def test_schedule_uses_calendar_week_boundaries_in_holiday_week() -> None:
    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"])
    )
    calendar = make_calendar(sessions)
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    valid = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-09")],
            "entry_date": [pd.Timestamp("2024-01-10")],
            "exit_date": [pd.Timestamp("2024-01-12")],
        }
    )
    compute_universe(master, prices, trading_calendar=calendar, schedule=valid)

    wrong_signal = valid.copy()
    wrong_signal["signal_date"] = pd.Timestamp("2024-01-10")
    wrong_signal["entry_date"] = pd.Timestamp("2024-01-11")
    with pytest.raises(SchemaError, match="first TSE session"):
        compute_universe(master, prices, trading_calendar=calendar, schedule=wrong_signal)

    wrong_exit = valid.copy()
    wrong_exit["exit_date"] = pd.Timestamp("2024-01-11")
    with pytest.raises(SchemaError, match="last TSE session"):
        compute_universe(master, prices, trading_calendar=calendar, schedule=wrong_exit)


def test_config_and_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "universe.json"
    config_path.write_text(
        json.dumps(
            {
                "min_listing_sessions_elapsed": 60,
                "liquidity_window_sessions": 20,
                "min_price_formation_days": 15,
                "min_median_turnover_yen": 100000000,
                "planned_investment_yen": 1000000,
                "max_participation_rate": 0.01,
                "domestic_common_product_category": "011",
            }
        ),
        encoding="utf-8",
    )
    assert load_config(config_path) == UniverseConfig()
    config_path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(config_path)
    with pytest.raises(ValueError, match="must not exceed"):
        UniverseConfig(min_price_formation_days=21)


def test_partitioned_cli_preserves_history_and_manifest(tmp_path: Path, capsys) -> None:
    sessions = pd.bdate_range("2023-10-02", periods=90)
    calendar = make_calendar(sessions)
    master = make_master(sessions, {"10000": "011", "20000": "120"})
    prices = make_prices(sessions, ("10000", "20000"))
    master_dir = tmp_path / "listed_info"
    price_dir = tmp_path / "prices"
    master_dir.mkdir()
    price_dir.mkdir()
    for year, partition in master.groupby(master["Date"].dt.year):
        partition.to_parquet(master_dir / f"{year}.parquet", index=False)
    for year, partition in prices.groupby(prices["date"].dt.year):
        partition.to_parquet(price_dir / f"{year}.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "universe"
    report_dir = tmp_path / "reports"

    status = main(
        [
            "--listed-info-dir",
            str(master_dir),
            "--prices-dir",
            str(price_dir),
            "--calendar",
            str(calendar_path),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ]
    )

    assert status == 0
    assert '"status": "complete"' in capsys.readouterr().out
    output_2024 = pd.read_parquet(output_dir / "2024.parquet")
    first_common = output_2024.loc[output_2024["code"].eq("10000")].iloc[0]
    assert first_common["listed_sessions_elapsed"] >= 60
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["output_rows"] == len(master)
    assert len(manifest["completed_partitions"]) == 2
    assert manifest["config"]["min_median_turnover_yen"] == 100_000_000.0

    with pytest.raises(FileExistsError):
        run_universe(
            listed_info_dir=master_dir,
            prices_dir=price_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "other-reports",
        )


def test_partitioned_runner_uses_year_without_signals_as_history_only(tmp_path: Path) -> None:
    sessions = pd.bdate_range("2023-10-02", periods=90)
    calendar = make_calendar(sessions)
    master = make_master(sessions, {"10000": "011"})
    prices = make_prices(sessions, ("10000",))
    master_dir = tmp_path / "listed_info"
    price_dir = tmp_path / "prices"
    master_dir.mkdir()
    price_dir.mkdir()
    for year, partition in master.groupby(master["Date"].dt.year):
        partition.to_parquet(master_dir / f"{year}.parquet", index=False)
    for year, partition in prices.groupby(prices["date"].dt.year):
        partition.to_parquet(price_dir / f"{year}.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)

    signal_date = next(day for day in sessions if day.year == 2024 and day.weekday() == 0)
    signal_position = sessions.get_loc(signal_date)
    schedule = pd.DataFrame(
        {
            "signal_date": [signal_date],
            "entry_date": [sessions[signal_position + 1]],
            "exit_date": [sessions[signal_position + 4]],
        }
    )
    schedule_path = tmp_path / "schedule.parquet"
    schedule.to_parquet(schedule_path, index=False)
    output_dir = tmp_path / "universe"
    report_dir = tmp_path / "reports"

    manifest = run_universe(
        listed_info_dir=master_dir,
        prices_dir=price_dir,
        calendar_path=calendar_path,
        schedule_path=schedule_path,
        output_dir=output_dir,
        report_dir=report_dir,
    )

    assert manifest["status"] == "complete"
    assert not (output_dir / "2023.parquet").exists()
    assert not (report_dir / "2023.json").exists()
    output_2024 = pd.read_parquet(output_dir / "2024.parquet")
    assert len(output_2024) == 1
    assert output_2024.iloc[0]["listed_sessions_elapsed"] >= 60
    assert manifest["skipped_partitions"] == [
        {
            "year": 2023,
            "reason": "no_configured_signal_dates",
            "master_input_rows": int(master["Date"].dt.year.eq(2023).sum()),
            "price_input_rows": int(prices["date"].dt.year.eq(2023).sum()),
        }
    ]
    assert [item["year"] for item in manifest["completed_partitions"]] == [2024]
