from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from stock_prediction.descriptors import (
    FEATURE_COLUMNS,
    DescriptorConfig,
    compute_descriptors,
    load_config,
    main,
    run_descriptors,
)
from stock_prediction.preprocessing import SchemaError


def make_calendar(start: str = "2023-09-01", end: str = "2024-04-30") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    divisions = np.where(dates.dayofweek < 5, "1", "0")
    return pd.DataFrame({"Date": dates, "HolDiv": divisions})


def session_dates(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(calendar.loc[calendar["HolDiv"].isin(["1", "2"]), "Date"])


def make_prices(
    dates: pd.DatetimeIndex,
    *,
    code: str = "10000",
    offset: float = 0.0,
) -> pd.DataFrame:
    position = np.arange(len(dates), dtype="float64")
    close = 100.0 + offset + position * 0.7 + np.sin(position / 3.0)
    return pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "adjusted_close": close,
            "adjusted_high": close + 2.0,
            "adjusted_low": close - 2.0,
            "adjusted_volume": 1_000.0 + position * 10.0,
            "turnover_value": 1_000_000.0 + position * 5_000.0,
            "has_price": True,
            "calendar_verified": True,
        }
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    return make_calendar()


@pytest.fixture
def prices(calendar: pd.DataFrame) -> pd.DataFrame:
    return make_prices(session_dates(calendar)[:90])


def make_topix(dates: pd.DatetimeIndex) -> pd.DataFrame:
    position = np.arange(len(dates), dtype="float64")
    return pd.DataFrame({"date": dates, "close": 2_000.0 + position * 2.5 + np.cos(position / 4.0)})


def test_known_values_match_readme_formulas_and_do_not_mutate_inputs(
    prices: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    topix = make_topix(pd.DatetimeIndex(prices["date"]))
    original_prices = prices.copy(deep=True)
    original_topix = topix.copy(deep=True)

    result = compute_descriptors(prices, trading_calendar=calendar, topix=topix)
    row = result.descriptors.iloc[-1]
    close = prices["adjusted_close"].to_numpy()
    log_returns = np.diff(np.log(close))
    expected_r20 = close[-1] / close[-21] - 1
    expected_topix_r20 = topix["close"].iloc[-1] / topix["close"].iloc[-21] - 1

    assert row["R_5"] == pytest.approx(close[-1] / close[-6] - 1)
    assert row["R_20"] == pytest.approx(expected_r20)
    assert row["R_60"] == pytest.approx(close[-1] / close[-61] - 1)
    assert row["PriceToSMA_20"] == pytest.approx(close[-1] / close[-20:].mean() - 1)
    assert row["SMA_5To20"] == pytest.approx(close[-5:].mean() / close[-20:].mean() - 1)
    expected_volatility_5 = log_returns[-5:].std(ddof=1)
    expected_volatility_20 = log_returns[-20:].std(ddof=1)
    assert row["Volatility_5To20"] == pytest.approx(expected_volatility_5 / expected_volatility_20)
    assert row["Volatility_20"] == pytest.approx(expected_volatility_20)
    low_20 = prices["adjusted_low"].iloc[-20:].min()
    high_20 = prices["adjusted_high"].iloc[-20:].max()
    assert row["RangePosition_20"] == pytest.approx((close[-1] - low_20) / (high_20 - low_20))
    assert row["VolumeRatio_5To20"] == pytest.approx(
        prices["adjusted_volume"].iloc[-5:].mean() / prices["adjusted_volume"].iloc[-20:].mean()
    )
    assert row["TurnoverRatio_5To20"] == pytest.approx(
        prices["turnover_value"].iloc[-5:].mean() / prices["turnover_value"].iloc[-20:].mean()
    )
    assert row["RelativeTOPIX_20"] == pytest.approx(expected_r20 - expected_topix_r20)
    assert result.report["complete_feature_rows"] == 30
    pdt.assert_frame_equal(prices, original_prices)
    pdt.assert_frame_equal(topix, original_topix)


def test_descriptors_never_cross_security_boundaries(calendar: pd.DataFrame) -> None:
    dates = session_dates(calendar)[:70]
    first = make_prices(dates, code="10000")
    second = make_prices(dates, code="20000", offset=500.0)
    frame = pd.concat([second, first], ignore_index=True)

    output = compute_descriptors(frame, trading_calendar=calendar).descriptors
    first_rows = output.loc[output["code"].eq("10000")].reset_index(drop=True)
    second_rows = output.loc[output["code"].eq("20000")].reset_index(drop=True)

    assert first_rows.loc[0, list(FEATURE_COLUMNS)].isna().all()
    assert second_rows.loc[0, list(FEATURE_COLUMNS)].isna().all()
    expected = second["adjusted_close"].iloc[-1] / second["adjusted_close"].iloc[-6] - 1
    assert second_rows.iloc[-1]["R_5"] == pytest.approx(expected)


def test_missing_observation_row_is_not_skipped_or_shortened(calendar: pd.DataFrame) -> None:
    dates = session_dates(calendar)[:30]
    frame = make_prices(dates)
    missing_index = 12
    frame.loc[missing_index, [
        "adjusted_close", "adjusted_high", "adjusted_low", "adjusted_volume", "turnover_value"
    ]] = np.nan
    frame.loc[missing_index, "has_price"] = False

    output = compute_descriptors(frame, trading_calendar=calendar).descriptors

    assert np.isnan(output.loc[20, "PriceToSMA_20"])
    assert np.isnan(output.loc[20, "Volatility_20"])
    assert np.isnan(output.loc[20, "RangePosition_20"])
    assert np.isnan(output.loc[20, "VolumeRatio_5To20"])
    # Endpoint momentum remains defined when t and t-5 exist; no filling occurs.
    expected = frame.loc[20, "adjusted_close"] / frame.loc[15, "adjusted_close"] - 1
    assert output.loc[20, "R_5"] == pytest.approx(expected)


def test_absent_session_row_invalidates_row_count_based_lag(calendar: pd.DataFrame) -> None:
    dates = session_dates(calendar)[:12]
    frame = make_prices(dates).drop(index=3).reset_index(drop=True)

    output = compute_descriptors(frame, trading_calendar=calendar).descriptors
    target = output.index[output["date"].eq(dates[8])][0]

    # The fifth previous *row* is six exchange sessions old and cannot stand in for t-5.
    assert np.isnan(output.loc[target, "R_5"])


def test_holiday_is_not_counted_as_a_trading_session() -> None:
    calendar = make_calendar("2024-01-01", "2024-02-29")
    holiday = pd.Timestamp("2024-01-15")
    calendar.loc[calendar["Date"].eq(holiday), "HolDiv"] = "0"
    dates = session_dates(calendar)
    frame = make_prices(dates)

    output = compute_descriptors(frame, trading_calendar=calendar).descriptors
    target = output.index[output["date"].eq(pd.Timestamp("2024-01-19"))][0]
    source = output.index[output["date"].eq(pd.Timestamp("2024-01-11"))][0]
    expected = frame.loc[target, "adjusted_close"] / frame.loc[source, "adjusted_close"] - 1

    assert output.loc[target, "R_5"] == pytest.approx(expected)


def test_constant_or_zero_denominators_become_missing(calendar: pd.DataFrame) -> None:
    frame = make_prices(session_dates(calendar)[:70])
    frame[["adjusted_close", "adjusted_high", "adjusted_low"]] = 100.0
    frame[["adjusted_volume", "turnover_value"]] = 0.0

    row = compute_descriptors(frame, trading_calendar=calendar).descriptors.iloc[-1]

    assert row["R_60"] == pytest.approx(0.0)
    assert row["PriceToSMA_20"] == pytest.approx(0.0)
    assert row["SMA_5To20"] == pytest.approx(0.0)
    assert np.isnan(row["Volatility_5To20"])
    assert row["Volatility_20"] == pytest.approx(0.0)
    assert np.isnan(row["RangePosition_20"])
    assert np.isnan(row["VolumeRatio_5To20"])
    assert np.isnan(row["TurnoverRatio_5To20"])


def test_previous_prices_warm_up_matches_single_frame(prices: pd.DataFrame, calendar: pd.DataFrame) -> None:
    split = 65
    expected = compute_descriptors(prices, trading_calendar=calendar).descriptors.iloc[split:].reset_index(drop=True)
    actual = compute_descriptors(
        prices.iloc[split:].reset_index(drop=True),
        trading_calendar=calendar,
        previous_prices=prices.iloc[:split].tail(60).reset_index(drop=True),
    ).descriptors

    pdt.assert_frame_equal(actual, expected)


def test_future_rows_cannot_change_past_descriptors(prices: pd.DataFrame, calendar: pd.DataFrame) -> None:
    cutoff = 70
    baseline = compute_descriptors(prices.iloc[:cutoff], trading_calendar=calendar).descriptors
    changed = prices.copy()
    changed.loc[cutoff:, [
        "adjusted_close", "adjusted_high", "adjusted_low", "adjusted_volume", "turnover_value"
    ]] *= 10.0

    output = compute_descriptors(changed, trading_calendar=calendar).descriptors.iloc[:cutoff].reset_index(drop=True)

    pdt.assert_frame_equal(output, baseline)


def test_topix_is_optional_but_can_be_required(prices: pd.DataFrame, calendar: pd.DataFrame) -> None:
    optional = compute_descriptors(prices, trading_calendar=calendar)
    assert optional.descriptors["RelativeTOPIX_20"].isna().all()
    assert optional.report["topix_supplied"] is False

    with pytest.raises(SchemaError, match="TOPIX input is required"):
        compute_descriptors(
            prices,
            trading_calendar=calendar,
            config=DescriptorConfig(require_topix=True),
        )


def test_unverified_calendar_duplicate_key_and_bad_topix_fail_early(
    prices: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    unverified = prices.copy()
    unverified["calendar_verified"] = False
    with pytest.raises(SchemaError, match="calendar_verified=False"):
        compute_descriptors(unverified, trading_calendar=calendar)

    duplicate = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate-key"):
        compute_descriptors(duplicate, trading_calendar=calendar)

    topix = make_topix(pd.DatetimeIndex(prices["date"])).rename(columns={"close": "Close"})
    with pytest.raises(SchemaError, match="exactly date and close"):
        compute_descriptors(prices, trading_calendar=calendar, topix=topix)


def test_invalid_previous_rows_do_not_overlap_current(prices: pd.DataFrame, calendar: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="precede every current"):
        compute_descriptors(
            prices.iloc[10:],
            trading_calendar=calendar,
            previous_prices=prices.iloc[:11],
        )


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"volatility_ddof": 1, "unknown": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(path)


def test_partitioned_cli_preserves_year_boundary_history_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calendar = make_calendar("2023-09-01", "2024-04-30")
    dates = session_dates(calendar)
    frame = make_prices(dates)
    prices_dir = tmp_path / "interim"
    prices_dir.mkdir()
    for year, partition in frame.groupby(frame["date"].dt.year):
        partition.to_parquet(prices_dir / f"{year}.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"

    status = main(
        [
            "--prices-dir", str(prices_dir),
            "--calendar", str(calendar_path),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ]
    )

    assert status == 0
    assert '"status": "complete"' in capsys.readouterr().out
    output_2024 = pd.read_parquet(output_dir / "2024.parquet")
    first_2024 = frame.index[frame["date"].dt.year.eq(2024)][0]
    expected_r60 = frame.loc[first_2024, "adjusted_close"] / frame.loc[first_2024 - 60, "adjusted_close"] - 1
    assert output_2024.iloc[0]["R_60"] == pytest.approx(expected_r60)
    manifest = pd.read_json(report_dir / "manifest.json", typ="series")
    assert manifest["status"] == "complete"
    assert manifest["output_rows"] == len(frame)
    assert manifest["topix_supplied"] is False
    assert len(manifest["completed_partitions"]) == 2

    with pytest.raises(FileExistsError):
        run_descriptors(
            prices_dir=prices_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "other-reports",
        )
