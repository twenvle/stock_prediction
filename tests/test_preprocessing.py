"""Safety checks for J-Quants daily-price preprocessing.

The synthetic data uses the schema inspected in the repository's raw Parquet
files.  Raw files used in integration tests are temporary, reproducible inputs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.preprocessing import (
    PreprocessingConfig,
    SchemaError,
    preprocess_prices,
    run_preprocessing,
)


PRICE_COLUMNS = (
    "O", "H", "L", "C", "Vo", "Va", "AdjFactor", "AdjO", "AdjH",
    "AdjL", "AdjC", "AdjVo",
)


def make_prices(
    observations: list[tuple[str, str, float]],
) -> pd.DataFrame:
    """Build valid price rows from (date, code, close), without forward fills."""
    frame = pd.DataFrame(
        [
            {
                "Date": date,
                "Code": code,
                "O": close,
                "H": close * 1.05,
                "L": close * 0.95,
                "C": close,
                "UL": "0",
                "LL": "0",
                "Vo": 1_000.0,
                "Va": close * 1_000.0,
                "AdjFactor": 1.0,
                "AdjO": close,
                "AdjH": close * 1.05,
                "AdjL": close * 0.95,
                "AdjC": close,
                "AdjVo": 1_000.0,
            }
            for date, code, close in observations
        ],
    )
    frame["Date"] = pd.to_datetime(frame["Date"]).astype("datetime64[ns]")
    frame[["Code", "UL", "LL"]] = frame[["Code", "UL", "LL"]].astype("string")
    frame[list(PRICE_COLUMNS)] = frame[list(PRICE_COLUMNS)].astype("float64")
    return frame


@pytest.fixture
def prices() -> pd.DataFrame:
    """Unsorted prices distinguish each issuer's history and cross a year end."""
    return make_prices(
        [
            ("2025-01-07", "20000", 550.0),
            ("2024-12-30", "10000", 100.0),
            ("2025-01-06", "20000", 500.0),
            ("2025-01-06", "10000", 110.0),
            ("2024-12-30", "20000", 500.0),
            ("2025-01-07", "10000", 121.0),
        ],
    )


def sha256(path: Path) -> str:
    """Return a file's digest to verify input bytes remain untouched."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unverified_config(**kwargs: object) -> PreprocessingConfig:
    return PreprocessingConfig(allow_unverified_calendar=True, **kwargs)


def make_calendar(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    return pd.DataFrame(
        {"Date": dates, "HolDiv": pd.array(["1" if d.weekday() < 5 else "0" for d in dates], dtype="string")},
    )


def test_notebook_api_does_not_mutate_input_and_sorts_by_issuer(prices: pd.DataFrame) -> None:
    original = prices.copy(deep=True)
    result = preprocess_prices(prices, config=unverified_config())
    assert_frame_equal(prices, original)
    assert len(result.prices) == 6
    assert result.quarantine.empty
    assert result.prices[["code", "date"]].equals(
        result.prices[["code", "date"]].sort_values(["code", "date"]).reset_index(drop=True),
    )
    assert not result.prices["calendar_verified"].any()
    assert result.prices["has_price"].all()
    assert set(result.prices["research_period"]) == {"development", "final_backtest"}


@pytest.mark.parametrize("column", ["Date", "Code", "AdjC", "Va"])
def test_missing_required_column_fails_early(prices: pd.DataFrame, column: str) -> None:
    with pytest.raises(SchemaError):
        preprocess_prices(prices.drop(columns=column), config=unverified_config())


def test_known_optional_columns_are_validated_but_not_copied_to_base_output(prices: pd.DataFrame) -> None:
    extended = prices.copy(deep=True)
    extended["MO"] = extended["O"]
    extended["MUL"] = pd.Series(["0"] * len(extended), dtype="string")
    extended["MktCap"] = extended["C"] * 1_000.0
    extended["ExRT"] = pd.Series([None, "1", None, "2", None, "3"], dtype="string")
    result = preprocess_prices(extended, config=unverified_config())
    assert len(result.prices) == len(prices)
    assert {"MO", "MUL", "MktCap", "ExRT"}.isdisjoint(result.prices.columns)


def test_unknown_extra_daily_bar_column_fails_early(prices: pd.DataFrame) -> None:
    prices["FutureField"] = 1.0
    with pytest.raises(SchemaError):
        preprocess_prices(prices, config=unverified_config())


@pytest.mark.parametrize(
    "column,value",
    [("Code", 10000), ("C", "100"), ("Vo", "1000"), ("MktCap", "1000"), ("ExRT", 1)],
)
def test_wrong_schema_type_is_not_silently_coerced(prices: pd.DataFrame, column: str, value: object) -> None:
    prices[column] = value
    with pytest.raises(SchemaError):
        preprocess_prices(prices, config=unverified_config())


def test_invalid_ex_rights_type_fails_early(prices: pd.DataFrame) -> None:
    prices["ExRT"] = pd.Series(["4"] * len(prices), dtype="string")
    with pytest.raises(SchemaError):
        preprocess_prices(prices, config=unverified_config())


@pytest.mark.parametrize("column", ["C", "AdjC"])
def test_complex_price_dtype_is_rejected_without_losing_imaginary_values(column: str) -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0)])
    frame[column] = np.array([100.0 + 1.0j], dtype="complex128")
    original = frame.copy(deep=True)
    with pytest.raises(SchemaError):
        preprocess_prices(frame, config=unverified_config())
    assert_frame_equal(frame, original)


def test_all_duplicate_key_members_are_quarantined(prices: pd.DataFrame) -> None:
    duplicate = prices.iloc[[0]].copy()
    duplicate["C"] = duplicate["O"] = 549.0
    duplicate["AdjC"] = duplicate["AdjO"] = 549.0
    combined = pd.concat([prices, duplicate], ignore_index=True)
    result = preprocess_prices(combined, config=unverified_config())
    assert len(result.prices) == 5
    assert len(result.quarantine) == 2
    assert result.quarantine["issue_reasons"].str.contains("duplicate_key").all()
    assert result.report["issue_counts"]["duplicate_key"] == 2


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("H", 50.0, "ohlc_inconsistent"),
        ("AdjL", 150.0, "adjusted_ohlc_inconsistent"),
        ("C", 0.0, "zero_price"),
        ("O", -1.0, "negative_price"),
        ("Vo", -1.0, "negative_quantity"),
        ("Va", -1.0, "negative_quantity"),
        ("AdjVo", -1.0, "negative_quantity"),
        ("AdjFactor", 0.0, "non_positive_adjustment_factor"),
        ("AdjFactor", -1.0, "non_positive_adjustment_factor"),
        ("AdjC", float("inf"), "non_finite"),
        ("Va", float("-inf"), "non_finite"),
        ("UL", "9", "invalid_upper_limit_flag"),
        ("LL", "2", "invalid_lower_limit_flag"),
    ],
)
def test_invalid_values_are_preserved_in_quarantine(column: str, value: object, reason: str) -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0)])
    frame.loc[0, column] = value
    result = preprocess_prices(frame, config=unverified_config())
    assert result.prices.empty
    assert len(result.quarantine) == 1
    assert reason in result.quarantine.iloc[0]["issue_reasons"].split(";")
    assert result.report["issue_counts"][reason] == 1


def test_multiple_issue_reasons_are_not_lost() -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0)])
    frame.loc[0, "H"] = 1.0
    frame.loc[0, "Va"] = -1.0
    result = preprocess_prices(frame, config=unverified_config())
    reasons = set(result.quarantine.iloc[0]["issue_reasons"].split(";"))
    assert {"ohlc_inconsistent", "negative_quantity"} <= reasons


@pytest.mark.parametrize("code", ["130A0", "10001"])
def test_alphanumeric_and_nonordinary_share_codes_are_retained(code: str) -> None:
    result = preprocess_prices(make_prices([("2024-12-30", code, 100.0)]), config=unverified_config())
    assert result.prices["code"].tolist() == [code]


@pytest.mark.parametrize("code", ["1000", "abc00", "100000", " 10000", "10000 "])
def test_invalid_security_codes_are_not_padded_or_trimmed(code: str) -> None:
    result = preprocess_prices(make_prices([("2024-12-30", code, 100.0)]), config=unverified_config())
    assert result.prices.empty
    assert "invalid_code" in result.quarantine.iloc[0]["issue_reasons"]


def test_null_key_is_quarantined() -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0), ("2024-12-30", "20000", 200.0)])
    frame.loc[0, "Date"] = pd.NaT
    frame.loc[1, "Code"] = pd.NA
    result = preprocess_prices(frame, config=unverified_config())
    assert result.prices.empty
    assert len(result.quarantine) == 2
    assert result.report["issue_counts"]["invalid_date"] == 1
    assert result.report["issue_counts"]["invalid_code"] == 1


def test_missing_prices_are_kept_without_forward_filling() -> None:
    frame = make_prices(
        [("2024-12-26", "10000", 100.0), ("2024-12-27", "10000", 100.0), ("2024-12-30", "10000", 110.0)],
    )
    missing_columns = ["O", "H", "L", "C", "Vo", "Va", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]
    frame.loc[1, missing_columns] = np.nan
    result = preprocess_prices(frame, config=unverified_config())
    assert len(result.prices) == 3
    assert result.quarantine.empty
    missing = result.prices.iloc[1]
    assert not missing["has_price"]
    assert pd.isna(missing["adjusted_close"])
    assert pd.isna(missing["close"])
    assert pd.isna(missing["observed_adjusted_close_return"])
    assert pd.isna(result.prices.iloc[2]["observed_adjusted_close_return"])


def test_quarantined_price_observation_breaks_the_return_chain() -> None:
    frame = make_prices(
        [("2024-12-26", "10000", 100.0), ("2024-12-27", "10000", 100.0), ("2024-12-30", "10000", 110.0)],
    )
    frame.loc[1, "Va"] = -1.0
    result = preprocess_prices(frame, config=unverified_config())
    assert len(result.prices) == 2
    assert len(result.quarantine) == 1
    assert pd.isna(result.prices.iloc[1]["observed_adjusted_close_return"])
    assert result.prices.iloc[1]["previous_observation_date"] == pd.Timestamp("2024-12-27")


def test_duplicate_key_observation_breaks_the_return_chain() -> None:
    frame = make_prices(
        [("2024-12-26", "10000", 100.0), ("2024-12-27", "10000", 100.0), ("2024-12-27", "10000", 101.0), ("2024-12-30", "10000", 110.0)],
    )
    result = preprocess_prices(frame, config=unverified_config())
    assert len(result.prices) == 2
    assert len(result.quarantine) == 2
    assert pd.isna(result.prices.iloc[1]["observed_adjusted_close_return"])
    assert result.prices.iloc[1]["previous_observation_date"] == pd.Timestamp("2024-12-27")


def test_partial_ohlcv_missingness_is_quarantined() -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0)])
    frame.loc[0, "AdjC"] = np.nan
    result = preprocess_prices(frame, config=unverified_config())
    assert result.prices.empty
    assert "partial_missing_ohlcv" in result.quarantine.iloc[0]["issue_reasons"]


def test_zero_quantities_and_adjustment_events_do_not_change_supplied_prices() -> None:
    frame = make_prices([("2024-12-30", "10000", 100.0)])
    frame.loc[0, ["Vo", "Va", "AdjVo"]] = 0.0
    frame.loc[0, "AdjFactor"] = 0.5
    frame.loc[0, ["AdjO", "AdjH", "AdjL", "AdjC"]] *= 0.5
    result = preprocess_prices(frame, config=unverified_config())
    assert result.quarantine.empty
    assert result.prices.iloc[0]["is_adjustment_event"]
    assert result.prices.iloc[0]["adjusted_close"] == 50.0
    assert result.prices.iloc[0]["close"] == 100.0


def test_returns_respect_security_and_year_boundaries(prices: pd.DataFrame) -> None:
    result = preprocess_prices(prices, config=unverified_config())
    first = result.prices.groupby("code", sort=False).head(1)
    assert first["observed_adjusted_close_return"].isna().all()
    assert first["previous_observation_date"].isna().all()
    issuer = result.prices.loc[result.prices["code"].eq("10000")]
    np.testing.assert_allclose(issuer["observed_adjusted_close_return"].iloc[1:], [0.1, 0.1])
    assert issuer.iloc[1]["previous_observation_date"] == pd.Timestamp("2024-12-30")


def test_extreme_returns_are_flagged_without_filtering() -> None:
    frame = make_prices([("2024-12-27", "10000", 100.0), ("2024-12-30", "10000", 200.0)])
    result = preprocess_prices(frame, config=unverified_config(abnormal_return_threshold=0.5))
    assert len(result.prices) == 2
    assert result.quarantine.empty
    assert result.prices["is_large_return"].tolist() == [False, True]
    assert result.prices.iloc[1]["observed_adjusted_close_return"] == 1.0


def test_finite_prices_whose_return_overflows_are_kept_with_explicit_audit_flag() -> None:
    frame = make_prices([("2024-12-27", "10000", 1e-300), ("2024-12-30", "10000", 1e300)])
    assert np.isfinite(frame[list(PRICE_COLUMNS)].to_numpy()).all()
    result = preprocess_prices(frame, config=unverified_config())
    assert len(result.prices) == 2
    assert result.quarantine.empty
    assert result.prices["has_price"].all()
    assert result.prices["observed_adjusted_close_return"].isna().all()
    assert result.prices["is_non_finite_return"].tolist() == [False, True]
    assert result.prices["is_large_return"].tolist() == [False, False]
    assert result.report["non_finite_return_rows"] == 1


def test_future_changes_do_not_change_earlier_output(prices: pd.DataFrame) -> None:
    before = preprocess_prices(prices, config=unverified_config()).prices
    changed = prices.copy(deep=True)
    future = changed["Date"].gt("2025-01-06")
    changed.loc[future, ["O", "H", "L", "C", "AdjO", "AdjH", "AdjL", "AdjC", "Va"]] *= 2.0
    after = preprocess_prices(changed, config=unverified_config()).prices
    assert_frame_equal(before.loc[before["date"].le("2025-01-06")], after.loc[after["date"].le("2025-01-06")])


def test_calendar_is_required_unless_explicitly_unverified(prices: pd.DataFrame) -> None:
    with pytest.raises(SchemaError):
        preprocess_prices(prices)


def test_verified_calendar_observes_holiday_and_keeps_absent_rows_absent() -> None:
    calendar = make_calendar("2025-01-06", "2025-01-10")
    calendar.loc[calendar["Date"].eq("2025-01-06"), "HolDiv"] = "0"
    frame = make_prices([("2025-01-06", "10000", 100.0), ("2025-01-07", "10000", 110.0), ("2025-01-10", "10000", 121.0)])
    result = preprocess_prices(frame, trading_calendar=calendar)
    assert result.prices["date"].tolist() == [pd.Timestamp("2025-01-07"), pd.Timestamp("2025-01-10")]
    assert result.prices["calendar_verified"].all()
    assert "non_trading_day" in result.quarantine.iloc[0]["issue_reasons"]
    assert len(result.prices) == 2


@pytest.mark.parametrize("holiday_division", ["1", "2"])
def test_calendar_recognizes_both_open_day_codes(holiday_division: str) -> None:
    calendar = make_calendar("2025-01-07", "2025-01-07")
    calendar["HolDiv"] = holiday_division
    result = preprocess_prices(make_prices([("2025-01-07", "10000", 100.0)]), trading_calendar=calendar)
    assert len(result.prices) == 1
    assert result.quarantine.empty


def test_calendar_coverage_must_include_closed_days_too() -> None:
    frame = make_prices([("2025-01-03", "10000", 100.0), ("2025-01-06", "10000", 110.0)])
    calendar = make_calendar("2025-01-03", "2025-01-06").drop(index=1)
    with pytest.raises(SchemaError):
        preprocess_prices(frame, trading_calendar=calendar)


def test_calendar_cannot_silently_leave_dates_unverified() -> None:
    frame = make_prices([("2025-01-06", "10000", 100.0)])
    with pytest.raises(SchemaError):
        preprocess_prices(frame, config=unverified_config(), trading_calendar=make_calendar("2025-01-07", "2025-01-08"))


@pytest.mark.parametrize("invalidity", ["duplicate", "unsupported_code", "missing_column"])
def test_invalid_calendar_fails_before_price_processing(invalidity: str) -> None:
    calendar = make_calendar("2025-01-06", "2025-01-07")
    if invalidity == "duplicate":
        calendar = pd.concat([calendar, calendar.iloc[[0]]], ignore_index=True)
    elif invalidity == "unsupported_code":
        calendar.loc[0, "HolDiv"] = "9"
    else:
        calendar = calendar.drop(columns="HolDiv")
    with pytest.raises(SchemaError):
        preprocess_prices(make_prices([("2025-01-06", "10000", 100.0)]), trading_calendar=calendar)


def test_empty_input_can_be_preprocessed(prices: pd.DataFrame) -> None:
    result = preprocess_prices(prices.iloc[:0], config=unverified_config())
    assert result.prices.empty
    assert result.quarantine.empty


def test_report_accounts_for_selected_excluded_and_quarantined_rows(prices: pd.DataFrame) -> None:
    prices.loc[3, "Vo"] = -1.0
    result = preprocess_prices(prices, config=unverified_config(start_date="2025-01-06", end_date="2025-01-06"))
    assert result.report["input_rows"] == 6
    assert result.report["selected_rows"] == 2
    assert result.report["outside_date_range_rows"] == 4
    assert result.report["output_rows"] == 1
    assert result.report["quarantined_rows"] == 1
    assert result.report["input_date_min"] == "2024-12-30"
    assert result.report["input_date_max"] == "2025-01-07"


def test_previous_prices_supports_yearly_partition_return_calculation() -> None:
    first = preprocess_prices(make_prices([("2024-12-30", "10000", 100.0), ("2024-12-30", "20000", 500.0)]), config=unverified_config())
    second = preprocess_prices(
        make_prices([("2025-01-06", "10000", 110.0), ("2025-01-06", "20000", 550.0)]),
        config=unverified_config(),
        previous_prices=first.prices,
    )
    np.testing.assert_allclose(second.prices["observed_adjusted_close_return"], [0.1, 0.1])
    assert second.prices["previous_observation_date"].eq(pd.Timestamp("2024-12-30")).all()


def test_quarantined_year_end_observation_breaks_next_year_return_chain() -> None:
    first_frame = make_prices([("2024-12-27", "10000", 100.0), ("2024-12-30", "10000", 100.0)])
    first_frame.loc[1, "Vo"] = -1.0
    first = preprocess_prices(first_frame, config=unverified_config())
    prior = pd.concat([first.prices, first.quarantine], ignore_index=True)
    second = preprocess_prices(
        make_prices([("2025-01-06", "10000", 110.0)]),
        config=unverified_config(),
        previous_prices=prior,
    )
    assert pd.isna(second.prices.iloc[0]["observed_adjusted_close_return"])
    assert second.prices.iloc[0]["previous_observation_date"] == pd.Timestamp("2024-12-30")


def test_previous_prices_cannot_contain_future_or_same_day_observations() -> None:
    frame = make_prices([("2025-01-06", "10000", 100.0)])
    prior = preprocess_prices(frame, config=unverified_config()).prices
    with pytest.raises(SchemaError):
        preprocess_prices(frame, config=unverified_config(), previous_prices=prior)


def test_previous_prices_must_have_unique_keys() -> None:
    prior = preprocess_prices(make_prices([("2024-12-30", "10000", 100.0)]), config=unverified_config()).prices
    prior = pd.concat([prior, prior], ignore_index=True)
    with pytest.raises(SchemaError):
        preprocess_prices(make_prices([("2025-01-06", "10000", 110.0)]), config=unverified_config(), previous_prices=prior)


def write_partitioned_prices(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True)
    inputs = [raw_dir / "2024.parquet", raw_dir / "2025.parquet"]
    make_prices([("2024-12-30", "10000", 100.0), ("2024-12-30", "20000", 500.0)]).to_parquet(inputs[0], index=False)
    make_prices([("2025-01-06", "10000", 110.0), ("2025-01-06", "20000", 550.0)]).to_parquet(inputs[1], index=False)
    return inputs


def test_run_processes_partitions_preserves_input_bytes_and_records_provenance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    inputs = write_partitioned_prices(raw_dir)
    digests = {path: sha256(path) for path in inputs}
    output_dir = tmp_path / "interim"
    report_dir = tmp_path / "reports"
    manifest = run_preprocessing(raw_dir=raw_dir, output_dir=output_dir, report_dir=report_dir, config=unverified_config())
    assert isinstance(manifest, dict)
    assert json.loads((report_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    for year in (2024, 2025):
        assert (output_dir / f"{year}.parquet").is_file()
        assert (output_dir / "quarantine" / f"{year}.parquet").is_file()
        report = json.loads((report_dir / f"{year}.json").read_text(encoding="utf-8"))
        assert report["input_rows"] == report["output_rows"] == 2
        assert report["quarantined_rows"] == 0
        assert (report_dir / f"{year}_missing_by_date.parquet").is_file()
        assert (report_dir / f"{year}_missing_by_code.parquet").is_file()
    second = pd.read_parquet(output_dir / "2025.parquet")
    np.testing.assert_allclose(second["observed_adjusted_close_return"], [0.1, 0.1])
    assert {path: sha256(path) for path in inputs} == digests
    assert all(digest in json.dumps(manifest) for digest in digests.values())


def test_run_preflights_and_reads_arrow_date32_partition(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    frame = make_prices([("2024-12-27", "10000", 100.0), ("2024-12-30", "10000", 110.0)])
    table = pa.Table.from_pandas(frame, preserve_index=False)
    date_position = table.schema.get_field_index("Date")
    table = table.set_column(date_position, "Date", pa.array(frame["Date"].dt.date, type=pa.date32()))
    input_path = raw_dir / "2024.parquet"
    pq.write_table(table, input_path)
    digest = sha256(input_path)
    output_dir = tmp_path / "interim"
    manifest = run_preprocessing(raw_dir=raw_dir, output_dir=output_dir, report_dir=tmp_path / "reports", config=unverified_config())
    assert manifest["inputs"][0]["schema"]["Date"] == "date32[day]"
    assert manifest["inputs"][0]["date_min"] == "2024-12-27"
    assert manifest["inputs"][0]["date_max"] == "2024-12-30"
    output = pd.read_parquet(output_dir / "2024.parquet")
    assert_frame_equal(output, preprocess_prices(frame, config=unverified_config()).prices)
    assert sha256(input_path) == digest


def test_run_accepts_official_supplemental_raw_columns(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    frame = make_prices([("2025-01-06", "10000", 100.0)])
    frame["MktCap"] = pd.Series([1_000.0], dtype="float64")
    frame["ExRT"] = pd.Series([None], dtype="string")
    frame.to_parquet(raw_dir / "2025.parquet", index=False)
    output_dir = tmp_path / "interim"
    report_dir = tmp_path / "reports"
    manifest = run_preprocessing(
        raw_dir=raw_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        config=unverified_config(),
    )
    assert manifest["inputs"][0]["schema"]["MktCap"] == "double"
    assert manifest["inputs"][0]["schema"]["ExRT"] in {"string", "large_string"}
    output = pd.read_parquet(output_dir / "2025.parquet")
    assert {"MktCap", "ExRT"}.isdisjoint(output.columns)


@pytest.mark.parametrize("existing_directory", ["interim", "reports"])
def test_existing_output_is_never_overwritten(tmp_path: Path, existing_directory: str) -> None:
    raw_dir = tmp_path / "raw"
    write_partitioned_prices(raw_dir)
    output_dir = tmp_path / "interim"
    report_dir = tmp_path / "reports"
    protected_dir = tmp_path / existing_directory
    protected_dir.mkdir()
    sentinel = protected_dir / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_preprocessing(raw_dir=raw_dir, output_dir=output_dir, report_dir=report_dir, config=unverified_config())
    assert sentinel.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("target", ["raw", "raw/derived"])
def test_run_refuses_output_under_raw(tmp_path: Path, target: str) -> None:
    raw_dir = tmp_path / "raw"
    inputs = write_partitioned_prices(raw_dir)
    before = {path: sha256(path) for path in inputs}
    with pytest.raises((ValueError, FileExistsError)):
        run_preprocessing(raw_dir=raw_dir, output_dir=tmp_path / target, report_dir=tmp_path / "reports", config=unverified_config())
    assert {path: sha256(path) for path in inputs} == before


def test_run_refuses_report_output_under_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    inputs = write_partitioned_prices(raw_dir)
    before = {path: sha256(path) for path in inputs}
    with pytest.raises(ValueError):
        run_preprocessing(raw_dir=raw_dir, output_dir=tmp_path / "interim", report_dir=raw_dir / "reports", config=unverified_config())
    assert {path: sha256(path) for path in inputs} == before


def test_run_rejects_file_year_and_actual_data_year_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_prices([("2025-01-06", "10000", 100.0)]).to_parquet(raw_dir / "2024.parquet", index=False)
    with pytest.raises(SchemaError):
        run_preprocessing(raw_dir=raw_dir, output_dir=tmp_path / "interim", report_dir=tmp_path / "reports", config=unverified_config())


def test_run_does_not_assume_nonannual_parquet_is_price_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_partitioned_prices(raw_dir)
    pd.DataFrame({"metadata": [1]}).to_parquet(raw_dir / "trading_calendar.parquet", index=False)
    run_preprocessing(raw_dir=raw_dir, output_dir=tmp_path / "interim", report_dir=tmp_path / "reports", config=unverified_config())
    assert not (tmp_path / "interim" / "trading_calendar.parquet").exists()


@pytest.mark.parametrize("exception_class,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
def test_run_preserves_completed_partition_and_progress_on_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_class: type[BaseException],
    status: str,
) -> None:
    module = importlib.import_module("stock_prediction.preprocessing")
    original_preprocess = module.preprocess_prices
    raw_dir = tmp_path / "raw"
    inputs = write_partitioned_prices(raw_dir)
    before = {path: sha256(path) for path in inputs}

    def stop_second_partition(frame: pd.DataFrame, **kwargs: object) -> object:
        if frame["Date"].dt.year.eq(2025).any():
            raise exception_class("test interruption")
        return original_preprocess(frame, **kwargs)

    monkeypatch.setattr(module, "preprocess_prices", stop_second_partition)
    output_dir = tmp_path / "interim"
    report_dir = tmp_path / "reports"
    with pytest.raises(exception_class):
        run_preprocessing(raw_dir=raw_dir, output_dir=output_dir, report_dir=report_dir, config=unverified_config())
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == status
    assert manifest["error_type"] == exception_class.__name__
    assert [part["year"] for part in manifest["completed_partitions"]] == [2024]
    assert manifest["output_rows"] == 2
    assert (output_dir / "2024.parquet").is_file()
    assert (report_dir / "2024.json").is_file()
    assert not (output_dir / "2025.parquet").exists()
    assert {path: sha256(path) for path in inputs} == before


def test_cli_matches_notebook_run_function(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    inputs = write_partitioned_prices(raw_dir)
    hashes = {path: sha256(path) for path in inputs}
    notebook_output = tmp_path / "notebook_interim"
    run_preprocessing(raw_dir=raw_dir, output_dir=notebook_output, report_dir=tmp_path / "notebook_reports", config=unverified_config())
    cli_output = tmp_path / "cli_interim"
    cli_reports = tmp_path / "cli_reports"
    completed = subprocess.run(
        [sys.executable, "-m", "stock_prediction.preprocessing", "--raw-dir", str(raw_dir), "--output-dir", str(cli_output), "--report-dir", str(cli_reports), "--allow-unverified-calendar"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr
    for year in (2024, 2025):
        assert_frame_equal(pd.read_parquet(notebook_output / f"{year}.parquet"), pd.read_parquet(cli_output / f"{year}.parquet"))
    assert (cli_reports / "manifest.json").is_file()
    assert {path: sha256(path) for path in inputs} == hashes


def test_cli_carries_prior_issuer_observation_across_an_absent_year(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    partitions = {
        2023: make_prices([("2023-12-29", "10000", 100.0)]),
        2024: make_prices([("2024-12-30", "20000", 200.0)]),
        2025: make_prices([("2025-01-06", "10000", 110.0)]),
    }
    for year, frame in partitions.items():
        frame.to_parquet(raw_dir / f"{year}.parquet", index=False)
    all_rows = pd.concat(list(partitions.values()), ignore_index=True)
    expected = preprocess_prices(all_rows, config=unverified_config()).prices
    expected = expected.loc[expected["date"].dt.year.eq(2025)].reset_index(drop=True)
    output_dir = tmp_path / "cli_interim"
    completed = subprocess.run(
        [sys.executable, "-m", "stock_prediction.preprocessing", "--raw-dir", str(raw_dir), "--output-dir", str(output_dir), "--report-dir", str(tmp_path / "cli_reports"), "--allow-unverified-calendar"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr
    actual = pd.read_parquet(output_dir / "2025.parquet")
    assert actual.iloc[0]["previous_observation_date"] == pd.Timestamp("2023-12-29")
    assert actual.iloc[0]["observed_adjusted_close_return"] == pytest.approx(0.1)
    assert_frame_equal(actual, expected)
