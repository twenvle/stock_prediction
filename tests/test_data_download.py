"""Tests for reproducible J-Quants V2 raw-data acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stock_prediction.data_download import (
    ApiResponseError,
    DownloadConfig,
    DownloadSchemaError,
    JQuantsV2Client,
    PRICE_REQUIRED_COLUMNS,
    download_jquants_raw,
    resolve_api_key,
    validate_calendar,
    validate_daily_prices,
)
from stock_prediction.preprocessing import preprocess_prices


def make_calendar(start: str, end: str, open_dates: set[str]) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "HolDiv": ["1" if value.date().isoformat() in open_dates else "0" for value in dates],
        },
    )


def make_prices(trading_date: str, codes: tuple[str, ...] = ("10000", "20000")) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, code in enumerate(codes, start=1):
        close = float(position * 100)
        records.append(
            {
                "Date": trading_date,
                "Code": code,
                "O": close,
                "H": close + 5.0,
                "L": close - 5.0,
                "C": close,
                "UL": "0",
                "LL": "0",
                "Vo": 1_000.0,
                "Va": close * 1_000.0,
                "AdjFactor": 1.0,
                "AdjO": close,
                "AdjH": close + 5.0,
                "AdjL": close - 5.0,
                "AdjC": close,
                "AdjVo": 1_000.0,
            },
        )
    return pd.DataFrame.from_records(records)


class FakeClient:
    def __init__(self, calendar: pd.DataFrame, prices: dict[str, pd.DataFrame | BaseException]) -> None:
        self.calendar = calendar
        self.prices = prices
        self.calendar_calls: list[tuple[str, str]] = []
        self.price_calls: list[str] = []

    def get_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calendar_calls.append((start_date, end_date))
        return self.calendar.copy(deep=True)

    def get_daily_prices(self, trading_date: str) -> pd.DataFrame:
        self.price_calls.append(trading_date)
        result = self.prices[trading_date]
        if isinstance(result, BaseException):
            raise result
        return result.copy(deep=True)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


class NoopRateLimiter:
    def wait(self) -> None:
        pass


@pytest.mark.parametrize(
    "start,end",
    [("2025/01/01", "2025-01-02"), ("2025-02-30", "2025-03-01"), ("2025-01-02", "2025-01-01")],
)
def test_download_config_rejects_invalid_ranges(start: str, end: str) -> None:
    with pytest.raises(ValueError):
        DownloadConfig(start_date=start, end_date=end)


def test_resolve_api_key_uses_environment_without_logging_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", "environment-secret")
    assert resolve_api_key() == "environment-secret"


def test_client_follows_pagination_without_changing_query() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"data": [{"Date": "2025-01-06"}], "pagination_key": "next-page"}),
            FakeResponse(200, {"data": [{"Date": "2025-01-06"}]}),
        ],
    )
    client = JQuantsV2Client(
        "top-secret",
        session=session,  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),  # type: ignore[arg-type]
        max_retries=0,
    )
    result = client.get_daily_prices("2025-01-06")
    assert len(result) == 2
    assert session.calls[0]["params"] == {"date": "2025-01-06"}
    assert session.calls[1]["params"] == {"date": "2025-01-06", "pagination_key": "next-page"}
    assert all(call["headers"]["x-api-key"] == "top-secret" for call in session.calls)


def test_client_waits_and_retries_rate_limit_without_exposing_key() -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [
            FakeResponse(429, {"message": "slow down top-secret"}),
            FakeResponse(200, {"data": []}),
        ],
    )
    client = JQuantsV2Client(
        "top-secret",
        session=session,  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),  # type: ignore[arg-type]
        max_retries=1,
        rate_limit_wait_seconds=65,
        sleeper=sleeps.append,
    )
    assert client.get_daily_prices("2025-01-06").empty
    assert sleeps == [65]

    forbidden = JQuantsV2Client(
        "top-secret",
        session=FakeSession([FakeResponse(403, {"message": "invalid top-secret"})]),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),  # type: ignore[arg-type]
        max_retries=0,
    )
    with pytest.raises(ApiResponseError) as error:
        forbidden.get_daily_prices("2025-01-06")
    assert "top-secret" not in str(error.value)
    assert "***" in str(error.value)


def test_calendar_requires_every_day_and_known_string_divisions() -> None:
    calendar = make_calendar("2025-01-01", "2025-01-03", {"2025-01-02"})
    validated = validate_calendar(calendar, "2025-01-01", "2025-01-03")
    assert validated["Date"].tolist() == list(pd.date_range("2025-01-01", "2025-01-03"))
    with pytest.raises(DownloadSchemaError):
        validate_calendar(calendar.drop(index=1), "2025-01-01", "2025-01-03")
    invalid = calendar.copy()
    invalid["HolDiv"] = invalid["HolDiv"].astype(object)
    invalid.loc[0, "HolDiv"] = 1
    with pytest.raises(DownloadSchemaError):
        validate_calendar(invalid, "2025-01-01", "2025-01-03")


@pytest.mark.parametrize("change", ["missing", "unknown", "duplicate", "wrong_date", "numeric_code"])
def test_daily_price_schema_fails_early(change: str) -> None:
    prices = make_prices("2025-01-06")
    if change == "missing":
        prices = prices.drop(columns="AdjC")
    elif change == "unknown":
        prices["FutureField"] = 1
    elif change == "duplicate":
        prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    elif change == "wrong_date":
        prices.loc[0, "Date"] = "2025-01-07"
    else:
        prices["Code"] = [10000, 20000]
    with pytest.raises(DownloadSchemaError):
        validate_daily_prices(prices, "2025-01-06")


def test_all_null_known_optional_columns_keep_explicit_parquet_compatible_types() -> None:
    prices = make_prices("2025-01-06")
    prices["MAdjC"] = None
    prices["MUL"] = None
    prices["MktCap"] = None
    prices["ExRT"] = None
    validated = validate_daily_prices(prices, "2025-01-06")
    assert pd.api.types.is_float_dtype(validated["MAdjC"].dtype)
    assert pd.api.types.is_string_dtype(validated["MUL"].dtype)
    assert pd.api.types.is_float_dtype(validated["MktCap"].dtype)
    assert pd.api.types.is_string_dtype(validated["ExRT"].dtype)


def test_official_supplemental_columns_are_validated_and_preserved() -> None:
    prices = make_prices("2025-01-06")
    prices["MktCap"] = [1_000.0, None]
    prices["ExRT"] = pd.Series([None, "1"], dtype="string")
    validated = validate_daily_prices(prices, "2025-01-06")
    assert list(validated.columns)[-2:] == ["MktCap", "ExRT"]
    assert validated["MktCap"].tolist()[0] == 1_000.0
    assert pd.isna(validated.loc[1, "MktCap"])
    assert pd.isna(validated.loc[0, "ExRT"])
    assert validated.loc[1, "ExRT"] == "1"


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("MktCap", ["1000", "2000"]),
        ("ExRT", [1, None]),
        ("ExRT", ["4", None]),
    ],
)
def test_official_supplemental_columns_reject_invalid_types_or_values(
    column: str,
    values: list[object],
) -> None:
    prices = make_prices("2025-01-06")
    prices[column] = values
    with pytest.raises(DownloadSchemaError):
        validate_daily_prices(prices, "2025-01-06")


def test_download_saves_calendar_year_partitions_and_auditable_manifest(tmp_path: Path) -> None:
    open_dates = {"2024-12-30", "2025-01-02", "2025-01-03"}
    calendar = make_calendar("2024-12-30", "2025-01-03", open_dates)
    price_responses = {day: make_prices(day) for day in open_dates}
    for response in price_responses.values():
        response["MO"] = response["O"]
        response["MUL"] = pd.Series(["0"] * len(response), dtype="string")
        response["MktCap"] = response["C"] * 1_000.0
        response["ExRT"] = pd.Series([None] * len(response), dtype="string")
    client = FakeClient(calendar, price_responses)
    raw_root = tmp_path / "data" / "raw"
    output_dir = raw_root / "snapshot"
    config = DownloadConfig(
        start_date="2024-12-30",
        end_date="2025-01-03",
        output_root=raw_root,
        output_dir=output_dir,
    )
    result = download_jquants_raw(config, api_key="must-not-be-saved", client=client)

    assert result.output_dir == output_dir.resolve()
    assert [path.name for path in result.price_paths] == ["2024.parquet", "2025.parquet"]
    assert len(pd.read_parquet(result.calendar_path)) == 5
    assert len(pd.read_parquet(output_dir / "prices" / "2024.parquet")) == 2
    assert len(pd.read_parquet(output_dir / "prices" / "2025.parquet")) == 4
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "must-not-be-saved" not in manifest_text
    assert manifest["status"] == "completed"
    assert manifest["credential_persisted"] is False
    assert manifest["calendar"]["rows"] == 5
    assert manifest["calendar"]["tse_trading_days"] == 3
    assert manifest["totals"] == {
        "requested_trading_dates": 3,
        "completed_trading_dates": 3,
        "completed_years": 2,
        "price_rows": 6,
    }
    assert [part["rows"] for part in manifest["completed_partitions"]] == [2, 4]
    assert all(part["duplicate_keys"] == 0 for part in manifest["completed_partitions"])
    assert client.price_calls == sorted(open_dates)
    assert list(pd.read_parquet(result.price_paths[0]).columns) == [
        *PRICE_REQUIRED_COLUMNS,
        "MO",
        "MUL",
        "MktCap",
        "ExRT",
    ]
    preprocessed = preprocess_prices(
        pd.read_parquet(output_dir / "prices" / "2025.parquet"),
        trading_calendar=pd.read_parquet(result.calendar_path),
    )
    assert len(preprocessed.prices) == 4
    assert preprocessed.quarantine.empty


def test_interruption_preserves_current_rows_and_progress(tmp_path: Path) -> None:
    open_dates = {"2025-01-06", "2025-01-07"}
    client = FakeClient(
        make_calendar("2025-01-06", "2025-01-07", open_dates),
        {
            "2025-01-06": make_prices("2025-01-06"),
            "2025-01-07": KeyboardInterrupt("stop now"),
        },
    )
    raw_root = tmp_path / "raw"
    output_dir = raw_root / "interrupted"
    config = DownloadConfig(
        start_date="2025-01-06",
        end_date="2025-01-07",
        output_root=raw_root,
        output_dir=output_dir,
    )
    with pytest.raises(KeyboardInterrupt):
        download_jquants_raw(config, api_key="secret", client=client)
    partial = pd.read_parquet(output_dir / "partial_prices" / "2025.parquet")
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert len(partial) == 2
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["partial_partition"]["completed_trading_dates"] == 1
    assert manifest["partial_partition"]["last_completed_date"] == "2025-01-06"
    assert "secret" not in manifest_text


def test_output_must_be_new_and_below_raw_root(tmp_path: Path) -> None:
    calendar = make_calendar("2025-01-06", "2025-01-06", {"2025-01-06"})
    client = FakeClient(calendar, {"2025-01-06": make_prices("2025-01-06")})
    raw_root = tmp_path / "raw"
    outside = tmp_path / "outside"
    config = DownloadConfig(
        start_date="2025-01-06",
        end_date="2025-01-06",
        output_root=raw_root,
        output_dir=outside,
    )
    with pytest.raises(ValueError):
        download_jquants_raw(config, client=client)
    outside.mkdir()
    assert not any(outside.iterdir())

    existing = raw_root / "existing"
    existing.mkdir(parents=True)
    config = DownloadConfig(
        start_date="2025-01-06",
        end_date="2025-01-06",
        output_root=raw_root,
        output_dir=existing,
    )
    with pytest.raises(FileExistsError):
        download_jquants_raw(config, client=client)


def test_weekend_only_range_completes_without_price_calls(tmp_path: Path) -> None:
    calendar = make_calendar("2025-01-04", "2025-01-05", set())
    client = FakeClient(calendar, {})
    raw_root = tmp_path / "raw"
    config = DownloadConfig(
        start_date="2025-01-04",
        end_date="2025-01-05",
        output_root=raw_root,
        output_dir="weekend",
    )
    result = download_jquants_raw(config, client=client)
    assert result.output_dir == (raw_root / "weekend").resolve()
    assert result.price_paths == ()
    assert result.manifest["status"] == "completed"
    assert result.manifest["totals"]["requested_trading_dates"] == 0
    assert client.price_calls == []
