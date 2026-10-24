"""Download immutable J-Quants API V2 price and calendar snapshots.

The module is shared by the command-line interface and notebooks.  It stores
provider fields without filling, correcting, or selecting securities.  API
keys are accepted only at runtime and are never written to output metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
import requests


LOGGER = logging.getLogger(__name__)

API_BASE_URL = "https://api.jquants.com/v2"
CALENDAR_PATH = "/markets/calendar"
DAILY_PRICES_PATH = "/equities/bars/daily"
DEFAULT_REQUESTS_PER_MINUTE = 5.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 65.0
OPEN_HOLIDAY_DIVISIONS = frozenset({"1", "2"})
HOLIDAY_DIVISIONS = frozenset({"0", "1", "2", "3"})

PRICE_REQUIRED_COLUMNS = (
    "Date",
    "Code",
    "O",
    "H",
    "L",
    "C",
    "UL",
    "LL",
    "Vo",
    "Va",
    "AdjFactor",
    "AdjO",
    "AdjH",
    "AdjL",
    "AdjC",
    "AdjVo",
)
PRICE_OPTIONAL_COLUMNS = (
    "MO",
    "MH",
    "ML",
    "MC",
    "MUL",
    "MLL",
    "MVo",
    "MVa",
    "MAdjO",
    "MAdjH",
    "MAdjL",
    "MAdjC",
    "MAdjVo",
    "AO",
    "AH",
    "AL",
    "AC",
    "AUL",
    "ALL",
    "AVo",
    "AVa",
    "AAdjO",
    "AAdjH",
    "AAdjL",
    "AAdjC",
    "AAdjVo",
    "MktCap",
    "ExRT",
)
PRICE_FLAG_COLUMNS = frozenset({"UL", "LL", "MUL", "MLL", "AUL", "ALL"})
PRICE_EX_RIGHTS_COLUMNS = frozenset({"ExRT"})
EX_RIGHTS_TYPES = frozenset({"1", "2", "3"})
PRICE_NUMERIC_COLUMNS = tuple(
    column
    for column in (*PRICE_REQUIRED_COLUMNS, *PRICE_OPTIONAL_COLUMNS)
    if column not in {"Date", "Code", *PRICE_FLAG_COLUMNS, *PRICE_EX_RIGHTS_COLUMNS}
)
PRICE_ALLOWED_COLUMNS = frozenset((*PRICE_REQUIRED_COLUMNS, *PRICE_OPTIONAL_COLUMNS))
CALENDAR_COLUMNS = ("Date", "HolDiv")


class RawDownloadError(RuntimeError):
    """Base error for raw-data acquisition."""


class ApiResponseError(RawDownloadError):
    """J-Quants returned an HTTP or JSON response that cannot be used safely."""


class DownloadSchemaError(RawDownloadError):
    """An API response violates the expected J-Quants V2 schema."""


class RawDataClient(Protocol):
    """Small interface used by the downloader and synthetic tests."""

    def get_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return all calendar days in the inclusive range."""

    def get_daily_prices(self, trading_date: str) -> pd.DataFrame:
        """Return every available stock-price row for one TSE trading date."""


@dataclass(frozen=True)
class DownloadConfig:
    """Runtime settings for one immutable raw-data snapshot.

    Dates are inclusive Japanese market dates.  ``output_dir`` is optional; if
    omitted, a timestamped directory is created below ``output_root``.  An
    explicitly supplied output directory must also be below ``output_root`` and
    must not already exist.
    """

    start_date: str
    end_date: str
    output_root: Path | str = Path("data/raw")
    output_dir: Path | str | None = None
    requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    rate_limit_wait_seconds: float = DEFAULT_RATE_LIMIT_WAIT_SECONDS

    def __post_init__(self) -> None:
        start = _parse_iso_date(self.start_date, "start_date")
        end = _parse_iso_date(self.end_date, "end_date")
        if start > end:
            raise ValueError("start_date must not be later than end_date")
        for name in ("requests_per_minute", "timeout_seconds", "rate_limit_wait_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")


@dataclass(frozen=True)
class DownloadResult:
    """Paths and final manifest produced by a completed download."""

    output_dir: Path
    calendar_path: Path
    price_paths: tuple[Path, ...]
    manifest: dict[str, Any]


class RequestRateLimiter:
    """Space every HTTP request so pagination also counts against the limit."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0 or not math.isfinite(requests_per_minute):
            raise ValueError("requests_per_minute must be positive and finite")
        self._interval = 60.0 / requests_per_minute
        self._sleep = sleeper
        self._monotonic = monotonic
        self._next_allowed: float | None = None

    def wait(self) -> None:
        """Wait until the next evenly spaced request slot."""
        now = self._monotonic()
        if self._next_allowed is not None and now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._monotonic()
        self._next_allowed = max(now, self._next_allowed or now) + self._interval


class JQuantsV2Client:
    """Minimal J-Quants V2 client with pagination, throttling, and retries."""

    def __init__(
        self,
        api_key: str,
        *,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit_wait_seconds: float = DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("A non-empty J-Quants API key is required")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._rate_limit_wait_seconds = rate_limit_wait_seconds
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._sleep = sleeper
        self._rate_limiter = rate_limiter or RequestRateLimiter(
            requests_per_minute,
            sleeper=sleeper,
        )

    def close(self) -> None:
        """Close the internally owned HTTP session."""
        if self._owns_session:
            self._session.close()

    def get_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch the complete TSE/OSE calendar, including non-business days."""
        records = self._get_paginated(
            CALENDAR_PATH,
            {"from": start_date, "to": end_date},
        )
        return pd.DataFrame.from_records(records)

    def get_daily_prices(self, trading_date: str) -> pd.DataFrame:
        """Fetch all available TSE daily-price rows for one date."""
        records = self._get_paginated(DAILY_PRICES_PATH, {"date": trading_date})
        return pd.DataFrame.from_records(records)

    def _get_paginated(self, path: str, params: Mapping[str, str]) -> list[dict[str, Any]]:
        query = dict(params)
        records: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        while True:
            payload = self._request_json(path, query)
            data = payload.get("data")
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise ApiResponseError("J-Quants response field 'data' must be a list of objects")
            records.extend(data)
            pagination_key = payload.get("pagination_key")
            if pagination_key is None:
                return records
            if not isinstance(pagination_key, str) or not pagination_key:
                raise ApiResponseError("J-Quants pagination_key must be a non-empty string")
            if pagination_key in seen_keys:
                raise ApiResponseError("J-Quants returned a repeated pagination_key")
            seen_keys.add(pagination_key)
            query["pagination_key"] = pagination_key

    def _request_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        url = f"{API_BASE_URL}{path}"
        for attempt in range(self._max_retries + 1):
            self._rate_limiter.wait()
            try:
                response = self._session.get(
                    url,
                    params=dict(params),
                    headers={
                        "x-api-key": self._api_key,
                        "Accept": "application/json",
                        "User-Agent": "stock-prediction-raw-downloader/1",
                    },
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < self._max_retries:
                    self._sleep(min(2.0**attempt, 30.0))
                    continue
                raise ApiResponseError(f"J-Quants request failed: {_redact(str(exc), self._api_key)}") from exc

            status = response.status_code
            if status == 429 or 500 <= status < 600:
                if attempt < self._max_retries:
                    wait_seconds = self._rate_limit_wait_seconds if status == 429 else min(2.0**attempt, 30.0)
                    self._sleep(wait_seconds)
                    continue
            if not 200 <= status < 300:
                detail = _response_detail(response)
                raise ApiResponseError(
                    f"J-Quants API returned HTTP {status}: {_redact(detail, self._api_key)}",
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ApiResponseError("J-Quants API returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ApiResponseError("J-Quants API response must be a JSON object")
            return payload
        raise AssertionError("unreachable retry loop")


def resolve_api_key(api_key: str | None = None, *, prompt: bool = False) -> str:
    """Resolve a key from an argument, environment, or an optional hidden prompt."""
    candidate = api_key if api_key is not None else os.environ.get("JQUANTS_API_KEY")
    if candidate is None and prompt:
        candidate = getpass.getpass("J-Quants API key: ")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            "J-Quants API key is required; set JQUANTS_API_KEY or use the hidden CLI prompt",
        )
    return candidate.strip()


def download_jquants_raw(
    config: DownloadConfig,
    *,
    api_key: str | None = None,
    client: RawDataClient | None = None,
) -> DownloadResult:
    """Download and save one immutable raw-data snapshot.

    Parameters
    ----------
    config:
        Inclusive market-date range, output location, and request controls.
    api_key:
        J-Quants V2 API key.  If omitted, ``JQUANTS_API_KEY`` is used.  It is
        ignored when a test/dummy ``client`` is supplied and never persisted.
    client:
        Optional compatible client for tests or controlled notebook use.

    Returns
    -------
    DownloadResult
        Completed output paths and manifest.  Exceptions are re-raised after
        partial rows and failure/interruption metadata have been saved.
    """
    secret = ""
    owned_client: JQuantsV2Client | None = None
    if client is None:
        secret = resolve_api_key(api_key)
        owned_client = JQuantsV2Client(
            secret,
            requests_per_minute=float(config.requests_per_minute),
            timeout_seconds=float(config.timeout_seconds),
            max_retries=config.max_retries,
            rate_limit_wait_seconds=float(config.rate_limit_wait_seconds),
        )
        client = owned_client
    elif api_key:
        secret = api_key

    output_dir = _create_output_directory(config)
    manifest_path = output_dir / "manifest.json"
    started_at = _utc_now()
    manifest = _initial_manifest(config, output_dir, started_at)
    _write_manifest(manifest_path, manifest)

    current_year: int | None = None
    current_frames: list[pd.DataFrame] = []
    current_completed_dates: list[str] = []
    calendar_path = output_dir / "trading_calendar.parquet"
    price_paths: list[Path] = []

    try:
        raw_calendar = client.get_calendar(config.start_date, config.end_date)
        calendar = validate_calendar(raw_calendar, config.start_date, config.end_date)
        _write_parquet_once(calendar, calendar_path)
        calendar_stats = _calendar_stats(calendar)
        calendar_stats["path"] = calendar_path.relative_to(output_dir).as_posix()
        calendar_stats["sha256"] = _sha256(calendar_path)
        manifest["calendar"] = calendar_stats

        open_dates = calendar.loc[
            calendar["HolDiv"].isin(OPEN_HOLIDAY_DIVISIONS),
            "Date",
        ].tolist()
        dates_by_year: dict[int, list[pd.Timestamp]] = {}
        for trading_date in open_dates:
            dates_by_year.setdefault(int(trading_date.year), []).append(trading_date)

        manifest["totals"]["requested_trading_dates"] = len(open_dates)
        _touch_manifest(manifest_path, manifest)

        for year, trading_dates in dates_by_year.items():
            current_year = year
            current_frames = []
            current_completed_dates = []
            manifest["current_partition"] = {
                "year": year,
                "requested_trading_dates": len(trading_dates),
                "completed_trading_dates": 0,
                "last_completed_date": None,
            }
            _touch_manifest(manifest_path, manifest)

            for position, trading_timestamp in enumerate(trading_dates, start=1):
                trading_date = trading_timestamp.date().isoformat()
                raw_prices = client.get_daily_prices(trading_date)
                prices = validate_daily_prices(raw_prices, trading_date)
                current_frames.append(prices)
                current_completed_dates.append(trading_date)
                manifest["current_partition"]["completed_trading_dates"] = position
                manifest["current_partition"]["last_completed_date"] = trading_date
                manifest["totals"]["completed_trading_dates"] += 1
                _touch_manifest(manifest_path, manifest)
                if position % 10 == 0 or position == len(trading_dates):
                    LOGGER.info("Downloaded %s: %s/%s trading dates", year, position, len(trading_dates))

            year_frame = _combine_price_frames(current_frames)
            year_path = output_dir / "prices" / f"{year}.parquet"
            _write_parquet_once(year_frame, year_path)
            price_paths.append(year_path)
            partition = _price_stats(year_frame)
            partition.update(
                {
                    "year": year,
                    "requested_trading_dates": len(trading_dates),
                    "completed_trading_dates": len(current_completed_dates),
                    "path": year_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(year_path),
                },
            )
            manifest["completed_partitions"].append(partition)
            manifest["totals"]["price_rows"] += len(year_frame)
            manifest["current_partition"] = None
            _touch_manifest(manifest_path, manifest)
            current_year = None
            current_frames = []
            current_completed_dates = []

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_iso(_utc_now())
        manifest["totals"]["completed_years"] = len(manifest["completed_partitions"])
        _touch_manifest(manifest_path, manifest)
        return DownloadResult(
            output_dir=output_dir,
            calendar_path=calendar_path,
            price_paths=tuple(price_paths),
            manifest=manifest,
        )
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = _redact(str(exc), secret)
        if current_year is not None and current_frames:
            try:
                partial_frame = _combine_price_frames(current_frames)
                partial_path = output_dir / "partial_prices" / f"{current_year}.parquet"
                _write_parquet_once(partial_frame, partial_path)
                manifest["partial_partition"] = {
                    "year": current_year,
                    "completed_trading_dates": len(current_completed_dates),
                    "last_completed_date": current_completed_dates[-1],
                    "path": partial_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(partial_path),
                    **_price_stats(partial_frame),
                }
            except Exception as partial_exc:
                manifest["partial_write_error"] = _redact(str(partial_exc), secret)
        manifest["completed_at_utc"] = _utc_iso(_utc_now())
        _touch_manifest(manifest_path, manifest)
        raise
    finally:
        if owned_client is not None:
            owned_client.close()


def validate_calendar(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Validate exact calendar schema, values, uniqueness, and full-day coverage."""
    if not isinstance(frame, pd.DataFrame):
        raise DownloadSchemaError("Trading calendar response must be a pandas DataFrame")
    actual = set(frame.columns)
    expected = set(CALENDAR_COLUMNS)
    if actual != expected:
        raise DownloadSchemaError(
            f"Trading calendar columns must be exactly {list(CALENDAR_COLUMNS)}; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}",
        )
    output = frame.loc[:, CALENDAR_COLUMNS].copy(deep=True)
    output["Date"] = _parse_date_series(output["Date"], "calendar.Date")
    _require_strings(output["HolDiv"], "calendar.HolDiv")
    if output["Date"].isna().any() or output["HolDiv"].isna().any():
        raise DownloadSchemaError("Trading calendar Date and HolDiv must not be null")
    invalid_divisions = set(output["HolDiv"]) - HOLIDAY_DIVISIONS
    if invalid_divisions:
        raise DownloadSchemaError(f"Unsupported HolDiv values: {sorted(invalid_divisions)}")
    output["HolDiv"] = output["HolDiv"].astype("string")
    if output["Date"].duplicated().any():
        raise DownloadSchemaError("Trading calendar contains duplicate dates")
    output.sort_values("Date", inplace=True, ignore_index=True)
    expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    same_dates = len(output) == len(expected_dates) and np.array_equal(
        output["Date"].to_numpy(dtype="datetime64[D]"),
        expected_dates.to_numpy(dtype="datetime64[D]"),
    )
    if not same_dates:
        actual_dates = pd.DatetimeIndex(output["Date"])
        missing = expected_dates.difference(actual_dates)
        outside = actual_dates.difference(expected_dates)
        raise DownloadSchemaError(
            "Trading calendar must contain every requested calendar day exactly once; "
            f"missing={list(missing.strftime('%Y-%m-%d'))[:10]}, "
            f"outside={list(outside.strftime('%Y-%m-%d'))[:10]}",
        )
    return output


def validate_daily_prices(frame: pd.DataFrame, requested_date: str) -> pd.DataFrame:
    """Validate one date of V2 daily bars without changing provider values."""
    if not isinstance(frame, pd.DataFrame):
        raise DownloadSchemaError("Daily-price response must be a pandas DataFrame")
    if frame.empty:
        raise DownloadSchemaError(
            f"No daily-price rows returned for trading date {requested_date}; "
            "check the subscription's available history",
        )
    actual = set(frame.columns)
    required = set(PRICE_REQUIRED_COLUMNS)
    missing = required - actual
    unexpected = actual - PRICE_ALLOWED_COLUMNS
    if missing or unexpected:
        raise DownloadSchemaError(
            "Unexpected daily-price schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )
    ordered_columns = [
        column
        for column in (*PRICE_REQUIRED_COLUMNS, *PRICE_OPTIONAL_COLUMNS)
        if column in frame.columns
    ]
    output = frame.loc[:, ordered_columns].copy(deep=True)
    output["Date"] = _parse_date_series(output["Date"], "prices.Date")
    if output["Date"].isna().any():
        raise DownloadSchemaError("prices.Date must not be null or invalid")
    expected_date = pd.Timestamp(requested_date)
    if not output["Date"].eq(expected_date).all():
        actual_dates = sorted(output["Date"].dt.strftime("%Y-%m-%d").unique().tolist())
        raise DownloadSchemaError(
            f"Daily-price response for {requested_date} contains dates {actual_dates}",
        )
    _require_strings(output["Code"], "prices.Code")
    if output["Code"].isna().any() or not output["Code"].str.fullmatch(r"[0-9A-Z]{5}").all():
        raise DownloadSchemaError("prices.Code must contain non-null five-character uppercase codes")
    output["Code"] = output["Code"].astype("string")
    for column in PRICE_FLAG_COLUMNS.intersection(output.columns):
        _require_strings(output[column], f"prices.{column}")
        invalid = set(output[column].dropna()) - {"0", "1"}
        if invalid:
            raise DownloadSchemaError(f"prices.{column} contains unsupported flags: {sorted(invalid)}")
        output[column] = output[column].astype("string")
    if "ExRT" in output.columns:
        _require_strings(output["ExRT"], "prices.ExRT")
        invalid = set(output["ExRT"].dropna()) - EX_RIGHTS_TYPES
        if invalid:
            raise DownloadSchemaError(
                f"prices.ExRT contains unsupported ex-rights types: {sorted(invalid)}",
            )
        output["ExRT"] = output["ExRT"].astype("string")
    for column in set(PRICE_NUMERIC_COLUMNS).intersection(output.columns):
        values = output[column].dropna()
        if not values.empty and not is_numeric_dtype(values.dtype):
            raise DownloadSchemaError(f"prices.{column} must contain numbers or nulls")
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise DownloadSchemaError(f"prices.{column} must not contain booleans")
        output[column] = output[column].astype("float64")
    if output.duplicated(["Date", "Code"], keep=False).any():
        raise DownloadSchemaError(f"Daily-price response for {requested_date} contains duplicate keys")
    output.sort_values(["Code", "Date"], inplace=True, ignore_index=True)
    return output


def _combine_price_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=PRICE_REQUIRED_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.duplicated(["Date", "Code"], keep=False).any():
        raise DownloadSchemaError("Combined price partition contains duplicate Date/Code keys")
    ordered = [
        column
        for column in (*PRICE_REQUIRED_COLUMNS, *PRICE_OPTIONAL_COLUMNS)
        if column in combined.columns
    ]
    combined = combined.loc[:, ordered]
    combined.sort_values(["Code", "Date"], inplace=True, ignore_index=True)
    return combined


def _initial_manifest(
    config: DownloadConfig,
    output_dir: Path,
    started_at: datetime,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": 1,
        "snapshot_id": output_dir.name,
        "status": "running",
        "started_at_utc": _utc_iso(started_at),
        "updated_at_utc": _utc_iso(started_at),
        "completed_at_utc": None,
        "provider": "J-Quants API",
        "api_version": "v2",
        "api_base_url": API_BASE_URL,
        "endpoints": {
            "trading_calendar": CALENDAR_PATH,
            "daily_prices": DAILY_PRICES_PATH,
        },
        "request": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "requests_per_minute": float(config.requests_per_minute),
            "timeout_seconds": float(config.timeout_seconds),
            "max_retries": config.max_retries,
            "rate_limit_wait_seconds": float(config.rate_limit_wait_seconds),
        },
        "credential_persisted": False,
        "calendar": None,
        "completed_partitions": [],
        "current_partition": None,
        "totals": {
            "requested_trading_dates": 0,
            "completed_trading_dates": 0,
            "completed_years": 0,
            "price_rows": 0,
        },
        "software": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "pyarrow": _package_version("pyarrow"),
            "requests": _package_version("requests"),
            "module_sha256": _sha256(Path(__file__)),
        },
    }


def _calendar_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "date_min": _date_or_none(frame["Date"].min()),
        "date_max": _date_or_none(frame["Date"].max()),
        "duplicate_dates": int(frame["Date"].duplicated(keep=False).sum()),
        "tse_trading_days": int(frame["HolDiv"].isin(OPEN_HOLIDAY_DIVISIONS).sum()),
        "holiday_division_counts": {
            str(key): int(value)
            for key, value in frame["HolDiv"].value_counts(dropna=False).sort_index().items()
        },
        "missing_rate": {
            column: float(value) for column, value in frame.isna().mean().items()
        },
    }


def _price_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "securities": int(frame["Code"].nunique(dropna=True)) if "Code" in frame else 0,
        "date_min": _date_or_none(frame["Date"].min()) if "Date" in frame and not frame.empty else None,
        "date_max": _date_or_none(frame["Date"].max()) if "Date" in frame and not frame.empty else None,
        "duplicate_keys": int(frame.duplicated(["Date", "Code"], keep=False).sum()) if not frame.empty else 0,
        "columns": list(frame.columns),
        "missing_rate": {
            column: float(value) for column, value in frame.isna().mean().items()
        },
    }


def _create_output_directory(config: DownloadConfig) -> Path:
    output_root = Path(config.output_root).expanduser().resolve()
    if config.output_dir is None:
        snapshot_name = f"jquants_v2_{_utc_now().strftime('%Y%m%dT%H%M%S%fZ')}"
        output_dir = output_root / snapshot_name
    else:
        supplied = Path(config.output_dir).expanduser()
        output_dir = supplied.resolve() if supplied.is_absolute() else (output_root / supplied).resolve()
    if output_dir == output_root or not output_dir.is_relative_to(output_root):
        raise ValueError("output_dir must be a new directory below output_root")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_parquet_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite raw data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _touch_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _utc_iso(_utc_now())
    _write_manifest(path, manifest)


def _parse_iso_date(value: str, name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid calendar date") from exc


def _parse_date_series(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DownloadSchemaError(f"{name} contains invalid dates") from exc
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    if not is_datetime64_any_dtype(parsed.dtype):
        raise DownloadSchemaError(f"{name} must contain dates")
    if not parsed.dt.normalize().equals(parsed):
        raise DownloadSchemaError(f"{name} must not contain times")
    return parsed.astype("datetime64[ns]")


def _require_strings(series: pd.Series, name: str) -> None:
    if not all(isinstance(value, str) for value in series.dropna()):
        raise DownloadSchemaError(f"{name} must contain strings without numeric coercion")


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "message" in payload:
            return str(payload["message"])
    except ValueError:
        pass
    return response.text[:1000] or "no response detail"


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_or_none(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download J-Quants V2 daily prices and trading calendar into a new raw snapshot.",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive market date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive market date (YYYY-MM-DD).")
    parser.add_argument("--output-root", default="data/raw", help="Raw-data root (default: data/raw).")
    parser.add_argument(
        "--output-dir",
        help="Optional new directory below output-root; it must not already exist.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Rate limit including pagination requests (default: 5, Free-plan safe).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        help="Wait before retrying HTTP 429 (default: 65).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prompt for the API key without echo when needed."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = DownloadConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root,
            output_dir=args.output_dir,
            requests_per_minute=args.requests_per_minute,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            rate_limit_wait_seconds=args.rate_limit_wait_seconds,
        )
        api_key = resolve_api_key(prompt=True)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        result = download_jquants_raw(config, api_key=api_key)
    except (ValueError, RawDownloadError, FileExistsError) as exc:
        parser.error(str(exc))
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
