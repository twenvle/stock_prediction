"""Validate J-Quants V2 daily bars without filling prices or selecting a universe.

The output is an observed-row intermediate dataset, not a training dataset.
Provider-adjusted prices are retained as supplied; their historical absolute
levels are not certified point-in-time. Returns here are quality diagnostics,
not labels or features. See docs/DATA_DICTIONARY.md for units and limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_numeric_dtype
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_START = pd.Timestamp("2016-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
FINAL_START = pd.Timestamp("2025-01-01")
FINAL_END = pd.Timestamp("2026-12-31")
DEFAULT_ABNORMAL_RETURN_THRESHOLD = 0.5
SCHEMA_ID = "jquants-v2-daily-bars-base-16-with-known-optional-columns"
RENAME_COLUMNS = {
    "Date": "date", "Code": "code", "O": "open", "H": "high",
    "L": "low", "C": "close", "UL": "upper_limit", "LL": "lower_limit",
    "Vo": "volume", "Va": "turnover_value", "AdjFactor": "adjustment_factor",
    "AdjO": "adjusted_open", "AdjH": "adjusted_high", "AdjL": "adjusted_low",
    "AdjC": "adjusted_close", "AdjVo": "adjusted_volume",
}
RAW_COLUMNS = tuple(RENAME_COLUMNS)
KNOWN_OPTIONAL_RAW_COLUMNS = (
    "MO", "MH", "ML", "MC", "MUL", "MLL", "MVo", "MVa",
    "MAdjO", "MAdjH", "MAdjL", "MAdjC", "MAdjVo",
    "AO", "AH", "AL", "AC", "AUL", "ALL", "AVo", "AVa",
    "AAdjO", "AAdjH", "AAdjL", "AAdjC", "AAdjVo",
    "MktCap", "ExRT",
)
OPTIONAL_RAW_FLAG_COLUMNS = ("MUL", "MLL", "AUL", "ALL")
OPTIONAL_RAW_ENUM_COLUMNS = ("ExRT",)
EX_RIGHTS_TYPES = frozenset({"1", "2", "3"})
OPTIONAL_RAW_NUMERIC_COLUMNS = tuple(
    column
    for column in KNOWN_OPTIONAL_RAW_COLUMNS
    if column not in (*OPTIONAL_RAW_FLAG_COLUMNS, *OPTIONAL_RAW_ENUM_COLUMNS)
)
CANONICAL_COLUMNS = tuple(RENAME_COLUMNS.values())
RAW_NUMERIC_COLUMNS = tuple(c for c in RAW_COLUMNS if c not in ("Date", "Code", "UL", "LL"))
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close")
QUANTITY_COLUMNS = ("volume", "turnover_value", "adjusted_volume")
OBSERVATION_COLUMNS = (*PRICE_COLUMNS, *QUANTITY_COLUMNS)
NUMERIC_COLUMNS = (*OBSERVATION_COLUMNS, "adjustment_factor")
OPEN_HOLIDAY_DIVISIONS = ("1", "2")
HOLIDAY_DIVISIONS = ("0", "1", "2", "3")


class SchemaError(ValueError):
    """The supplied schema, calendar coverage or partition layout is unsupported."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Inclusive date selection and an audit-only return threshold.

    Calendar-free processing must be explicitly enabled. It never establishes
    trading-day eligibility. No setting fits a statistic to the final period.
    """

    start_date: str | None = None
    end_date: str | None = None
    abnormal_return_threshold: float = DEFAULT_ABNORMAL_RETURN_THRESHOLD
    allow_unverified_calendar: bool = False

    def __post_init__(self) -> None:
        for name in ("start_date", "end_date"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError(f"{name} must be an ISO date YYYY-MM-DD")
                pd.Timestamp(value)  # Reject nonexistent dates.
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date")
        threshold = self.abnormal_return_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("abnormal_return_threshold must be positive and finite")
        if not isinstance(self.allow_unverified_calendar, bool):
            raise ValueError("allow_unverified_calendar must be boolean")


@dataclass
class PreprocessingResult:
    """Accepted observations, rejected observations, and input-quality reports."""

    prices: pd.DataFrame
    quarantine: pd.DataFrame
    report: dict[str, Any]
    missing_by_date: pd.DataFrame
    missing_by_code: pd.DataFrame


def load_config(path: Path | str) -> PreprocessingConfig:
    """Read a JSON configuration; fail on unknown fields rather than ignore them."""
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(PreprocessingConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return PreprocessingConfig(**values)


def _require_string_values(series: pd.Series, name: str) -> None:
    if not all(isinstance(value, str) for value in series.dropna()):
        raise SchemaError(f"{name} must contain strings, not coerced numeric codes/flags")


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        return series.astype("datetime64[ns]")
    _require_string_values(series, name)
    # Only documented ISO dates, not ambiguous locale formats or intraday times.
    valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    return pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce").astype("datetime64[ns]")


def _normalize(prices: pd.DataFrame) -> pd.DataFrame:
    if not prices.columns.is_unique:
        raise SchemaError("Column names must be unique")
    missing = set(RAW_COLUMNS) - set(prices.columns)
    extra = set(prices.columns) - set(RAW_COLUMNS) - set(KNOWN_OPTIONAL_RAW_COLUMNS)
    if missing or extra:
        raise SchemaError(f"Unsupported daily-bars columns; missing={sorted(missing)}, extra={sorted(extra)}")
    for name in set(OPTIONAL_RAW_NUMERIC_COLUMNS).intersection(prices.columns):
        if not (is_float_dtype(prices[name].dtype) or is_integer_dtype(prices[name].dtype)) or is_bool_dtype(prices[name].dtype):
            raise SchemaError(f"{name} must be numeric (null allowed), got {prices[name].dtype}")
    for name in RAW_NUMERIC_COLUMNS:
        if not (is_float_dtype(prices[name].dtype) or is_integer_dtype(prices[name].dtype)) or is_bool_dtype(prices[name].dtype):
            raise SchemaError(f"{name} must be numeric (null allowed), got {prices[name].dtype}")
    optional_strings = set((*OPTIONAL_RAW_FLAG_COLUMNS, *OPTIONAL_RAW_ENUM_COLUMNS)).intersection(prices.columns)
    for name in ("Code", "UL", "LL", *optional_strings):
        _require_string_values(prices[name], name)
    if "ExRT" in prices.columns:
        invalid_ex_rights = set(prices["ExRT"].dropna()) - EX_RIGHTS_TYPES
        if invalid_ex_rights:
            raise SchemaError(f"ExRT contains unsupported values: {sorted(invalid_ex_rights)}")
    result = prices.loc[:, list(RAW_COLUMNS)].rename(columns=RENAME_COLUMNS).copy().reset_index(drop=True)
    result["date"] = _parse_dates(prices["Date"].reset_index(drop=True), "Date")
    for name in ("code", "upper_limit", "lower_limit"):
        result[name] = result[name].astype("string")
    result[list(NUMERIC_COLUMNS)] = result[list(NUMERIC_COLUMNS)].astype("float64")
    return result


def validate_trading_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Validate a V2 Date/HolDiv calendar, including closed calendar dates.

    HolDiv 1/2 are TSE sessions; 0/3 are closed for cash equities. Missing
    calendar dates are not guessed from weekdays or equity observations.
    """
    if not calendar.columns.is_unique or set(calendar.columns) != {"Date", "HolDiv"}:
        raise SchemaError("Calendar columns must be exactly Date and HolDiv")
    _require_string_values(calendar["HolDiv"], "HolDiv")
    dates = _parse_dates(calendar["Date"].reset_index(drop=True), "calendar Date")
    if dates.isna().any() or (dates != dates.dt.normalize()).any() or dates.duplicated().any():
        raise SchemaError("Calendar dates must be unique, valid midnight dates")
    divisions = calendar["HolDiv"].reset_index(drop=True).astype("string")
    if not divisions.isin(HOLIDAY_DIVISIONS).all():
        raise SchemaError("HolDiv must be one of the string values 0, 1, 2, 3")
    result = pd.DataFrame({"date": dates, "is_session": divisions.isin(OPEN_HOLIDAY_DIVISIONS)})
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty or len(result) != len(pd.date_range(result.date.min(), result.date.max(), freq="D")):
        raise SchemaError("Calendar must cover every calendar date, including holidays")
    return result


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _missingness(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = frame.groupby(key, dropna=False, sort=True)
    rates = frame.loc[:, list(CANONICAL_COLUMNS)].isna().groupby(frame[key], dropna=False, sort=True).mean()
    rates.columns = [f"{name}_missing_rate" for name in rates.columns]
    return grouped.size().rename("row_count").to_frame().join(rates).reset_index()


def _quality_returns(frame: pd.DataFrame, previous_prices: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["date", "code", "adjusted_close", "has_price"]
    context = frame.loc[:, columns].copy()
    context["_current_index"] = np.arange(len(context))
    if previous_prices is not None and not previous_prices.empty:
        if not set(columns).issubset(previous_prices.columns):
            raise SchemaError(f"previous_prices requires canonical columns {columns}")
        previous = previous_prices.loc[:, columns].copy()
        previous["date"] = _parse_dates(previous["date"], "previous date")
        _require_string_values(previous["code"], "previous code")
        if previous["date"].isna().any() or previous["code"].isna().any() or previous.duplicated(["date", "code"]).any():
            raise SchemaError("previous_prices requires valid unique keys")
        if previous.date.ne(previous.date.dt.normalize()).any() or not previous.code.str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
            raise SchemaError("previous_prices requires valid midnight dates and five-character codes")
        if not frame.empty and (previous["date"] >= frame["date"].min()).any():
            raise SchemaError("previous_prices must precede every current observation")
        if not is_bool_dtype(previous["has_price"].dtype) or previous["has_price"].isna().any():
            raise SchemaError("previous_prices has_price must be non-null boolean")
        if not is_numeric_dtype(previous["adjusted_close"].dtype):
            raise SchemaError("previous_prices adjusted_close must be numeric")
        known_close = previous.loc[previous.has_price, "adjusted_close"]
        if not (known_close.notna() & np.isfinite(known_close) & known_close.gt(0)).all():
            raise SchemaError("previous_prices with has_price=True requires positive finite adjusted_close")
        previous["_current_index"] = -1
        context = pd.concat([previous, context], ignore_index=True)
    context = context.sort_values(["code", "date"], kind="stable")
    groups = context.groupby("code", sort=False)
    prior_close = groups["adjusted_close"].shift(1)
    prior_has_price = groups["has_price"].shift(1).eq(True)
    context["previous_observation_date"] = groups["date"].shift(1)
    eligible = context["has_price"] & prior_has_price & prior_close.gt(0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        returns = context["adjusted_close"] / prior_close - 1
    context["observed_adjusted_close_return"] = returns.where(eligible & np.isfinite(returns))
    context["is_non_finite_return"] = eligible & ~np.isfinite(returns)
    current = context.loc[context["_current_index"] >= 0].sort_values("_current_index")
    frame["previous_observation_date"] = current["previous_observation_date"].to_numpy()
    frame["observed_adjusted_close_return"] = current["observed_adjusted_close_return"].to_numpy()
    frame["is_non_finite_return"] = current["is_non_finite_return"].to_numpy()
    return frame


def preprocess_prices(
    prices: pd.DataFrame,
    *,
    config: PreprocessingConfig | None = None,
    trading_calendar: pd.DataFrame | None = None,
    previous_prices: pd.DataFrame | None = None,
) -> PreprocessingResult:
    """Normalize observed daily bars and quarantine invalid rows without repair.

    Input dates are Japanese session dates, prices/turnover are JPY, and volume
    retains its provider unit. All-null OHLCV rows are preserved as unavailable
    prices. Duplicate keys are all rejected. No calendar rows, stock universe,
    forward-filled prices, scalers, labels or features are constructed.

    Raises SchemaError on unexpected columns/types, insufficient calendar
    coverage, missing calendar permission, or non-past previous_prices.
    """
    config = config or PreprocessingConfig()
    frame = _normalize(prices)
    input_rows = len(frame)
    minimum, maximum = frame.date.min(), frame.date.max()
    input_codes = int(frame.code.nunique())
    # Keep invalid dates for quarantine; valid rows outside the requested range
    # are counted explicitly instead of silently disappearing.
    in_range = pd.Series(True, index=frame.index)
    if config.start_date:
        in_range &= frame.date.isna() | frame.date.ge(pd.Timestamp(config.start_date))
    if config.end_date:
        in_range &= frame.date.isna() | frame.date.le(pd.Timestamp(config.end_date))
    frame = frame.loc[in_range].reset_index(drop=True)
    issues: dict[str, pd.Series] = {}

    def flag(name: str, mask: pd.Series) -> None:
        issues[name] = mask.fillna(False).astype(bool)

    flag("invalid_date", frame.date.isna() | frame.date.ne(frame.date.dt.normalize()))
    flag("invalid_code", ~frame.code.str.fullmatch(r"[0-9A-Z]{5}", na=False))
    flag("duplicate_key", frame.duplicated(["date", "code"], keep=False))
    for column in ("upper_limit", "lower_limit"):
        flag(f"invalid_{column}_flag", frame[column].notna() & ~frame[column].isin(["0", "1"]))
    numeric = frame.loc[:, list(NUMERIC_COLUMNS)]
    flag("non_finite", (numeric.notna() & ~np.isfinite(numeric)).any(axis=1))
    flag("zero_price", frame.loc[:, list(PRICE_COLUMNS)].eq(0).any(axis=1))
    flag("negative_price", frame.loc[:, list(PRICE_COLUMNS)].lt(0).any(axis=1))
    flag("negative_quantity", frame.loc[:, list(QUANTITY_COLUMNS)].lt(0).any(axis=1))
    flag("non_positive_adjustment_factor", frame.adjustment_factor.isna() | frame.adjustment_factor.le(0))
    missing_observations = frame.loc[:, list(OBSERVATION_COLUMNS)].isna()
    flag("partial_missing_ohlcv", missing_observations.any(axis=1) & ~missing_observations.all(axis=1))
    for prefix, reason in (("", "ohlc_inconsistent"), ("adjusted_", "adjusted_ohlc_inconsistent")):
        o, h, low, c = (frame[f"{prefix}{name}"] for name in ("open", "high", "low", "close"))
        flag(reason, low.gt(h) | o.lt(low) | o.gt(h) | c.lt(low) | c.gt(h))

    calendar_verified = trading_calendar is not None
    calendar_session_count: int | None = None
    absent_market_sessions: list[str] = []
    if trading_calendar is None:
        if not config.allow_unverified_calendar:
            raise SchemaError("A trading calendar is required; explicitly allow_unverified_calendar for interim audit only")
    else:
        calendar = validate_trading_calendar(trading_calendar)
        dated = frame.loc[~issues["invalid_date"], "date"]
        if not dated.empty:
            if dated.min() < calendar.date.min() or dated.max() > calendar.date.max():
                raise SchemaError("Calendar does not cover the selected observations")
            sessions = calendar.loc[calendar.is_session, "date"]
            flag("non_trading_day", frame.date.notna() & ~frame.date.isin(sessions))
            in_period_sessions = sessions.loc[sessions.between(dated.min(), dated.max())]
            calendar_session_count = len(in_period_sessions)
            observed_dates = set(dated)
            absent_market_sessions = [_date_string(day) for day in in_period_sessions if day not in observed_dates]

    bad = pd.Series(False, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype="string")
    for name, mask in issues.items():
        bad |= mask
        reasons.loc[mask] = reasons.loc[mask] + name + ";"
    frame["has_price"] = ~missing_observations.any(axis=1) & ~bad
    frame["is_adjustment_event"] = frame.adjustment_factor.notna() & frame.adjustment_factor.ne(1)
    frame["calendar_verified"] = calendar_verified
    frame["research_period"] = "outside_research_period"
    frame.loc[frame.date.between(DEVELOPMENT_START, DEVELOPMENT_END), "research_period"] = "development"
    frame.loc[frame.date.between(FINAL_START, FINAL_END), "research_period"] = "final_backtest"
    # Invalid dates/codes cannot participate in grouped time calculations.
    # Duplicate and other bad rows remain with has_price=False to block bridging.
    key_valid = ~(issues["invalid_date"] | issues["invalid_code"])
    temporal = _quality_returns(frame.loc[key_valid].reset_index(drop=True), previous_prices)
    frame["previous_observation_date"] = pd.NaT
    frame["observed_adjusted_close_return"] = np.nan
    frame["is_non_finite_return"] = False
    frame.loc[key_valid, "previous_observation_date"] = temporal.previous_observation_date.to_numpy()
    frame.loc[key_valid, "observed_adjusted_close_return"] = temporal.observed_adjusted_close_return.to_numpy()
    frame.loc[key_valid, "is_non_finite_return"] = temporal.is_non_finite_return.to_numpy()
    frame["is_large_return"] = frame.observed_adjusted_close_return.abs().gt(config.abnormal_return_threshold)
    quarantine = frame.loc[bad].copy()
    quarantine["issue_reasons"] = reasons.loc[bad].str.rstrip(";")
    accepted = frame.loc[~bad].copy()
    sort_columns = ["code", "date"]
    accepted = accepted.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    quarantine = quarantine.sort_values(sort_columns, kind="stable", na_position="last").reset_index(drop=True)
    report = {
        "schema_id": SCHEMA_ID,
        "input_rows": input_rows,
        "selected_rows": len(frame),
        "outside_date_range_rows": input_rows - len(frame),
        "input_date_min": _date_string(minimum), "input_date_max": _date_string(maximum),
        "selected_date_min": _date_string(frame.date.min()), "selected_date_max": _date_string(frame.date.max()),
        "input_codes": input_codes, "output_codes": int(accepted.code.nunique()),
        "output_rows": len(accepted), "quarantined_rows": len(quarantine),
        "issue_counts": {name: int(mask.sum()) for name, mask in issues.items()},
        "missing_rate_by_column": {name: (float(frame[name].isna().mean()) if len(frame) else None) for name in CANONICAL_COLUMNS},
        "calendar_verified": calendar_verified,
        "calendar_session_count": calendar_session_count,
        "market_sessions_without_any_observation": absent_market_sessions,
        "no_price_rows": int(missing_observations.all(axis=1).sum()),
        "large_return_rows": int(accepted.is_large_return.sum()),
        "non_finite_return_rows": int(accepted.is_non_finite_return.sum()),
        "zero_quantity_rows": int(accepted.loc[:, list(QUANTITY_COLUMNS)].eq(0).any(axis=1).sum()),
        "adjustment_event_rows": int(accepted.is_adjustment_event.sum()),
        "config": asdict(config),
        "limitations": [
            "Observed rows only; absence of an entire row cannot establish listing or suspension status.",
            "Provider-adjusted absolute levels are snapshot values, not point-in-time certified features.",
            "Observed close returns are audit diagnostics; gaps are not one-session returns.",
            "No universe, earnings exclusion, liquidity eligibility, features or labels have been generated.",
        ],
    }
    if not calendar_verified:
        report["limitations"].append("TSE session dates have not been verified against an official calendar.")
    return PreprocessingResult(accepted, quarantine, report, _missingness(frame, "date"), _missingness(frame, "code"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inspect_partition(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    missing = set(RAW_COLUMNS) - set(schema.names)
    extra = set(schema.names) - set(RAW_COLUMNS) - set(KNOWN_OPTIONAL_RAW_COLUMNS)
    if len(schema.names) != len(set(schema.names)) or missing or extra:
        raise SchemaError(f"{path.name}: unsupported columns {schema.names}")
    for name in schema.names:
        dtype = schema.field(name).type
        if name == "Date":
            valid = pa.types.is_timestamp(dtype) and dtype.tz is None or pa.types.is_date(dtype) or pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
        elif name in ("Code", "UL", "LL", *OPTIONAL_RAW_FLAG_COLUMNS, *OPTIONAL_RAW_ENUM_COLUMNS):
            valid = pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
        else:
            valid = pa.types.is_floating(dtype) or pa.types.is_integer(dtype)
        if not valid:
            raise SchemaError(f"{path.name}: {name} has unsupported Arrow type {dtype}")
    # Scan Date only: selection/partition integrity is based on actual values,
    # never on filenames. This also detects keys that filters would discard.
    date_values = parquet.read(columns=["Date"]).to_pandas()["Date"]
    if pa.types.is_date(schema.field("Date").type):
        date_values = pd.to_datetime(date_values)
    dates = _parse_dates(date_values, "Date")
    expected_year = int(path.stem)
    if dates.isna().any() or dates.ne(dates.dt.normalize()).any() or dates.dt.year.ne(expected_year).any():
        raise SchemaError(f"{path.name}: dates must be valid midnight dates within actual partition year {expected_year}")
    return {
        "file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size,
        "raw_rows": parquet.metadata.num_rows,
        "date_min": _date_string(dates.min()), "date_max": _date_string(dates.max()),
        "schema": {name: str(schema.field(name).type) for name in schema.names},
        "api_acquired_at": None, "api_version": None,
    }


def _read_selected(path: Path, config: PreprocessingConfig) -> pd.DataFrame:
    schema = pq.read_schema(path)
    dtype = schema.field("Date").type
    filters: list[tuple[str, str, Any]] = []
    for operator, bound in ((">=", config.start_date), ("<=", config.end_date)):
        if bound is not None:
            value: Any = bound
            if pa.types.is_timestamp(dtype):
                value = pd.Timestamp(bound).to_pydatetime()
            elif pa.types.is_date(dtype):
                value = pd.Timestamp(bound).date()
            filters.append(("Date", operator, value))
    frame = pq.read_table(path, columns=list(RAW_COLUMNS), filters=filters or None).to_pandas()
    # Arrow date32/date64 may materialize Python date objects; normalize their
    # supported logical type before the stricter DataFrame API validation.
    if pa.types.is_date(dtype):
        frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def _read_calendar(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        dtype = pq.read_schema(path).field("Date").type
        if pa.types.is_date(dtype):
            frame["Date"] = pd.to_datetime(frame["Date"])
        return frame
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"Date": "string", "HolDiv": "string"})
    raise SchemaError("calendar_path must be a V2 calendar .csv or .parquet file")


def _ensure_output_paths(raw_dir: Path, output_dir: Path, report_dir: Path, calendar_path: Path | None) -> None:
    for target in (output_dir, report_dir):
        if target == raw_dir or raw_dir in target.parents or target in raw_dir.parents:
            raise ValueError("Outputs must not overlap data/raw or an ancestor of the input directory")
        if target.exists():
            raise FileExistsError(f"Use a new output directory; already exists: {target}")
        if calendar_path is not None and (target == calendar_path or target in calendar_path.parents):
            raise ValueError("Outputs must not contain the input calendar")
    if output_dir == report_dir or output_dir in report_dir.parents or report_dir in output_dir.parents:
        raise ValueError("output_dir and report_dir must be separate non-overlapping directories")
    protected_raw = (PROJECT_ROOT / "data" / "raw").resolve()
    for target in (output_dir, report_dir):
        if target == protected_raw or protected_raw in target.parents:
            raise ValueError("data/raw is read-only even when a different raw_dir is supplied")


def run_preprocessing(
    *, raw_dir: Path | str, output_dir: Path | str, report_dir: Path | str,
    config: PreprocessingConfig | None = None, calendar_path: Path | str | None = None,
) -> dict[str, Any]:
    """Process annual Parquet files one at a time and persist an audit manifest.

    Annual file dates must agree with their year partitions, making duplicate
    keys across partitions impossible. Previous observed rows are carried
    across consecutive processed partitions (including missing/invalid prices).
    Existing outputs and any writes within raw inputs are rejected. A failure
    or KeyboardInterrupt preserves completed files and a partial manifest.
    """
    config = config or PreprocessingConfig()
    raw_dir, output_dir, report_dir = (Path(path).resolve() for path in (raw_dir, output_dir, report_dir))
    calendar_path = Path(calendar_path).resolve() if calendar_path is not None else None
    _ensure_output_paths(raw_dir, output_dir, report_dir, calendar_path)
    files = sorted(path for path in raw_dir.glob("*.parquet") if re.fullmatch(r"\d{4}", path.stem))
    if not files:
        raise FileNotFoundError(f"No annual YYYY.parquet files found in {raw_dir}")
    if calendar_path is None and not config.allow_unverified_calendar:
        raise SchemaError("Supply --calendar or explicitly use --allow-unverified-calendar for interim audit")
    calendar = _read_calendar(calendar_path) if calendar_path else None
    if calendar is not None:
        validate_trading_calendar(calendar)
    # Preflight schemas and actual date ranges before creating any outputs.
    inputs = [_inspect_partition(path) for path in files]
    selected = [(path, info) for path, info in zip(files, inputs) if info["date_min"] is not None
                and (config.start_date is None or info["date_max"] >= config.start_date)
                and (config.end_date is None or info["date_min"] <= config.end_date)]
    if not selected:
        raise ValueError("No observations overlap the selected date range")
    code_files = [Path(__file__), PROJECT_ROOT / "src" / "stock_prediction" / "__init__.py", PROJECT_ROOT / "pyproject.toml"]
    code_hashes = {path.name: _sha256(path) for path in code_files if path.is_file()}
    git_revision = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False)
        if git.returncode == 0:
            git_revision = git.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    manifest: dict[str, Any] = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": SCHEMA_ID, "config": asdict(config), "inputs": inputs,
        "ignored_files": sorted(path.name for path in raw_dir.iterdir() if path.is_file() and path not in files),
        "calendar": {"file": calendar_path.name, "sha256": _sha256(calendar_path)} if calendar_path else None,
        "code_sha256": code_hashes, "git_revision": git_revision, "random_seed": None,
        "versions": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")},
        "completed_partitions": [], "output_rows": 0, "quarantined_rows": 0,
        "api_metadata_note": "Acquisition timestamps/API version were not supplied; file mtime is not acquisition time.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "quarantine").mkdir()
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    previous: pd.DataFrame | None = None
    try:
        for path, info in selected:
            frame = _read_selected(path, config)
            result = preprocess_prices(frame, config=config, trading_calendar=calendar, previous_prices=previous)
            result.report["input_file"] = info
            result.report["raw_file_rows"] = info["raw_rows"]
            result.report["date_filter_excluded_rows"] = info["raw_rows"] - len(frame)
            result.prices.to_parquet(output_dir / path.name, index=False)
            result.quarantine.to_parquet(output_dir / "quarantine" / path.name, index=False)
            result.missing_by_date.to_parquet(report_dir / f"{path.stem}_missing_by_date.parquet", index=False)
            result.missing_by_code.to_parquet(report_dir / f"{path.stem}_missing_by_code.parquet", index=False)
            _write_json(report_dir / f"{path.stem}.json", result.report)
            # Keep the last *observed* row, including missing or quarantined
            # prices. Do not replace an unavailable observation by an older one.
            combined = pd.concat([result.prices, result.quarantine.drop(columns=["issue_reasons"])], ignore_index=True)
            if previous is not None:
                # An issuer can be absent for a whole annual partition. Retain
                # its prior observation to match the unpartitioned function.
                combined = pd.concat([previous, combined], ignore_index=True)
            combined = combined.loc[combined.date.notna() & combined.date.eq(combined.date.dt.normalize())
                                    & combined.code.str.fullmatch(r"[0-9A-Z]{5}", na=False)]
            # Duplicate-key members are all unavailable (has_price=False).
            # Their date must still block the next row from bridging the gap.
            previous = combined.sort_values(["code", "date"], kind="stable").groupby("code", sort=False).tail(1)
            manifest["completed_partitions"].append({"year": int(path.stem), "output_rows": len(result.prices),
                "quarantined_rows": len(result.quarantine), "output_sha256": _sha256(output_dir / path.name),
                "quarantine_sha256": _sha256(output_dir / "quarantine" / path.name)})
            manifest["output_rows"] += len(result.prices)
            manifest["quarantined_rows"] += len(result.quarantine)
            _write_json(manifest_path, manifest)
            LOGGER.info("%s: rows=%d accepted=%d quarantined=%d no_price=%d large_return=%d calendar_verified=%s",
                        path.name, len(frame), len(result.prices), len(result.quarantine), result.report["no_price_rows"],
                        result.report["large_return_rows"], result.report["calendar_verified"])
            del frame, result, combined
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; errors produce a nonzero status without silently repairing data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, help="New data/interim run directory")
    parser.add_argument("--report-dir", type=Path, help="New reports run directory")
    parser.add_argument("--config", type=Path, help="JSON PreprocessingConfig")
    parser.add_argument("--start-date", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--calendar", type=Path, help="V2 Date/HolDiv calendar CSV or Parquet")
    parser.add_argument("--allow-unverified-calendar", action="store_true", default=None,
                        help="Allow intermediate audit without confirming TSE sessions")
    parser.add_argument("--abnormal-return-threshold", type=float, help="Absolute observed-return audit threshold; no clipping")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        values = asdict(load_config(args.config) if args.config else PreprocessingConfig())
        for field in fields(PreprocessingConfig):
            override = getattr(args, field.name)
            if override is not None:
                values[field.name] = override
        config = PreprocessingConfig(**values)
        if config.allow_unverified_calendar and args.calendar is None:
            LOGGER.warning("Trading calendar unavailable: output is unverified intermediate observations")
        manifest = run_preprocessing(raw_dir=args.raw_dir,
            output_dir=args.output_dir or PROJECT_ROOT / "data" / "interim" / f"prices_{run_id}",
            report_dir=args.report_dir or PROJECT_ROOT / "reports" / f"preprocessing_{run_id}",
            config=config, calendar_path=args.calendar)
    except (ValueError, OSError, pa.ArrowException) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"status": manifest["status"], "output_rows": manifest["output_rows"],
                      "quarantined_rows": manifest["quarantined_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
