"""Create point-in-time daily technical descriptors from audited price rows.

The module implements the eleven descriptors fixed in README section 7.  It
does not select a security universe, create weekly labels, fill missing prices,
fit preprocessing statistics, or standardize features.  Trading-session
windows are verified against an explicit J-Quants V2 calendar.
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
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)
import pyarrow as pa
import pyarrow.parquet as pq

from stock_prediction.preprocessing import SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "daily-technical-descriptors-v1"

MOMENTUM_WINDOWS = (5, 20, 60)
SMA_SHORT_WINDOW = 5
SMA_LONG_WINDOW = 20
VOLATILITY_SHORT_WINDOW = 5
VOLATILITY_LONG_WINDOW = 20
RANGE_WINDOW = 20
VOLUME_SHORT_WINDOW = 5
VOLUME_LONG_WINDOW = 20
TURNOVER_SHORT_WINDOW = 5
TURNOVER_LONG_WINDOW = 20
TOPIX_WINDOW = 20
MAX_HISTORY_SESSIONS = max(
    *MOMENTUM_WINDOWS,
    SMA_LONG_WINDOW - 1,
    VOLATILITY_LONG_WINDOW,
    RANGE_WINDOW - 1,
    VOLUME_LONG_WINDOW - 1,
    TURNOVER_LONG_WINDOW - 1,
)

PRICE_INPUT_COLUMNS = (
    "date",
    "code",
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "adjusted_volume",
    "turnover_value",
    "has_price",
    "calendar_verified",
)
NUMERIC_INPUT_COLUMNS = (
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "adjusted_volume",
    "turnover_value",
)
FEATURE_COLUMNS = (
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
)
NON_TOPIX_FEATURE_COLUMNS = tuple(name for name in FEATURE_COLUMNS if name != "RelativeTOPIX_20")
OUTPUT_COLUMNS = ("date", "code", *FEATURE_COLUMNS)


@dataclass(frozen=True)
class DescriptorConfig:
    """Settings which do not fit statistics or inspect the final test period.

    ``volatility_ddof=1`` produces the sample standard deviation documented in
    the implementation plan.  ``require_topix`` lets production runs reject a
    missing TOPIX input instead of emitting an explicitly all-missing relative
    descriptor.
    """

    volatility_ddof: int = 1
    require_topix: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.volatility_ddof, bool) or self.volatility_ddof not in (0, 1):
            raise ValueError("volatility_ddof must be 0 or 1")
        if not isinstance(self.require_topix, bool):
            raise ValueError("require_topix must be boolean")


@dataclass
class DescriptorResult:
    """Descriptor rows and an auditable quality summary for the current input."""

    descriptors: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path | str) -> DescriptorConfig:
    """Read a strict JSON configuration and reject unknown keys."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    expected = {field.name for field in fields(DescriptorConfig)}
    unknown = set(values) - expected
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return DescriptorConfig(**values)


def _normalize_dates(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        result = series.astype("datetime64[ns]")
    else:
        if not all(isinstance(value, str) for value in series.dropna()):
            raise SchemaError(f"{name} must contain ISO dates or timezone-naive datetimes")
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        result = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce")
    return result.astype("datetime64[ns]")


def _require_numeric(series: pd.Series, name: str) -> None:
    if isinstance(series.dtype, pd.api.extensions.ExtensionDtype):
        # Nullable integer/float extension arrays are accepted and normalized.
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
    else:
        numeric = is_float_dtype(series.dtype) or is_integer_dtype(series.dtype)
    if not numeric or is_bool_dtype(series.dtype):
        raise SchemaError(f"{name} must be numeric (null allowed), got {series.dtype}")


def _normalize_prices(prices: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if not prices.columns.is_unique:
        raise SchemaError(f"{source} column names must be unique")
    missing = set(PRICE_INPUT_COLUMNS) - set(prices.columns)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {sorted(missing)}")
    frame = prices.loc[:, list(PRICE_INPUT_COLUMNS)].copy().reset_index(drop=True)
    frame["date"] = _normalize_dates(frame["date"], f"{source} date")
    if frame["date"].isna().any() or frame["date"].ne(frame["date"].dt.normalize()).any():
        raise SchemaError(f"{source} dates must be valid midnight session dates")
    if not all(isinstance(value, str) for value in frame["code"]):
        raise SchemaError(f"{source} code must contain strings")
    frame["code"] = frame["code"].astype("string")
    if not frame["code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError(f"{source} code must contain five uppercase alphanumeric characters")
    if frame.duplicated(["date", "code"]).any():
        count = int(frame.duplicated(["date", "code"], keep=False).sum())
        raise SchemaError(f"{source} contains {count} duplicate-key rows")
    for name in NUMERIC_INPUT_COLUMNS:
        _require_numeric(frame[name], f"{source} {name}")
        frame[name] = frame[name].astype("float64")
        if (frame[name].notna() & ~np.isfinite(frame[name])).any():
            raise SchemaError(f"{source} {name} contains non-finite values")
    for name in ("has_price", "calendar_verified"):
        if not is_bool_dtype(frame[name].dtype) or frame[name].isna().any():
            raise SchemaError(f"{source} {name} must be non-null boolean")
        frame[name] = frame[name].astype(bool)
    if not frame["calendar_verified"].all():
        raise SchemaError(
            f"{source} contains calendar_verified=False; descriptors require an explicit trading calendar"
        )
    known = frame["has_price"]
    required_when_known = frame.loc[known, list(NUMERIC_INPUT_COLUMNS)]
    if required_when_known.isna().any().any():
        raise SchemaError(f"{source} has_price=True requires all descriptor inputs")
    prices_known = frame.loc[known, ["adjusted_close", "adjusted_high", "adjusted_low"]]
    if prices_known.le(0).any().any():
        raise SchemaError(f"{source} adjusted prices must be positive when has_price=True")
    quantities_known = frame.loc[known, ["adjusted_volume", "turnover_value"]]
    if quantities_known.lt(0).any().any():
        raise SchemaError(f"{source} volume and turnover must be non-negative when has_price=True")
    if (
        frame.loc[known, "adjusted_low"].gt(frame.loc[known, "adjusted_close"]).any()
        or frame.loc[known, "adjusted_close"].gt(frame.loc[known, "adjusted_high"]).any()
    ):
        raise SchemaError(f"{source} adjusted OHLC values are inconsistent")
    unavailable = frame.loc[~known, list(NUMERIC_INPUT_COLUMNS)]
    if not unavailable.empty and unavailable.notna().any().any():
        raise SchemaError(f"{source} has_price=False rows must not expose descriptor inputs")
    return frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def _normalize_topix(topix: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    if not topix.columns.is_unique or set(topix.columns) != {"date", "close"}:
        raise SchemaError("TOPIX columns must be exactly date and close")
    result = topix.loc[:, ["date", "close"]].copy().reset_index(drop=True)
    result["date"] = _normalize_dates(result["date"], "TOPIX date")
    if result["date"].isna().any() or result["date"].ne(result["date"].dt.normalize()).any():
        raise SchemaError("TOPIX dates must be valid midnight session dates")
    if result["date"].duplicated().any():
        raise SchemaError("TOPIX dates must be unique")
    _require_numeric(result["close"], "TOPIX close")
    result["close"] = result["close"].astype("float64")
    if (result["close"].notna() & (~np.isfinite(result["close"]) | result["close"].le(0))).any():
        raise SchemaError("TOPIX close must be positive and finite when present")
    session_ids = sessions.set_index("date")["session_id"]
    if not result["date"].isin(session_ids.index).all():
        raise SchemaError("TOPIX contains dates which are not TSE trading sessions")
    result["session_id"] = result["date"].map(session_ids).astype("int64")
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def _sessions(calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    validated = validate_trading_calendar(calendar)
    sessions = validated.loc[validated["is_session"], ["date"]].reset_index(drop=True)
    sessions["session_id"] = np.arange(len(sessions), dtype=np.int64)
    report = {
        "calendar_date_min": _date_string(validated["date"].min()),
        "calendar_date_max": _date_string(validated["date"].max()),
        "calendar_rows": int(len(validated)),
        "calendar_sessions": int(len(sessions)),
    }
    return sessions, report


def _finite_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = numerator.to_numpy(dtype="float64", na_value=np.nan)
    denom = denominator.to_numpy(dtype="float64", na_value=np.nan)
    values = np.full(len(numer), np.nan, dtype="float64")
    valid = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values[valid] = numer[valid] / denom[valid]
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=numerator.index, dtype="float64")


def _rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: Literal["mean", "min", "max", "std"],
    *,
    ddof: int = 1,
) -> pd.Series:
    rolling = frame.groupby("code", sort=False)[column].rolling(window=window, min_periods=window)
    if operation == "std":
        values = rolling.std(ddof=ddof)
    else:
        values = getattr(rolling, operation)()
    values = values.reset_index(level=0, drop=True).reindex(frame.index).astype("float64")
    oldest_session = frame.groupby("code", sort=False)["session_id"].shift(window - 1)
    continuous = frame["session_id"].sub(oldest_session).eq(window - 1)
    return values.where(continuous)


def _topix_return(topix: pd.DataFrame, window: int) -> pd.Series:
    prior_close = topix["close"].shift(window)
    prior_session = topix["session_id"].shift(window)
    continuous = topix["session_id"].sub(prior_session).eq(window)
    result = _finite_ratio(topix["close"], prior_close).sub(1).where(continuous)
    return pd.Series(result.to_numpy(), index=topix["date"], dtype="float64")


def _calculate(frame: pd.DataFrame, *, volatility_ddof: int, topix: pd.DataFrame | None) -> pd.DataFrame:
    groups = frame.groupby("code", sort=False)
    output = frame.loc[:, ["date", "code"]].copy()
    for window in MOMENTUM_WINDOWS:
        prior_close = groups["adjusted_close"].shift(window)
        prior_session = groups["session_id"].shift(window)
        prior_has_price = groups["has_price"].shift(window).eq(True)
        continuous = frame["session_id"].sub(prior_session).eq(window)
        output[f"R_{window}"] = (
            _finite_ratio(frame["adjusted_close"], prior_close).sub(1)
            .where(frame["has_price"] & prior_has_price & continuous)
        )

    sma_short = _rolling(frame, "adjusted_close", SMA_SHORT_WINDOW, "mean")
    sma_long = _rolling(frame, "adjusted_close", SMA_LONG_WINDOW, "mean")
    output["PriceToSMA_20"] = _finite_ratio(frame["adjusted_close"], sma_long).sub(1)
    output["SMA_5To20"] = _finite_ratio(sma_short, sma_long).sub(1)

    previous_close = groups["adjusted_close"].shift(1)
    previous_session = groups["session_id"].shift(1)
    previous_has_price = groups["has_price"].shift(1).eq(True)
    consecutive = frame["session_id"].sub(previous_session).eq(1)
    close_ratio = _finite_ratio(frame["adjusted_close"], previous_close)
    log_return = pd.Series(np.nan, index=frame.index, dtype="float64")
    log_eligible = frame["has_price"] & previous_has_price & consecutive & close_ratio.gt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_return.loc[log_eligible] = np.log(close_ratio.loc[log_eligible])
    frame = frame.copy()
    frame["_log_return"] = log_return
    volatility_short = _rolling(
        frame, "_log_return", VOLATILITY_SHORT_WINDOW, "std", ddof=volatility_ddof
    )
    volatility_long = _rolling(
        frame, "_log_return", VOLATILITY_LONG_WINDOW, "std", ddof=volatility_ddof
    )
    output["Volatility_5To20"] = _finite_ratio(volatility_short, volatility_long)
    output["Volatility_20"] = volatility_long

    low_20 = _rolling(frame, "adjusted_low", RANGE_WINDOW, "min")
    high_20 = _rolling(frame, "adjusted_high", RANGE_WINDOW, "max")
    output["RangePosition_20"] = _finite_ratio(frame["adjusted_close"].sub(low_20), high_20.sub(low_20))

    volume_short = _rolling(frame, "adjusted_volume", VOLUME_SHORT_WINDOW, "mean")
    volume_long = _rolling(frame, "adjusted_volume", VOLUME_LONG_WINDOW, "mean")
    output["VolumeRatio_5To20"] = _finite_ratio(volume_short, volume_long)

    turnover_short = _rolling(frame, "turnover_value", TURNOVER_SHORT_WINDOW, "mean")
    turnover_long = _rolling(frame, "turnover_value", TURNOVER_LONG_WINDOW, "mean")
    output["TurnoverRatio_5To20"] = _finite_ratio(turnover_short, turnover_long)

    if topix is None:
        output["RelativeTOPIX_20"] = np.nan
    else:
        market_return = output["date"].map(_topix_return(topix, TOPIX_WINDOW))
        output["RelativeTOPIX_20"] = output["R_20"].sub(market_return)

    output = output.loc[:, list(OUTPUT_COLUMNS)]
    for name in FEATURE_COLUMNS:
        output[name] = output[name].where(np.isfinite(output[name])).astype("float64")
    return output


def compute_descriptors(
    prices: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    topix: pd.DataFrame | None = None,
    config: DescriptorConfig | None = None,
    previous_prices: pd.DataFrame | None = None,
) -> DescriptorResult:
    """Compute README-defined daily descriptors without mutating inputs.

    Parameters
    ----------
    prices:
        Current audited intermediate rows.  Dates are Japanese TSE session
        dates; adjusted prices are JPY, adjusted volume retains the provider
        unit, and turnover is JPY.  Every row must have been calendar-verified.
    trading_calendar:
        J-Quants V2 ``Date`` / ``HolDiv`` calendar covering all input dates and
        intervening holidays.  ``HolDiv`` 1 and 2 are TSE sessions.
    topix:
        Optional normalized ``date`` / ``close`` frame.  No values are filled.
        If omitted and not required by config, ``RelativeTOPIX_20`` is missing.
    previous_prices:
        Optional strictly earlier audited rows used only as rolling warm-up.
        Results are returned for ``prices`` rows, never for warm-up rows.

    Returns
    -------
    DescriptorResult
        One output row per current ``(date, code)`` and an audit report.

    Raises
    ------
    SchemaError
        On an unsupported schema, unverified calendar state, duplicate key,
        non-session date, non-past warm-up row, or invalid numeric input.
    """

    config = config or DescriptorConfig()
    current = _normalize_prices(prices, source="prices")
    previous = None
    if previous_prices is not None:
        previous = _normalize_prices(previous_prices, source="previous_prices")
        if not current.empty and not previous.empty and previous["date"].max() >= current["date"].min():
            raise SchemaError("previous_prices must precede every current observation")
    sessions, calendar_report = _sessions(trading_calendar)
    session_ids = sessions.set_index("date")["session_id"]
    context_parts: list[pd.DataFrame] = []
    if previous is not None and not previous.empty:
        previous = previous.copy()
        previous["_is_current"] = False
        context_parts.append(previous)
    current = current.copy()
    current["_is_current"] = True
    context_parts.append(current)
    context = pd.concat(context_parts, ignore_index=True)
    if context.duplicated(["date", "code"]).any():
        raise SchemaError("prices and previous_prices overlap on a primary key")
    if not context.empty and not context["date"].isin(session_ids.index).all():
        raise SchemaError("Price input contains dates which are not TSE trading sessions")
    context["session_id"] = context["date"].map(session_ids).astype("int64")
    context = context.sort_values(["code", "date"], kind="stable").reset_index(drop=True)

    normalized_topix = None
    if topix is None:
        if config.require_topix:
            raise SchemaError("TOPIX input is required by configuration")
    else:
        normalized_topix = _normalize_topix(topix, sessions)
    calculated = _calculate(
        context,
        volatility_ddof=config.volatility_ddof,
        topix=normalized_topix,
    )
    result = calculated.loc[context["_is_current"].to_numpy()].reset_index(drop=True)
    missing_rate = {name: float(result[name].isna().mean()) for name in FEATURE_COLUMNS}
    available_rows = {name: int(result[name].notna().sum()) for name in FEATURE_COLUMNS}
    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "input_rows": int(len(current)),
        "history_rows": int(0 if previous is None else len(previous)),
        "output_rows": int(len(result)),
        "securities": int(current["code"].nunique()),
        "date_min": _date_string(current["date"].min()),
        "date_max": _date_string(current["date"].max()),
        "duplicate_keys": 0,
        "rows_without_price": int((~current["has_price"]).sum()),
        "feature_missing_rate": missing_rate,
        "feature_available_rows": available_rows,
        "complete_non_topix_feature_rows": int(result[list(NON_TOPIX_FEATURE_COLUMNS)].notna().all(axis=1).sum()),
        "complete_feature_rows": int(result[list(FEATURE_COLUMNS)].notna().all(axis=1).sum()),
        "topix_supplied": topix is not None,
        "topix_rows": None if normalized_topix is None else int(len(normalized_topix)),
        **calendar_report,
    }
    return DescriptorResult(descriptors=result, report=report)


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _read_calendar(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"Date": "string", "HolDiv": "string"})
    raise ValueError("Calendar must be a .parquet or .csv file")


def _read_topix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("TOPIX input must be a .parquet or .csv file")


def _annual_partitions(prices_dir: Path) -> tuple[list[Path], list[str]]:
    paths = sorted(path for path in prices_dir.glob("*.parquet") if re.fullmatch(r"\d{4}\.parquet", path.name))
    if not paths:
        raise ValueError(f"No annual YYYY.parquet partitions found in {prices_dir}")
    ignored = sorted(path.name for path in prices_dir.iterdir() if path.is_file() and path not in paths)
    return paths, ignored


def _inspect_partition(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    missing = set(PRICE_INPUT_COLUMNS) - names
    if missing:
        raise SchemaError(f"{path.name} is missing required columns: {sorted(missing)}")
    return {
        "file": path.name,
        "rows": int(parquet.metadata.num_rows),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _read_prices(path: Path) -> pd.DataFrame:
    return pq.read_table(path, columns=list(PRICE_INPUT_COLUMNS)).to_pandas()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ensure_output_paths(prices_dir: Path, output_dir: Path, report_dir: Path) -> None:
    prices_resolved = prices_dir.resolve()
    output_resolved = output_dir.resolve()
    report_resolved = report_dir.resolve()
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Output and report directories must both be new")
    if (
        _is_relative_to(output_resolved, prices_resolved)
        or _is_relative_to(report_resolved, prices_resolved)
        or _is_relative_to(prices_resolved, output_resolved)
        or _is_relative_to(prices_resolved, report_resolved)
    ):
        raise ValueError("Outputs must not contain or be contained by the input price directory")
    if _is_relative_to(output_resolved, report_resolved) or _is_relative_to(report_resolved, output_resolved):
        raise ValueError("Output and report directories must not overlap")


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("pandas", "numpy", "pyarrow")
    }


def run_descriptors(
    *,
    prices_dir: Path | str,
    calendar_path: Path | str,
    output_dir: Path | str,
    report_dir: Path | str,
    topix_path: Path | str | None = None,
    config: DescriptorConfig | None = None,
) -> dict[str, Any]:
    """Process annual intermediate partitions with bounded rolling history.

    Completed partitions and manifest progress are retained if execution is
    interrupted.  Existing output directories are never overwritten.
    """

    config = config or DescriptorConfig()
    prices_dir = Path(prices_dir)
    calendar_path = Path(calendar_path)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    topix_path = None if topix_path is None else Path(topix_path)
    if not prices_dir.is_dir():
        raise FileNotFoundError(f"Price directory does not exist: {prices_dir}")
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    if topix_path is not None and not topix_path.is_file():
        raise FileNotFoundError(f"TOPIX input does not exist: {topix_path}")
    _ensure_output_paths(prices_dir, output_dir, report_dir)
    partitions, ignored = _annual_partitions(prices_dir)
    inputs = [_inspect_partition(path) for path in partitions]
    calendar = _read_calendar(calendar_path)
    _sessions(calendar)  # Validate before creating any output.
    topix = None if topix_path is None else _read_topix(topix_path)
    if topix is not None:
        sessions, _ = _sessions(calendar)
        _normalize_topix(topix, sessions)
    elif config.require_topix:
        raise SchemaError("TOPIX input is required by configuration")

    source_path = Path(__file__)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "prices_dir": str(prices_dir),
        "calendar": {
            "path": str(calendar_path),
            "sha256": _sha256(calendar_path),
        },
        "topix_supplied": topix_path is not None,
        "topix": None if topix_path is None else {"path": str(topix_path), "sha256": _sha256(topix_path)},
        "inputs": inputs,
        "ignored_files": ignored,
        "completed_partitions": [],
        "output_rows": 0,
        "code_sha256": {
            "descriptors.py": _sha256(source_path),
            "pyproject.toml": _sha256(PROJECT_ROOT / "pyproject.toml"),
        },
        "versions": _versions(),
        "random_seed": None,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    previous: pd.DataFrame | None = None
    try:
        for path, info in zip(partitions, inputs, strict=True):
            current = _read_prices(path)
            normalized_current = _normalize_prices(current, source=path.name)
            expected_year = int(path.stem)
            if not normalized_current.empty and not normalized_current["date"].dt.year.eq(expected_year).all():
                raise SchemaError(f"{path.name} contains observations outside calendar year {expected_year}")
            result = compute_descriptors(
                normalized_current,
                trading_calendar=calendar,
                topix=topix,
                config=config,
                previous_prices=previous,
            )
            output_path = output_dir / path.name
            result.descriptors.to_parquet(output_path, index=False)
            partition_report = {
                **result.report,
                "input": info,
                "output_file": output_path.name,
                "output_sha256": _sha256(output_path),
            }
            _write_json(report_dir / f"{path.stem}.json", partition_report)
            manifest["completed_partitions"].append(
                {
                    "year": expected_year,
                    "input_rows": int(len(normalized_current)),
                    "output_rows": int(len(result.descriptors)),
                    "output_sha256": partition_report["output_sha256"],
                    "date_min": result.report["date_min"],
                    "date_max": result.report["date_max"],
                    "securities": result.report["securities"],
                    "rows_without_price": result.report["rows_without_price"],
                    "feature_missing_rate": result.report["feature_missing_rate"],
                }
            )
            manifest["output_rows"] += len(result.descriptors)
            _write_json(manifest_path, manifest)
            history_parts = [normalized_current] if previous is None else [previous, normalized_current]
            history = pd.concat(history_parts, ignore_index=True)
            previous = (
                history.sort_values(["code", "date"], kind="stable")
                .groupby("code", sort=False)
                .tail(MAX_HISTORY_SESSIONS)
                .reset_index(drop=True)
            )
            LOGGER.info(
                "%s: rows=%d securities=%d complete_non_topix=%d topix_supplied=%s",
                path.name,
                len(result.descriptors),
                result.report["securities"],
                result.report["complete_non_topix_feature_rows"],
                result.report["topix_supplied"],
            )
            del current, normalized_current, result, history
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
    """CLI entry point shared with the Notebook-facing implementation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-dir", type=Path, required=True, help="Annual audited interim YYYY.parquet files")
    parser.add_argument("--calendar", type=Path, required=True, help="J-Quants V2 Date/HolDiv CSV or Parquet")
    parser.add_argument("--topix", type=Path, help="Optional normalized date/close TOPIX CSV or Parquet")
    parser.add_argument("--output-dir", type=Path, help="New data/processed descriptor run directory")
    parser.add_argument("--report-dir", type=Path, help="New reports descriptor run directory")
    parser.add_argument("--config", type=Path, help="JSON DescriptorConfig")
    parser.add_argument("--volatility-ddof", type=int, choices=(0, 1), help="Override volatility ddof")
    parser.add_argument("--require-topix", action="store_true", default=None, help="Fail unless --topix is supplied")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        values = asdict(load_config(args.config) if args.config else DescriptorConfig())
        if args.volatility_ddof is not None:
            values["volatility_ddof"] = args.volatility_ddof
        if args.require_topix is not None:
            values["require_topix"] = args.require_topix
        config = DescriptorConfig(**values)
        manifest = run_descriptors(
            prices_dir=args.prices_dir,
            calendar_path=args.calendar,
            topix_path=args.topix,
            output_dir=args.output_dir or PROJECT_ROOT / "data" / "processed" / f"descriptors_{run_id}",
            report_dir=args.report_dir or PROJECT_ROOT / "reports" / f"descriptors_{run_id}",
            config=config,
        )
    except (ValueError, OSError, pa.ArrowException) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"status": manifest["status"], "output_rows": manifest["output_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
