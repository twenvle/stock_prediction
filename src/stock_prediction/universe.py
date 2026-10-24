"""Build a point-in-time Japanese-equity universe without future leakage.

The point-in-time listed-issue master is the universe spine.  Audited daily
prices are joined without forward filling, so a missing observation remains a
missing/non-trading day.  Signal-time eligibility is deliberately separated
from ex-post execution proxies that use the next-session open and evaluation-
day close.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
import pyarrow as pa
import pyarrow.parquet as pq

from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "point-in-time-universe-v1"
DOMESTIC_COMMON_PRODUCT_CATEGORY = "011"
MAX_HISTORY_SESSIONS = 60

MASTER_REQUIRED_COLUMNS = ("Date", "Code", "ProdCat")
MASTER_OPTIONAL_OUTPUT_COLUMNS = ("CoName", "Mkt", "MktNm")
PRICE_REQUIRED_COLUMNS = (
    "date",
    "code",
    "open",
    "close",
    "upper_limit",
    "lower_limit",
    "volume",
    "turnover_value",
    "has_price",
    "calendar_verified",
)
SCHEDULE_COLUMNS = ("signal_date", "entry_date", "exit_date")
OUTPUT_COLUMNS = (
    "signal_date",
    "code",
    "company_name",
    "market_code",
    "market_name",
    "product_category",
    "is_domestic_common_stock",
    "listed_sessions_elapsed",
    "has_min_listing_history",
    "liquidity_window_observations",
    "price_formation_days_20",
    "median_turnover_value_20",
    "max_order_value_yen_20",
    "has_full_liquidity_window",
    "meets_price_formation_days",
    "meets_median_turnover",
    "meets_participation_limit",
    "is_liquid",
    "signal_close",
    "signal_price_available",
    "signal_price_formed",
    "is_eligible",
    "exclusion_reasons",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "entry_price_available",
    "exit_price_available",
    "entry_not_limit_up",
    "exit_not_limit_down",
    "is_entry_executable",
    "is_exit_executable",
    "is_executable",
)


@dataclass(frozen=True)
class UniverseConfig:
    """Fixed universe rules; currency values are Japanese yen.

    Liquidity uses the signal session and the preceding 19 TSE sessions.  The
    turnover median is computed over sessions with an actual positive-volume,
    positive-turnover price formation; missing observations are not filled.
    """

    domestic_common_product_category: str = DOMESTIC_COMMON_PRODUCT_CATEGORY
    min_listing_sessions_elapsed: int = 60
    liquidity_window_sessions: int = 20
    min_price_formation_days: int = 15
    min_median_turnover_yen: float = 100_000_000.0
    planned_investment_yen: float = 1_000_000.0
    max_participation_rate: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.domestic_common_product_category, str) or not re.fullmatch(
            r"\d{3}", self.domestic_common_product_category
        ):
            raise ValueError("domestic_common_product_category must be a three-digit string")
        for name in ("min_listing_sessions_elapsed", "liquidity_window_sessions", "min_price_formation_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_price_formation_days > self.liquidity_window_sessions:
            raise ValueError("min_price_formation_days must not exceed liquidity_window_sessions")
        for name in ("min_median_turnover_yen", "planned_investment_yen", "max_participation_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_participation_rate > 1:
            raise ValueError("max_participation_rate must not exceed 1")


@dataclass
class UniverseResult:
    """Point-in-time decisions and an audit report."""

    universe: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path | str) -> UniverseConfig:
    """Read JSON configuration and reject unknown fields."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(UniverseConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return UniverseConfig(**values)


def _require_string_values(series: pd.Series, name: str) -> None:
    if not all(isinstance(value, str) for value in series.dropna()):
        raise SchemaError(f"{name} must contain strings, not coerced numeric values")


def _parse_dates(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must be timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        parsed = series.astype("datetime64[ns]")
    else:
        _require_string_values(series, name)
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=nullable)
        parsed = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce").astype("datetime64[ns]")
    if (not nullable and parsed.isna().any()) or parsed.dropna().ne(parsed.dropna().dt.normalize()).any():
        raise SchemaError(f"{name} must contain valid midnight dates")
    return parsed


def _normalize_master(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise SchemaError(f"{source} contains duplicate column names")
    missing = set(MASTER_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {sorted(missing)}")
    output_columns = [*MASTER_REQUIRED_COLUMNS, *[c for c in MASTER_OPTIONAL_OUTPUT_COLUMNS if c in frame.columns]]
    output = frame.loc[:, output_columns].copy().reset_index(drop=True)
    output["Date"] = _parse_dates(output["Date"], f"{source}.Date")
    for column in output_columns:
        if column != "Date":
            _require_string_values(output[column], f"{source}.{column}")
            output[column] = output[column].astype("string")
    if output["Code"].isna().any() or not output["Code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError(f"{source}.Code must contain non-null five-character codes")
    if output["ProdCat"].isna().any() or not output["ProdCat"].str.fullmatch(r"\d{3}", na=False).all():
        raise SchemaError(f"{source}.ProdCat must contain non-null three-digit product categories")
    if output.duplicated(["Date", "Code"], keep=False).any():
        raise SchemaError(f"{source} contains duplicate Date/Code keys")
    return output.sort_values(["Code", "Date"], kind="stable").reset_index(drop=True)


def _normalize_prices(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise SchemaError(f"{source} contains duplicate column names")
    missing = set(PRICE_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {sorted(missing)}")
    output = frame.loc[:, list(PRICE_REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    output["date"] = _parse_dates(output["date"], f"{source}.date")
    _require_string_values(output["code"], f"{source}.code")
    output["code"] = output["code"].astype("string")
    if output["code"].isna().any() or not output["code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError(f"{source}.code must contain non-null five-character codes")
    if output.duplicated(["date", "code"], keep=False).any():
        raise SchemaError(f"{source} contains duplicate date/code keys")
    for column in ("open", "close", "volume", "turnover_value"):
        if not is_numeric_dtype(output[column].dtype) or pd.api.types.is_bool_dtype(output[column].dtype):
            raise SchemaError(f"{source}.{column} must be numeric")
        output[column] = output[column].astype("float64")
        if (output[column].notna() & ~np.isfinite(output[column])).any():
            raise SchemaError(f"{source}.{column} contains non-finite values")
    for column in ("upper_limit", "lower_limit"):
        _require_string_values(output[column], f"{source}.{column}")
        output[column] = output[column].astype("string")
        if not output[column].dropna().isin(["0", "1"]).all():
            raise SchemaError(f"{source}.{column} must be 0, 1, or null")
    for column in ("has_price", "calendar_verified"):
        if not is_bool_dtype(output[column].dtype) or output[column].isna().any():
            raise SchemaError(f"{source}.{column} must be non-null boolean")
    if not output["calendar_verified"].all():
        raise SchemaError(f"{source} contains calendar_verified=False")
    known = output["has_price"]
    if not (
        output.loc[known, ["open", "close"]].notna().all(axis=1)
        & output.loc[known, ["open", "close"]].gt(0).all(axis=1)
        & output.loc[known, ["volume", "turnover_value"]].notna().all(axis=1)
        & output.loc[known, ["volume", "turnover_value"]].ge(0).all(axis=1)
    ).all():
        raise SchemaError(f"{source} has_price=True requires positive prices and non-negative volume/turnover")
    return output.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def _normalize_schedule(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    if not frame.columns.is_unique or set(frame.columns) != set(SCHEDULE_COLUMNS):
        raise SchemaError(f"schedule columns must be exactly {list(SCHEDULE_COLUMNS)}")
    output = frame.loc[:, list(SCHEDULE_COLUMNS)].copy().reset_index(drop=True)
    for column in SCHEDULE_COLUMNS:
        output[column] = _parse_dates(output[column], f"schedule.{column}")
        if not output[column].isin(sessions).all():
            raise SchemaError(f"schedule.{column} contains non-session dates")
    if output["signal_date"].duplicated().any():
        raise SchemaError("schedule.signal_date must be unique")
    session_ids = pd.Series(np.arange(len(sessions), dtype="int64"), index=sessions)
    signal_ids = output["signal_date"].map(session_ids)
    entry_ids = output["entry_date"].map(session_ids)
    if not entry_ids.eq(signal_ids + 1).all():
        raise SchemaError("entry_date must be the TSE session immediately after signal_date")
    session_frame = pd.DataFrame({"date": sessions})
    session_frame["week"] = session_frame["date"].dt.to_period("W-SUN")
    weekly = session_frame.groupby("week", observed=True)["date"].agg(first="min", last="max")
    signal_weeks = output["signal_date"].dt.to_period("W-SUN")
    expected_signals = signal_weeks.map(weekly["first"])
    expected_exits = signal_weeks.map(weekly["last"])
    if not output["signal_date"].eq(expected_signals).all():
        raise SchemaError("signal_date must be the first TSE session of its calendar week")
    if not output["exit_date"].eq(expected_exits).all():
        raise SchemaError("exit_date must be the last TSE session of the signal calendar week")
    if not (output["exit_date"] >= output["entry_date"]).all():
        raise SchemaError("the signal week must contain an entry session no later than exit_date")
    return output.sort_values("signal_date").reset_index(drop=True)


def _calendar_sessions(calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    validated = validate_trading_calendar(calendar)
    session_frame = validated.loc[validated["is_session"], ["date"]].reset_index(drop=True)
    session_frame["session_id"] = np.arange(len(session_frame), dtype="int64")
    return session_frame, pd.DatetimeIndex(session_frame["date"])


def _rolling_by_listing_run(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
    operation: str,
    min_periods: int,
) -> pd.Series:
    grouped = frame.groupby(["Code", "_listing_run"], sort=False)[column]
    rolling = grouped.rolling(window=window, min_periods=min_periods)
    calculated = getattr(rolling, operation)().reset_index(level=[0, 1], drop=True)
    return calculated.reindex(frame.index)


def _reason_column(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype="string")
    checks = (
        (~frame["is_domestic_common_stock"], "not_domestic_common_stock"),
        (~frame["has_min_listing_history"], "listing_history_under_60_sessions"),
        (~frame["has_full_liquidity_window"], "liquidity_history_under_20_sessions"),
        (~frame["meets_price_formation_days"], "insufficient_price_formation_days"),
        (~frame["meets_median_turnover"], "median_turnover_below_threshold"),
        (~frame["meets_participation_limit"], "participation_limit_exceeded"),
        (~frame["signal_price_available"], "signal_close_unavailable"),
        (frame["signal_price_available"] & ~frame["signal_price_formed"], "no_signal_session_trade"),
    )
    for mask, label in checks:
        reasons.loc[mask.fillna(True)] = reasons.loc[mask.fillna(True)] + label + ";"
    return reasons.str.rstrip(";")


def _attach_execution(
    output: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None,
    execution_prices: pd.DataFrame,
) -> pd.DataFrame:
    result = output.copy()
    if schedule is None:
        result["entry_date"] = pd.NaT
        result["exit_date"] = pd.NaT
        result["entry_price"] = np.nan
        result["exit_price"] = np.nan
        for column in (
            "entry_price_available",
            "exit_price_available",
            "entry_not_limit_up",
            "exit_not_limit_down",
            "is_entry_executable",
            "is_exit_executable",
            "is_executable",
        ):
            result[column] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        return result

    result = result.merge(schedule, on="signal_date", how="left", validate="many_to_one")
    lookup = execution_prices.set_index(["date", "code"])

    def values(date_column: str, value_column: str) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([result[date_column], result["code"]])
        return lookup[value_column].reindex(keys).to_numpy()

    entry_has_price = pd.Series(values("entry_date", "has_price"), index=result.index).fillna(False).astype(bool)
    exit_has_price = pd.Series(values("exit_date", "has_price"), index=result.index).fillna(False).astype(bool)
    result["entry_price"] = pd.to_numeric(values("entry_date", "open"), errors="coerce").astype("float64")
    result["exit_price"] = pd.to_numeric(values("exit_date", "close"), errors="coerce").astype("float64")
    entry_volume = pd.to_numeric(values("entry_date", "volume"), errors="coerce")
    exit_volume = pd.to_numeric(values("exit_date", "volume"), errors="coerce")
    entry_turnover = pd.to_numeric(values("entry_date", "turnover_value"), errors="coerce")
    exit_turnover = pd.to_numeric(values("exit_date", "turnover_value"), errors="coerce")
    entry_limit = pd.Series(values("entry_date", "upper_limit"), index=result.index, dtype="string")
    exit_limit = pd.Series(values("exit_date", "lower_limit"), index=result.index, dtype="string")
    result["entry_price_available"] = entry_has_price & np.isfinite(result["entry_price"]) & result["entry_price"].gt(0)
    result["exit_price_available"] = exit_has_price & np.isfinite(result["exit_price"]) & result["exit_price"].gt(0)
    result["entry_not_limit_up"] = entry_limit.eq("0").fillna(False)
    result["exit_not_limit_down"] = exit_limit.eq("0").fillna(False)
    result["is_entry_executable"] = (
        result["entry_price_available"]
        & pd.Series(entry_volume, index=result.index).gt(0)
        & pd.Series(entry_turnover, index=result.index).gt(0)
        & result["entry_not_limit_up"]
    )
    result["is_exit_executable"] = (
        result["exit_price_available"]
        & pd.Series(exit_volume, index=result.index).gt(0)
        & pd.Series(exit_turnover, index=result.index).gt(0)
        & result["exit_not_limit_down"]
    )
    result["is_executable"] = result["is_entry_executable"] & result["is_exit_executable"]
    return result


def compute_universe(
    listed_info: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    config: UniverseConfig | None = None,
    previous_listed_info: pd.DataFrame | None = None,
    previous_prices: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
    execution_prices: pd.DataFrame | None = None,
) -> UniverseResult:
    """Compute point-in-time eligibility and optional ex-post execution flags.

    ``listed_info`` and ``prices`` contain the current output period.
    ``previous_*`` inputs must be strictly earlier and are used only as rolling
    warm-up.  ``schedule`` is an optional exact mapping of signal, next-session
    entry, and evaluation dates.  Future ``execution_prices`` affect only the
    execution proxy columns, never ``is_eligible``.
    """

    config = config or UniverseConfig()
    current_master = _normalize_master(listed_info, source="listed_info")
    current_prices = _normalize_prices(prices, source="prices")
    history_master = None
    history_prices = None
    if previous_listed_info is not None:
        history_master = _normalize_master(previous_listed_info, source="previous_listed_info")
    if previous_prices is not None:
        history_prices = _normalize_prices(previous_prices, source="previous_prices")
    if not current_master.empty:
        current_min = current_master["Date"].min()
        if history_master is not None and not history_master.empty and history_master["Date"].max() >= current_min:
            raise SchemaError("previous_listed_info must precede every current master row")
        if history_prices is not None and not history_prices.empty and history_prices["date"].max() >= current_min:
            raise SchemaError("previous_prices must precede every current master row")

    session_frame, sessions = _calendar_sessions(trading_calendar)
    session_ids = session_frame.set_index("date")["session_id"]
    normalized_schedule = None if schedule is None else _normalize_schedule(schedule, sessions)
    if normalized_schedule is not None:
        current_dates = set(current_master["Date"])
        missing_signals = set(normalized_schedule["signal_date"]) - current_dates
        if missing_signals:
            raise SchemaError(f"schedule signal dates are absent from listed_info: {sorted(missing_signals)}")

    master_parts: list[pd.DataFrame] = []
    if history_master is not None and not history_master.empty:
        previous = history_master.copy()
        previous["_is_current"] = False
        master_parts.append(previous)
    current = current_master.copy()
    current["_is_current"] = True
    master_parts.append(current)
    master = pd.concat(master_parts, ignore_index=True)
    if master.duplicated(["Date", "Code"]).any():
        raise SchemaError("current and previous listed-info inputs overlap")
    if not master.empty and not master["Date"].isin(sessions).all():
        raise SchemaError("listed_info contains non-session dates")
    if not master.empty:
        expected_master_dates = set(sessions[sessions.slice_indexer(master["Date"].min(), master["Date"].max())])
        actual_master_dates = set(master["Date"])
        missing_master_dates = expected_master_dates - actual_master_dates
        if missing_master_dates:
            raise SchemaError(f"listed_info is missing TSE session snapshots: {sorted(missing_master_dates)}")
    master["session_id"] = master["Date"].map(session_ids).astype("int64")
    master = master.sort_values(["Code", "Date"], kind="stable").reset_index(drop=True)
    discontinuity = master.groupby("Code", sort=False)["session_id"].diff().ne(1)
    master["_listing_run"] = discontinuity.groupby(master["Code"], sort=False).cumsum().astype("int64")
    master["listed_sessions_elapsed"] = (
        master.groupby(["Code", "_listing_run"], sort=False).cumcount().astype("int64")
    )

    price_parts: list[pd.DataFrame] = []
    if history_prices is not None and not history_prices.empty:
        price_parts.append(history_prices)
    price_parts.append(current_prices)
    liquidity_prices = pd.concat(price_parts, ignore_index=True)
    if liquidity_prices.duplicated(["date", "code"]).any():
        raise SchemaError("current and previous price inputs overlap")
    if not liquidity_prices.empty and not liquidity_prices["date"].isin(sessions).all():
        raise SchemaError("prices contains non-session dates")
    join_prices = liquidity_prices.rename(columns={"date": "Date", "code": "Code"})
    grid = master.merge(join_prices, on=["Date", "Code"], how="left", validate="one_to_one")
    grid["has_price"] = grid["has_price"].fillna(False).astype(bool)
    grid["calendar_verified"] = grid["calendar_verified"].fillna(False).astype(bool)
    grid["_price_formed"] = (
        grid["has_price"]
        & grid["volume"].gt(0)
        & grid["turnover_value"].gt(0)
        & grid["close"].gt(0)
    )
    grid["_master_observation"] = 1
    grid["_turnover_when_formed"] = grid["turnover_value"].where(grid["_price_formed"])
    window = config.liquidity_window_sessions
    grid["liquidity_window_observations"] = _rolling_by_listing_run(
        grid, "_master_observation", window=window, operation="sum", min_periods=1
    ).astype("int64")
    grid["price_formation_days_20"] = _rolling_by_listing_run(
        grid, "_price_formed", window=window, operation="sum", min_periods=1
    ).astype("int64")
    grid["median_turnover_value_20"] = _rolling_by_listing_run(
        grid, "_turnover_when_formed", window=window, operation="median", min_periods=1
    ).astype("float64")

    current_grid = grid.loc[grid["_is_current"]].copy()
    if normalized_schedule is not None:
        current_grid = current_grid.loc[current_grid["Date"].isin(normalized_schedule["signal_date"])].copy()
    current_grid.rename(
        columns={
            "Date": "signal_date",
            "Code": "code",
            "CoName": "company_name",
            "Mkt": "market_code",
            "MktNm": "market_name",
            "ProdCat": "product_category",
            "close": "signal_close",
        },
        inplace=True,
    )
    for column in ("company_name", "market_code", "market_name"):
        if column not in current_grid:
            current_grid[column] = pd.Series(pd.NA, index=current_grid.index, dtype="string")
    current_grid["is_domestic_common_stock"] = current_grid["product_category"].eq(
        config.domestic_common_product_category
    )
    # The bounded annual pipeline retains exactly the amount of history needed
    # for eligibility.  Values above the threshold are therefore intentionally
    # capped instead of implying a listing date earlier than the input proves.
    current_grid["listed_sessions_elapsed"] = current_grid["listed_sessions_elapsed"].clip(
        upper=config.min_listing_sessions_elapsed
    )
    current_grid["has_min_listing_history"] = current_grid["listed_sessions_elapsed"].ge(
        config.min_listing_sessions_elapsed
    )
    current_grid["has_full_liquidity_window"] = current_grid["liquidity_window_observations"].eq(window)
    current_grid["meets_price_formation_days"] = current_grid["price_formation_days_20"].ge(
        config.min_price_formation_days
    )
    current_grid["meets_median_turnover"] = current_grid["median_turnover_value_20"].ge(
        config.min_median_turnover_yen
    )
    current_grid["max_order_value_yen_20"] = (
        current_grid["median_turnover_value_20"] * config.max_participation_rate
    )
    current_grid["meets_participation_limit"] = current_grid["max_order_value_yen_20"].ge(
        config.planned_investment_yen
    )
    current_grid["is_liquid"] = (
        current_grid["has_full_liquidity_window"]
        & current_grid["meets_price_formation_days"]
        & current_grid["meets_median_turnover"]
        & current_grid["meets_participation_limit"]
    )
    current_grid["signal_price_available"] = (
        current_grid["has_price"]
        & np.isfinite(current_grid["signal_close"])
        & current_grid["signal_close"].gt(0)
    )
    current_grid["signal_price_formed"] = current_grid["_price_formed"]
    current_grid["is_eligible"] = (
        current_grid["is_domestic_common_stock"]
        & current_grid["has_min_listing_history"]
        & current_grid["is_liquid"]
        & current_grid["signal_price_formed"]
    )
    current_grid["exclusion_reasons"] = _reason_column(current_grid)

    execution_parts = [liquidity_prices]
    if execution_prices is not None:
        normalized_execution = _normalize_prices(execution_prices, source="execution_prices")
        execution_parts.append(normalized_execution)
    all_execution_prices = pd.concat(execution_parts, ignore_index=True)
    if all_execution_prices.duplicated(["date", "code"]).any():
        raise SchemaError("execution_prices overlaps current or previous price inputs")
    current_grid = _attach_execution(
        current_grid,
        schedule=normalized_schedule,
        execution_prices=all_execution_prices,
    )
    result = current_grid.loc[:, list(OUTPUT_COLUMNS)].sort_values(
        ["signal_date", "code"], kind="stable"
    ).reset_index(drop=True)

    criteria = (
        "is_domestic_common_stock",
        "has_min_listing_history",
        "has_full_liquidity_window",
        "meets_price_formation_days",
        "meets_median_turnover",
        "meets_participation_limit",
        "signal_price_formed",
        "is_liquid",
        "is_eligible",
    )
    exclusion_counts: dict[str, int] = {}
    for reasons in result.loc[~result["is_eligible"], "exclusion_reasons"]:
        for reason in str(reasons).split(";"):
            if reason:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "input_master_rows": int(len(current_master)),
        "history_master_rows": int(0 if history_master is None else len(history_master)),
        "input_price_rows": int(len(current_prices)),
        "history_price_rows": int(0 if history_prices is None else len(history_prices)),
        "output_rows": int(len(result)),
        "signal_dates": int(result["signal_date"].nunique()),
        "securities": int(result["code"].nunique()),
        "date_min": _date_string(result["signal_date"].min()),
        "date_max": _date_string(result["signal_date"].max()),
        "criterion_true_counts": {column: int(result[column].sum()) for column in criteria},
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "schedule_supplied": normalized_schedule is not None,
        "executable_rows": None if normalized_schedule is None else int(result["is_executable"].sum()),
        "master_rows_without_price": int((grid["_is_current"] & ~grid["has_price"]).sum()),
        "price_rows_not_in_master": int(
            len(
                current_prices.merge(
                    current_master[["Date", "Code"]].rename(columns={"Date": "date", "Code": "code"}),
                    on=["date", "code"],
                    how="left",
                    indicator=True,
                ).query("_merge == 'left_only'")
            )
        ),
        "limitations": [
            "Daily OHLCV and limit flags provide conservative execution proxies, not order-book fill certainty.",
            "Supervisory/special-treatment designations are unavailable in the current input schema.",
            "Future entry/exit observations never contribute to is_eligible.",
            "The initial master snapshot does not reveal earlier listing history; 60 observed sessions are required.",
        ],
    }
    return UniverseResult(universe=result, report=report)


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


def _read_table(path: Path, *, string_columns: Sequence[str] = ()) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={column: "string" for column in string_columns})
    raise ValueError(f"Unsupported table format: {path}; expected .parquet or .csv")


def _annual_partitions(directory: Path, label: str) -> dict[int, Path]:
    paths = {
        int(path.stem): path
        for path in directory.glob("*.parquet")
        if re.fullmatch(r"\d{4}\.parquet", path.name)
    }
    if not paths:
        raise ValueError(f"No annual YYYY.parquet {label} partitions found in {directory}")
    return dict(sorted(paths.items()))


def _inspect_partition(path: Path, required_columns: Sequence[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    missing = set(required_columns) - set(parquet.schema_arrow.names)
    if missing:
        raise SchemaError(f"{path.name} is missing required columns: {sorted(missing)}")
    return {
        "file": str(path),
        "rows": int(parquet.metadata.num_rows),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _read_master_partition(path: Path) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [*MASTER_REQUIRED_COLUMNS, *[name for name in MASTER_OPTIONAL_OUTPUT_COLUMNS if name in names]]
    return pq.read_table(path, columns=columns).to_pandas()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ensure_output_paths(inputs: Sequence[Path], output_dir: Path, report_dir: Path) -> None:
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("Output and report directories must both be new")
    resolved_outputs = (output_dir.resolve(), report_dir.resolve())
    for source in inputs:
        resolved_source = source.resolve()
        if any(
            _is_relative_to(destination, resolved_source) or _is_relative_to(resolved_source, destination)
            for destination in resolved_outputs
        ):
            raise ValueError("Outputs must not contain or be contained by an input path")
    if _is_relative_to(resolved_outputs[0], resolved_outputs[1]) or _is_relative_to(
        resolved_outputs[1], resolved_outputs[0]
    ):
        raise ValueError("Output and report directories must not overlap")


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")}


def run_universe(
    *,
    listed_info_dir: Path | str,
    prices_dir: Path | str,
    calendar_path: Path | str,
    output_dir: Path | str,
    report_dir: Path | str,
    schedule_path: Path | str | None = None,
    config: UniverseConfig | None = None,
) -> dict[str, Any]:
    """Process annual files with bounded 60-session state and manifests."""

    config = config or UniverseConfig()
    listed_info_dir = Path(listed_info_dir)
    prices_dir = Path(prices_dir)
    calendar_path = Path(calendar_path)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    schedule_path = None if schedule_path is None else Path(schedule_path)
    for path, label in ((listed_info_dir, "listed-info"), (prices_dir, "price")):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    if schedule_path is not None and not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule does not exist: {schedule_path}")
    inputs = [listed_info_dir, prices_dir, calendar_path]
    if schedule_path is not None:
        inputs.append(schedule_path)
    _ensure_output_paths(inputs, output_dir, report_dir)

    master_paths = _annual_partitions(listed_info_dir, "listed-info")
    price_paths = _annual_partitions(prices_dir, "price")
    missing_price_years = set(master_paths) - set(price_paths)
    if missing_price_years:
        raise ValueError(f"Missing price partitions for listed-info years: {sorted(missing_price_years)}")
    master_inputs = {year: _inspect_partition(path, MASTER_REQUIRED_COLUMNS) for year, path in master_paths.items()}
    price_inputs = {year: _inspect_partition(price_paths[year], PRICE_REQUIRED_COLUMNS) for year in master_paths}
    calendar = _read_table(calendar_path, string_columns=("Date", "HolDiv"))
    _, sessions = _calendar_sessions(calendar)
    schedule = None
    if schedule_path is not None:
        schedule = _normalize_schedule(_read_table(schedule_path), sessions)

    source_path = Path(__file__)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "listed_info_dir": str(listed_info_dir),
        "prices_dir": str(prices_dir),
        "calendar": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "schedule": None if schedule_path is None else {"path": str(schedule_path), "sha256": _sha256(schedule_path)},
        "master_inputs": master_inputs,
        "price_inputs": price_inputs,
        "completed_partitions": [],
        "skipped_partitions": [],
        "output_rows": 0,
        "eligible_rows": 0,
        "code_sha256": {
            "universe.py": _sha256(source_path),
            "pyproject.toml": _sha256(PROJECT_ROOT / "pyproject.toml"),
        },
        "versions": _versions(),
        "random_seed": None,
        "limitations": [
            "Execution columns are ex-post conservative proxies and are excluded from is_eligible.",
            "Supervisory/special-treatment designations and order-book fills are not available.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    previous_master: pd.DataFrame | None = None
    previous_prices: pd.DataFrame | None = None
    try:
        for year, master_path in master_paths.items():
            current_master = _read_master_partition(master_path)
            current_prices = pq.read_table(price_paths[year], columns=list(PRICE_REQUIRED_COLUMNS)).to_pandas()
            if not current_master.empty and not pd.to_datetime(current_master["Date"]).dt.year.eq(year).all():
                raise SchemaError(f"{master_path.name} contains rows outside {year}")
            if not current_prices.empty and not pd.to_datetime(current_prices["date"]).dt.year.eq(year).all():
                raise SchemaError(f"{price_paths[year].name} contains rows outside {year}")
            year_schedule = None
            future_prices = None
            if schedule is not None:
                year_schedule = schedule.loc[schedule["signal_date"].dt.year.eq(year)].reset_index(drop=True)
                execution_years = sorted(
                    set(year_schedule["entry_date"].dt.year) | set(year_schedule["exit_date"].dt.year)
                )
                future_frames: list[pd.DataFrame] = []
                for execution_year in execution_years:
                    if execution_year == year:
                        continue
                    if execution_year not in price_paths:
                        raise ValueError(f"Missing price partition for execution year {execution_year}")
                    future_frames.append(
                        pq.read_table(price_paths[execution_year], columns=list(PRICE_REQUIRED_COLUMNS)).to_pandas()
                    )
                if future_frames:
                    future_prices = pd.concat(future_frames, ignore_index=True)

            process_partition = year_schedule is None or not year_schedule.empty
            if process_partition:
                result = compute_universe(
                    current_master,
                    current_prices,
                    trading_calendar=calendar,
                    config=config,
                    previous_listed_info=previous_master,
                    previous_prices=previous_prices,
                    schedule=year_schedule,
                    execution_prices=future_prices,
                )
                output_path = output_dir / f"{year}.parquet"
                result.universe.to_parquet(output_path, index=False)
                partition_report = {
                    **result.report,
                    "master_input": master_inputs[year],
                    "price_input": price_inputs[year],
                    "output_file": output_path.name,
                    "output_sha256": _sha256(output_path),
                }
                _write_json(report_dir / f"{year}.json", partition_report)
                eligible_rows = int(result.universe["is_eligible"].sum())
                manifest["completed_partitions"].append(
                    {
                        "year": year,
                        "output_rows": int(len(result.universe)),
                        "eligible_rows": eligible_rows,
                        "signal_dates": result.report["signal_dates"],
                        "date_min": result.report["date_min"],
                        "date_max": result.report["date_max"],
                        "exclusion_reason_counts": result.report["exclusion_reason_counts"],
                        "output_sha256": partition_report["output_sha256"],
                    }
                )
                manifest["output_rows"] += len(result.universe)
                manifest["eligible_rows"] += eligible_rows
            else:
                manifest["skipped_partitions"].append(
                    {
                        "year": year,
                        "reason": "no_configured_signal_dates",
                        "master_input_rows": master_inputs[year]["rows"],
                        "price_input_rows": price_inputs[year]["rows"],
                    }
                )
            _write_json(manifest_path, manifest)

            normalized_master = _normalize_master(current_master, source=master_path.name)
            master_history = normalized_master if previous_master is None else pd.concat(
                [previous_master, normalized_master], ignore_index=True
            )
            previous_master = (
                master_history.sort_values(["Code", "Date"], kind="stable")
                .groupby("Code", sort=False)
                .tail(MAX_HISTORY_SESSIONS)
                .reset_index(drop=True)
            )
            normalized_prices = _normalize_prices(current_prices, source=price_paths[year].name)
            price_history = normalized_prices if previous_prices is None else pd.concat(
                [previous_prices, normalized_prices], ignore_index=True
            )
            if previous_master.empty:
                previous_prices = price_history.iloc[0:0].copy()
            else:
                cutoff = previous_master["Date"].min()
                previous_prices = price_history.loc[price_history["date"].ge(cutoff)].reset_index(drop=True)
            if process_partition:
                LOGGER.info(
                    "%s: rows=%d eligible=%d signal_dates=%d",
                    year,
                    len(result.universe),
                    eligible_rows,
                    result.report["signal_dates"],
                )
                del result
            else:
                LOGGER.info("%s: no configured signal dates; retained as history only", year)
            del current_master, current_prices, master_history, price_history
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
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listed-info-dir", type=Path, required=True, help="Annual point-in-time listed_info files")
    parser.add_argument("--prices-dir", type=Path, required=True, help="Annual audited interim price files")
    parser.add_argument("--calendar", type=Path, required=True, help="J-Quants V2 Date/HolDiv CSV or Parquet")
    parser.add_argument("--schedule", type=Path, help="Optional signal_date/entry_date/exit_date CSV or Parquet")
    parser.add_argument("--output-dir", type=Path, help="New data/processed universe run directory")
    parser.add_argument("--report-dir", type=Path, help="New reports universe run directory")
    parser.add_argument("--config", type=Path, help="JSON UniverseConfig")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        config = load_config(args.config) if args.config else UniverseConfig()
        manifest = run_universe(
            listed_info_dir=args.listed_info_dir,
            prices_dir=args.prices_dir,
            calendar_path=args.calendar,
            schedule_path=args.schedule,
            output_dir=args.output_dir or PROJECT_ROOT / "data" / "processed" / f"universe_{run_id}",
            report_dir=args.report_dir or PROJECT_ROOT / "reports" / f"universe_{run_id}",
            config=config,
        )
    except (ValueError, OSError, pa.ArrowException) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_rows": manifest["output_rows"],
                "eligible_rows": manifest["eligible_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
