"""Build leakage-aware weekly Japanese-equity research datasets.

The module creates a calendar-derived weekly schedule, computes close-to-close
targets from adjusted prices, and joins signal-date descriptors, the point-in-
time universe, and point-in-time earnings exclusions.  Signal eligibility is
kept separate from future target availability and ex-post execution fields.
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
import pyarrow.parquet as pq

from stock_prediction.descriptors import FEATURE_COLUMNS
from stock_prediction.earnings_events import APPENDED_COLUMNS as EARNINGS_COLUMNS
from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "weekly-research-dataset-v2"
SCHEDULE_SCHEMA_ID = "weekly-trading-schedule-v2"
SCHEDULE_COLUMNS = ("signal_date", "entry_date", "exit_date")
PRICE_COLUMNS = ("date", "code", "adjusted_close", "has_price", "calendar_verified")
DESCRIPTOR_COLUMNS = ("date", "code", *FEATURE_COLUMNS)
UNIVERSE_REQUIRED_COLUMNS = (
    "signal_date",
    "code",
    "entry_date",
    "exit_date",
    "is_eligible",
    "exclusion_reasons",
)
EARNINGS_REQUIRED_COLUMNS = ("signal_date", "code", *EARNINGS_COLUMNS)
TARGET_COLUMNS = (
    "signal_adjusted_close",
    "evaluation_adjusted_close",
    "target_return",
    "target_topix_return",
    "target_excess_return",
    "has_target_return",
    "has_target_excess_return",
    "target_missing_reasons",
)
DERIVED_COLUMNS = (
    "trading_sessions_in_week",
    "has_descriptor_row",
    "feature_missing_count",
    "has_complete_features",
    "is_signal_eligible_without_cooldown",
    "is_signal_eligible",
    "is_target_return_sample",
    "is_target_excess_return_sample",
)


@dataclass(frozen=True)
class WeeklyDatasetConfig:
    """Calendar and target settings fixed before final-period evaluation.

    ``min_trading_sessions_per_week=4`` includes ordinary five-session weeks
    and Monday- or Friday-holiday four-session weeks.  Weeks starting after
    Tuesday or ending before Thursday are excluded independently.

    ``signal_start_date`` is the first permitted weekly signal date.  Earlier
    source rows remain available to upstream feature, universe-history, and
    earnings-cooldown calculations, but are not emitted as research weeks.
    """

    min_trading_sessions_per_week: int = 4
    require_topix: bool = True
    signal_start_date: str | None = None

    def __post_init__(self) -> None:
        value = self.min_trading_sessions_per_week
        if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 5:
            raise ValueError("min_trading_sessions_per_week must be an integer from 2 through 5")
        if not isinstance(self.require_topix, bool):
            raise ValueError("require_topix must be boolean")
        if self.signal_start_date is not None:
            _parse_config_date(self.signal_start_date, "signal_start_date")


@dataclass
class WeeklyScheduleResult:
    """Included schedule rows, all calendar-week audit rows, and a summary."""

    schedule: pd.DataFrame
    audit: pd.DataFrame
    report: dict[str, Any]


@dataclass
class WeeklyDatasetResult:
    """Joined weekly rows and an auditable quality summary."""

    dataset: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path | str) -> WeeklyDatasetConfig:
    """Read a strict JSON configuration and reject unknown fields."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(WeeklyDatasetConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return WeeklyDatasetConfig(**values)


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _parse_config_date(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{name} must be an ISO date string in YYYY-MM-DD form")
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed) or pd.Timestamp(parsed).strftime("%Y-%m-%d") != value:
        raise ValueError(f"{name} must be a valid ISO date")
    return pd.Timestamp(parsed)


def _parse_dates(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        result = series.astype("datetime64[ns]")
    else:
        if not all(isinstance(value, str) for value in series.dropna()):
            raise SchemaError(f"{name} must contain ISO dates or timezone-naive datetimes")
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        result = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce")
    if (not nullable and result.isna().any()) or result.dropna().ne(result.dropna().dt.normalize()).any():
        qualifier = "valid midnight dates or null" if nullable else "valid non-null midnight dates"
        raise SchemaError(f"{name} must contain {qualifier}")
    return result.astype("datetime64[ns]")


def _normalize_codes(series: pd.Series, name: str) -> pd.Series:
    if not all(isinstance(value, str) for value in series):
        raise SchemaError(f"{name} must contain strings")
    result = series.astype("string")
    if not result.str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError(f"{name} must contain non-null five-character uppercase codes")
    return result


def _require_columns(frame: pd.DataFrame, required: Sequence[str], source: str) -> None:
    if not frame.columns.is_unique:
        raise SchemaError(f"{source} column names must be unique")
    missing = set(required) - set(frame.columns)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {sorted(missing)}")


def _require_boolean(series: pd.Series, name: str, *, nullable: bool = False) -> None:
    if not is_bool_dtype(series.dtype):
        raise SchemaError(f"{name} must be boolean")
    if not nullable and series.isna().any():
        raise SchemaError(f"{name} must be non-null boolean")


def _validate_key(frame: pd.DataFrame, date_column: str, source: str) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output[date_column] = _parse_dates(output[date_column], f"{source}.{date_column}")
    output["code"] = _normalize_codes(output["code"], f"{source}.code")
    if output.duplicated([date_column, "code"]).any():
        count = int(output.duplicated([date_column, "code"], keep=False).sum())
        raise SchemaError(f"{source} contains {count} duplicate {date_column}/code rows")
    return output


def build_weekly_schedule(
    trading_calendar: pd.DataFrame,
    *,
    config: WeeklyDatasetConfig | None = None,
) -> WeeklyScheduleResult:
    """Create valid weekly signal, next-session entry, and evaluation dates.

    The input must be a continuous J-Quants ``Date``/``HolDiv`` calendar.
    Only complete Monday-through-Sunday calendar weeks are considered.  The
    first TSE session must be Monday or Tuesday, the last must be Thursday or
    Friday, and the configured minimum number of TSE sessions must be present.
    """

    config = config or WeeklyDatasetConfig()
    calendar = validate_trading_calendar(trading_calendar)
    signal_start = (
        None
        if config.signal_start_date is None
        else _parse_config_date(config.signal_start_date, "signal_start_date")
    )
    minimum = calendar["date"].min()
    maximum = calendar["date"].max()
    work = calendar.copy()
    work["calendar_week"] = work["date"].dt.to_period("W-SUN")
    records: list[dict[str, Any]] = []
    for week, group in work.groupby("calendar_week", sort=True, observed=True):
        sessions = group.loc[group["is_session"], "date"].sort_values().reset_index(drop=True)
        week_start = week.start_time.normalize()
        week_end = week.end_time.normalize()
        full_week = bool(week_start >= minimum and week_end <= maximum)
        signal = sessions.iloc[0] if len(sessions) else pd.NaT
        entry = sessions.iloc[1] if len(sessions) >= 2 else pd.NaT
        exit_date = sessions.iloc[-1] if len(sessions) else pd.NaT
        enough_sessions = len(sessions) >= config.min_trading_sessions_per_week
        allowed_signal = bool(pd.notna(signal) and pd.Timestamp(signal).weekday() in (0, 1))
        allowed_exit = bool(pd.notna(exit_date) and pd.Timestamp(exit_date).weekday() in (3, 4))
        has_entry = bool(pd.notna(entry) and pd.notna(exit_date) and entry <= exit_date)
        on_or_after_signal_start = bool(
            signal_start is None or (pd.notna(signal) and pd.Timestamp(signal) >= signal_start)
        )
        included = (
            full_week
            and enough_sessions
            and allowed_signal
            and allowed_exit
            and has_entry
            and on_or_after_signal_start
        )
        reasons: list[str] = []
        if not full_week:
            reasons.append("calendar_boundary_incomplete")
        if not enough_sessions:
            reasons.append("fewer_than_minimum_sessions")
        if not allowed_signal:
            reasons.append("signal_weekday_not_monday_or_tuesday")
        if not allowed_exit:
            reasons.append("evaluation_weekday_not_thursday_or_friday")
        if not has_entry:
            reasons.append("next_session_entry_unavailable")
        if signal_start is not None and pd.notna(signal) and pd.Timestamp(signal) < signal_start:
            reasons.append("signal_before_configured_start")
        records.append(
            {
                "calendar_week": str(week),
                "week_start": week_start,
                "week_end": week_end,
                "trading_sessions_in_week": int(len(sessions)),
                "signal_date": signal,
                "entry_date": entry,
                "exit_date": exit_date,
                "is_included": included,
                "exclusion_reasons": ";".join(reasons),
            }
        )
    audit = pd.DataFrame.from_records(records)
    if audit.empty:
        audit = pd.DataFrame(
            columns=(
                "calendar_week",
                "week_start",
                "week_end",
                "trading_sessions_in_week",
                *SCHEDULE_COLUMNS,
                "is_included",
                "exclusion_reasons",
            )
        )
    schedule = audit.loc[audit["is_included"], list(SCHEDULE_COLUMNS)].reset_index(drop=True)
    for column in SCHEDULE_COLUMNS:
        schedule[column] = schedule[column].astype("datetime64[ns]")
    reason_counts: dict[str, int] = {}
    for value in audit.loc[~audit["is_included"], "exclusion_reasons"]:
        for reason in str(value).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "schema_id": SCHEDULE_SCHEMA_ID,
        "config": asdict(config),
        "calendar_date_min": _date_string(minimum),
        "calendar_date_max": _date_string(maximum),
        "calendar_rows": int(len(calendar)),
        "calendar_sessions": int(calendar["is_session"].sum()),
        "configured_signal_start_date": config.signal_start_date,
        "calendar_weeks": int(len(audit)),
        "included_weeks": int(len(schedule)),
        "excluded_weeks": int((~audit["is_included"]).sum()),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "signal_date_min": _date_string(schedule["signal_date"].min()),
        "signal_date_max": _date_string(schedule["signal_date"].max()),
    }
    return WeeklyScheduleResult(schedule=schedule, audit=audit, report=report)


def _normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, PRICE_COLUMNS, "prices")
    output = _validate_key(frame.loc[:, list(PRICE_COLUMNS)], "date", "prices")
    if not is_numeric_dtype(output["adjusted_close"].dtype) or is_bool_dtype(output["adjusted_close"].dtype):
        raise SchemaError("prices.adjusted_close must be numeric")
    output["adjusted_close"] = output["adjusted_close"].astype("float64")
    for column in ("has_price", "calendar_verified"):
        _require_boolean(output[column], f"prices.{column}")
        output[column] = output[column].astype(bool)
    if not output["calendar_verified"].all():
        raise SchemaError("prices contain calendar_verified=False; weekly targets require the official calendar")
    known = output["has_price"]
    if not (
        output.loc[known, "adjusted_close"].notna()
        & np.isfinite(output.loc[known, "adjusted_close"])
        & output.loc[known, "adjusted_close"].gt(0)
    ).all():
        raise SchemaError("prices has_price=True requires positive finite adjusted_close")
    if output.loc[~known, "adjusted_close"].notna().any():
        raise SchemaError("prices has_price=False rows must not expose adjusted_close")
    return output.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _normalize_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, DESCRIPTOR_COLUMNS, "descriptors")
    output = _validate_key(frame.loc[:, list(DESCRIPTOR_COLUMNS)], "date", "descriptors")
    for column in FEATURE_COLUMNS:
        if not is_numeric_dtype(output[column].dtype) or is_bool_dtype(output[column].dtype):
            raise SchemaError(f"descriptors.{column} must be numeric")
        output[column] = output[column].astype("float64")
        if (output[column].notna() & ~np.isfinite(output[column])).any():
            raise SchemaError(f"descriptors.{column} contains non-finite values")
    return output.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _normalize_universe(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, UNIVERSE_REQUIRED_COLUMNS, "universe")
    output = _validate_key(frame, "signal_date", "universe")
    for column in ("entry_date", "exit_date"):
        output[column] = _parse_dates(output[column], f"universe.{column}")
    _require_boolean(output["is_eligible"], "universe.is_eligible")
    output["is_eligible"] = output["is_eligible"].astype(bool)
    if not all(isinstance(value, str) for value in output["exclusion_reasons"]):
        raise SchemaError("universe.exclusion_reasons must contain non-null strings")
    output["exclusion_reasons"] = output["exclusion_reasons"].astype("string")
    if not (output["signal_date"] < output["entry_date"]).all():
        raise SchemaError("universe.entry_date must follow signal_date")
    if not (output["entry_date"] <= output["exit_date"]).all():
        raise SchemaError("universe.exit_date must not precede entry_date")
    return output.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)


def _normalize_earnings(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, EARNINGS_REQUIRED_COLUMNS, "earnings")
    output = _validate_key(frame, "signal_date", "earnings")
    date_columns = (
        "first_scheduled_earnings_date",
        "first_actual_earnings_date",
        "last_known_earnings_date",
    )
    boolean_columns = (
        "scheduled_earnings_during_holding",
        "active_schedule_date_missing",
        "actual_earnings_during_holding",
        "is_in_post_earnings_cooldown",
        "is_earnings_eligible_without_cooldown",
        "is_earnings_eligible",
        "is_earnings_eligible_ex_post",
    )
    for column in date_columns:
        output[column] = _parse_dates(output[column], f"earnings.{column}", nullable=True)
    for column in boolean_columns:
        _require_boolean(output[column], f"earnings.{column}")
        output[column] = output[column].astype(bool)
    for column in ("scheduled_earnings_count", "actual_earnings_count"):
        if (
            not is_numeric_dtype(output[column].dtype)
            or is_bool_dtype(output[column].dtype)
            or output[column].isna().any()
        ):
            raise SchemaError(f"earnings.{column} must be non-null numeric")
        values = output[column].astype("int64")
        if not values.astype("float64").eq(pd.to_numeric(output[column]).astype("float64")).all() or values.lt(0).any():
            raise SchemaError(f"earnings.{column} must contain non-negative integers")
        output[column] = values
    cooldown = output["trading_sessions_since_last_earnings"]
    if not is_numeric_dtype(cooldown.dtype) or is_bool_dtype(cooldown.dtype):
        raise SchemaError("earnings.trading_sessions_since_last_earnings must be numeric or null")
    numeric_cooldown = pd.to_numeric(cooldown, errors="coerce")
    if (numeric_cooldown.dropna() < 0).any() or not numeric_cooldown.dropna().mod(1).eq(0).all():
        raise SchemaError("earnings.trading_sessions_since_last_earnings must contain non-negative integers or null")
    output["trading_sessions_since_last_earnings"] = numeric_cooldown.astype("Int64")
    for column in ("earnings_exclusion_reasons", "earnings_ex_post_exclusion_reasons"):
        if not all(isinstance(value, str) for value in output[column]):
            raise SchemaError(f"earnings.{column} must contain non-null strings")
        output[column] = output[column].astype("string")
    return output.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)


def _normalize_topix(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    if not frame.columns.is_unique or set(frame.columns) != {"date", "close"}:
        raise SchemaError("TOPIX columns must be exactly date and close")
    output = frame.loc[:, ["date", "close"]].copy().reset_index(drop=True)
    output["date"] = _parse_dates(output["date"], "TOPIX.date")
    if output["date"].duplicated().any():
        raise SchemaError("TOPIX dates must be unique")
    if not output["date"].isin(sessions).all():
        raise SchemaError("TOPIX contains dates which are not TSE trading sessions")
    if not is_numeric_dtype(output["close"].dtype) or is_bool_dtype(output["close"].dtype):
        raise SchemaError("TOPIX.close must be numeric")
    output["close"] = output["close"].astype("float64")
    if (output["close"].notna() & (~np.isfinite(output["close"]) | output["close"].le(0))).any():
        raise SchemaError("TOPIX.close must be positive and finite when present")
    return output.sort_values("date", kind="stable").reset_index(drop=True)


def _assert_matching_keys(left: pd.DataFrame, right: pd.DataFrame, right_name: str) -> None:
    keys = ["signal_date", "code"]
    comparison = left.loc[:, keys].merge(right.loc[:, keys], on=keys, how="outer", indicator=True)
    missing = int(comparison["_merge"].eq("left_only").sum())
    extra = int(comparison["_merge"].eq("right_only").sum())
    if missing or extra:
        raise SchemaError(
            f"{right_name} key coverage must exactly match universe; missing={missing}, extra={extra}"
        )


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if is_datetime64_any_dtype(left.dtype) or is_datetime64_any_dtype(right.dtype):
        left_values = pd.to_datetime(left, errors="coerce")
        right_values = pd.to_datetime(right, errors="coerce")
        return bool((left_values.eq(right_values) | (left_values.isna() & right_values.isna())).all())
    return bool((left.eq(right) | (left.isna() & right.isna())).all())


def _validate_earnings_candidate_copy(universe: pd.DataFrame, earnings: pd.DataFrame) -> None:
    keys = ["signal_date", "code"]
    overlap = [
        column
        for column in universe.columns
        if column in earnings.columns and column not in keys and column not in EARNINGS_COLUMNS
    ]
    if not overlap:
        return
    joined = universe.loc[:, [*keys, *overlap]].merge(
        earnings.loc[:, [*keys, *overlap]],
        on=keys,
        how="inner",
        suffixes=("_universe", "_earnings"),
        validate="one_to_one",
    )
    mismatches = [
        column
        for column in overlap
        if not _series_equal(joined[f"{column}_universe"], joined[f"{column}_earnings"])
    ]
    if mismatches:
        raise SchemaError(f"earnings candidate columns disagree with universe: {sorted(mismatches)}")


def _lookup_prices(
    rows: pd.DataFrame,
    prices: pd.DataFrame,
    date_column: str,
) -> tuple[pd.Series, pd.Series]:
    lookup = prices.set_index(["date", "code"])
    keys = pd.MultiIndex.from_arrays([rows[date_column], rows["code"]])
    has_price = pd.Series(lookup["has_price"].reindex(keys).fillna(False).to_numpy(), index=rows.index, dtype="bool")
    close = pd.Series(
        pd.to_numeric(lookup["adjusted_close"].reindex(keys), errors="coerce").to_numpy(),
        index=rows.index,
        dtype="float64",
    )
    available = has_price & np.isfinite(close) & close.gt(0)
    return close.where(available), available


def _target_missing_reasons(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype="string")
    checks = (
        (frame["signal_adjusted_close"].isna(), "signal_adjusted_close_unavailable"),
        (frame["evaluation_adjusted_close"].isna(), "evaluation_adjusted_close_unavailable"),
        (~frame["has_target_excess_return"] & frame["has_target_return"], "topix_return_unavailable"),
    )
    for mask, label in checks:
        reasons.loc[mask] = reasons.loc[mask] + label + ";"
    return reasons.str.rstrip(";")


def _reason_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        for reason in str(value).split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def build_weekly_dataset(
    prices: pd.DataFrame,
    descriptors: pd.DataFrame,
    universe: pd.DataFrame,
    earnings: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    topix: pd.DataFrame | None,
    config: WeeklyDatasetConfig | None = None,
) -> WeeklyDatasetResult:
    """Join weekly components without fitting preprocessing or selecting a target.

    ``is_signal_eligible`` uses only the point-in-time universe and main
    earnings flag.  Future adjusted closes determine target columns and sample
    availability flags only.  Missing features are retained for fold-local
    preprocessing; missing target prices are reported and never filled.
    """

    config = config or WeeklyDatasetConfig()
    schedule_result = build_weekly_schedule(trading_calendar, config=config)
    schedule = schedule_result.schedule
    calendar = validate_trading_calendar(trading_calendar)
    sessions = pd.DatetimeIndex(calendar.loc[calendar["is_session"], "date"])
    normalized_prices = _normalize_prices(prices)
    normalized_descriptors = _normalize_descriptors(descriptors)
    normalized_universe = _normalize_universe(universe)
    normalized_earnings = _normalize_earnings(earnings)
    if not normalized_prices["date"].isin(sessions).all():
        raise SchemaError("prices contain dates which are not TSE trading sessions")
    if not normalized_descriptors["date"].isin(sessions).all():
        raise SchemaError("descriptors contain dates which are not TSE trading sessions")
    if topix is None:
        if config.require_topix:
            raise SchemaError("TOPIX input is required by configuration")
        normalized_topix = None
    else:
        normalized_topix = _normalize_topix(topix, sessions)

    valid_dates = set(schedule["signal_date"])
    invalid_signals = sorted(set(normalized_universe["signal_date"]) - valid_dates)
    if invalid_signals:
        raise SchemaError(f"universe contains signal dates excluded by weekly schedule: {invalid_signals[:5]}")
    expected = normalized_universe.loc[:, ["signal_date", "entry_date", "exit_date"]].drop_duplicates()
    if expected["signal_date"].duplicated().any():
        raise SchemaError("universe has inconsistent entry_date/exit_date within a signal date")
    expected = expected.merge(schedule, on="signal_date", how="left", suffixes=("_input", "_calendar"))
    for column in ("entry_date", "exit_date"):
        if not expected[f"{column}_input"].eq(expected[f"{column}_calendar"]).all():
            raise SchemaError(f"universe.{column} disagrees with the official weekly schedule")

    _assert_matching_keys(normalized_universe, normalized_earnings, "earnings")
    _validate_earnings_candidate_copy(normalized_universe, normalized_earnings)
    output = normalized_universe.merge(
        normalized_earnings.loc[:, list(EARNINGS_REQUIRED_COLUMNS)],
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
    )
    session_counts = schedule_result.audit.loc[
        schedule_result.audit["is_included"]
    ].set_index("signal_date")["trading_sessions_in_week"]
    output["trading_sessions_in_week"] = output["signal_date"].map(session_counts).astype("int64")

    signal_descriptors = normalized_descriptors.rename(columns={"date": "signal_date"})
    signal_descriptors["has_descriptor_row"] = True
    output = output.merge(
        signal_descriptors,
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
    )
    output["has_descriptor_row"] = output["has_descriptor_row"].fillna(False).astype(bool)
    missing_eligible_descriptors = output["is_eligible"] & ~output["has_descriptor_row"]
    if missing_eligible_descriptors.any():
        raise SchemaError(
            "descriptors are missing for "
            f"{int(missing_eligible_descriptors.sum())} universe-eligible signal rows"
        )
    output["feature_missing_count"] = output.loc[:, list(FEATURE_COLUMNS)].isna().sum(axis=1).astype("int64")
    output["has_complete_features"] = output["feature_missing_count"].eq(0)

    signal_close, signal_available = _lookup_prices(output, normalized_prices, "signal_date")
    evaluation_close, evaluation_available = _lookup_prices(output, normalized_prices, "exit_date")
    output["signal_adjusted_close"] = signal_close
    output["evaluation_adjusted_close"] = evaluation_close
    missing_eligible_signal_price = output["is_eligible"] & ~signal_available
    if missing_eligible_signal_price.any():
        raise SchemaError(
            "adjusted signal close is missing for "
            f"{int(missing_eligible_signal_price.sum())} universe-eligible rows"
        )
    values = np.full(len(output), np.nan, dtype="float64")
    target_available = signal_available & evaluation_available
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values[target_available] = (
            output.loc[target_available, "evaluation_adjusted_close"].to_numpy()
            / output.loc[target_available, "signal_adjusted_close"].to_numpy()
            - 1
        )
    values[~np.isfinite(values)] = np.nan
    output["target_return"] = values
    output["has_target_return"] = np.isfinite(output["target_return"])

    if normalized_topix is None:
        output["target_topix_return"] = np.nan
    else:
        topix_lookup = normalized_topix.set_index("date")["close"]
        signal_topix = output["signal_date"].map(topix_lookup).astype("float64")
        evaluation_topix = output["exit_date"].map(topix_lookup).astype("float64")
        topix_values = np.full(len(output), np.nan, dtype="float64")
        topix_available = (
            np.isfinite(signal_topix)
            & signal_topix.gt(0)
            & np.isfinite(evaluation_topix)
            & evaluation_topix.gt(0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            topix_values[topix_available] = (
                evaluation_topix.loc[topix_available].to_numpy()
                / signal_topix.loc[topix_available].to_numpy()
                - 1
            )
        topix_values[~np.isfinite(topix_values)] = np.nan
        output["target_topix_return"] = topix_values
        if config.require_topix:
            affected_weeks = output.loc[~np.isfinite(output["target_topix_return"]), "signal_date"].nunique()
            if affected_weeks:
                raise SchemaError(f"TOPIX close coverage is incomplete for {affected_weeks} dataset weeks")
    output["target_excess_return"] = output["target_return"] - output["target_topix_return"]
    output["has_target_excess_return"] = np.isfinite(output["target_excess_return"])
    output["target_missing_reasons"] = _target_missing_reasons(output)

    output["is_signal_eligible_without_cooldown"] = (
        output["is_eligible"] & output["is_earnings_eligible_without_cooldown"]
    )
    output["is_signal_eligible"] = output["is_eligible"] & output["is_earnings_eligible"]
    output["is_target_return_sample"] = output["is_signal_eligible"] & output["has_target_return"]
    output["is_target_excess_return_sample"] = (
        output["is_signal_eligible"] & output["has_target_excess_return"]
    )
    output = output.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)

    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "schedule": schedule_result.report,
        "input_price_rows": int(len(normalized_prices)),
        "input_price_securities": int(normalized_prices["code"].nunique()),
        "input_price_date_min": _date_string(normalized_prices["date"].min()),
        "input_price_date_max": _date_string(normalized_prices["date"].max()),
        "input_price_unavailable_rows": int((~normalized_prices["has_price"]).sum()),
        "input_descriptor_rows": int(len(normalized_descriptors)),
        "input_descriptor_securities": int(normalized_descriptors["code"].nunique()),
        "input_descriptor_date_min": _date_string(normalized_descriptors["date"].min()),
        "input_descriptor_date_max": _date_string(normalized_descriptors["date"].max()),
        "input_universe_rows": int(len(normalized_universe)),
        "input_universe_date_min": _date_string(normalized_universe["signal_date"].min()),
        "input_universe_date_max": _date_string(normalized_universe["signal_date"].max()),
        "input_earnings_rows": int(len(normalized_earnings)),
        "input_topix_rows": None if normalized_topix is None else int(len(normalized_topix)),
        "duplicate_key_counts": {"prices": 0, "descriptors": 0, "universe": 0, "earnings": 0},
        "output_rows": int(len(output)),
        "signal_dates": int(output["signal_date"].nunique()),
        "securities": int(output["code"].nunique()),
        "date_min": _date_string(output["signal_date"].min()),
        "date_max": _date_string(output["signal_date"].max()),
        "universe_eligible_rows": int(output["is_eligible"].sum()),
        "earnings_eligible_rows": int(output["is_earnings_eligible"].sum()),
        "signal_eligible_rows": int(output["is_signal_eligible"].sum()),
        "signal_eligible_without_cooldown_rows": int(output["is_signal_eligible_without_cooldown"].sum()),
        "target_return_rows": int(output["has_target_return"].sum()),
        "target_excess_return_rows": int(output["has_target_excess_return"].sum()),
        "target_return_sample_rows": int(output["is_target_return_sample"].sum()),
        "target_excess_return_sample_rows": int(output["is_target_excess_return_sample"].sum()),
        "signal_eligible_missing_target_rows": int(
            (output["is_signal_eligible"] & ~output["has_target_return"]).sum()
        ),
        "missing_descriptor_rows": int((~output["has_descriptor_row"]).sum()),
        "complete_feature_rows": int(output["has_complete_features"].sum()),
        "universe_exclusion_reason_counts": _reason_counts(
            output.loc[~output["is_eligible"], "exclusion_reasons"]
        ),
        "earnings_exclusion_reason_counts": _reason_counts(
            output.loc[~output["is_earnings_eligible"], "earnings_exclusion_reasons"]
        ),
        "active_schedule_date_missing_rows": int(output["active_schedule_date_missing"].sum()),
        "feature_missing_rate": {
            column: float(output[column].isna().mean()) for column in FEATURE_COLUMNS
        },
        "target_missing_reason_counts": _reason_counts(output["target_missing_reasons"]),
        "limitations": [
            "Missing evaluation prices are reported, not filled; delisting outcomes require a later explicit policy.",
            "is_executable and is_earnings_eligible_ex_post are retained for audits but never affect signal eligibility.",
            "Feature imputation, outlier handling, standardization, and target selection remain fold-local later steps.",
            "Adjusted prices may be revised retrospectively after corporate actions; input snapshots must remain identified.",
        ],
    }
    return WeeklyDatasetResult(dataset=output, report=report)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    try:
        temporary.replace(path)
    except PermissionError:
        # Some managed Windows filesystems permit file writes but deny atomic
        # rename.  Preserve progress with a direct replacement as a fallback.
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink()


def _read_table(path: Path, *, string_columns: Sequence[str] = ()) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={column: "string" for column in string_columns})
    raise ValueError(f"Unsupported file type: {path}")


def _annual_partitions(directory: Path, label: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    paths: dict[int, Path] = {}
    for path in sorted(directory.glob("*.parquet")):
        if not re.fullmatch(r"\d{4}", path.stem):
            raise ValueError(f"{label} contains a non-annual Parquet file: {path.name}")
        paths[int(path.stem)] = path
    if not paths:
        raise FileNotFoundError(f"{label} has no YYYY.parquet files: {directory}")
    return paths


def _inspect_partition(path: Path, required: Sequence[str]) -> dict[str, Any]:
    metadata = pq.ParquetFile(path)
    names = metadata.schema_arrow.names
    missing = set(required) - set(names)
    if missing:
        raise SchemaError(f"{path} is missing required columns: {sorted(missing)}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "rows": int(metadata.metadata.num_rows),
        "columns": names,
        "sha256": _sha256(path),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_new_outputs(inputs: Sequence[Path], output_dir: Path, report_dir: Path) -> None:
    if output_dir.resolve() == report_dir.resolve():
        raise ValueError("output_dir and report_dir must differ")
    for path in (output_dir, report_dir):
        if path.exists():
            raise FileExistsError(f"Output path already exists: {path}")
        for source in inputs:
            if source.is_dir() and _is_relative_to(path, source):
                raise ValueError(f"Output path must not be inside an input directory: {path}")


def _versions() -> dict[str, str]:
    return {
        "python": __import__("platform").python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def run_weekly_schedule(
    *,
    calendar_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
    config: WeeklyDatasetConfig | None = None,
) -> dict[str, Any]:
    """Write a new reusable schedule Parquet and JSON audit report."""

    config = config or WeeklyDatasetConfig()
    calendar_path = Path(calendar_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Output path already exists: {path}")
    if output_path.resolve() == calendar_path.resolve() or report_path.resolve() == calendar_path.resolve():
        raise ValueError("Schedule outputs must not overwrite the input calendar")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    calendar = _read_table(calendar_path, string_columns=("Date", "HolDiv"))
    result = build_weekly_schedule(calendar, config=config)
    result.schedule.to_parquet(output_path, index=False)
    report = {
        **result.report,
        "calendar": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "output": {"path": str(output_path), "rows": int(len(result.schedule)), "sha256": _sha256(output_path)},
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }
    _write_json(report_path, report)
    return report


def run_weekly_dataset(
    *,
    prices_dir: Path | str,
    descriptors_dir: Path | str,
    universe_dir: Path | str,
    earnings_dir: Path | str,
    calendar_path: Path | str,
    topix_path: Path | str | None,
    output_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    config: WeeklyDatasetConfig | None = None,
) -> dict[str, Any]:
    """Build annual weekly Parquet files and retain manifest progress on interruption."""

    config = config or WeeklyDatasetConfig()
    prices_dir = Path(prices_dir)
    descriptors_dir = Path(descriptors_dir)
    universe_dir = Path(universe_dir)
    earnings_dir = Path(earnings_dir)
    calendar_path = Path(calendar_path)
    topix_path = None if topix_path is None else Path(topix_path)
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    if topix_path is not None and not topix_path.is_file():
        raise FileNotFoundError(f"TOPIX input does not exist: {topix_path}")
    if config.require_topix and topix_path is None:
        raise FileNotFoundError("TOPIX input is required by configuration")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "data" / "processed" / f"weekly_dataset_{run_id}"
    report_dir = Path(report_dir) if report_dir is not None else PROJECT_ROOT / "reports" / f"weekly_dataset_{run_id}"
    inputs = [prices_dir, descriptors_dir, universe_dir, earnings_dir, calendar_path]
    if topix_path is not None:
        inputs.append(topix_path)
    _ensure_new_outputs(inputs, output_dir, report_dir)

    price_paths = _annual_partitions(prices_dir, "prices")
    descriptor_paths = _annual_partitions(descriptors_dir, "descriptors")
    universe_paths = _annual_partitions(universe_dir, "universe")
    earnings_paths = _annual_partitions(earnings_dir, "earnings")
    years = sorted(universe_paths)
    for year in years:
        for label, paths in (
            ("prices", price_paths),
            ("descriptors", descriptor_paths),
            ("earnings", earnings_paths),
        ):
            if year not in paths:
                raise FileNotFoundError(f"{label} partition is missing for universe year {year}")
    price_inputs = {year: _inspect_partition(price_paths[year], PRICE_COLUMNS) for year in years}
    descriptor_inputs = {
        year: _inspect_partition(descriptor_paths[year], DESCRIPTOR_COLUMNS) for year in years
    }
    universe_inputs = {
        year: _inspect_partition(universe_paths[year], UNIVERSE_REQUIRED_COLUMNS) for year in years
    }
    earnings_inputs = {
        year: _inspect_partition(earnings_paths[year], EARNINGS_REQUIRED_COLUMNS) for year in years
    }
    calendar = _read_table(calendar_path, string_columns=("Date", "HolDiv"))
    topix = None if topix_path is None else _read_table(topix_path)
    schedule_report = build_weekly_schedule(calendar, config=config).report

    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "prices_dir": str(prices_dir),
        "descriptors_dir": str(descriptors_dir),
        "universe_dir": str(universe_dir),
        "earnings_dir": str(earnings_dir),
        "calendar": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "topix": None if topix_path is None else {"path": str(topix_path), "sha256": _sha256(topix_path)},
        "schedule": schedule_report,
        "price_inputs": price_inputs,
        "descriptor_inputs": descriptor_inputs,
        "universe_inputs": universe_inputs,
        "earnings_inputs": earnings_inputs,
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "completed_partitions": [],
        "current_partition": None,
        "output_rows": 0,
        "signal_eligible_rows": 0,
        "target_return_sample_rows": 0,
        "target_excess_return_sample_rows": 0,
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
        "random_seed": None,
    }
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    try:
        for year in years:
            manifest["current_partition"] = year
            _write_json(manifest_path, manifest)
            prices = pq.read_table(price_paths[year], columns=list(PRICE_COLUMNS)).to_pandas()
            descriptors = pq.read_table(descriptor_paths[year], columns=list(DESCRIPTOR_COLUMNS)).to_pandas()
            universe = pd.read_parquet(universe_paths[year])
            earnings = pd.read_parquet(earnings_paths[year])
            result = build_weekly_dataset(
                prices,
                descriptors,
                universe,
                earnings,
                trading_calendar=calendar,
                topix=topix,
                config=config,
            )
            if not result.dataset["signal_date"].dt.year.eq(year).all():
                raise SchemaError(f"universe partition {year} contains signal dates from another year")
            output_path = output_dir / f"{year}.parquet"
            result.dataset.to_parquet(output_path, index=False)
            partition_report = {
                **result.report,
                "year": year,
                "output_path": str(output_path),
                "output_sha256": _sha256(output_path),
            }
            _write_json(report_dir / f"{year}.json", partition_report)
            summary = {
                "year": year,
                "output_rows": result.report["output_rows"],
                "signal_dates": result.report["signal_dates"],
                "signal_eligible_rows": result.report["signal_eligible_rows"],
                "target_return_sample_rows": result.report["target_return_sample_rows"],
                "target_excess_return_sample_rows": result.report["target_excess_return_sample_rows"],
                "output_sha256": partition_report["output_sha256"],
            }
            manifest["completed_partitions"].append(summary)
            for column in (
                "output_rows",
                "signal_eligible_rows",
                "target_return_sample_rows",
                "target_excess_return_sample_rows",
            ):
                manifest[column] += summary[column]
            manifest["current_partition"] = None
            _write_json(manifest_path, manifest)
            LOGGER.info(
                "%s: rows=%d signal_eligible=%d return_samples=%d excess_samples=%d",
                year,
                summary["output_rows"],
                summary["signal_eligible_rows"],
                summary["target_return_sample_rows"],
                summary["target_excess_return_sample_rows"],
            )
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for reusable schedule creation and dataset assembly."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    schedule_parser = subparsers.add_parser("schedule", help="Create the reusable weekly schedule")
    schedule_parser.add_argument("--calendar", type=Path, required=True)
    schedule_parser.add_argument("--output", type=Path, required=True)
    schedule_parser.add_argument("--report", type=Path, required=True)
    schedule_parser.add_argument("--config", type=Path)

    build_parser = subparsers.add_parser("build", help="Join annual weekly research datasets")
    build_parser.add_argument("--prices-dir", type=Path, required=True)
    build_parser.add_argument("--descriptors-dir", type=Path, required=True)
    build_parser.add_argument("--universe-dir", type=Path, required=True)
    build_parser.add_argument("--earnings-dir", type=Path, required=True)
    build_parser.add_argument("--calendar", type=Path, required=True)
    build_parser.add_argument("--topix", type=Path)
    build_parser.add_argument("--output-dir", type=Path)
    build_parser.add_argument("--report-dir", type=Path)
    build_parser.add_argument("--config", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config(args.config) if args.config else WeeklyDatasetConfig()
    if args.command == "schedule":
        report = run_weekly_schedule(
            calendar_path=args.calendar,
            output_path=args.output,
            report_path=args.report,
            config=config,
        )
        print(json.dumps({"status": "complete", "included_weeks": report["included_weeks"]}, ensure_ascii=False))
        return 0
    manifest = run_weekly_dataset(
        prices_dir=args.prices_dir,
        descriptors_dir=args.descriptors_dir,
        universe_dir=args.universe_dir,
        earnings_dir=args.earnings_dir,
        calendar_path=args.calendar,
        topix_path=args.topix,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_rows": manifest["output_rows"],
                "signal_eligible_rows": manifest["signal_eligible_rows"],
                "target_return_sample_rows": manifest["target_return_sample_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
