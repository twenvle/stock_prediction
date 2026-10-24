"""Apply point-in-time earnings exclusions to weekly Japanese-equity candidates.

The signal-time decision uses only schedule publications available by the
signal date and actual earnings disclosures available by the historical TSE
close.  Later disclosures during the intended holding period are retained as
a separate ex-post audit and never alter ``is_earnings_eligible``.

Both the notebook-facing DataFrame API and the annual-file CLI use
``compute_earnings_exclusions``.  Raw inputs are never modified or overwritten.
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
from pandas.api.types import is_datetime64_any_dtype
import pyarrow as pa
import pyarrow.parquet as pq

from stock_prediction.auxiliary_data_download import (
    EARNINGS_SCHEDULE_COLUMNS,
    FINANCIAL_RESULTS_COLUMNS,
    FINANCIAL_RESULTS_REQUIRED_COLUMNS,
)
from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "earnings-exclusions-v3"
DEFAULT_EARNINGS_DOC_TYPE_PREFIXES = (
    "1QFinancialStatements",
    "2QFinancialStatements",
    "3QFinancialStatements",
    "FYFinancialStatements",
    "OtherPeriodFinancialStatements",
)
TSE_CLOSE_EXTENSION_DATE = pd.Timestamp("2024-11-05")
TSE_CLOSE_SECONDS_BEFORE_EXTENSION = 15 * 60 * 60
TSE_CLOSE_SECONDS_AFTER_EXTENSION = 15 * 60 * 60 + 30 * 60
CANDIDATE_REQUIRED_COLUMNS = ("signal_date", "code", "exit_date")
SCHEDULE_EVENT_KEY = ("Code", "FQName", "FYE")
SCHEDULE_PUBLICATION_KEY = (*SCHEDULE_EVENT_KEY, "PubDate")
APPENDED_COLUMNS = (
    "scheduled_earnings_during_holding",
    "scheduled_earnings_count",
    "first_scheduled_earnings_date",
    "active_schedule_date_missing",
    "actual_earnings_during_holding",
    "actual_earnings_count",
    "first_actual_earnings_date",
    "last_known_earnings_date",
    "trading_sessions_since_last_earnings",
    "is_in_post_earnings_cooldown",
    "is_earnings_eligible_without_cooldown",
    "is_earnings_eligible",
    "is_earnings_eligible_ex_post",
    "earnings_exclusion_reasons",
    "earnings_ex_post_exclusion_reasons",
)


@dataclass(frozen=True)
class EarningsEventConfig:
    """Fixed earnings-event rules.

    ``cooldown_trading_sessions`` excludes the first N official TSE sessions
    after an actual earnings disclosure.  A signal on session N remains in the
    cooldown and session N+1 is eligible again.

    ``earnings_doc_type_prefixes`` identifies earnings statements in the
    J-Quants financial-summary ``DocType`` field.  Forecast/dividend revisions
    are intentionally not earnings events unless explicitly configured.
    """

    cooldown_trading_sessions: int = 25
    earnings_doc_type_prefixes: tuple[str, ...] = DEFAULT_EARNINGS_DOC_TYPE_PREFIXES

    def __post_init__(self) -> None:
        if (
            isinstance(self.cooldown_trading_sessions, bool)
            or not isinstance(self.cooldown_trading_sessions, int)
            or self.cooldown_trading_sessions <= 0
        ):
            raise ValueError("cooldown_trading_sessions must be a positive integer")
        prefixes = self.earnings_doc_type_prefixes
        if not isinstance(prefixes, tuple) or not prefixes:
            raise ValueError("earnings_doc_type_prefixes must be a non-empty tuple")
        if any(not isinstance(value, str) or not value.strip() for value in prefixes):
            raise ValueError("earnings_doc_type_prefixes must contain non-empty strings")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("earnings_doc_type_prefixes must not contain duplicates")


@dataclass
class EarningsEventResult:
    """Annotated candidate rows and their audit report."""

    exclusions: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path | str) -> EarningsEventConfig:
    """Read JSON configuration and reject unknown fields."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(EarningsEventConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if "earnings_doc_type_prefixes" in values:
        prefixes = values["earnings_doc_type_prefixes"]
        if not isinstance(prefixes, list):
            raise ValueError("earnings_doc_type_prefixes must be a JSON array")
        values["earnings_doc_type_prefixes"] = tuple(prefixes)
    return EarningsEventConfig(**values)


def _require_string_values(series: pd.Series, name: str) -> None:
    if not all(isinstance(value, str) for value in series.dropna()):
        raise SchemaError(f"{name} must contain strings, not coerced numeric values")


def _parse_dates(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must be timezone-naive dates")
    if is_datetime64_any_dtype(series.dtype):
        parsed = series.astype("datetime64[ns]")
    else:
        _require_string_values(series, name)
        normalized = series.replace("", pd.NA).astype("string")
        valid = normalized.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=nullable)
        parsed = pd.to_datetime(normalized.where(valid), format="%Y-%m-%d", errors="coerce")
        parsed = parsed.astype("datetime64[ns]")
    if (not nullable and parsed.isna().any()) or parsed.dropna().ne(parsed.dropna().dt.normalize()).any():
        raise SchemaError(f"{name} must contain valid midnight dates")
    return parsed


def _normalize_candidates(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise SchemaError("candidates must be a DataFrame with unique column names")
    missing = set(CANDIDATE_REQUIRED_COLUMNS) - set(frame.columns)
    conflicts = set(APPENDED_COLUMNS).intersection(frame.columns)
    if missing or conflicts:
        raise SchemaError(
            f"Unsupported candidate schema; missing={sorted(missing)}, output_conflicts={sorted(conflicts)}"
        )
    output = frame.copy(deep=True).reset_index(drop=True)
    for column in ("signal_date", "exit_date"):
        output[column] = _parse_dates(output[column], f"candidates.{column}")
        if not output[column].isin(sessions).all():
            raise SchemaError(f"candidates.{column} contains non-session dates")
    _require_string_values(output["code"], "candidates.code")
    output["code"] = output["code"].astype("string")
    if output["code"].isna().any() or not output["code"].str.fullmatch(r"[0-9A-Z]{5}").all():
        raise SchemaError("candidates.code must contain non-null five-character uppercase codes")
    if output.duplicated(["signal_date", "code"], keep=False).any():
        raise SchemaError("candidates contains duplicate signal_date/code keys")
    if not output["signal_date"].lt(output["exit_date"]).all():
        raise SchemaError("candidate exit_date must be later than signal_date")

    session_frame = pd.DataFrame({"date": sessions})
    session_frame["week"] = session_frame["date"].dt.to_period("W-SUN")
    weekly = session_frame.groupby("week", observed=True)["date"].agg(first="min", last="max")
    weeks = output["signal_date"].dt.to_period("W-SUN")
    if not output["signal_date"].eq(weeks.map(weekly["first"])).all():
        raise SchemaError("candidate signal_date must be the first TSE session of its calendar week")
    if not output["exit_date"].eq(weeks.map(weekly["last"])).all():
        raise SchemaError("candidate exit_date must be the last TSE session of the signal calendar week")

    if "entry_date" in output.columns:
        output["entry_date"] = _parse_dates(output["entry_date"], "candidates.entry_date")
        if not output["entry_date"].isin(sessions).all():
            raise SchemaError("candidates.entry_date contains non-session dates")
        session_ids = pd.Series(np.arange(len(sessions), dtype="int64"), index=sessions)
        if not output["entry_date"].map(session_ids).eq(output["signal_date"].map(session_ids) + 1).all():
            raise SchemaError("candidate entry_date must be the TSE session immediately after signal_date")
        if not output["entry_date"].le(output["exit_date"]).all():
            raise SchemaError("candidate entry_date must not be later than exit_date")
    return output.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)


def _normalize_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise SchemaError("earnings_schedule must be a DataFrame with unique column names")
    if set(frame.columns) != set(EARNINGS_SCHEDULE_COLUMNS):
        missing = set(EARNINGS_SCHEDULE_COLUMNS) - set(frame.columns)
        extra = set(frame.columns) - set(EARNINGS_SCHEDULE_COLUMNS)
        raise SchemaError(f"Unsupported earnings_schedule schema; missing={sorted(missing)}, extra={sorted(extra)}")
    output = frame.loc[:, list(EARNINGS_SCHEDULE_COLUMNS)].copy().reset_index(drop=True)
    output["PubDate"] = _parse_dates(output["PubDate"], "earnings_schedule.PubDate")
    output["SchDate"] = _parse_dates(output["SchDate"], "earnings_schedule.SchDate", nullable=True)
    for column in set(EARNINGS_SCHEDULE_COLUMNS) - {"PubDate", "SchDate"}:
        _require_string_values(output[column], f"earnings_schedule.{column}")
        output[column] = output[column].astype("string")
    if output["Code"].isna().any() or not output["Code"].str.fullmatch(r"[0-9A-Z]{5}").all():
        raise SchemaError("earnings_schedule.Code must contain non-null five-character uppercase codes")
    if output.duplicated(keep=False).any():
        raise SchemaError("earnings_schedule contains exact duplicate rows")
    return output.sort_values(
        [*SCHEDULE_PUBLICATION_KEY, "SchDate"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _same_day_schedule_state_stats(schedule: pd.DataFrame) -> dict[str, int]:
    """Count publication-date groups with more than one reported state."""

    duplicated = schedule.duplicated(list(SCHEDULE_PUBLICATION_KEY), keep=False)
    same_day_rows = schedule.loc[duplicated]
    groups = same_day_rows.loc[:, list(SCHEDULE_PUBLICATION_KEY)].drop_duplicates()
    groups_with_missing_date = same_day_rows.loc[
        same_day_rows["SchDate"].isna(), list(SCHEDULE_PUBLICATION_KEY)
    ].drop_duplicates()
    return {
        "same_day_schedule_state_groups": int(len(groups)),
        "same_day_schedule_state_rows": int(len(same_day_rows)),
        "same_day_schedule_state_groups_with_missing_date": int(len(groups_with_missing_date)),
    }


def _normalize_financial_results(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise SchemaError("financial_results must be a DataFrame with unique column names")
    missing = set(FINANCIAL_RESULTS_REQUIRED_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(FINANCIAL_RESULTS_COLUMNS)
    if missing or extra:
        raise SchemaError(f"Unsupported financial_results schema; missing={sorted(missing)}, extra={sorted(extra)}")
    output = frame.copy(deep=True).reset_index(drop=True)
    output["DiscDate"] = _parse_dates(output["DiscDate"], "financial_results.DiscDate")
    for column in ("DiscTime", "Code", "DiscNo", "DocType"):
        _require_string_values(output[column], f"financial_results.{column}")
        output[column] = output[column].astype("string")
    if output["Code"].isna().any() or not output["Code"].str.fullmatch(r"[0-9A-Z]{5}").all():
        raise SchemaError("financial_results.Code must contain non-null five-character uppercase codes")
    times = output["DiscTime"].dropna()
    if not times.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?").all():
        raise SchemaError("financial_results.DiscTime must be an HH:MM or HH:MM:SS string")
    if output["DiscNo"].isna().any() or output.duplicated(["DiscDate", "DiscNo"], keep=False).any():
        raise SchemaError("financial_results requires unique, non-null DiscDate/DiscNo keys")
    return output.sort_values(["DiscDate", "DiscTime", "Code", "DiscNo"], kind="stable").reset_index(drop=True)


def _schedule_annotations(candidates: pd.DataFrame, schedule: pd.DataFrame) -> dict[str, pd.Series]:
    counts = np.zeros(len(candidates), dtype="int64")
    first_dates = np.full(len(candidates), np.datetime64("NaT"), dtype="datetime64[ns]")
    missing_active = np.zeros(len(candidates), dtype=bool)
    candidate_groups = candidates.groupby("code", sort=False).indices
    for code, revisions_for_code in schedule.groupby("Code", sort=False):
        positions = candidate_groups.get(str(code))
        if positions is None:
            continue
        positions = np.asarray(positions, dtype="int64")
        signals = candidates.loc[positions, "signal_date"].to_numpy(dtype="datetime64[ns]")
        exits = candidates.loc[positions, "exit_date"].to_numpy(dtype="datetime64[ns]")
        for _, revisions in revisions_for_code.groupby(["FQName", "FYE"], dropna=False, sort=False):
            publication_dates = revisions["PubDate"].drop_duplicates().sort_values().reset_index(drop=True)
            next_publications = publication_dates.shift(-1)
            for publication_position, publication_value in publication_dates.items():
                publication = np.datetime64(publication_value, "ns")
                active = signals >= publication
                next_publication = next_publications.iloc[publication_position]
                if not pd.isna(next_publication):
                    active &= signals < np.datetime64(next_publication, "ns")
                same_day_states = revisions.loc[revisions["PubDate"].eq(publication_value)]
                if same_day_states["SchDate"].isna().any():
                    missing_active[positions[active]] = True
                scheduled_dates = same_day_states["SchDate"].dropna().drop_duplicates()
                event_hits = np.zeros(len(positions), dtype=bool)
                first_event_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
                for scheduled in scheduled_dates:
                    scheduled_date = np.datetime64(scheduled, "ns")
                    hits = active & (signals <= scheduled_date) & (scheduled_date <= exits)
                    event_hits |= hits
                    replace = hits & (np.isnat(first_event_dates) | (scheduled_date < first_event_dates))
                    first_event_dates[replace] = scheduled_date
                hit_positions = positions[event_hits]
                if not len(hit_positions):
                    continue
                counts[hit_positions] += 1
                existing = first_dates[hit_positions]
                event_dates = first_event_dates[event_hits]
                replace = np.isnat(existing) | (event_dates < existing)
                first_dates[hit_positions[replace]] = event_dates[replace]
    return {
        "scheduled_earnings_count": pd.Series(counts, dtype="int64"),
        "first_scheduled_earnings_date": pd.Series(first_dates, dtype="datetime64[ns]"),
        "active_schedule_date_missing": pd.Series(missing_active, dtype="bool"),
    }


def _actual_annotations(
    candidates: pd.DataFrame,
    actual_events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cooldown_sessions: int,
) -> dict[str, pd.Series]:
    n_rows = len(candidates)
    during_counts = np.zeros(n_rows, dtype="int64")
    first_during = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    last_before = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
    elapsed = np.full(n_rows, -1, dtype="int64")
    candidate_groups = candidates.groupby("code", sort=False).indices
    session_ids = pd.Series(np.arange(len(sessions), dtype="int64"), index=sessions)
    signal_session_ids = candidates["signal_date"].map(session_ids).to_numpy(dtype="int64")
    session_values = sessions.to_numpy(dtype="datetime64[ns]")
    signal_dates = candidates["signal_date"].to_numpy(dtype="datetime64[ns]")
    extension_date = np.datetime64(TSE_CLOSE_EXTENSION_DATE, "ns")
    signal_close_seconds = np.where(
        signal_dates < extension_date,
        TSE_CLOSE_SECONDS_BEFORE_EXTENSION,
        TSE_CLOSE_SECONDS_AFTER_EXTENSION,
    )

    for code, events_for_code in actual_events.groupby("Code", sort=False):
        positions = candidate_groups.get(str(code))
        if positions is None:
            continue
        positions = np.asarray(positions, dtype="int64")
        event_days = events_for_code.loc[:, ["DiscDate", "DiscTime"]].copy()
        split_times = event_days["DiscTime"].str.extract(
            r"(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
        )
        event_days["_time_seconds"] = (
            pd.to_numeric(split_times["hour"], errors="coerce") * 60 * 60
            + pd.to_numeric(split_times["minute"], errors="coerce") * 60
            + pd.to_numeric(split_times["second"], errors="coerce").fillna(0)
        )
        by_date = event_days.groupby("DiscDate", sort=True, as_index=False)["_time_seconds"].min()
        event_dates = by_date["DiscDate"].to_numpy(dtype="datetime64[ns]")
        event_time_seconds = by_date["_time_seconds"].to_numpy(dtype="float64")
        signals = candidates.loc[positions, "signal_date"].to_numpy(dtype="datetime64[ns]")
        exits = candidates.loc[positions, "exit_date"].to_numpy(dtype="datetime64[ns]")
        same_day_indices = np.searchsorted(event_dates, signals, side="left")
        has_same_day = same_day_indices < len(event_dates)
        has_same_day &= np.where(
            has_same_day,
            event_dates[np.minimum(same_day_indices, len(event_dates) - 1)] == signals,
            False,
        )
        same_day_known = np.zeros(len(positions), dtype=bool)
        if has_same_day.any():
            same_times = event_time_seconds[same_day_indices[has_same_day]]
            same_day_known[has_same_day] = (
                np.isfinite(same_times) & (same_times <= signal_close_seconds[positions[has_same_day]])
            )

        first_indices = np.searchsorted(event_dates, signals, side="right")
        after_indices = np.searchsorted(event_dates, exits, side="right")
        same_day_after_close = has_same_day & ~same_day_known
        counts = after_indices - first_indices + same_day_after_close.astype("int64")
        during_counts[positions] = counts
        has_during = counts > 0
        if has_during.any():
            first_values = np.full(has_during.sum(), np.datetime64("NaT"), dtype="datetime64[ns]")
            during_local_positions = np.flatnonzero(has_during)
            is_same_day_first = same_day_after_close[during_local_positions]
            first_values[is_same_day_first] = signals[during_local_positions[is_same_day_first]]
            later_local_positions = during_local_positions[~is_same_day_first]
            if len(later_local_positions):
                first_values[~is_same_day_first] = event_dates[first_indices[later_local_positions]]
            first_during[positions[has_during]] = first_values

        prior_indices = np.searchsorted(event_dates, signals, side="left") - 1
        has_prior = prior_indices >= 0
        if has_prior.any():
            prior_positions = positions[has_prior]
            prior_dates = event_dates[prior_indices[has_prior]]
            last_before[prior_positions] = prior_dates
            sessions_through_event = np.searchsorted(session_values, prior_dates, side="right")
            elapsed[prior_positions] = signal_session_ids[prior_positions] + 1 - sessions_through_event
        if same_day_known.any():
            known_positions = positions[same_day_known]
            last_before[known_positions] = signals[same_day_known]
            elapsed[known_positions] = 0

    elapsed_series = pd.Series(pd.array(elapsed, dtype="Int64"))
    elapsed_series = elapsed_series.mask(elapsed_series.lt(0), pd.NA)
    cooldown = elapsed_series.notna() & elapsed_series.le(cooldown_sessions)
    return {
        "actual_earnings_count": pd.Series(during_counts, dtype="int64"),
        "first_actual_earnings_date": pd.Series(first_during, dtype="datetime64[ns]"),
        "last_known_earnings_date": pd.Series(last_before, dtype="datetime64[ns]"),
        "trading_sessions_since_last_earnings": elapsed_series,
        "is_in_post_earnings_cooldown": cooldown.astype(bool),
    }


def _reason_column(frame: pd.DataFrame, *, ex_post: bool) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype="string")
    conditions = [
        (frame["scheduled_earnings_during_holding"], "scheduled_earnings_during_holding"),
        (frame["is_in_post_earnings_cooldown"], "post_earnings_cooldown"),
    ]
    if ex_post:
        conditions.append((frame["actual_earnings_during_holding"], "actual_earnings_during_holding_ex_post"))
    for mask, name in conditions:
        reasons.loc[mask] = reasons.loc[mask] + name + ";"
    return reasons.str.rstrip(";")


def compute_earnings_exclusions(
    candidates: pd.DataFrame,
    earnings_schedule: pd.DataFrame,
    financial_results: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    config: EarningsEventConfig | None = None,
) -> EarningsEventResult:
    """Annotate weekly candidate rows with point-in-time earnings exclusions.

    Required candidate columns are ``signal_date``, ``code`` and ``exit_date``.
    If ``entry_date`` is supplied, it is validated as the next TSE session.
    Other candidate columns are preserved.  Inputs are copied and never
    modified.

    The main ``is_earnings_eligible`` flag excludes an as-of scheduled event
    in the inclusive date interval ``[signal_date, exit_date]`` and excludes
    signals on cooldown day zero through the configured number of TSE sessions
    after an actual earnings disclosure available by the historical TSE close.
    The inclusive schedule start is conservative because that input has no
    announcement time.  Later actual events affect only the explicitly
    ex-post audit columns.
    """

    config = config or EarningsEventConfig()
    validated_calendar = validate_trading_calendar(trading_calendar)
    sessions = pd.DatetimeIndex(validated_calendar.loc[validated_calendar["is_session"], "date"])
    normalized_candidates = _normalize_candidates(candidates, sessions)
    normalized_schedule = _normalize_schedule(earnings_schedule)
    same_day_schedule_stats = _same_day_schedule_state_stats(normalized_schedule)
    normalized_financials = _normalize_financial_results(financial_results)
    if normalized_candidates.empty:
        raise SchemaError("candidates must contain at least one row")
    if normalized_candidates["signal_date"].min() < validated_calendar["date"].min() or normalized_candidates[
        "exit_date"
    ].max() > validated_calendar["date"].max():
        raise SchemaError("trading calendar does not cover all candidate signal/exit dates")
    first_signal_id = int(sessions.get_loc(normalized_candidates["signal_date"].min()))
    if first_signal_id < config.cooldown_trading_sessions:
        raise SchemaError(
            "trading calendar must contain at least cooldown_trading_sessions sessions before the first signal"
        )

    schedule_annotations = _schedule_annotations(normalized_candidates, normalized_schedule)
    doc_type_counts = {
        str(key): int(value)
        for key, value in normalized_financials["DocType"].fillna("<missing>").value_counts().sort_index().items()
    }
    is_earnings_statement = normalized_financials["DocType"].fillna("").str.startswith(
        config.earnings_doc_type_prefixes
    )
    statement_like = normalized_financials["DocType"].fillna("").str.contains(
        "FinancialStatements", regex=False
    )
    unmatched_statement_types = sorted(
        normalized_financials.loc[statement_like & ~is_earnings_statement, "DocType"].dropna().unique().tolist()
    )
    if unmatched_statement_types:
        raise SchemaError(
            "financial_results contains FinancialStatements DocType values not matched by "
            f"earnings_doc_type_prefixes: {unmatched_statement_types}"
        )
    actual_events = normalized_financials.loc[
        is_earnings_statement, ["DiscDate", "DiscTime", "Code", "DiscNo", "DocType"]
    ].copy()
    actual_annotations = _actual_annotations(
        normalized_candidates,
        actual_events,
        sessions,
        config.cooldown_trading_sessions,
    )

    output = normalized_candidates.copy()
    for name, values in (*schedule_annotations.items(), *actual_annotations.items()):
        output[name] = values.reset_index(drop=True)
    output["scheduled_earnings_during_holding"] = output["scheduled_earnings_count"].gt(0)
    output["actual_earnings_during_holding"] = output["actual_earnings_count"].gt(0)
    output["is_earnings_eligible_without_cooldown"] = ~output["scheduled_earnings_during_holding"]
    output["is_earnings_eligible"] = (
        output["is_earnings_eligible_without_cooldown"] & ~output["is_in_post_earnings_cooldown"]
    )
    output["is_earnings_eligible_ex_post"] = (
        output["is_earnings_eligible"] & ~output["actual_earnings_during_holding"]
    )
    output["earnings_exclusion_reasons"] = _reason_column(output, ex_post=False)
    output["earnings_ex_post_exclusion_reasons"] = _reason_column(output, ex_post=True)
    ordered_columns = [*normalized_candidates.columns, *APPENDED_COLUMNS]
    output = output.loc[:, ordered_columns].sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)

    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "candidate_rows": int(len(output)),
        "signal_dates": int(output["signal_date"].nunique()),
        "securities": int(output["code"].nunique()),
        "date_min": _date_string(output["signal_date"].min()),
        "date_max": _date_string(output["signal_date"].max()),
        "earnings_schedule_rows": int(len(normalized_schedule)),
        "schedule_publication_date_min": _date_string(normalized_schedule["PubDate"].min()),
        "schedule_publication_date_max": _date_string(normalized_schedule["PubDate"].max()),
        "schedule_missing_date_rows": int(normalized_schedule["SchDate"].isna().sum()),
        **same_day_schedule_stats,
        "financial_result_rows": int(len(normalized_financials)),
        "actual_earnings_event_rows": int(len(actual_events)),
        "financial_doc_type_counts": doc_type_counts,
        "scheduled_crossing_rows": int(output["scheduled_earnings_during_holding"].sum()),
        "actual_crossing_rows_ex_post": int(output["actual_earnings_during_holding"].sum()),
        "cooldown_rows": int(output["is_in_post_earnings_cooldown"].sum()),
        "eligible_without_cooldown_rows": int(output["is_earnings_eligible_without_cooldown"].sum()),
        "eligible_rows": int(output["is_earnings_eligible"].sum()),
        "eligible_rows_ex_post": int(output["is_earnings_eligible_ex_post"].sum()),
        "active_schedule_date_missing_rows": int(output["active_schedule_date_missing"].sum()),
        "limitations": [
            "Scheduled dates have no announcement time, so signal-date events are conservatively included.",
            "Multiple states on one publication date are retained as a set because their within-day order is unavailable.",
            "Same-day disclosures at/before the historical TSE close are cooldown day 0; later or missing times are ex-post events.",
            "Actual holding-period disclosures never affect is_earnings_eligible.",
            "Input event-history completeness must be established from the immutable source snapshot manifest.",
        ],
    }
    return EarningsEventResult(exclusions=output, report=report)


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_table(path: Path, *, string_columns: Sequence[str] = ()) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={column: "string" for column in string_columns})
    else:
        raise ValueError(f"Unsupported table extension: {path}")
    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def _annual_partitions(directory: Path, label: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in directory.glob("*.parquet"):
        if not re.fullmatch(r"\d{4}\.parquet", path.name):
            continue
        year = int(path.stem)
        paths[year] = path
    if not paths:
        raise FileNotFoundError(f"No YYYY.parquet files found in {label}: {directory}")
    return dict(sorted(paths.items()))


def _inspect_partition(path: Path, required_columns: Sequence[str]) -> dict[str, Any]:
    metadata = pq.read_metadata(path)
    columns = metadata.schema.names
    missing = set(required_columns) - set(columns)
    if missing:
        raise SchemaError(f"{path} is missing required columns: {sorted(missing)}")
    return {"path": str(path), "rows": int(metadata.num_rows), "sha256": _sha256(path)}


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


def _read_event_partitions(
    paths: dict[int, Path],
    *,
    columns: Sequence[str],
    partition_date_column: str,
    label: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year, path in paths.items():
        names = set(pq.read_schema(path).names)
        selected = [column for column in columns if column in names]
        frame = pq.read_table(path, columns=selected).to_pandas()
        if not frame.empty:
            dates = _parse_dates(frame[partition_date_column], f"{path.name}.{partition_date_column}")
            if not dates.dt.year.eq(year).all():
                raise SchemaError(f"{path.name} contains {label} rows outside {year}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def run_earnings_events(
    *,
    candidates_dir: Path | str,
    earnings_schedule_dir: Path | str,
    financial_results_dir: Path | str,
    calendar_path: Path | str,
    output_dir: Path | str,
    report_dir: Path | str,
    config: EarningsEventConfig | None = None,
) -> dict[str, Any]:
    """Process annual candidate files and save reproducible outputs/reports."""

    config = config or EarningsEventConfig()
    candidates_dir = Path(candidates_dir)
    earnings_schedule_dir = Path(earnings_schedule_dir)
    financial_results_dir = Path(financial_results_dir)
    calendar_path = Path(calendar_path)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    for path, label in (
        (candidates_dir, "candidates"),
        (earnings_schedule_dir, "earnings schedule"),
        (financial_results_dir, "financial results"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    _ensure_output_paths(
        [candidates_dir, earnings_schedule_dir, financial_results_dir, calendar_path],
        output_dir,
        report_dir,
    )

    candidate_paths = _annual_partitions(candidates_dir, "candidates")
    schedule_paths = _annual_partitions(earnings_schedule_dir, "earnings schedule")
    financial_paths = _annual_partitions(financial_results_dir, "financial results")
    candidate_inputs = {
        year: _inspect_partition(path, CANDIDATE_REQUIRED_COLUMNS) for year, path in candidate_paths.items()
    }
    schedule_inputs = {
        year: _inspect_partition(path, EARNINGS_SCHEDULE_COLUMNS) for year, path in schedule_paths.items()
    }
    financial_inputs = {
        year: _inspect_partition(path, FINANCIAL_RESULTS_REQUIRED_COLUMNS)
        for year, path in financial_paths.items()
    }
    calendar = _read_table(calendar_path, string_columns=("Date", "HolDiv"))
    earnings_schedule = _read_event_partitions(
        schedule_paths,
        columns=EARNINGS_SCHEDULE_COLUMNS,
        partition_date_column="PubDate",
        label="publication",
    )
    financial_results = _read_event_partitions(
        financial_paths,
        columns=FINANCIAL_RESULTS_REQUIRED_COLUMNS,
        partition_date_column="DiscDate",
        label="disclosure",
    )
    same_day_schedule_stats = _same_day_schedule_state_stats(earnings_schedule)

    source_path = Path(__file__)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "candidates_dir": str(candidates_dir),
        "earnings_schedule_dir": str(earnings_schedule_dir),
        "financial_results_dir": str(financial_results_dir),
        "calendar": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "candidate_inputs": candidate_inputs,
        "earnings_schedule_inputs": schedule_inputs,
        "financial_results_inputs": financial_inputs,
        "completed_partitions": [],
        "current_partition": None,
        "output_rows": 0,
        "eligible_rows": 0,
        "scheduled_crossing_rows": 0,
        "cooldown_rows": 0,
        "actual_crossing_rows_ex_post": 0,
        **same_day_schedule_stats,
        "code_sha256": {
            "earnings_events.py": _sha256(source_path),
            "pyproject.toml": _sha256(PROJECT_ROOT / "pyproject.toml"),
        },
        "versions": _versions(),
        "random_seed": None,
        "limitations": [
            "The CLI reads only event columns across annual event files; candidate files are processed one year at a time.",
            "Event-history completeness must be checked against the auxiliary-download manifest.",
            "Multiple schedule states on one publication date are retained as an unordered set.",
            "Future actual disclosures are ex-post audits and are excluded from is_earnings_eligible.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    try:
        for year, candidate_path in candidate_paths.items():
            manifest["current_partition"] = year
            _write_json(manifest_path, manifest)
            candidates = pd.read_parquet(candidate_path)
            if not candidates.empty:
                signal_dates = _parse_dates(candidates["signal_date"], f"{candidate_path.name}.signal_date")
                if not signal_dates.dt.year.eq(year).all():
                    raise SchemaError(f"{candidate_path.name} contains candidate rows outside {year}")
            result = compute_earnings_exclusions(
                candidates,
                earnings_schedule,
                financial_results,
                trading_calendar=calendar,
                config=config,
            )
            output_path = output_dir / f"{year}.parquet"
            result.exclusions.to_parquet(output_path, index=False)
            partition_report = {
                **result.report,
                "candidate_input": candidate_inputs[year],
                "output_file": output_path.name,
                "output_sha256": _sha256(output_path),
            }
            _write_json(report_dir / f"{year}.json", partition_report)
            summary = {
                "year": year,
                "output_rows": result.report["candidate_rows"],
                "eligible_rows": result.report["eligible_rows"],
                "scheduled_crossing_rows": result.report["scheduled_crossing_rows"],
                "cooldown_rows": result.report["cooldown_rows"],
                "actual_crossing_rows_ex_post": result.report["actual_crossing_rows_ex_post"],
                "date_min": result.report["date_min"],
                "date_max": result.report["date_max"],
                "output_sha256": partition_report["output_sha256"],
            }
            manifest["completed_partitions"].append(summary)
            for key in (
                "output_rows",
                "eligible_rows",
                "scheduled_crossing_rows",
                "cooldown_rows",
                "actual_crossing_rows_ex_post",
            ):
                manifest[key] += summary[key]
            manifest["current_partition"] = None
            _write_json(manifest_path, manifest)
            LOGGER.info(
                "%s: rows=%d eligible=%d scheduled=%d cooldown=%d actual_ex_post=%d",
                year,
                summary["output_rows"],
                summary["eligible_rows"],
                summary["scheduled_crossing_rows"],
                summary["cooldown_rows"],
                summary["actual_crossing_rows_ex_post"],
            )
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
    parser.add_argument("--candidates-dir", type=Path, required=True, help="Annual weekly candidate Parquet files")
    parser.add_argument(
        "--earnings-schedule-dir", type=Path, required=True, help="Annual J-Quants earnings_schedule files"
    )
    parser.add_argument(
        "--financial-results-dir", type=Path, required=True, help="Annual J-Quants financial_results files"
    )
    parser.add_argument("--calendar", type=Path, required=True, help="J-Quants V2 Date/HolDiv CSV or Parquet")
    parser.add_argument("--output-dir", type=Path, help="New data/processed earnings-events run directory")
    parser.add_argument("--report-dir", type=Path, help="New reports earnings-events run directory")
    parser.add_argument("--config", type=Path, help="JSON EarningsEventConfig")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        config = load_config(args.config) if args.config else EarningsEventConfig()
        manifest = run_earnings_events(
            candidates_dir=args.candidates_dir,
            earnings_schedule_dir=args.earnings_schedule_dir,
            financial_results_dir=args.financial_results_dir,
            calendar_path=args.calendar,
            output_dir=args.output_dir or PROJECT_ROOT / "data" / "processed" / f"earnings_events_{run_id}",
            report_dir=args.report_dir or PROJECT_ROOT / "reports" / f"earnings_events_{run_id}",
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
                "scheduled_crossing_rows": manifest["scheduled_crossing_rows"],
                "cooldown_rows": manifest["cooldown_rows"],
                "same_day_schedule_state_groups": manifest["same_day_schedule_state_groups"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
