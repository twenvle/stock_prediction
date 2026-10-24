"""Executable weekly long-only portfolio backtest.

Selections are made after the signal close from saved model predictions and a
point-in-time weekly candidate table.  Entry is attempted once at the next
TSE-session open.  The default exit is the evaluation-session close; blocked
sell orders are retried on later session closes so positions never disappear
from the accounting ledger.  Raw prices are used for cash execution, while an
adjustment factor changes held share counts across splits/consolidations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from stock_prediction.preprocessing import SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "weekly-executable-backtest-v1"

PREDICTION_COLUMNS = (
    "signal_date",
    "code",
    "model",
    "target_column",
    "prediction",
)
PRICE_COLUMNS = (
    "date",
    "code",
    "open",
    "close",
    "upper_limit",
    "lower_limit",
    "volume",
    "turnover_value",
    "adjustment_factor",
    "has_price",
    "calendar_verified",
)
DELISTING_COLUMNS = (
    "code",
    "delisting_date",
    "settlement_date",
    "settlement_value_per_share",
)
EXIT_PRICE_MODES = frozenset({"evaluation_close", "next_session_open"})
DELISTING_POLICIES = frozenset({"error", "write_off"})


class UnresolvedDelistingError(RuntimeError):
    """A held delisted security has no supplied economic settlement value."""

    def __init__(self, *, code: str, delisting_date: pd.Timestamp, position_id: str) -> None:
        self.code = code
        self.delisting_date = delisting_date
        self.position_id = position_id
        super().__init__(
            "Unresolved delisting for held security "
            f"{code} on {delisting_date:%Y-%m-%d}; provide settlement_value_per_share "
            "or rerun an explicitly labelled write_off sensitivity"
        )


@dataclass(frozen=True)
class BacktestConfig:
    """Portfolio and execution assumptions fixed before final-period use.

    Costs are one-way basis points.  The defaults (5/5/10 bps) are explicit
    research assumptions, not a claim about a universal market cost.  Entry
    participation capacity is based on signal-time trailing median turnover,
    never the future entry-session turnover.
    """

    model_name: str = "linear"
    target_column: str = "target_return"
    eligibility_column: str = "is_signal_eligible"
    liquidity_column: str = "median_turnover_value_20"
    lot_size_column: str = "lot_size"
    initial_cash_yen: float = 10_000_000.0
    top_n: int = 10
    default_lot_size: int = 100
    max_participation_rate: float = 0.01
    commission_bps: float = 5.0
    half_spread_bps: float = 5.0
    slippage_bps: float = 10.0
    exit_price_mode: str = "evaluation_close"
    max_exit_wait_sessions: int = 20
    unresolved_delisting_policy: str = "error"
    annualization_sessions: int = 252

    def __post_init__(self) -> None:
        for name in ("model_name", "target_column", "eligibility_column", "liquidity_column", "lot_size_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("top_n", "default_lot_size", "max_exit_wait_sessions", "annualization_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "initial_cash_yen",
            "max_participation_rate",
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.initial_cash_yen <= 0:
            raise ValueError("initial_cash_yen must be positive")
        if not 0 < self.max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if min(self.commission_bps, self.half_spread_bps, self.slippage_bps) < 0:
            raise ValueError("cost assumptions must be non-negative")
        if self.commission_bps >= 10_000 or self.half_spread_bps + self.slippage_bps >= 10_000:
            raise ValueError("cost assumptions must keep commission below 100% and sell price positive")
        if self.exit_price_mode not in EXIT_PRICE_MODES:
            raise ValueError(f"exit_price_mode must be one of {sorted(EXIT_PRICE_MODES)}")
        if self.unresolved_delisting_policy not in DELISTING_POLICIES:
            raise ValueError(
                f"unresolved_delisting_policy must be one of {sorted(DELISTING_POLICIES)}"
            )


@dataclass
class BacktestResult:
    """Auditable ledgers, portfolio path, metrics, and run report."""

    orders: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    corporate_actions: pd.DataFrame
    daily_portfolio: pd.DataFrame
    weekly_metrics: pd.DataFrame
    yearly_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    report: dict[str, Any]


@dataclass
class _Position:
    position_id: str
    signal_date: pd.Timestamp
    code: str
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    exit_start_date: pd.Timestamp
    liquidity_yen: float
    lot_size: int
    selection_rank: int
    prediction: float
    shares: float
    entry_shares: float
    entry_reference_price: float
    entry_effective_price: float
    entry_gross_notional: float
    entry_cash_out: float
    entry_cost: float
    exit_gross_notional: float = 0.0
    exit_cash_in: float = 0.0
    exit_cost: float = 0.0
    exit_attempts: int = 0
    final_exit_date: pd.Timestamp | None = None
    status: str = "open"
    exit_reason: str | None = None
    pending_settlement_value: float | None = None
    settlement_date: pd.Timestamp | None = None


def load_config(path: Path | str) -> BacktestConfig:
    """Load a strict JSON configuration with no ignored keys."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Backtest configuration must be a JSON object")
    known = set(BacktestConfig.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown backtest configuration keys: {sorted(unknown)}")
    return BacktestConfig(**payload)


def _parse_dates(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        raise SchemaError(f"{name} must be timezone-naive")
    invalid = parsed.isna() & series.notna()
    if invalid.any() or (not nullable and parsed.isna().any()):
        raise SchemaError(f"{name} contains invalid or missing dates")
    normalized = parsed.dt.normalize()
    if not parsed.dropna().eq(normalized.dropna()).all():
        raise SchemaError(f"{name} must contain dates without intraday times")
    return normalized.astype("datetime64[ns]")


def _normalize_codes(series: pd.Series, name: str) -> pd.Series:
    if not (isinstance(series.dtype, pd.StringDtype) or series.dtype == object):
        raise SchemaError(f"{name} must be string-like")
    output = series.astype("string")
    if output.isna().any() or not output.str.fullmatch(r"[0-9A-Z]{5}").all():
        raise SchemaError(f"{name} must contain non-null five-character uppercase codes")
    return output


def _require_columns(frame: pd.DataFrame, required: Sequence[str], source: str) -> None:
    if not frame.columns.is_unique:
        raise SchemaError(f"{source} contains duplicate column names")
    missing = set(required) - set(frame.columns)
    if missing:
        raise SchemaError(f"{source} is missing required columns: {sorted(missing)}")


def _normalize_predictions(predictions: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    output = predictions.loc[:, list(PREDICTION_COLUMNS)].copy().reset_index(drop=True)
    output["signal_date"] = _parse_dates(output["signal_date"], "predictions.signal_date")
    output["code"] = _normalize_codes(output["code"], "predictions.code")
    for column in ("model", "target_column"):
        output[column] = output[column].astype("string")
        if output[column].isna().any() or output[column].str.len().eq(0).any():
            raise SchemaError(f"predictions.{column} must contain non-empty strings")
    if not is_numeric_dtype(output["prediction"].dtype) or is_bool_dtype(output["prediction"].dtype):
        raise SchemaError("predictions.prediction must be numeric")
    output["prediction"] = pd.to_numeric(output["prediction"], errors="coerce").astype("float64")
    selected = output.loc[
        output["model"].eq(config.model_name)
        & output["target_column"].eq(config.target_column)
    ].copy()
    if selected.empty:
        raise SchemaError(
            f"predictions contain no rows for model={config.model_name!r}, "
            f"target={config.target_column!r}"
        )
    duplicates = selected.duplicated(["signal_date", "code"], keep=False)
    if duplicates.any():
        raise SchemaError(
            f"predictions contain {int(duplicates.sum())} duplicate signal_date/code rows "
            "for the selected model and target"
        )
    selected.sort_values(["signal_date", "code"], inplace=True, ignore_index=True)
    return selected


def _normalize_candidates(candidates: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    required = (
        "signal_date",
        "code",
        "entry_date",
        "exit_date",
        config.eligibility_column,
        config.liquidity_column,
    )
    _require_columns(candidates, required, "candidates")
    columns = list(required)
    if config.lot_size_column in candidates.columns:
        columns.append(config.lot_size_column)
    output = candidates.loc[:, columns].copy().reset_index(drop=True)
    for column in ("signal_date", "entry_date", "exit_date"):
        output[column] = _parse_dates(output[column], f"candidates.{column}")
    output["code"] = _normalize_codes(output["code"], "candidates.code")
    eligible = output[config.eligibility_column]
    if not is_bool_dtype(eligible.dtype) or eligible.isna().any():
        raise SchemaError(f"candidates.{config.eligibility_column} must be non-null boolean")
    liquidity = output[config.liquidity_column]
    if not is_numeric_dtype(liquidity.dtype) or is_bool_dtype(liquidity.dtype):
        raise SchemaError(f"candidates.{config.liquidity_column} must be numeric")
    output[config.liquidity_column] = pd.to_numeric(liquidity, errors="coerce").astype("float64")
    invalid_liquidity = output[config.eligibility_column] & (
        ~np.isfinite(output[config.liquidity_column]) | output[config.liquidity_column].le(0)
    )
    if invalid_liquidity.any():
        raise SchemaError(
            f"{int(invalid_liquidity.sum())} eligible candidate rows have invalid signal-time liquidity"
        )
    if config.lot_size_column in output:
        lots = pd.to_numeric(output[config.lot_size_column], errors="coerce")
        invalid_lots = lots.notna() & (lots.le(0) | lots.mod(1).ne(0))
        if invalid_lots.any():
            raise SchemaError(f"candidates.{config.lot_size_column} must be positive integers or null")
        output[config.lot_size_column] = lots.astype("Int64")
    else:
        output[config.lot_size_column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    duplicates = output.duplicated(["signal_date", "code"], keep=False)
    if duplicates.any():
        raise SchemaError(f"candidates contain {int(duplicates.sum())} duplicate signal_date/code rows")
    if output["entry_date"].le(output["signal_date"]).any():
        raise SchemaError("candidate entry_date must be after signal_date")
    if output["exit_date"].lt(output["entry_date"]).any():
        raise SchemaError("candidate exit_date must not precede entry_date")
    schedule_counts = output.groupby("signal_date")[["entry_date", "exit_date"]].nunique()
    if schedule_counts.gt(1).to_numpy().any():
        raise SchemaError("candidates must have one entry_date and exit_date per signal_date")
    return output


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    _require_columns(prices, PRICE_COLUMNS, "prices")
    output = prices.loc[:, list(PRICE_COLUMNS)].copy().reset_index(drop=True)
    output["date"] = _parse_dates(output["date"], "prices.date")
    output["code"] = _normalize_codes(output["code"], "prices.code")
    for column in ("upper_limit", "lower_limit"):
        output[column] = output[column].astype("string")
        invalid = ~output[column].isin(["0", "1"])
        if invalid.any():
            raise SchemaError(f"prices.{column} must contain only '0' or '1'")
    for column in ("has_price", "calendar_verified"):
        if not is_bool_dtype(output[column].dtype) or output[column].isna().any():
            raise SchemaError(f"prices.{column} must be non-null boolean")
    for column in ("open", "close", "volume", "turnover_value", "adjustment_factor"):
        if not is_numeric_dtype(output[column].dtype) or is_bool_dtype(output[column].dtype):
            raise SchemaError(f"prices.{column} must be numeric")
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    if (~output["calendar_verified"]).any():
        raise SchemaError("prices contain rows without verified trading-calendar membership")
    if (output["adjustment_factor"].isna() | output["adjustment_factor"].le(0)).any():
        raise SchemaError("prices.adjustment_factor must be positive and finite")
    duplicates = output.duplicated(["date", "code"], keep=False)
    if duplicates.any():
        raise SchemaError(f"prices contain {int(duplicates.sum())} duplicate date/code rows")
    return output.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _normalize_delistings(delistings: pd.DataFrame | None) -> pd.DataFrame:
    if delistings is None:
        return pd.DataFrame(columns=list(DELISTING_COLUMNS))
    _require_columns(delistings, DELISTING_COLUMNS, "delistings")
    output = delistings.loc[:, list(DELISTING_COLUMNS)].copy().reset_index(drop=True)
    output["code"] = _normalize_codes(output["code"], "delistings.code")
    output["delisting_date"] = _parse_dates(output["delisting_date"], "delistings.delisting_date")
    output["settlement_date"] = _parse_dates(
        output["settlement_date"], "delistings.settlement_date", nullable=True
    )
    values = output["settlement_value_per_share"]
    if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
        raise SchemaError("delistings.settlement_value_per_share must be numeric (null allowed)")
    output["settlement_value_per_share"] = pd.to_numeric(values, errors="coerce").astype("float64")
    invalid_values = output["settlement_value_per_share"].notna() & (
        ~np.isfinite(output["settlement_value_per_share"])
        | output["settlement_value_per_share"].lt(0)
    )
    if invalid_values.any():
        raise SchemaError("delisting settlement values must be non-negative and finite")
    known = output["settlement_value_per_share"].notna()
    if (known & output["settlement_date"].isna()).any():
        raise SchemaError("known delisting settlement values require settlement_date")
    if (
        output["settlement_date"].notna()
        & output["settlement_date"].lt(output["delisting_date"])
    ).any():
        raise SchemaError("delisting settlement_date must not precede delisting_date")
    if output["code"].duplicated().any():
        raise SchemaError("delistings must contain at most one event per code in a backtest run")
    return output.sort_values(["delisting_date", "code"], kind="stable").reset_index(drop=True)


def _select_orders(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    config: BacktestConfig,
) -> pd.DataFrame:
    merged = predictions.merge(
        candidates,
        on=["signal_date", "code"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        count = int(merged["_merge"].ne("both").sum())
        raise SchemaError(f"{count} prediction rows have no exact signal_date/code candidate row")
    merged.drop(columns="_merge", inplace=True)
    session_position = pd.Series(np.arange(len(sessions)), index=sessions)
    for row in merged[["signal_date", "entry_date", "exit_date"]].drop_duplicates().itertuples(index=False):
        if row.signal_date not in session_position or row.entry_date not in session_position or row.exit_date not in session_position:
            raise SchemaError("signal, entry, and exit dates must be official TSE sessions")
        if int(session_position[row.entry_date]) != int(session_position[row.signal_date]) + 1:
            raise SchemaError("entry_date must be the next TSE session after signal_date")
    eligible = merged.loc[
        merged[config.eligibility_column]
        & np.isfinite(merged["prediction"])
    ].copy()
    eligible.sort_values(
        ["signal_date", "prediction", "code"],
        ascending=[True, False, True],
        kind="stable",
        inplace=True,
    )
    eligible["selection_rank"] = eligible.groupby("signal_date", sort=True).cumcount() + 1
    selected = eligible.loc[eligible["selection_rank"].le(config.top_n)].copy()
    selected["used_default_lot_size"] = selected[config.lot_size_column].isna()
    selected["lot_size"] = selected[config.lot_size_column].fillna(config.default_lot_size).astype("int64")
    selected.rename(columns={config.liquidity_column: "signal_median_turnover_yen"}, inplace=True)
    return selected.sort_values(["entry_date", "selection_rank", "code"], kind="stable").reset_index(drop=True)


def _execution_block_reason(row: pd.Series | None, *, side: str, price_column: str) -> str | None:
    if row is None or not bool(row["has_price"]):
        return "price_unavailable"
    price = float(row[price_column])
    if not np.isfinite(price) or price <= 0:
        return "price_unavailable"
    if not np.isfinite(row["volume"]) or row["volume"] <= 0 or not np.isfinite(row["turnover_value"]) or row["turnover_value"] <= 0:
        return "no_trade_or_suspension"
    if side == "buy" and row["upper_limit"] != "0":
        return "limit_up"
    if side == "sell" and row["lower_limit"] != "0":
        return "limit_down"
    return None


def _costed_price(reference_price: float, *, side: str, config: BacktestConfig) -> tuple[float, float, float]:
    spread_rate = config.half_spread_bps / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    direction = 1.0 if side == "buy" else -1.0
    effective = reference_price * (1.0 + direction * (spread_rate + slippage_rate))
    return effective, reference_price * spread_rate, reference_price * slippage_rate


def _lot_floor(shares: float, lot_size: int) -> float:
    return float(math.floor((shares + 1e-12) / lot_size) * lot_size)


def backtest_portfolio(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame,
    delistings: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the executable portfolio simulation used by notebooks and the CLI.

    The input DataFrames are copied and never modified.  Prediction targets are
    deliberately not read: selection depends only on the configured prediction,
    exact-date eligibility, and information known at the signal close.
    """

    config = config or BacktestConfig()
    calendar = validate_trading_calendar(trading_calendar)
    sessions = pd.DatetimeIndex(calendar.loc[calendar["is_session"], "date"])
    normalized_predictions = _normalize_predictions(predictions, config)
    normalized_candidates = _normalize_candidates(candidates, config)
    normalized_prices = _normalize_prices(prices)
    normalized_delistings = _normalize_delistings(delistings)
    selected = _select_orders(normalized_predictions, normalized_candidates, sessions, config)
    if selected.empty:
        raise SchemaError("No finite, eligible predictions were selected")

    first_signal = selected["signal_date"].min()
    last_scheduled_exit = selected["exit_date"].max()
    first_position = sessions.get_indexer([first_signal])[0]
    last_exit_position = sessions.get_indexer([last_scheduled_exit])[0]
    if first_position < 0 or last_exit_position < 0:
        raise SchemaError("selected signal/exit dates are absent from the official calendar")
    required_end_position = min(
        len(sessions) - 1,
        last_exit_position + config.max_exit_wait_sessions + (1 if config.exit_price_mode == "next_session_open" else 0),
    )
    relevant_codes = set(selected["code"].astype(str))
    relevant_delistings = normalized_delistings.loc[
        normalized_delistings["code"].isin(relevant_codes)
    ].copy()
    if not relevant_delistings.empty:
        event_dates = pd.concat(
            [
                relevant_delistings["delisting_date"],
                relevant_delistings["settlement_date"].dropna(),
            ],
            ignore_index=True,
        )
        event_max = event_dates.max()
        eligible_sessions = np.flatnonzero(sessions >= event_max)
        if len(eligible_sessions):
            required_end_position = max(required_end_position, int(eligible_sessions[0]))
    simulation_sessions = sessions[first_position : required_end_position + 1]
    date_min = simulation_sessions.min()
    date_max = simulation_sessions.max()
    normalized_prices = normalized_prices.loc[
        normalized_prices["code"].isin(relevant_codes)
        & normalized_prices["date"].between(date_min, date_max)
    ].copy()
    if not normalized_prices["date"].isin(sessions).all():
        raise SchemaError("prices contain dates outside official TSE sessions")
    price_lookup = {
        (row.date, str(row.code)): pd.Series(row._asdict())
        for row in normalized_prices.itertuples(index=False)
    }
    delisting_lookup = {
        str(row.code): row for row in relevant_delistings.itertuples(index=False)
    }

    session_position = {day: index for index, day in enumerate(sessions)}
    selected = selected.copy()
    if config.exit_price_mode == "evaluation_close":
        selected["exit_start_date"] = selected["exit_date"]
    else:
        positions = selected["exit_date"].map(session_position).astype(int) + 1
        if positions.ge(len(sessions)).any():
            raise SchemaError("next-session-open exit needs a calendar session after every exit_date")
        selected["exit_start_date"] = [sessions[value] for value in positions]
    normal_simulation_end = selected["exit_start_date"].max()

    entry_groups = {
        day: group.copy() for day, group in selected.groupby("entry_date", sort=True)
    }
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    corporate_actions: list[dict[str, Any]] = []
    all_positions: list[_Position] = []
    open_positions: dict[str, _Position] = {}
    last_close: dict[str, float] = {}
    cash = float(config.initial_cash_yen)
    position_counter = 0
    order_counter = 0
    trade_counter = 0

    def add_order(
        *,
        position_id: str | None,
        signal_date: pd.Timestamp,
        code: str,
        side: str,
        order_date: pd.Timestamp,
        status: str,
        reason: str,
        requested_shares: float,
        filled_shares: float,
        reference_price: float | None,
        effective_price: float | None,
        participation_capacity_shares: float,
        gross_notional: float,
        commission: float,
        spread_cost: float,
        slippage_cost: float,
    ) -> None:
        nonlocal order_counter
        order_counter += 1
        orders.append(
            {
                "order_id": f"O{order_counter:08d}",
                "position_id": position_id,
                "signal_date": signal_date,
                "code": code,
                "side": side,
                "order_date": order_date,
                "status": status,
                "reason": reason,
                "requested_shares": requested_shares,
                "filled_shares": filled_shares,
                "unfilled_shares": max(requested_shares - filled_shares, 0.0),
                "reference_price": reference_price,
                "effective_price": effective_price,
                "participation_capacity_shares": participation_capacity_shares,
                "gross_notional": gross_notional,
                "commission": commission,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "total_cost": commission + spread_cost + slippage_cost,
            }
        )

    def add_trade(
        *,
        position: _Position,
        side: str,
        trade_date: pd.Timestamp,
        shares: float,
        reference_price: float,
        effective_price: float,
        gross_notional: float,
        commission: float,
        spread_cost: float,
        slippage_cost: float,
        reason: str,
    ) -> None:
        nonlocal trade_counter
        trade_counter += 1
        trades.append(
            {
                "trade_id": f"T{trade_counter:08d}",
                "position_id": position.position_id,
                "signal_date": position.signal_date,
                "code": position.code,
                "side": side,
                "trade_date": trade_date,
                "shares": shares,
                "reference_price": reference_price,
                "effective_price": effective_price,
                "gross_notional": gross_notional,
                "commission": commission,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "total_cost": commission + spread_cost + slippage_cost,
                "reason": reason,
            }
        )

    def settle_position(position: _Position, day: pd.Timestamp, value: float, reason: str) -> None:
        nonlocal cash
        shares = position.shares
        gross = shares * value
        cash += gross
        add_order(
            position_id=position.position_id,
            signal_date=position.signal_date,
            code=position.code,
            side="sell",
            order_date=day,
            status="filled",
            reason=reason,
            requested_shares=shares,
            filled_shares=shares,
            reference_price=value,
            effective_price=value,
            participation_capacity_shares=shares,
            gross_notional=gross,
            commission=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
        )
        add_trade(
            position=position,
            side="sell",
            trade_date=day,
            shares=shares,
            reference_price=value,
            effective_price=value,
            gross_notional=gross,
            commission=0.0,
            spread_cost=0.0,
            slippage_cost=0.0,
            reason=reason,
        )
        position.exit_gross_notional += gross
        position.exit_cash_in += gross
        position.shares = 0.0
        position.final_exit_date = day
        position.status = "closed"
        position.exit_reason = reason
        open_positions.pop(position.position_id, None)

    def attempt_exits(day: pd.Timestamp, *, price_column: str) -> None:
        nonlocal cash
        for position in sorted(list(open_positions.values()), key=lambda value: value.position_id):
            if position.pending_settlement_value is not None:
                continue
            if day < position.exit_start_date or position.exit_attempts >= config.max_exit_wait_sessions:
                continue
            position.exit_attempts += 1
            row = price_lookup.get((day, position.code))
            block = _execution_block_reason(row, side="sell", price_column=price_column)
            capacity = 0.0
            if block is None and row is not None:
                reference_price = float(row[price_column])
                capacity = _lot_floor(
                    position.liquidity_yen * config.max_participation_rate / reference_price,
                    position.lot_size,
                )
                if capacity <= 0:
                    block = "below_participation_lot"
            if block is not None:
                add_order(
                    position_id=position.position_id,
                    signal_date=position.signal_date,
                    code=position.code,
                    side="sell",
                    order_date=day,
                    status="unfilled",
                    reason=block,
                    requested_shares=position.shares,
                    filled_shares=0.0,
                    reference_price=None if row is None else float(row[price_column]) if np.isfinite(row[price_column]) else None,
                    effective_price=None,
                    participation_capacity_shares=capacity,
                    gross_notional=0.0,
                    commission=0.0,
                    spread_cost=0.0,
                    slippage_cost=0.0,
                )
                continue
            reference_price = float(row[price_column])
            filled = min(position.shares, capacity)
            filled = _lot_floor(filled, position.lot_size)
            if filled <= 0:
                continue
            effective_price, spread_per_share, slippage_per_share = _costed_price(
                reference_price, side="sell", config=config
            )
            gross = reference_price * filled
            effective_notional = effective_price * filled
            commission = effective_notional * config.commission_bps / 10_000.0
            spread_cost = spread_per_share * filled
            slippage_cost = slippage_per_share * filled
            net = effective_notional - commission
            cash += net
            remaining_before = position.shares
            position.shares -= filled
            position.exit_gross_notional += gross
            position.exit_cash_in += net
            position.exit_cost += commission + spread_cost + slippage_cost
            complete = position.shares <= 1e-9
            status = "filled" if complete else "partially_filled"
            add_order(
                position_id=position.position_id,
                signal_date=position.signal_date,
                code=position.code,
                side="sell",
                order_date=day,
                status=status,
                reason="scheduled_exit" if day == position.exit_start_date else "delayed_exit_retry",
                requested_shares=remaining_before,
                filled_shares=filled,
                reference_price=reference_price,
                effective_price=effective_price,
                participation_capacity_shares=capacity,
                gross_notional=gross,
                commission=commission,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
            )
            add_trade(
                position=position,
                side="sell",
                trade_date=day,
                shares=filled,
                reference_price=reference_price,
                effective_price=effective_price,
                gross_notional=gross,
                commission=commission,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                reason="scheduled_exit" if day == position.exit_start_date else "delayed_exit_retry",
            )
            if complete:
                position.shares = 0.0
                position.final_exit_date = day
                position.status = "closed"
                position.exit_reason = "scheduled_exit" if day == position.exit_start_date else "delayed_exit_retry"
                open_positions.pop(position.position_id, None)

    daily_rows: list[dict[str, Any]] = []
    for day in simulation_sessions:
        # Corporate-action factors become effective before the session's open.
        for position in sorted(list(open_positions.values()), key=lambda value: value.position_id):
            row = price_lookup.get((day, position.code))
            if row is None:
                continue
            factor = float(row["adjustment_factor"])
            if factor != 1.0:
                before = position.shares
                after = before / factor
                position.shares = after
                corporate_actions.append(
                    {
                        "position_id": position.position_id,
                        "code": position.code,
                        "date": day,
                        "adjustment_factor": factor,
                        "shares_before": before,
                        "shares_after": after,
                        "is_integral_after": bool(np.isclose(after, round(after))),
                        "is_round_lot_after": bool(np.isclose(after % position.lot_size, 0.0)),
                    }
                )

        # A known delisting removes the position from exchange execution on and
        # after its effective date.  No event is inferred from price absence.
        for position in sorted(list(open_positions.values()), key=lambda value: value.position_id):
            event = delisting_lookup.get(position.code)
            if event is None or day < event.delisting_date or position.pending_settlement_value is not None:
                continue
            if pd.isna(event.settlement_value_per_share):
                if config.unresolved_delisting_policy == "error":
                    raise UnresolvedDelistingError(
                        code=position.code,
                        delisting_date=event.delisting_date,
                        position_id=position.position_id,
                    )
                value = 0.0
                settlement_date = day
                reason = "unresolved_delisting_write_off_sensitivity"
            else:
                value = float(event.settlement_value_per_share)
                settlement_date = event.settlement_date
                reason = "delisting_settlement"
            position.pending_settlement_value = value
            position.settlement_date = settlement_date
            position.status = "pending_delisting_settlement"
            position.exit_reason = reason

        for position in sorted(list(open_positions.values()), key=lambda value: value.position_id):
            if position.pending_settlement_value is not None and day >= position.settlement_date:
                settle_position(position, day, position.pending_settlement_value, position.exit_reason or "delisting_settlement")

        if config.exit_price_mode == "next_session_open":
            attempt_exits(day, price_column="open")

        group = entry_groups.get(day)
        if group is not None:
            group = group.sort_values(["selection_rank", "code"], kind="stable")
            cash_snapshot = cash
            per_security_budget = cash_snapshot / len(group)
            held_codes = {position.code for position in open_positions.values()}
            for row in group.itertuples(index=False):
                position_counter += 1
                position_id = f"P{position_counter:08d}"
                code = str(row.code)
                price_row = price_lookup.get((day, code))
                event = delisting_lookup.get(code)
                block = None
                if code in held_codes:
                    block = "already_held"
                elif event is not None and day >= event.delisting_date:
                    block = "delisted_before_entry"
                else:
                    block = _execution_block_reason(price_row, side="buy", price_column="open")
                capacity = 0.0
                desired = 0.0
                if block is None and price_row is not None:
                    reference_price = float(price_row["open"])
                    effective_price, spread_per_share, slippage_per_share = _costed_price(
                        reference_price, side="buy", config=config
                    )
                    unit_cash = effective_price * (1.0 + config.commission_bps / 10_000.0)
                    desired = _lot_floor(per_security_budget / unit_cash, int(row.lot_size))
                    affordable = _lot_floor(cash / unit_cash, int(row.lot_size))
                    capacity = _lot_floor(
                        float(row.signal_median_turnover_yen)
                        * config.max_participation_rate
                        / reference_price,
                        int(row.lot_size),
                    )
                    filled = min(desired, affordable, capacity)
                    if desired <= 0 or affordable <= 0:
                        block = "insufficient_cash_for_round_lot"
                    elif capacity <= 0:
                        block = "below_participation_lot"
                if block is not None:
                    add_order(
                        position_id=None,
                        signal_date=row.signal_date,
                        code=code,
                        side="buy",
                        order_date=day,
                        status="unfilled",
                        reason=block,
                        requested_shares=desired,
                        filled_shares=0.0,
                        reference_price=None if price_row is None else float(price_row["open"]) if np.isfinite(price_row["open"]) else None,
                        effective_price=None,
                        participation_capacity_shares=capacity,
                        gross_notional=0.0,
                        commission=0.0,
                        spread_cost=0.0,
                        slippage_cost=0.0,
                    )
                    continue
                filled = float(filled)
                gross = reference_price * filled
                effective_notional = effective_price * filled
                commission = effective_notional * config.commission_bps / 10_000.0
                spread_cost = spread_per_share * filled
                slippage_cost = slippage_per_share * filled
                cash_out = effective_notional + commission
                cash -= cash_out
                position = _Position(
                    position_id=position_id,
                    signal_date=row.signal_date,
                    code=code,
                    entry_date=day,
                    scheduled_exit_date=row.exit_date,
                    exit_start_date=row.exit_start_date,
                    liquidity_yen=float(row.signal_median_turnover_yen),
                    lot_size=int(row.lot_size),
                    selection_rank=int(row.selection_rank),
                    prediction=float(row.prediction),
                    shares=filled,
                    entry_shares=filled,
                    entry_reference_price=reference_price,
                    entry_effective_price=effective_price,
                    entry_gross_notional=gross,
                    entry_cash_out=cash_out,
                    entry_cost=commission + spread_cost + slippage_cost,
                )
                all_positions.append(position)
                open_positions[position_id] = position
                held_codes.add(code)
                last_close.setdefault(code, reference_price)
                status = "filled" if np.isclose(filled, desired) else "partially_filled"
                add_order(
                    position_id=position_id,
                    signal_date=row.signal_date,
                    code=code,
                    side="buy",
                    order_date=day,
                    status=status,
                    reason="next_session_open",
                    requested_shares=desired,
                    filled_shares=filled,
                    reference_price=reference_price,
                    effective_price=effective_price,
                    participation_capacity_shares=capacity,
                    gross_notional=gross,
                    commission=commission,
                    spread_cost=spread_cost,
                    slippage_cost=slippage_cost,
                )
                add_trade(
                    position=position,
                    side="buy",
                    trade_date=day,
                    shares=filled,
                    reference_price=reference_price,
                    effective_price=effective_price,
                    gross_notional=gross,
                    commission=commission,
                    spread_cost=spread_cost,
                    slippage_cost=slippage_cost,
                    reason="next_session_open",
                )

        if config.exit_price_mode == "evaluation_close":
            attempt_exits(day, price_column="close")

        for code in relevant_codes:
            row = price_lookup.get((day, code))
            if row is not None and bool(row["has_price"]) and np.isfinite(row["close"]) and row["close"] > 0:
                last_close[code] = float(row["close"])
        market_value = 0.0
        stale_marks = 0
        for position in open_positions.values():
            row = price_lookup.get((day, position.code))
            fresh = row is not None and bool(row["has_price"]) and np.isfinite(row["close"]) and row["close"] > 0
            if position.pending_settlement_value is not None:
                mark = position.pending_settlement_value
                fresh = False
            else:
                mark = last_close.get(position.code, position.entry_reference_price)
            market_value += position.shares * mark
            stale_marks += int(not fresh)
        equity = cash + market_value
        daily_rows.append(
            {
                "date": day,
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "open_positions": len(open_positions),
                "stale_mark_positions": stale_marks,
            }
        )
        if day >= normal_simulation_end and not open_positions:
            break

    order_columns = (
        "order_id", "position_id", "signal_date", "code", "side", "order_date", "status", "reason",
        "requested_shares", "filled_shares", "unfilled_shares", "reference_price", "effective_price",
        "participation_capacity_shares", "gross_notional", "commission", "spread_cost", "slippage_cost", "total_cost",
    )
    trade_columns = (
        "trade_id", "position_id", "signal_date", "code", "side", "trade_date", "shares",
        "reference_price", "effective_price", "gross_notional", "commission", "spread_cost", "slippage_cost",
        "total_cost", "reason",
    )
    action_columns = (
        "position_id", "code", "date", "adjustment_factor", "shares_before", "shares_after",
        "is_integral_after", "is_round_lot_after",
    )
    orders_frame = pd.DataFrame.from_records(orders, columns=list(order_columns))
    trades_frame = pd.DataFrame.from_records(trades, columns=list(trade_columns))
    actions_frame = pd.DataFrame.from_records(corporate_actions, columns=list(action_columns))
    position_rows: list[dict[str, Any]] = []
    for position in all_positions:
        closed = position.status == "closed"
        pnl = position.exit_cash_in - position.entry_cash_out if closed else np.nan
        position_rows.append(
            {
                "position_id": position.position_id,
                "signal_date": position.signal_date,
                "code": position.code,
                "selection_rank": position.selection_rank,
                "prediction": position.prediction,
                "entry_date": position.entry_date,
                "scheduled_exit_date": position.scheduled_exit_date,
                "exit_start_date": position.exit_start_date,
                "final_exit_date": position.final_exit_date,
                "status": position.status,
                "exit_reason": position.exit_reason,
                "entry_shares": position.entry_shares,
                "remaining_shares": position.shares,
                "lot_size": position.lot_size,
                "entry_reference_price": position.entry_reference_price,
                "entry_effective_price": position.entry_effective_price,
                "entry_gross_notional": position.entry_gross_notional,
                "exit_gross_notional": position.exit_gross_notional,
                "entry_cost": position.entry_cost,
                "exit_cost": position.exit_cost,
                "net_pnl": pnl,
                "net_return": pnl / position.entry_cash_out if closed and position.entry_cash_out else np.nan,
                "exit_attempts": position.exit_attempts,
            }
        )
    position_columns = (
        "position_id", "signal_date", "code", "selection_rank", "prediction", "entry_date",
        "scheduled_exit_date", "exit_start_date", "final_exit_date", "status", "exit_reason",
        "entry_shares", "remaining_shares", "lot_size", "entry_reference_price", "entry_effective_price",
        "entry_gross_notional", "exit_gross_notional", "entry_cost", "exit_cost", "net_pnl", "net_return",
        "exit_attempts",
    )
    positions_frame = pd.DataFrame.from_records(position_rows, columns=list(position_columns))
    daily = pd.DataFrame.from_records(daily_rows)
    daily["daily_return"] = daily["equity"].pct_change().fillna(0.0)
    daily["cumulative_return"] = daily["equity"] / config.initial_cash_yen - 1.0
    daily["running_peak_equity"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["running_peak_equity"] - 1.0

    weekly_records: list[dict[str, Any]] = []
    signal_equity = daily.set_index("date")["equity"]
    for signal_date, group in selected.groupby("signal_date", sort=True):
        signal_orders = orders_frame.loc[
            orders_frame["signal_date"].eq(signal_date) & orders_frame["side"].eq("buy")
        ]
        signal_positions = positions_frame.loc[positions_frame["signal_date"].eq(signal_date)]
        signal_trades = trades_frame.loc[trades_frame["signal_date"].eq(signal_date)]
        exchange_trades = signal_trades.loc[
            ~signal_trades["reason"].isin(
                ["delisting_settlement", "unresolved_delisting_write_off_sensitivity"]
            )
        ]
        traded = float(exchange_trades["gross_notional"].sum()) if len(exchange_trades) else 0.0
        denominator = float(signal_equity.get(signal_date, config.initial_cash_yen))
        weekly_records.append(
            {
                "signal_date": signal_date,
                "selected_count": int(len(group)),
                "filled_entry_count": int(signal_orders["filled_shares"].gt(0).sum()),
                "entry_fill_coverage": float(signal_orders["filled_shares"].gt(0).mean()),
                "closed_position_count": int(signal_positions["status"].eq("closed").sum()),
                "open_position_count": int(signal_positions["status"].ne("closed").sum()),
                "buy_gross_notional": float(exchange_trades.loc[exchange_trades["side"].eq("buy"), "gross_notional"].sum()),
                "sell_gross_notional": float(exchange_trades.loc[exchange_trades["side"].eq("sell"), "gross_notional"].sum()),
                "delisting_settlement_cash": float(
                    signal_trades.loc[
                        signal_trades["reason"].isin(
                            ["delisting_settlement", "unresolved_delisting_write_off_sensitivity"]
                        ),
                        "gross_notional",
                    ].sum()
                ),
                "transaction_cost": float(signal_trades["total_cost"].sum()),
                "realized_net_pnl": float(signal_positions["net_pnl"].sum(min_count=1)) if signal_positions["net_pnl"].notna().any() else np.nan,
                "two_way_turnover": traded / denominator if denominator else np.nan,
                "one_way_turnover": 0.5 * traded / denominator if denominator else np.nan,
            }
        )
    weekly = pd.DataFrame.from_records(weekly_records)

    yearly_records: list[dict[str, Any]] = []
    for year, year_daily in daily.groupby(daily["date"].dt.year, sort=True):
        year_daily = year_daily.sort_values("date", kind="stable")
        earlier = daily.loc[daily["date"].lt(year_daily["date"].min()), "equity"]
        starting_equity = float(earlier.iloc[-1]) if len(earlier) else config.initial_cash_yen
        ending_equity = float(year_daily["equity"].iloc[-1])
        returns = year_daily["daily_return"]
        standard_deviation = float(returns.std(ddof=1)) if len(returns) >= 2 else np.nan
        volatility = (
            standard_deviation * math.sqrt(config.annualization_sessions)
            if np.isfinite(standard_deviation)
            else np.nan
        )
        year_sharpe = (
            float(returns.mean() / standard_deviation * math.sqrt(config.annualization_sessions))
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else np.nan
        )
        peak_values = np.maximum.accumulate(
            np.concatenate(([starting_equity], year_daily["equity"].to_numpy(dtype="float64")))
        )[1:]
        year_drawdown = year_daily["equity"].to_numpy(dtype="float64") / peak_values - 1.0
        year_trades = (
            trades_frame
            if trades_frame.empty
            else trades_frame.loc[trades_frame["trade_date"].dt.year.eq(year)]
        )
        year_weekly = weekly.loc[weekly["signal_date"].dt.year.eq(year)]
        yearly_records.append(
            {
                "year": int(year),
                "date_min": year_daily["date"].min(),
                "date_max": year_daily["date"].max(),
                "trading_sessions": int(len(year_daily)),
                "starting_equity_yen": starting_equity,
                "ending_equity_yen": ending_equity,
                "return": ending_equity / starting_equity - 1.0,
                "annualized_volatility": volatility,
                "sharpe_ratio_zero_risk_free": year_sharpe,
                "maximum_drawdown": float(np.min(year_drawdown)),
                "transaction_cost_yen": float(year_trades["total_cost"].sum()),
                "trade_count": int(len(year_trades)),
                "two_way_turnover": float(year_weekly["two_way_turnover"].sum()),
                "one_way_turnover": float(year_weekly["one_way_turnover"].sum()),
            }
        )
    yearly = pd.DataFrame.from_records(yearly_records)

    daily_returns = daily["daily_return"].iloc[1:]
    periods = len(daily_returns)
    end_equity = float(daily["equity"].iloc[-1])
    annualized_return = (
        (end_equity / config.initial_cash_yen) ** (config.annualization_sessions / periods) - 1.0
        if periods and end_equity > 0
        else np.nan
    )
    annualized_volatility = (
        float(daily_returns.std(ddof=1) * math.sqrt(config.annualization_sessions))
        if len(daily_returns) >= 2
        else np.nan
    )
    daily_standard_deviation = float(daily_returns.std(ddof=1)) if len(daily_returns) >= 2 else np.nan
    sharpe = (
        float(daily_returns.mean() / daily_standard_deviation * math.sqrt(config.annualization_sessions))
        if np.isfinite(daily_standard_deviation) and daily_standard_deviation > 0
        else np.nan
    )
    maximum_drawdown = float(daily["drawdown"].min())
    calmar = annualized_return / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
    closed_positions = positions_frame.loc[positions_frame["status"].eq("closed")]
    winners = closed_positions.loc[closed_positions["net_pnl"].gt(0), "net_pnl"]
    losers = closed_positions.loc[closed_positions["net_pnl"].lt(0), "net_pnl"]
    profit_loss_ratio = winners.mean() / abs(losers.mean()) if len(winners) and len(losers) else np.nan
    summary_record = {
        "initial_cash_yen": config.initial_cash_yen,
        "ending_equity_yen": end_equity,
        "cumulative_return": end_equity / config.initial_cash_yen - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio_zero_risk_free": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "selected_orders": int(len(selected)),
        "filled_entries": int((orders_frame["side"].eq("buy") & orders_frame["filled_shares"].gt(0)).sum()),
        "entry_fill_coverage": float((orders_frame.loc[orders_frame["side"].eq("buy"), "filled_shares"].gt(0)).mean()),
        "trade_count": int(len(trades_frame)),
        "closed_positions": int(len(closed_positions)),
        "open_positions": int(len(open_positions)),
        "win_rate": float(closed_positions["net_pnl"].gt(0).mean()) if len(closed_positions) else np.nan,
        "profit_loss_ratio": profit_loss_ratio,
        "total_transaction_cost_yen": float(trades_frame["total_cost"].sum()),
        "total_two_way_turnover": float(weekly["two_way_turnover"].sum()),
        "total_one_way_turnover": float(weekly["one_way_turnover"].sum()),
        "main_result_complete": not bool(open_positions),
    }
    summary = pd.DataFrame([summary_record])
    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "input": {
            "prediction_rows_selected_model_target": int(len(normalized_predictions)),
            "candidate_rows": int(len(normalized_candidates)),
            "price_rows_in_simulation": int(len(normalized_prices)),
            "delisting_event_rows": int(len(relevant_delistings)),
            "signal_dates": int(selected["signal_date"].nunique()),
            "date_min": date_min.strftime("%Y-%m-%d"),
            "date_max": daily["date"].max().strftime("%Y-%m-%d"),
        },
        "selection": {
            "selected_rows": int(len(selected)),
            "top_n": config.top_n,
            "tie_break": "prediction descending, code ascending",
            "target_values_used": False,
        },
        "execution": {
            "entry": "next TSE session open; unfilled remainder expires that session",
            "exit": config.exit_price_mode,
            "blocked_exit": f"retry up to {config.max_exit_wait_sessions} TSE sessions",
            "participation_denominator": config.liquidity_column,
            "future_entry_day_turnover_used_for_sizing": False,
            "default_lot_size_rows": int(selected["used_default_lot_size"].sum()),
        },
        "outputs": {
            "order_rows": int(len(orders_frame)),
            "trade_rows": int(len(trades_frame)),
            "position_rows": int(len(positions_frame)),
            "corporate_action_rows": int(len(actions_frame)),
            "daily_rows": int(len(daily)),
            "weekly_rows": int(len(weekly)),
            "yearly_rows": int(len(yearly)),
        },
        "main_result_complete": not bool(open_positions),
        "open_position_ids": sorted(open_positions),
        "limitations": [
            "Daily OHLCV and limit flags are conservative execution proxies, not order-book fills.",
            "A limit-up buy or limit-down sell day is treated as wholly unexecutable.",
            "Dividends and taxes are not included; results are price-return portfolio results.",
            "Delisting is never inferred from a missing price or listed-master disappearance.",
            "A missing point-in-time lot size uses the configured default and is counted in the report.",
            "Open positions after the retry horizon make the main result incomplete.",
        ],
        "random_seed": None,
    }
    return BacktestResult(
        orders=orders_frame,
        trades=trades_frame,
        positions=positions_frame,
        corporate_actions=actions_frame,
        daily_portfolio=daily,
        weekly_metrics=weekly,
        yearly_metrics=yearly,
        summary_metrics=summary,
        report=report,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_table(
    path: Path,
    *,
    string_columns: Sequence[str] = (),
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=None if columns is None else list(columns))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            usecols=None if columns is None else list(columns),
            dtype={name: "string" for name in string_columns},
        )
    raise SchemaError(f"Unsupported input format for {path}; use .parquet or .csv")


def _annual_partitions(directory: Path, years: Sequence[int], label: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    available = {
        int(path.stem): path
        for path in directory.glob("*.parquet")
        if re.fullmatch(r"\d{4}", path.stem)
    }
    missing = set(years) - set(available)
    if missing:
        raise FileNotFoundError(f"{label} partitions are missing for years: {sorted(missing)}")
    return [available[year] for year in sorted(set(years))]


def _read_partitions(paths: Sequence[Path], columns: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        schema_names = pq.read_schema(path).names
        missing = set(columns) - set(schema_names)
        if missing:
            raise SchemaError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(pd.read_parquet(path, columns=list(columns)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


def _read_price_partitions(
    paths: Sequence[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    codes: Sequence[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    code_values = sorted(set(codes))
    for path in paths:
        schema = pq.read_schema(path)
        missing = set(PRICE_COLUMNS) - set(schema.names)
        if missing:
            raise SchemaError(f"{path.name} is missing columns: {sorted(missing)}")
        date_type = schema.field("date").type
        if pa.types.is_timestamp(date_type):
            lower: Any = start.to_pydatetime()
            upper: Any = end.to_pydatetime()
        elif pa.types.is_date(date_type):
            lower = start.date()
            upper = end.date()
        else:
            raise SchemaError(f"{path.name}.date must be a Parquet date or timestamp")
        table = pq.read_table(
            path,
            columns=list(PRICE_COLUMNS),
            filters=[("date", ">=", lower), ("date", "<=", upper), ("code", "in", code_values)],
        )
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(PRICE_COLUMNS))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_new_outputs(inputs: Sequence[Path], output_dir: Path, report_dir: Path) -> None:
    if output_dir.resolve() == report_dir.resolve():
        raise ValueError("output_dir and report_dir must differ")
    for target in (output_dir, report_dir):
        if target.exists():
            raise FileExistsError(f"Output path already exists: {target}")
        for source in inputs:
            parent = source if source.is_dir() else source.parent
            if _is_relative_to(target, parent):
                raise ValueError("Backtest outputs must not be inside an input directory")


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def run_backtest(
    *,
    predictions_path: Path | str,
    candidates_dir: Path | str,
    prices_dir: Path | str,
    calendar_path: Path | str,
    delistings_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Read saved inputs, run the simulation, and persist immutable outputs."""

    config = config or BacktestConfig()
    predictions_path = Path(predictions_path)
    candidates_dir = Path(candidates_dir)
    prices_dir = Path(prices_dir)
    calendar_path = Path(calendar_path)
    delistings_path = Path(delistings_path) if delistings_path is not None else None
    required_files = [predictions_path, calendar_path]
    if delistings_path is not None:
        required_files.append(delistings_path)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "artifacts" / f"backtest_{run_id}"
    report_dir = Path(report_dir) if report_dir is not None else PROJECT_ROOT / "reports" / f"backtest_{run_id}"
    input_paths = [predictions_path, candidates_dir, prices_dir, calendar_path]
    if delistings_path is not None:
        input_paths.append(delistings_path)
    _ensure_new_outputs(input_paths, output_dir, report_dir)

    raw_predictions = _read_table(
        predictions_path,
        string_columns=("code", "model", "target_column"),
        columns=PREDICTION_COLUMNS,
    )
    predictions = _normalize_predictions(raw_predictions, config)
    delistings = (
        _read_table(delistings_path, string_columns=("code",), columns=DELISTING_COLUMNS)
        if delistings_path is not None
        else None
    )
    normalized_delistings = _normalize_delistings(delistings)
    prediction_years = sorted(predictions["signal_date"].dt.year.unique().tolist())
    candidate_columns = (
        "signal_date", "code", "entry_date", "exit_date", config.eligibility_column,
        config.liquidity_column,
    )
    candidate_paths = _annual_partitions(candidates_dir, prediction_years, "candidate")
    # lot_size is optional; read it only when every selected partition provides it.
    lot_size_presence = [config.lot_size_column in pq.read_schema(path).names for path in candidate_paths]
    if any(lot_size_presence) and not all(lot_size_presence):
        raise SchemaError("candidate partitions disagree on lot-size column presence")
    has_lot_sizes = all(lot_size_presence)
    columns = (*candidate_columns, *((config.lot_size_column,) if has_lot_sizes else ()))
    candidates = _read_partitions(candidate_paths, columns)
    candidates = candidates.loc[candidates["signal_date"].astype("datetime64[ns]").dt.year.isin(prediction_years)]
    calendar = _read_table(calendar_path)
    normalized_candidates = _normalize_candidates(candidates, config)
    relevant_candidates = normalized_candidates.merge(
        predictions[["signal_date", "code"]], on=["signal_date", "code"], how="inner"
    )
    if relevant_candidates.empty:
        raise SchemaError("No candidates match the selected predictions")
    calendar_checked = validate_trading_calendar(calendar)
    sessions = pd.DatetimeIndex(calendar_checked.loc[calendar_checked["is_session"], "date"])
    start = relevant_candidates["signal_date"].min()
    final_exit = relevant_candidates["exit_date"].max()
    final_position = sessions.get_indexer([final_exit])[0]
    if final_position < 0:
        raise SchemaError("candidate exit dates are absent from the official calendar")
    end_position = min(
        len(sessions) - 1,
        final_position + config.max_exit_wait_sessions + (1 if config.exit_price_mode == "next_session_open" else 0),
    )
    relevant_events = normalized_delistings.loc[
        normalized_delistings["code"].isin(set(predictions["code"].astype(str)))
    ]
    if not relevant_events.empty:
        event_end = pd.concat(
            [relevant_events["delisting_date"], relevant_events["settlement_date"].dropna()],
            ignore_index=True,
        ).max()
        event_session_positions = np.flatnonzero(sessions >= event_end)
        if not len(event_session_positions):
            raise SchemaError("trading calendar does not extend through the last relevant delisting event")
        end_position = max(end_position, int(event_session_positions[0]))
    end = sessions[end_position]
    price_years = list(range(start.year, end.year + 1))
    price_paths = _annual_partitions(prices_dir, price_years, "price")
    relevant_codes = set(predictions["code"].astype(str))
    prices = _read_price_partitions(
        price_paths,
        start=start,
        end=end,
        codes=sorted(relevant_codes),
    )
    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest_path = report_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "inputs": {
            "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
            "candidate_partitions": [{"path": str(path), "sha256": _sha256(path)} for path in candidate_paths],
            "price_partitions": [{"path": str(path), "sha256": _sha256(path)} for path in price_paths],
            "calendar": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
            "delistings": None if delistings_path is None else {"path": str(delistings_path), "sha256": _sha256(delistings_path)},
        },
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "completed_outputs": [],
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }
    _write_json(manifest_path, _json_safe(manifest))
    try:
        result = backtest_portfolio(
            predictions,
            candidates,
            prices,
            trading_calendar=calendar,
            delistings=delistings,
            config=config,
        )
        manifest.update(result.report)
        outputs = (
            ("orders", result.orders, output_dir / "orders.parquet"),
            ("trades", result.trades, output_dir / "trades.parquet"),
            ("positions", result.positions, output_dir / "positions.parquet"),
            ("corporate_actions", result.corporate_actions, output_dir / "corporate_actions.parquet"),
            ("daily_portfolio", result.daily_portfolio, output_dir / "daily_portfolio.parquet"),
            ("weekly_metrics", result.weekly_metrics, output_dir / "weekly_metrics.parquet"),
            ("yearly_metrics", result.yearly_metrics, output_dir / "yearly_metrics.parquet"),
            ("summary_metrics", result.summary_metrics, output_dir / "summary_metrics.parquet"),
        )
        for name, frame, path in outputs:
            frame.to_parquet(path, index=False)
            manifest["completed_outputs"].append(
                {"name": name, "path": str(path), "rows": int(len(frame)), "sha256": _sha256(path)}
            )
            _write_json(manifest_path, _json_safe(manifest))
        manifest["status"] = "complete" if result.report["main_result_complete"] else "incomplete_open_positions"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        if isinstance(exc, UnresolvedDelistingError):
            manifest["unresolved_delisting"] = {
                "code": exc.code,
                "delisting_date": exc.delisting_date,
                "position_id": exc.position_id,
                "sensitivity_instruction": "rerun with unresolved_delisting_policy='write_off'",
            }
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, _json_safe(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the executable weekly backtest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--prices-dir", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--delistings", type=Path)
    parser.add_argument("--config", type=Path, help="Strict JSON backtest configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config(args.config) if args.config else BacktestConfig()
    manifest = run_backtest(
        predictions_path=args.predictions,
        candidates_dir=args.candidates_dir,
        prices_dir=args.prices_dir,
        calendar_path=args.calendar,
        delistings_path=args.delistings,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_rows": manifest["selection"]["selected_rows"],
                "trade_rows": manifest["outputs"]["trade_rows"],
                "main_result_complete": manifest["main_result_complete"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
