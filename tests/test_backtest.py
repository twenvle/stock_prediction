"""Execution, accounting, delisting, and CLI tests for the backtest."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.backtest import (
    BacktestConfig,
    UnresolvedDelistingError,
    backtest_portfolio,
    load_config,
    main,
    run_backtest,
)
from stock_prediction.preprocessing import SchemaError


def make_calendar(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.date_range(sessions.min(), sessions.max(), freq="D")
    open_dates = set(sessions)
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(["1" if day in open_dates else "0" for day in dates], dtype="string"),
        }
    )


def make_predictions(codes: tuple[str, ...] = ("10000", "20000")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-08"] * len(codes)),
            "code": pd.array(codes, dtype="string"),
            "model": pd.array(["linear"] * len(codes), dtype="string"),
            "target_column": pd.array(["target_return"] * len(codes), dtype="string"),
            "prediction": [0.2 - index * 0.1 for index in range(len(codes))],
            "actual_target": [999.0] * len(codes),
        }
    )


def make_candidates(codes: tuple[str, ...] = ("10000", "20000")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-08"] * len(codes)),
            "code": pd.array(codes, dtype="string"),
            "entry_date": pd.to_datetime(["2024-01-09"] * len(codes)),
            "exit_date": pd.to_datetime(["2024-01-12"] * len(codes)),
            "is_signal_eligible": [True] * len(codes),
            "median_turnover_value_20": [100_000_000.0] * len(codes),
            "lot_size": pd.array([100] * len(codes), dtype="Int64"),
        }
    )


def make_prices(
    sessions: pd.DatetimeIndex,
    codes: tuple[str, ...] = ("10000", "20000"),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        for day in sessions:
            base = 100.0 + code_index * 10
            records.append(
                {
                    "date": day,
                    "code": code,
                    "open": base,
                    "close": base,
                    "upper_limit": "0",
                    "lower_limit": "0",
                    "volume": 1_000_000.0,
                    "turnover_value": 100_000_000.0,
                    "adjustment_factor": 1.0,
                    "has_price": True,
                    "calendar_verified": True,
                }
            )
    frame = pd.DataFrame.from_records(records)
    for column in ("code", "upper_limit", "lower_limit"):
        frame[column] = frame[column].astype("string")
    return frame


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-08", "2024-02-16")


def test_costs_round_lots_and_turnover_have_known_values(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-12")), "close"] = 110.0
    config = BacktestConfig(initial_cash_yen=100_000.0, top_n=1)

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=config,
    )

    assert list(result.trades["side"]) == ["buy", "sell"]
    assert list(result.trades["shares"]) == [900.0, 900.0]
    assert result.trades.iloc[0]["effective_price"] == pytest.approx(100.15)
    assert result.trades.iloc[1]["effective_price"] == pytest.approx(109.835)
    assert result.summary_metrics.iloc[0]["ending_equity_yen"] == pytest.approx(108_622.00675)
    assert result.summary_metrics.iloc[0]["total_transaction_cost_yen"] == pytest.approx(377.99325)
    assert result.weekly_metrics.iloc[0]["two_way_turnover"] == pytest.approx(1.89)
    assert result.yearly_metrics.iloc[0]["year"] == 2024
    assert result.yearly_metrics.iloc[0]["ending_equity_yen"] == pytest.approx(108_622.00675)
    assert result.summary_metrics.iloc[0]["main_result_complete"]


def test_limit_up_entry_is_unfilled_and_expires(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-09")), "upper_limit"] = "1"

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(top_n=1),
    )

    assert result.positions.empty
    assert len(result.orders) == 1
    assert result.orders.iloc[0]["reason"] == "limit_up"
    assert result.orders.iloc[0]["status"] == "unfilled"
    assert result.summary_metrics.iloc[0]["ending_equity_yen"] == pytest.approx(10_000_000.0)


def test_participation_cap_creates_round_lot_partial_fill(sessions: pd.DatetimeIndex) -> None:
    candidates = make_candidates(("10000",))
    candidates["median_turnover_value_20"] = 1_000_000.0
    result = backtest_portfolio(
        make_predictions(("10000",)),
        candidates,
        make_prices(sessions, ("10000",)),
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=1_000_000.0, top_n=1, max_participation_rate=0.01),
    )

    entry = result.orders.query("side == 'buy'").iloc[0]
    assert entry["requested_shares"] == 9_900.0
    assert entry["participation_capacity_shares"] == 100.0
    assert entry["filled_shares"] == 100.0
    assert entry["status"] == "partially_filled"


def test_limit_down_exit_retries_without_dropping_position(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-12")), "lower_limit"] = "1"
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-15")), "close"] = 105.0

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=100_000.0, top_n=1),
    )

    exits = result.orders.query("side == 'sell'").reset_index(drop=True)
    assert list(exits["reason"]) == ["limit_down", "delayed_exit_retry"]
    assert result.positions.iloc[0]["final_exit_date"] == pd.Timestamp("2024-01-15")
    assert result.positions.iloc[0]["exit_attempts"] == 2


def test_zero_volume_exit_is_treated_as_suspension_proxy_and_retried(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    exit_mask = prices["date"].eq(pd.Timestamp("2024-01-12"))
    prices.loc[exit_mask, ["volume", "turnover_value"]] = 0.0

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=100_000.0, top_n=1),
    )

    exits = result.orders.query("side == 'sell'").reset_index(drop=True)
    assert list(exits["reason"]) == ["no_trade_or_suspension", "delayed_exit_retry"]
    assert result.summary_metrics.iloc[0]["main_result_complete"]


def test_unfilled_exit_remains_open_and_marks_main_result_incomplete(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].ge(pd.Timestamp("2024-01-12")), "lower_limit"] = "1"

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=100_000.0, top_n=1, max_exit_wait_sessions=2),
    )

    assert len(result.orders.query("side == 'sell'")) == 2
    assert result.positions.iloc[0]["status"] == "open"
    assert result.summary_metrics.iloc[0]["open_positions"] == 1
    assert not result.summary_metrics.iloc[0]["main_result_complete"]
    assert not result.report["main_result_complete"]


def test_next_session_open_exit_is_configurable(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-12")), "close"] = 999.0
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-15")), "open"] = 120.0

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(
            initial_cash_yen=100_000.0,
            top_n=1,
            exit_price_mode="next_session_open",
        ),
    )

    sell = result.trades.query("side == 'sell'").iloc[0]
    assert sell["trade_date"] == pd.Timestamp("2024-01-15")
    assert sell["reference_price"] == 120.0


def test_known_delisting_settlement_and_unresolved_write_off(sessions: pd.DatetimeIndex) -> None:
    events = pd.DataFrame(
        {
            "code": pd.array(["10000"], dtype="string"),
            "delisting_date": pd.to_datetime(["2024-01-11"]),
            "settlement_date": pd.to_datetime(["2024-01-15"]),
            "settlement_value_per_share": [80.0],
        }
    )
    common = dict(
        predictions=make_predictions(("10000",)),
        candidates=make_candidates(("10000",)),
        prices=make_prices(sessions, ("10000",)),
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=100_000.0, top_n=1),
    )
    settled = backtest_portfolio(**common, delistings=events)
    position = settled.positions.iloc[0]
    assert position["exit_reason"] == "delisting_settlement"
    assert position["final_exit_date"] == pd.Timestamp("2024-01-15")
    assert settled.trades.query("side == 'sell'").iloc[0]["total_cost"] == 0.0

    unresolved = events.copy()
    unresolved["settlement_date"] = pd.NaT
    unresolved["settlement_value_per_share"] = np.nan
    with pytest.raises(UnresolvedDelistingError):
        backtest_portfolio(**common, delistings=unresolved)

    sensitivity = backtest_portfolio(
        **{**common, "config": replace(common["config"], unresolved_delisting_policy="write_off")},
        delistings=unresolved,
    )
    assert sensitivity.positions.iloc[0]["exit_reason"] == "unresolved_delisting_write_off_sensitivity"
    assert sensitivity.trades.query("side == 'sell'").iloc[0]["gross_notional"] == 0.0


def test_split_factor_changes_held_shares_before_exit(sessions: pd.DatetimeIndex) -> None:
    prices = make_prices(sessions, ("10000",))
    prices.loc[prices["date"].eq(pd.Timestamp("2024-01-10")), ["adjustment_factor", "open", "close"]] = [0.5, 50.0, 50.0]
    prices.loc[prices["date"].gt(pd.Timestamp("2024-01-10")), ["open", "close"]] = [50.0, 50.0]

    result = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        prices,
        trading_calendar=make_calendar(sessions),
        config=BacktestConfig(initial_cash_yen=100_000.0, top_n=1),
    )

    assert len(result.corporate_actions) == 1
    action = result.corporate_actions.iloc[0]
    assert action["shares_after"] == action["shares_before"] * 2
    assert result.trades.query("side == 'sell'").iloc[0]["shares"] == 1_800.0


def test_schema_rejects_random_entry_date_and_prediction_duplicates(sessions: pd.DatetimeIndex) -> None:
    candidates = make_candidates(("10000",))
    candidates["entry_date"] = pd.Timestamp("2024-01-10")
    with pytest.raises(SchemaError, match="next TSE session"):
        backtest_portfolio(
            make_predictions(("10000",)),
            candidates,
            make_prices(sessions, ("10000",)),
            trading_calendar=make_calendar(sessions),
        )

    duplicate = pd.concat([make_predictions(("10000",))] * 2, ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        backtest_portfolio(
            duplicate,
            make_candidates(("10000",)),
            make_prices(sessions, ("10000",)),
            trading_calendar=make_calendar(sessions),
        )


def test_actual_target_values_never_change_orders_or_results(sessions: pd.DatetimeIndex) -> None:
    predictions = make_predictions(("10000",))
    changed = predictions.copy()
    changed["actual_target"] = -999_999.0
    arguments = {
        "candidates": make_candidates(("10000",)),
        "prices": make_prices(sessions, ("10000",)),
        "trading_calendar": make_calendar(sessions),
        "config": BacktestConfig(initial_cash_yen=100_000.0, top_n=1),
    }

    baseline = backtest_portfolio(predictions, **arguments)
    future_changed = backtest_portfolio(changed, **arguments)

    assert_frame_equal(baseline.orders, future_changed.orders)
    assert_frame_equal(baseline.daily_portfolio, future_changed.daily_portfolio)
    assert_frame_equal(baseline.summary_metrics, future_changed.summary_metrics)


def test_config_defaults_and_strict_loader(tmp_path: Path) -> None:
    path = tmp_path / "backtest.json"
    path.write_text(
        json.dumps(
            {
                "commission_bps": 5.0,
                "half_spread_bps": 5.0,
                "slippage_bps": 10.0,
                "exit_price_mode": "evaluation_close",
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert (config.commission_bps, config.half_spread_bps, config.slippage_bps) == (5.0, 5.0, 10.0)

    path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def _write_cli_inputs(tmp_path: Path, sessions: pd.DatetimeIndex) -> tuple[Path, Path, Path, Path]:
    inputs = tmp_path / "inputs"
    candidates_dir = inputs / "candidates"
    prices_dir = inputs / "prices"
    candidates_dir.mkdir(parents=True)
    prices_dir.mkdir(parents=True)
    predictions_path = inputs / "predictions.parquet"
    calendar_path = inputs / "calendar.parquet"
    make_predictions(("10000",)).to_parquet(predictions_path, index=False)
    make_candidates(("10000",)).to_parquet(candidates_dir / "2024.parquet", index=False)
    make_prices(sessions, ("10000",)).to_parquet(prices_dir / "2024.parquet", index=False)
    make_calendar(sessions).to_parquet(calendar_path, index=False)
    return predictions_path, candidates_dir, prices_dir, calendar_path


def test_cli_runner_matches_dataframe_api_and_preserves_manifest(tmp_path: Path, sessions: pd.DatetimeIndex, capsys) -> None:
    predictions_path, candidates_dir, prices_dir, calendar_path = _write_cli_inputs(tmp_path, sessions)
    output_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    config = BacktestConfig(top_n=1)
    expected = backtest_portfolio(
        make_predictions(("10000",)),
        make_candidates(("10000",)),
        make_prices(sessions, ("10000",)),
        trading_calendar=make_calendar(sessions),
        config=config,
    )

    manifest = run_backtest(
        predictions_path=predictions_path,
        candidates_dir=candidates_dir,
        prices_dir=prices_dir,
        calendar_path=calendar_path,
        output_dir=output_dir,
        report_dir=report_dir,
        config=config,
    )
    assert manifest["status"] == "complete"
    assert_frame_equal(pd.read_parquet(output_dir / "trades.parquet"), expected.trades)
    assert_frame_equal(pd.read_parquet(output_dir / "summary_metrics.parquet"), expected.summary_metrics)
    saved = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert len(saved["completed_outputs"]) == 8

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"top_n": 1}), encoding="utf-8")
    assert main(
        [
            "--predictions", str(predictions_path),
            "--candidates-dir", str(candidates_dir),
            "--prices-dir", str(prices_dir),
            "--calendar", str(calendar_path),
            "--config", str(config_path),
            "--output-dir", str(tmp_path / "artifacts_main"),
            "--report-dir", str(tmp_path / "reports_main"),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"

    with pytest.raises(FileExistsError):
        run_backtest(
            predictions_path=predictions_path,
            candidates_dir=candidates_dir,
            prices_dir=prices_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "another_report",
            config=config,
        )


def test_failed_unresolved_delisting_is_recorded_in_manifest(tmp_path: Path, sessions: pd.DatetimeIndex) -> None:
    predictions_path, candidates_dir, prices_dir, calendar_path = _write_cli_inputs(tmp_path, sessions)
    delistings_path = tmp_path / "inputs" / "delistings.parquet"
    pd.DataFrame(
        {
            "code": pd.array(["10000"], dtype="string"),
            "delisting_date": pd.to_datetime(["2024-01-11"]),
            "settlement_date": pd.to_datetime([pd.NaT]),
            "settlement_value_per_share": [np.nan],
        }
    ).to_parquet(delistings_path, index=False)
    report_dir = tmp_path / "reports"
    with pytest.raises(UnresolvedDelistingError):
        run_backtest(
            predictions_path=predictions_path,
            candidates_dir=candidates_dir,
            prices_dir=prices_dir,
            calendar_path=calendar_path,
            delistings_path=delistings_path,
            output_dir=tmp_path / "artifacts",
            report_dir=report_dir,
            config=BacktestConfig(top_n=1),
        )
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "UnresolvedDelistingError"
    assert manifest["unresolved_delisting"]["code"] == "10000"


def test_interruption_manifest_preserves_completed_state(
    tmp_path: Path,
    sessions: pd.DatetimeIndex,
    monkeypatch,
) -> None:
    predictions_path, candidates_dir, prices_dir, calendar_path = _write_cli_inputs(tmp_path, sessions)
    report_dir = tmp_path / "reports"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("stock_prediction.backtest.backtest_portfolio", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_backtest(
            predictions_path=predictions_path,
            candidates_dir=candidates_dir,
            prices_dir=prices_dir,
            calendar_path=calendar_path,
            output_dir=tmp_path / "artifacts",
            report_dir=report_dir,
            config=BacktestConfig(top_n=1),
        )
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["completed_outputs"] == []
