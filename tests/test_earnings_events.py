"""Point-in-time, cooldown, and CLI tests for earnings-event exclusions."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.auxiliary_data_download import EARNINGS_SCHEDULE_COLUMNS
from stock_prediction.earnings_events import (
    EarningsEventConfig,
    compute_earnings_exclusions,
    load_config,
    main,
    run_earnings_events,
)
from stock_prediction.preprocessing import SchemaError


FINANCIAL_COLUMNS = ("DiscDate", "DiscTime", "Code", "DiscNo", "DocType")


def make_calendar(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.date_range(sessions.min(), sessions.max(), freq="D")
    open_dates = set(sessions)
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(["1" if value in open_dates else "0" for value in dates], dtype="string"),
        }
    )


def make_candidates(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["signal_date", "code", "exit_date"])


def make_schedule(rows: list[tuple[str, str | None, str, str, str]]) -> pd.DataFrame:
    records = []
    for publication, scheduled, fiscal_quarter, fiscal_year_end, code in rows:
        records.append(
            {
                "PubDate": publication,
                "SchDate": scheduled,
                "FQName": fiscal_quarter,
                "FYE": fiscal_year_end,
                "Code": code,
                "CoName": f"会社{code}",
                "CoNameEn": f"Company {code}",
            }
        )
    return pd.DataFrame.from_records(records, columns=EARNINGS_SCHEDULE_COLUMNS)


def make_financials(rows: list[tuple[str, str | None, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=FINANCIAL_COLUMNS)


def empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(columns=EARNINGS_SCHEDULE_COLUMNS)


def empty_financials() -> pd.DataFrame:
    return pd.DataFrame(columns=FINANCIAL_COLUMNS)


def test_schedule_revisions_are_applied_only_from_publication_date() -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    calendar = make_calendar(sessions)
    candidates = make_candidates(
        [
            ("2024-01-08", "10000", "2024-01-12"),
            ("2024-01-15", "10000", "2024-01-19"),
        ]
    )
    schedule = make_schedule(
        [
            ("2024-01-02", "2024-01-10", "3Q", "2024-03", "10000"),
            ("2024-01-09", "2024-01-30", "3Q", "2024-03", "10000"),
        ]
    )
    original_candidates = candidates.copy(deep=True)
    original_schedule = schedule.copy(deep=True)

    output = compute_earnings_exclusions(
        candidates,
        schedule,
        empty_financials(),
        trading_calendar=calendar,
    ).exclusions

    assert_frame_equal(candidates, original_candidates)
    assert_frame_equal(schedule, original_schedule)
    first, second = output.iloc[0], output.iloc[1]
    assert first["scheduled_earnings_during_holding"]
    assert first["first_scheduled_earnings_date"] == pd.Timestamp("2024-01-10")
    assert not first["is_earnings_eligible"]
    assert not second["scheduled_earnings_during_holding"]
    assert second["is_earnings_eligible"]

    changed_future = schedule.copy()
    changed_future.loc[changed_future["PubDate"].eq("2024-01-09"), "SchDate"] = "2024-01-16"
    changed = compute_earnings_exclusions(
        candidates,
        changed_future,
        empty_financials(),
        trading_calendar=calendar,
    ).exclusions
    historical_columns = [
        "signal_date",
        "code",
        "scheduled_earnings_during_holding",
        "first_scheduled_earnings_date",
        "is_earnings_eligible",
    ]
    assert_frame_equal(output.iloc[[0]][historical_columns], changed.iloc[[0]][historical_columns])
    assert changed.iloc[1]["scheduled_earnings_during_holding"]


def test_cooldown_excludes_sessions_one_through_twenty_five_and_respects_code_boundary() -> None:
    sessions = pd.bdate_range("2023-10-02", "2024-03-29")
    signal = pd.Timestamp("2024-02-13")  # Tuesday after a deliberately closed Monday.
    sessions = sessions.difference(pd.DatetimeIndex([pd.Timestamp("2024-02-12")]))
    signal_id = sessions.get_loc(signal)
    event_25 = sessions[signal_id - 25]
    event_26 = sessions[signal_id - 26]
    candidates = make_candidates(
        [
            (signal.strftime("%Y-%m-%d"), "10000", "2024-02-16"),
            (signal.strftime("%Y-%m-%d"), "20000", "2024-02-16"),
            (signal.strftime("%Y-%m-%d"), "30000", "2024-02-16"),
        ]
    )
    financials = make_financials(
        [
            (event_25.strftime("%Y-%m-%d"), "15:00:00", "10000", "A", "3QFinancialStatements_Consolidated_JP"),
            (event_26.strftime("%Y-%m-%d"), "15:00:00", "20000", "B", "FYFinancialStatements_Consolidated_JP"),
            (event_25.strftime("%Y-%m-%d"), "15:00:00", "30000", "C", "EarningsForecastRevision"),
        ]
    )

    output = compute_earnings_exclusions(
        candidates,
        empty_schedule(),
        financials,
        trading_calendar=make_calendar(sessions),
    ).exclusions.set_index("code")

    assert output.loc["10000", "trading_sessions_since_last_earnings"] == 25
    assert str(output["trading_sessions_since_last_earnings"].dtype) == "Int64"
    assert output.loc["10000", "is_in_post_earnings_cooldown"]
    assert not output.loc["10000", "is_earnings_eligible"]
    assert output.loc["10000", "is_earnings_eligible_without_cooldown"]
    assert output.loc["20000", "trading_sessions_since_last_earnings"] == 26
    assert not output.loc["20000", "is_in_post_earnings_cooldown"]
    assert output.loc["20000", "is_earnings_eligible"]
    assert pd.isna(output.loc["30000", "last_known_earnings_date"])
    assert output.loc["30000", "is_earnings_eligible"]


def test_actual_holding_event_is_ex_post_only_and_does_not_leak_into_main_flag() -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    actual = make_financials(
        [("2024-01-10", "15:00:00", "10000", "A", "3QFinancialStatements_Consolidated_JP")]
    )
    baseline = compute_earnings_exclusions(
        candidates,
        empty_schedule(),
        actual,
        trading_calendar=make_calendar(sessions),
    ).exclusions.iloc[0]
    assert baseline["is_earnings_eligible"]
    assert baseline["actual_earnings_during_holding"]
    assert not baseline["is_earnings_eligible_ex_post"]

    changed_future = actual.copy()
    changed_future["DocType"] = "EarningsForecastRevision"
    changed = compute_earnings_exclusions(
        candidates,
        empty_schedule(),
        changed_future,
        trading_calendar=make_calendar(sessions),
    ).exclusions.iloc[0]
    assert changed["is_earnings_eligible"] == baseline["is_earnings_eligible"]
    assert not changed["actual_earnings_during_holding"]
    assert changed["is_earnings_eligible_ex_post"]


def test_same_day_actual_uses_historical_tse_close_time() -> None:
    sessions = pd.bdate_range("2024-09-02", "2024-11-29")
    candidates = make_candidates(
        [
            ("2024-10-28", "10000", "2024-11-01"),
            ("2024-11-11", "20000", "2024-11-15"),
            ("2024-11-18", "30000", "2024-11-22"),
            ("2024-11-25", "40000", "2024-11-29"),
        ]
    )
    actual = make_financials(
        [
            ("2024-10-28", "15:10:00", "10000", "A", "2QFinancialStatements_Consolidated_JP"),
            ("2024-11-11", "15:20:00", "20000", "B", "2QFinancialStatements_Consolidated_JP"),
            ("2024-11-18", "15:40:00", "30000", "C", "2QFinancialStatements_Consolidated_JP"),
            ("2024-11-25", None, "40000", "D", "2QFinancialStatements_Consolidated_JP"),
        ]
    )
    output = compute_earnings_exclusions(
        candidates,
        empty_schedule(),
        actual,
        trading_calendar=make_calendar(sessions),
    ).exclusions.set_index("code")

    # Before 2024-11-05 the close was 15:00, so 15:10 is after the signal cutoff.
    assert output.loc["10000", "is_earnings_eligible"]
    assert output.loc["10000", "actual_earnings_during_holding"]
    # From 2024-11-05 the close is 15:30; 15:20 is known and starts cooldown day 0.
    assert output.loc["20000", "trading_sessions_since_last_earnings"] == 0
    assert output.loc["20000", "is_in_post_earnings_cooldown"]
    assert not output.loc["20000", "actual_earnings_during_holding"]
    assert not output.loc["20000", "is_earnings_eligible"]
    # After-close and missing-time rows cannot be used at the signal cutoff.
    assert output.loc["30000", "is_earnings_eligible"]
    assert output.loc["30000", "actual_earnings_during_holding"]
    assert output.loc["40000", "is_earnings_eligible"]
    assert output.loc["40000", "actual_earnings_during_holding"]


def test_signal_date_schedule_is_conservatively_treated_as_crossing() -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    schedule = make_schedule([("2024-01-08", "2024-01-08", "3Q", "2024-03", "10000")])
    row = compute_earnings_exclusions(
        candidates,
        schedule,
        empty_financials(),
        trading_calendar=make_calendar(sessions),
    ).exclusions.iloc[0]
    assert row["scheduled_earnings_during_holding"]
    assert not row["is_earnings_eligible"]


def test_missing_active_schedule_date_is_reported_without_silent_exclusion() -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    schedule = make_schedule([("2024-01-02", None, "3Q", "2024-03", "10000")])
    result = compute_earnings_exclusions(
        candidates,
        schedule,
        empty_financials(),
        trading_calendar=make_calendar(sessions),
    )
    assert result.exclusions.loc[0, "active_schedule_date_missing"]
    assert result.exclusions.loc[0, "is_earnings_eligible"]
    assert result.report["active_schedule_date_missing_rows"] == 1


def test_same_day_schedule_states_are_unordered_conservative_set() -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    candidates = make_candidates(
        [
            ("2024-01-08", "10000", "2024-01-12"),
            ("2024-01-15", "10000", "2024-01-19"),
        ]
    )
    schedule = make_schedule(
        [
            ("2024-01-02", None, "3Q", "2024-03", "10000"),
            ("2024-01-02", "2024-01-11", "3Q", "2024-03", "10000"),
            ("2024-01-02", "2024-01-10", "3Q", "2024-03", "10000"),
            ("2024-01-09", "2024-01-30", "3Q", "2024-03", "10000"),
        ]
    )

    result = compute_earnings_exclusions(
        candidates,
        schedule,
        empty_financials(),
        trading_calendar=make_calendar(sessions),
    )
    reversed_result = compute_earnings_exclusions(
        candidates,
        schedule.iloc[::-1].reset_index(drop=True),
        empty_financials(),
        trading_calendar=make_calendar(sessions),
    )

    assert_frame_equal(result.exclusions, reversed_result.exclusions)
    first, second = result.exclusions.iloc[0], result.exclusions.iloc[1]
    assert first["active_schedule_date_missing"]
    assert first["scheduled_earnings_count"] == 1
    assert first["first_scheduled_earnings_date"] == pd.Timestamp("2024-01-10")
    assert not first["is_earnings_eligible"]
    assert not second["active_schedule_date_missing"]
    assert second["scheduled_earnings_count"] == 0
    assert second["is_earnings_eligible"]
    assert result.report["same_day_schedule_state_groups"] == 1
    assert result.report["same_day_schedule_state_rows"] == 3
    assert result.report["same_day_schedule_state_groups_with_missing_date"] == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda candidates, schedule, financials: (
                pd.concat([candidates, candidates], ignore_index=True),
                schedule,
                financials,
            ),
            "duplicate signal_date/code",
        ),
        (
            lambda candidates, schedule, financials: (
                candidates,
                pd.concat([schedule, schedule], ignore_index=True),
                financials,
            ),
            "exact duplicate rows",
        ),
        (
            lambda candidates, schedule, financials: (
                candidates,
                schedule,
                financials.assign(FutureField="x"),
            ),
            "Unsupported financial_results schema",
        ),
        (
            lambda candidates, schedule, financials: (
                candidates,
                schedule,
                financials.assign(DocType="FinancialStatements_3Q"),
            ),
            "not matched by earnings_doc_type_prefixes",
        ),
    ],
)
def test_schema_errors_fail_early(mutation, match: str) -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    schedule = make_schedule([("2024-01-02", "2024-01-10", "3Q", "2024-03", "10000")])
    financials = make_financials(
        [("2024-01-02", "15:00:00", "10000", "A", "3QFinancialStatements_Consolidated_JP")]
    )
    changed = mutation(candidates, schedule, financials)
    with pytest.raises(SchemaError, match=match):
        compute_earnings_exclusions(*changed, trading_calendar=make_calendar(sessions))


def test_calendar_requires_cooldown_history() -> None:
    sessions = pd.bdate_range("2024-01-08", "2024-01-12")
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    with pytest.raises(SchemaError, match="at least cooldown_trading_sessions"):
        compute_earnings_exclusions(
            candidates,
            empty_schedule(),
            empty_financials(),
            trading_calendar=make_calendar(sessions),
        )


def test_config_and_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "earnings_events.json"
    config_path.write_text(
        json.dumps(
            {
                "cooldown_trading_sessions": 25,
                "earnings_doc_type_prefixes": [
                    "1QFinancialStatements",
                    "2QFinancialStatements",
                    "3QFinancialStatements",
                    "FYFinancialStatements",
                    "OtherPeriodFinancialStatements",
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_config(config_path) == EarningsEventConfig()
    config_path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(config_path)
    with pytest.raises(ValueError, match="positive integer"):
        EarningsEventConfig(cooldown_trading_sessions=0)


def test_partitioned_cli_matches_dataframe_api_and_writes_manifest(tmp_path: Path, capsys) -> None:
    sessions = pd.bdate_range("2023-11-01", "2024-02-02")
    calendar = make_calendar(sessions)
    candidates = make_candidates([("2024-01-08", "10000", "2024-01-12")])
    schedule = make_schedule([("2024-01-02", "2024-01-10", "3Q", "2024-03", "10000")])
    financials = make_financials(
        [("2024-01-02", "15:00:00", "10000", "A", "3QFinancialStatements_Consolidated_JP")]
    )
    candidate_dir = tmp_path / "candidates"
    schedule_dir = tmp_path / "earnings_schedule"
    financial_dir = tmp_path / "financial_results"
    for path in (candidate_dir, schedule_dir, financial_dir):
        path.mkdir()
    candidates.to_parquet(candidate_dir / "2024.parquet", index=False)
    schedule.to_parquet(schedule_dir / "2024.parquet", index=False)
    financials.to_parquet(financial_dir / "2024.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"

    status = main(
        [
            "--candidates-dir",
            str(candidate_dir),
            "--earnings-schedule-dir",
            str(schedule_dir),
            "--financial-results-dir",
            str(financial_dir),
            "--calendar",
            str(calendar_path),
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert status == 0
    assert '"status": "complete"' in capsys.readouterr().out
    cli_output = pd.read_parquet(output_dir / "2024.parquet")
    api_output = compute_earnings_exclusions(
        candidates,
        schedule,
        financials,
        trading_calendar=calendar,
    ).exclusions
    assert_frame_equal(cli_output, api_output)
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["output_rows"] == 1
    assert manifest["scheduled_crossing_rows"] == 1
    assert manifest["cooldown_rows"] == 1
    assert manifest["eligible_rows"] == 0
    assert manifest["same_day_schedule_state_groups"] == 0
    assert manifest["same_day_schedule_state_rows"] == 0
    assert manifest["same_day_schedule_state_groups_with_missing_date"] == 0

    with pytest.raises(FileExistsError):
        run_earnings_events(
            candidates_dir=candidate_dir,
            earnings_schedule_dir=schedule_dir,
            financial_results_dir=financial_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "other-reports",
        )


def test_interruption_preserves_completed_partition_and_manifest_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("stock_prediction.earnings_events")
    original_compute = module.compute_earnings_exclusions
    sessions = pd.bdate_range("2023-10-02", "2024-01-31")
    calendar = make_calendar(sessions)
    candidate_dir = tmp_path / "candidates"
    schedule_dir = tmp_path / "earnings_schedule"
    financial_dir = tmp_path / "financial_results"
    for path in (candidate_dir, schedule_dir, financial_dir):
        path.mkdir()
    make_candidates([("2023-12-18", "10000", "2023-12-22")]).to_parquet(
        candidate_dir / "2023.parquet", index=False
    )
    make_candidates([("2024-01-08", "10000", "2024-01-12")]).to_parquet(
        candidate_dir / "2024.parquet", index=False
    )
    make_schedule([("2023-12-01", "2023-12-20", "3Q", "2024-03", "10000")]).to_parquet(
        schedule_dir / "2023.parquet", index=False
    )
    make_financials(
        [("2023-11-01", "15:00:00", "10000", "A", "2QFinancialStatements_Consolidated_JP")]
    ).to_parquet(financial_dir / "2023.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"

    def stop_second_year(candidates: pd.DataFrame, *args, **kwargs):
        if pd.to_datetime(candidates["signal_date"]).dt.year.eq(2024).any():
            raise KeyboardInterrupt("test interruption")
        return original_compute(candidates, *args, **kwargs)

    monkeypatch.setattr(module, "compute_earnings_exclusions", stop_second_year)
    with pytest.raises(KeyboardInterrupt):
        run_earnings_events(
            candidates_dir=candidate_dir,
            earnings_schedule_dir=schedule_dir,
            financial_results_dir=financial_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=report_dir,
        )

    assert (output_dir / "2023.parquet").is_file()
    assert not (output_dir / "2024.parquet").exists()
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["current_partition"] == 2024
    assert [item["year"] for item in manifest["completed_partitions"]] == [2023]
