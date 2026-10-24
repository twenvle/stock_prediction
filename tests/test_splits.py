"""Purged walk-forward, isolation, schema, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.preprocessing import SchemaError
from stock_prediction.splits import (
    SplitConfig,
    build_walk_forward_splits,
    load_config,
    main,
    run_splits,
)


def make_calendar(start: str = "2015-12-01", end: str = "2027-01-15") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(["1" if value.weekday() < 5 else "0" for value in dates], dtype="string"),
        }
    )


def make_dataset() -> pd.DataFrame:
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for year in range(2016, 2027):
        sessions = pd.bdate_range(f"{year}-06-01", periods=5)
        periods.append((sessions[0], sessions[-1]))
    # The 2020 validation starts on 2020-01-03.  This late-2019 sample does
    # not overlap that label period, but its signal is inside the four-session
    # purge window and must therefore be removed from the first training fold.
    periods.extend(
        [
            (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02")),
            (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-09")),
        ]
    )
    records: list[dict[str, object]] = []
    for signal_date, exit_date in periods:
        for code in ("10000", "20000"):
            records.append({"signal_date": signal_date, "code": code, "exit_date": exit_date, "feature": 1.0})
    return pd.DataFrame.from_records(records).sort_values(["signal_date", "code"]).reset_index(drop=True)


def test_expanding_folds_purge_and_isolation_keep_signal_groups_together() -> None:
    dataset = make_dataset()
    original = dataset.copy(deep=True)
    result = build_walk_forward_splits(dataset, make_calendar())

    assert_frame_equal(dataset, original)
    assert [fold.validation_year for fold in result.folds] == [2020, 2021, 2022, 2023, 2024]
    assert result.report["maximum_development_label_horizon_sessions"] == 4
    assert set(dataset.iloc[result.isolated_positions]["signal_date"].dt.year) == {2025, 2026}
    assert not len(result.out_of_scope_positions)

    isolated = set(result.isolated_positions.tolist())
    for fold in result.folds:
        train = dataset.iloc[fold.train_positions]
        validation = dataset.iloc[fold.validation_positions]
        assert validation["signal_date"].dt.year.eq(fold.validation_year).all()
        assert train["signal_date"].max() < validation["signal_date"].min()
        assert train["exit_date"].max() < validation["signal_date"].min()
        assert isolated.isdisjoint(fold.train_positions.tolist())
        assert isolated.isdisjoint(fold.validation_positions.tolist())
        for signal_date, group in dataset.groupby("signal_date"):
            group_positions = set(group.index.tolist())
            memberships = [
                bool(group_positions & set(fold.train_positions.tolist())),
                bool(group_positions & set(fold.validation_positions.tolist())),
                bool(group_positions & set(fold.purged_positions.tolist())),
            ]
            assert sum(memberships) <= 1
            if any(memberships):
                selected = (
                    fold.train_positions
                    if memberships[0]
                    else fold.validation_positions
                    if memberships[1]
                    else fold.purged_positions
                )
                assert group_positions.issubset(set(selected.tolist()))

    first_fold = result.folds[0]
    purged_dates = set(dataset.iloc[first_fold.purged_positions]["signal_date"])
    assert pd.Timestamp("2019-12-31") in purged_dates
    assert pd.Timestamp("2019-06-03") in set(dataset.iloc[first_fold.train_positions]["signal_date"])


def test_isolation_bounds_are_selectable_and_unselected_future_stays_out_of_development() -> None:
    dataset = make_dataset()
    result = build_walk_forward_splits(
        dataset,
        make_calendar(),
        config=SplitConfig(isolation_start_date="2025-04-01", isolation_end_date="2025-12-31"),
    )

    isolated = dataset.iloc[result.isolated_positions]
    outside = dataset.iloc[result.out_of_scope_positions]
    assert isolated["signal_date"].dt.year.eq(2025).all()
    assert set(outside["signal_date"].dt.year) == {2026}
    assert result.report["out_of_scope"]["reason_signal_date_counts"] == {"after_selected_isolation": 1}
    all_fold_positions = np.concatenate(
        [np.concatenate([fold.train_positions, fold.validation_positions]) for fold in result.folds]
    )
    assert dataset.iloc[all_fold_positions]["signal_date"].dt.year.le(2024).all()


def test_labels_crossing_development_or_isolation_end_are_out_of_scope() -> None:
    dataset = make_dataset()
    crossing = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-12-31", "2024-12-31", "2026-12-30", "2026-12-30"]
            ),
            "code": ["10000", "20000", "10000", "20000"],
            "exit_date": pd.to_datetime(
                ["2025-01-03", "2025-01-03", "2027-01-04", "2027-01-04"]
            ),
        }
    )
    result = build_walk_forward_splits(pd.concat([dataset, crossing], ignore_index=True), make_calendar())

    assert result.report["out_of_scope"]["rows"] == 4
    assert result.report["out_of_scope"]["reason_signal_date_counts"] == {
        "label_crosses_development_end": 1,
        "label_crosses_isolation_end": 1,
    }
    assigned = np.concatenate(
        [
            *[np.concatenate([fold.train_positions, fold.validation_positions]) for fold in result.folds],
            result.isolated_positions,
        ]
    )
    assert set(result.out_of_scope_positions).isdisjoint(assigned.tolist())


def test_appending_post_isolation_rows_does_not_change_development_definitions() -> None:
    dataset = make_dataset()
    baseline = build_walk_forward_splits(dataset, make_calendar())
    future = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2027-01-04", "2027-01-04"]),
            "code": ["10000", "20000"],
            "exit_date": pd.to_datetime(["2027-01-08", "2027-01-08"]),
            "feature": [999.0, -999.0],
        }
    )
    extended = build_walk_forward_splits(pd.concat([dataset, future], ignore_index=True), make_calendar())

    assert_frame_equal(baseline.fold_definitions, extended.fold_definitions)
    for before, after in zip(baseline.folds, extended.folds):
        np.testing.assert_array_equal(before.train_positions, after.train_positions)
        np.testing.assert_array_equal(before.validation_positions, after.validation_positions)
        np.testing.assert_array_equal(before.purged_positions, after.purged_positions)
    assert len(extended.out_of_scope_positions) == 2


def test_purge_must_cover_observed_label_horizon() -> None:
    with pytest.raises(ValueError, match=r"maximum development label horizon \(4 TSE sessions\)"):
        build_walk_forward_splits(
            make_dataset(),
            make_calendar(),
            config=SplitConfig(purge_trading_sessions=3),
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("missing", "missing required columns"),
        ("duplicate", "duplicate signal_date/code"),
        ("conflicting_exit", "must map to one exit_date"),
        ("non_session", "official TSE trading session"),
    ],
)
def test_schema_errors_fail_early(change: str, match: str) -> None:
    dataset = make_dataset()
    calendar = make_calendar()
    if change == "missing":
        dataset = dataset.drop(columns="exit_date")
    elif change == "duplicate":
        dataset = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
    elif change == "conflicting_exit":
        row = dataset.iloc[[0]].copy()
        row["code"] = "30000"
        row["exit_date"] = row["exit_date"] + pd.Timedelta(days=3)
        dataset = pd.concat([dataset, row], ignore_index=True)
    else:
        dataset.loc[0:1, "signal_date"] = pd.Timestamp("2016-06-04")
    with pytest.raises(SchemaError, match=match):
        build_walk_forward_splits(dataset, calendar)


def test_config_is_strict_and_keeps_isolation_within_research_period(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(
        json.dumps(
            {
                "isolation_start_date": "2025-04-01",
                "isolation_end_date": "2026-03-31",
                "purge_trading_sessions": 5,
            }
        ),
        encoding="utf-8",
    )
    assert load_config(valid_path) == SplitConfig("2025-04-01", "2026-03-31", 5)

    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text('{"unknown": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(unknown_path)
    with pytest.raises(ValueError, match="within 2025-01-01 through 2026-12-31"):
        SplitConfig(isolation_start_date="2024-12-31")
    with pytest.raises(ValueError, match="must not follow"):
        SplitConfig(isolation_start_date="2026-01-01", isolation_end_date="2025-12-31")
    with pytest.raises(ValueError, match="non-negative integer"):
        SplitConfig(purge_trading_sessions=True)


def test_cli_outputs_match_dataframe_api(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = make_dataset()
    input_dir = tmp_path / "weekly"
    input_dir.mkdir()
    for year, part in dataset.groupby(dataset["signal_date"].dt.year, sort=True):
        part.to_parquet(input_dir / f"{year}.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    make_calendar().to_parquet(calendar_path, index=False)
    config_path = tmp_path / "splits.json"
    config_path.write_text(
        json.dumps(
            {
                "isolation_start_date": "2025-01-01",
                "isolation_end_date": "2026-12-31",
                "purge_trading_sessions": 4,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--calendar",
                str(calendar_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--report-dir",
                str(report_dir),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    direct = build_walk_forward_splits(dataset, make_calendar())
    assert printed["status"] == "complete"
    assert manifest["status"] == "complete"
    assert manifest["input"]["rows"] == len(dataset)
    assert len(manifest["completed_outputs"]) == 3
    assert_frame_equal(pd.read_parquet(output_dir / "folds.parquet"), direct.fold_definitions)
    assert_frame_equal(pd.read_parquet(output_dir / "isolated.parquet"), direct.isolated_definitions)
    assert_frame_equal(pd.read_parquet(output_dir / "out_of_scope.parquet"), direct.out_of_scope_definitions)

    with pytest.raises(FileExistsError, match="already exists"):
        run_splits(
            input_dir=input_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "other_reports",
        )


def test_interruption_preserves_completed_output_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset()
    input_dir = tmp_path / "weekly"
    input_dir.mkdir()
    for year, part in dataset.groupby(dataset["signal_date"].dt.year, sort=True):
        part.to_parquet(input_dir / f"{year}.parquet", index=False)
    calendar_path = tmp_path / "calendar.parquet"
    make_calendar().to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "output"
    report_dir = tmp_path / "reports"
    original_to_parquet = pd.DataFrame.to_parquet

    def interrupt_on_isolation(self, path, *args, **kwargs):
        if Path(path).name == "isolated.parquet":
            raise KeyboardInterrupt
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupt_on_isolation)
    with pytest.raises(KeyboardInterrupt):
        run_splits(
            input_dir=input_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=report_dir,
        )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert [item["name"] for item in manifest["completed_outputs"]] == ["folds"]
    assert (output_dir / "folds.parquet").is_file()
    assert not (output_dir / "isolated.parquet").exists()
