"""Fold-local preprocessing, baseline model, metric, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.descriptors import FEATURE_COLUMNS
from stock_prediction.modeling import (
    MODEL_NAMES,
    ModelingConfig,
    build_walk_forward_models,
    load_config,
    main,
    run_modeling,
)
from stock_prediction.preprocessing import SchemaError


def make_calendar(start: str = "2015-12-01", end: str = "2027-01-15") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "HolDiv": pd.array(
                ["1" if value.weekday() < 5 else "0" for value in dates], dtype="string"
            ),
        }
    )


def make_dataset() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    codes = ("10000", "20000", "30000", "40000")
    for year in range(2016, 2027):
        sessions = pd.bdate_range(f"{year}-06-01", periods=10)
        for week, (signal_date, exit_date) in enumerate(
            ((sessions[0], sessions[4]), (sessions[5], sessions[9]))
        ):
            for code_index, code in enumerate(codes):
                base = float((year - 2015) * 10 + week * 2 + code_index)
                target = 0.002 * base - 0.01
                record: dict[str, object] = {
                    "signal_date": signal_date,
                    "code": code,
                    "exit_date": exit_date,
                    "is_signal_eligible": code != "40000",
                    "target_return": target,
                    "target_excess_return": target - 0.001,
                }
                for feature_index, name in enumerate(FEATURE_COLUMNS):
                    record[name] = base / 100.0 + feature_index / 1000.0
                records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame["is_target_return_sample"] = (
        frame["is_signal_eligible"] & frame["target_return"].notna()
    )
    frame["is_target_excess_return_sample"] = (
        frame["is_signal_eligible"] & frame["target_excess_return"].notna()
    )
    frame.loc[
        (frame["signal_date"].dt.year == 2017) & (frame["code"] == "10000"), "R_5"
    ] = np.nan
    frame.loc[
        (frame["signal_date"].dt.year == 2020)
        & (frame["code"] == "30000")
        & (frame["signal_date"].dt.day > 7),
        "target_return",
    ] = np.nan
    frame["is_target_return_sample"] = (
        frame["is_signal_eligible"] & frame["target_return"].notna()
    )
    return frame.sort_values(["signal_date", "code"]).reset_index(drop=True)


def test_three_models_share_validation_candidates_and_fit_training_rows_only() -> None:
    dataset = make_dataset()
    original = dataset.copy(deep=True)
    result = build_walk_forward_models(
        dataset,
        make_calendar(),
        config=ModelingConfig(top_n=2),
    )

    assert_frame_equal(dataset, original)
    assert set(result.fitted_models) == {f"validation_{year}" for year in range(2020, 2025)}
    assert set(result.fold_metrics["model"]) == set(MODEL_NAMES)
    assert len(result.fold_metrics) == 15
    for (_, _), group in result.predictions.groupby(["fold_id", "signal_date"]):
        by_model = {name: set(part["code"]) for name, part in group.groupby("model")}
        assert by_model["constant"] == by_model["momentum"] == by_model["linear"]

    first_fold = result.split_result.folds[0]
    sample = dataset.iloc[first_fold.train_positions]
    sample = sample.loc[sample["is_target_return_sample"]]
    fitted = result.fitted_models["validation_2020"]
    expected_medians = sample.loc[:, list(FEATURE_COLUMNS)].median().to_numpy(dtype=np.float64)
    np.testing.assert_allclose(fitted.linear.preprocessor.medians, expected_medians)
    assert fitted.momentum.training_median == pytest.approx(float(sample["R_20"].median()))
    assert fitted.constant.value == pytest.approx(float(sample["target_return"].mean()))
    assert result.report["isolation"]["targets_or_features_inspected"] is False


def test_validation_outliers_do_not_change_that_folds_fitted_preprocessing() -> None:
    dataset = make_dataset()
    baseline = build_walk_forward_models(dataset, make_calendar())
    changed = dataset.copy()
    changed.loc[changed["signal_date"].dt.year.eq(2020), list(FEATURE_COLUMNS)] = 1.0e9
    changed_result = build_walk_forward_models(changed, make_calendar())

    before = baseline.fitted_models["validation_2020"].linear.preprocessor
    after = changed_result.fitted_models["validation_2020"].linear.preprocessor
    np.testing.assert_array_equal(before.medians, after.medians)
    np.testing.assert_array_equal(before.means, after.means)
    np.testing.assert_array_equal(before.scales, after.scales)
    before_predictions = baseline.predictions.query(
        "fold_id == 'validation_2020' and model == 'linear'"
    )["prediction"]
    after_predictions = changed_result.predictions.query(
        "fold_id == 'validation_2020' and model == 'linear'"
    )["prediction"]
    assert not np.allclose(before_predictions, after_predictions)


def test_isolated_values_cannot_change_development_models_predictions_or_metrics() -> None:
    dataset = make_dataset()
    baseline = build_walk_forward_models(dataset, make_calendar())
    changed = dataset.copy()
    future = changed["signal_date"].dt.year.ge(2025)
    changed.loc[future, list(FEATURE_COLUMNS)] = 1.0e12
    changed.loc[future, ["target_return", "target_excess_return"]] = -1.0e12
    changed_result = build_walk_forward_models(changed, make_calendar())

    assert_frame_equal(baseline.predictions, changed_result.predictions)
    assert_frame_equal(baseline.fold_metrics, changed_result.fold_metrics)
    assert_frame_equal(baseline.daily_metrics, changed_result.daily_metrics)
    assert baseline.report == changed_result.report
    for fold_id in baseline.fitted_models:
        assert (
            baseline.fitted_models[fold_id].to_dict()
            == changed_result.fitted_models[fold_id].to_dict()
        )


def test_target_coverage_keeps_missing_outcomes_separate_from_predictions() -> None:
    result = build_walk_forward_models(make_dataset(), make_calendar())
    metrics = result.fold_metrics.query("validation_year == 2020")

    assert metrics["validation_eligible_rows"].nunique() == 1
    assert metrics["target_observed_rows"].nunique() == 1
    assert metrics["prediction_rows"].nunique() == 1
    assert (metrics["prediction_coverage"] == 1.0).all()
    assert (metrics["target_coverage"] < 1.0).all()
    missing = result.predictions.query(
        "validation_year == 2020 and code == '30000' and not target_observed"
    )
    assert set(missing["model"]) == set(MODEL_NAMES)
    assert np.isfinite(missing["prediction"]).all()


def test_excess_return_target_is_configurable_without_changing_target_definition() -> None:
    config = ModelingConfig(target_column="target_excess_return", top_n=2)
    result = build_walk_forward_models(make_dataset(), make_calendar(), config=config)

    assert set(result.predictions["target_column"]) == {"target_excess_return"}
    assert set(result.fold_metrics["target_column"]) == {"target_excess_return"}
    first_fold = result.split_result.folds[0]
    train = make_dataset().iloc[first_fold.train_positions]
    train = train.loc[train["is_target_excess_return_sample"]]
    assert result.fitted_models["validation_2020"].constant.value == pytest.approx(
        float(train["target_excess_return"].mean())
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda frame: frame.assign(
                is_target_return_sample=~frame["is_target_return_sample"]
            ),
            "disagrees with eligibility",
        ),
        (
            lambda frame: frame.assign(
                R_20=np.where(frame.index == 0, np.inf, frame["R_20"])
            ),
            "contains non-finite",
        ),
        (
            lambda frame: frame.drop(columns=["R_20"]),
            "missing required modeling columns",
        ),
    ],
)
def test_schema_errors_are_early_and_explicit(mutate, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        build_walk_forward_models(mutate(make_dataset()), make_calendar())


def test_all_missing_training_feature_is_not_silently_imputed() -> None:
    dataset = make_dataset()
    dataset.loc[dataset["signal_date"].dt.year.le(2019), "R_60"] = np.nan
    with pytest.raises(SchemaError, match="all-missing features.*R_60"):
        build_walk_forward_models(dataset, make_calendar())


def test_strict_config_loader(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "target_column": "target_excess_return",
                "feature_columns": list(FEATURE_COLUMNS),
                "momentum_feature": "R_20",
                "top_n": 3,
            }
        ),
        encoding="utf-8",
    )
    assert load_config(valid) == ModelingConfig(
        target_column="target_excess_return", top_n=3
    )
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"unexpected": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(unknown)


def _write_annual_inputs(root: Path, dataset: pd.DataFrame) -> Path:
    input_dir = root / "weekly"
    input_dir.mkdir()
    for year, group in dataset.groupby(dataset["signal_date"].dt.year):
        group.to_parquet(input_dir / f"{year}.parquet", index=False)
    # These deliberately invalid final-period files prove that the modeling
    # CLI selects 2016-2024 by filename before inspecting schemas or values.
    pd.DataFrame({"do_not_open": [2025]}).to_parquet(input_dir / "2025.parquet", index=False)
    pd.DataFrame({"do_not_open": [2026]}).to_parquet(input_dir / "2026.parquet", index=False)
    return input_dir


def test_cli_reads_only_development_partitions_and_matches_dataframe_api(tmp_path: Path) -> None:
    dataset = make_dataset().loc[lambda value: value["signal_date"].dt.year.le(2024)].copy()
    input_dir = _write_annual_inputs(tmp_path, dataset)
    calendar_path = tmp_path / "calendar.parquet"
    make_calendar().to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"

    manifest = run_modeling(
        input_dir=input_dir,
        calendar_path=calendar_path,
        output_dir=output_dir,
        report_dir=report_dir,
        config=ModelingConfig(top_n=2),
    )
    expected = build_walk_forward_models(
        dataset.reset_index(drop=True), make_calendar(), config=ModelingConfig(top_n=2)
    )

    assert manifest["status"] == "complete"
    assert manifest["final_period_partitions_loaded"] is False
    assert {Path(value["path"]).stem for value in manifest["development_inputs"]} == {
        str(year) for year in range(2016, 2025)
    }
    assert_frame_equal(pd.read_parquet(output_dir / "predictions.parquet"), expected.predictions)
    assert_frame_equal(pd.read_parquet(output_dir / "fold_metrics.parquet"), expected.fold_metrics)
    assert_frame_equal(pd.read_parquet(output_dir / "daily_metrics.parquet"), expected.daily_metrics)
    parameters = json.loads((output_dir / "model_parameters.json").read_text(encoding="utf-8"))
    assert set(parameters["folds"]) == {f"validation_{year}" for year in range(2020, 2025)}
    saved_manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["status"] == "complete"
    assert len(saved_manifest["completed_outputs"]) == 4


def test_main_and_existing_output_refusal(tmp_path: Path, capsys) -> None:
    dataset = make_dataset().loc[lambda value: value["signal_date"].dt.year.le(2024)].copy()
    input_dir = _write_annual_inputs(tmp_path, dataset)
    calendar_path = tmp_path / "calendar.parquet"
    make_calendar().to_parquet(calendar_path, index=False)
    config_path = tmp_path / "modeling.json"
    config_path.write_text(
        json.dumps({**ModelingConfig().__dict__, "feature_columns": list(FEATURE_COLUMNS)}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
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
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    with pytest.raises(FileExistsError, match="Output path already exists"):
        run_modeling(
            input_dir=input_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=tmp_path / "another_report",
        )


def test_interruption_manifest_preserves_progress(tmp_path: Path, monkeypatch) -> None:
    dataset = make_dataset().loc[lambda value: value["signal_date"].dt.year.le(2024)].copy()
    input_dir = _write_annual_inputs(tmp_path, dataset)
    calendar_path = tmp_path / "calendar.parquet"
    make_calendar().to_parquet(calendar_path, index=False)
    output_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("stock_prediction.modeling.build_walk_forward_models", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_modeling(
            input_dir=input_dir,
            calendar_path=calendar_path,
            output_dir=output_dir,
            report_dir=report_dir,
        )
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["completed_outputs"] == []
    assert manifest["final_period_partitions_loaded"] is False
