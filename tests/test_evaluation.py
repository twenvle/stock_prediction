"""Cross-sectional metric, point-in-time industry, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_prediction.evaluation import (
    EvaluationConfig,
    evaluate_predictions,
    load_config,
    main,
    run_evaluation,
)
from stock_prediction.preprocessing import SchemaError


def make_predictions(*, two_models: bool = False) -> pd.DataFrame:
    dates = (pd.Timestamp("2020-01-06"), pd.Timestamp("2021-01-04"))
    codes = ("10000", "20000", "30000", "40000")
    records: list[dict[str, object]] = []
    models = ("linear", "momentum") if two_models else ("linear",)
    actual_by_date = ((0.4, 0.3, 0.2, 0.1), (0.1, 0.2, 0.3, 0.4))
    for signal_date, actual_values in zip(dates, actual_by_date, strict=True):
        for model in models:
            for code, actual, prediction in zip(
                codes, actual_values, (4.0, 3.0, 2.0, 1.0), strict=True
            ):
                records.append(
                    {
                        "fold_id": f"validation_{signal_date.year}",
                        "validation_year": signal_date.year,
                        "signal_date": signal_date,
                        "code": code,
                        "model": model,
                        "target_column": "target_return",
                        "actual_target": actual,
                        "target_observed": True,
                        "prediction": prediction,
                    }
                )
    return pd.DataFrame.from_records(records)


def make_listed_info() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for signal_date in (pd.Timestamp("2020-01-06"), pd.Timestamp("2021-01-04")):
        for code, industry_code, industry_name in (
            ("10000", "10", "業種A"),
            ("20000", "10", "業種A"),
            ("30000", "20", "業種B"),
            ("40000", "20", "業種B"),
        ):
            records.append(
                {
                    "Date": signal_date,
                    "Code": code,
                    "S33": industry_code,
                    "S33Nm": industry_name,
                }
            )
    return pd.DataFrame.from_records(records)


def test_known_ic_top_n_spread_and_time_weighted_aggregates() -> None:
    result = evaluate_predictions(
        make_predictions(),
        make_listed_info(),
        config=EvaluationConfig(top_n=1),
    )

    daily = result.daily_metrics.sort_values("signal_date").reset_index(drop=True)
    np.testing.assert_allclose(daily["spearman_ic"], [1.0, -1.0])
    np.testing.assert_allclose(daily["top_n_mean_target"], [0.4, 0.1])
    np.testing.assert_allclose(daily["bottom_n_mean_target"], [0.1, 0.4])
    np.testing.assert_allclose(daily["top_bottom_spread"], [0.3, -0.3])
    assert daily["top_n_complete"].all()

    summary = result.summary_metrics.iloc[0]
    assert summary["spearman_ic_mean"] == pytest.approx(0.0)
    assert summary["top_n_mean_target_mean"] == pytest.approx(0.25)
    assert summary["top_bottom_spread_mean"] == pytest.approx(0.0)
    assert summary["signal_dates"] == 2
    assert len(result.yearly_metrics) == 2
    assert set(result.industry_metrics["industry_name"]) == {"業種A", "業種B"}
    assert len(result.industry_yearly_metrics) == 4


def test_target_prediction_scoring_and_industry_coverage_stay_separate() -> None:
    predictions = make_predictions().loc[lambda frame: frame["validation_year"].eq(2020)].copy()
    predictions.loc[predictions["code"].eq("30000"), ["actual_target", "target_observed"]] = [
        np.nan,
        False,
    ]
    predictions.loc[predictions["code"].eq("40000"), "prediction"] = np.nan
    listed = make_listed_info().loc[
        lambda frame: frame["Date"].eq(pd.Timestamp("2020-01-06"))
        & ~frame["Code"].eq("40000")
    ]

    result = evaluate_predictions(
        predictions,
        listed,
        config=EvaluationConfig(top_n=3),
    )
    daily = result.daily_metrics.iloc[0]
    assert daily["eligible_rows"] == 4
    assert daily["target_coverage"] == pytest.approx(0.75)
    assert daily["prediction_coverage"] == pytest.approx(0.75)
    assert daily["scoring_coverage"] == pytest.approx(0.5)
    assert daily["industry_coverage"] == pytest.approx(0.75)
    assert daily["selection_count"] == 3
    assert bool(daily["top_n_complete"])
    assert daily["top_n_target_coverage"] == pytest.approx(2 / 3)
    assert pd.isna(daily["top_n_mean_target"])
    assert result.report["industry_join"]["missing_candidate_rows"] == 1


def test_equal_predictions_use_code_ascending_as_deterministic_tie_break() -> None:
    predictions = make_predictions().loc[lambda frame: frame["validation_year"].eq(2020)].copy()
    predictions["prediction"] = 1.0
    result = evaluate_predictions(
        predictions,
        make_listed_info(),
        config=EvaluationConfig(top_n=1),
    )
    daily = result.daily_metrics.iloc[0]
    assert pd.isna(daily["spearman_ic"])
    assert daily["top_n_mean_target"] == pytest.approx(0.4)
    assert daily["bottom_n_mean_target"] == pytest.approx(0.1)


def test_future_industry_rows_cannot_change_past_evaluation() -> None:
    predictions = make_predictions().loc[lambda frame: frame["validation_year"].eq(2020)].copy()
    listed = make_listed_info().copy()
    baseline = evaluate_predictions(predictions, listed, config=EvaluationConfig(top_n=1))
    changed = listed.copy()
    future = changed["Date"].eq(pd.Timestamp("2021-01-04"))
    changed.loc[future, ["S33", "S33Nm"]] = ["99", "未来分類"]
    changed_result = evaluate_predictions(predictions, changed, config=EvaluationConfig(top_n=1))

    assert_frame_equal(baseline.predictions_with_industry, changed_result.predictions_with_industry)
    assert_frame_equal(baseline.industry_metrics, changed_result.industry_metrics)
    assert baseline.report == changed_result.report


def test_missing_exact_date_industries_are_reported_without_forward_fill() -> None:
    predictions = make_predictions().loc[lambda frame: frame["validation_year"].eq(2020)].copy()
    future_only = make_listed_info().loc[lambda frame: frame["Date"].dt.year.eq(2021)].copy()
    result = evaluate_predictions(predictions, future_only)

    assert result.daily_metrics.iloc[0]["industry_coverage"] == pytest.approx(0.0)
    assert result.industry_daily_metrics.empty
    assert result.industry_metrics.empty
    assert "industry_code" in result.industry_metrics.columns
    assert result.report["industry_join"]["classified_candidate_rows"] == 0


@pytest.mark.parametrize(
    ("mutate_predictions", "mutate_industry", "match"),
    [
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            lambda frame: frame,
            "duplicate prediction-key",
        ),
        (
            lambda frame: frame.assign(
                target_observed=np.where(frame.index == 0, False, frame["target_observed"])
            ),
            lambda frame: frame,
            "disagrees with actual_target",
        ),
        (
            lambda frame: frame.assign(
                prediction=np.where(frame.index == 0, np.inf, frame["prediction"])
            ),
            lambda frame: frame,
            "non-finite",
        ),
        (
            lambda frame: frame,
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate signal_date/code",
        ),
    ],
)
def test_schema_errors_are_early_and_explicit(
    mutate_predictions,
    mutate_industry,
    match: str,
) -> None:
    with pytest.raises(SchemaError, match=match):
        evaluate_predictions(
            mutate_predictions(make_predictions()),
            mutate_industry(make_listed_info()),
        )


def test_model_candidate_sets_must_match_within_each_cross_section() -> None:
    predictions = make_predictions(two_models=True)
    predictions = predictions.loc[
        ~(
            predictions["validation_year"].eq(2020)
            & predictions["model"].eq("momentum")
            & predictions["code"].eq("40000")
        )
    ]
    with pytest.raises(SchemaError, match="eligible code set differs across models"):
        evaluate_predictions(predictions, make_listed_info())

    different_target = make_predictions(two_models=True)
    row = different_target["model"].eq("momentum") & different_target.index.to_series().eq(4)
    different_target.loc[row, "actual_target"] = 99.0
    with pytest.raises(SchemaError, match="actual target differs across models"):
        evaluate_predictions(different_target, make_listed_info())


def test_strict_config_loader_and_17_sector_override(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps(
            {
                "top_n": 5,
                "industry_date_column": "Date",
                "industry_security_column": "Code",
                "industry_code_column": "S17",
                "industry_name_column": "S17Nm",
            }
        ),
        encoding="utf-8",
    )
    assert load_config(path) == EvaluationConfig(
        top_n=5,
        industry_code_column="S17",
        industry_name_column="S17Nm",
    )
    path.write_text('{"unexpected": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        load_config(path)


def _write_cli_inputs(root: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    predictions = make_predictions().loc[lambda frame: frame["validation_year"].eq(2020)].copy()
    listed = make_listed_info().loc[lambda frame: frame["Date"].dt.year.eq(2020)].copy()
    input_dir = root / "inputs"
    industry_dir = input_dir / "listed_info"
    industry_dir.mkdir(parents=True)
    predictions_path = input_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    listed.to_parquet(industry_dir / "2020.parquet", index=False)
    # A non-required future partition with an invalid schema proves selection
    # by prediction year occurs before industry schema inspection.
    pd.DataFrame({"do_not_open": [2021]}).to_parquet(
        industry_dir / "2021.parquet", index=False
    )
    return predictions_path, industry_dir, predictions, listed


def test_cli_reads_only_prediction_years_and_matches_dataframe_api(tmp_path: Path) -> None:
    predictions_path, industry_dir, predictions, listed = _write_cli_inputs(tmp_path)
    output_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    config = EvaluationConfig(top_n=1)
    manifest = run_evaluation(
        predictions_path=predictions_path,
        industry_dir=industry_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        config=config,
    )
    expected = evaluate_predictions(predictions, listed, config=config)

    assert manifest["status"] == "complete"
    assert manifest["industry_years_loaded"] == [2020]
    assert len(manifest["completed_outputs"]) == 7
    assert_frame_equal(pd.read_parquet(output_dir / "daily_metrics.parquet"), expected.daily_metrics)
    assert_frame_equal(pd.read_parquet(output_dir / "yearly_metrics.parquet"), expected.yearly_metrics)
    assert_frame_equal(
        pd.read_parquet(output_dir / "industry_metrics.parquet"), expected.industry_metrics
    )
    saved = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete"


def test_main_existing_output_refusal_and_interruption_manifest(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    predictions_path, industry_dir, _, _ = _write_cli_inputs(tmp_path)
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(json.dumps(EvaluationConfig().__dict__), encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    assert (
        main(
            [
                "--predictions",
                str(predictions_path),
                "--industry-dir",
                str(industry_dir),
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
        run_evaluation(
            predictions_path=predictions_path,
            industry_dir=industry_dir,
            output_dir=output_dir,
            report_dir=tmp_path / "another_report",
        )

    interrupted_output = tmp_path / "interrupted_artifacts"
    interrupted_report = tmp_path / "interrupted_reports"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("stock_prediction.evaluation._evaluate_normalized_predictions", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_evaluation(
            predictions_path=predictions_path,
            industry_dir=industry_dir,
            output_dir=interrupted_output,
            report_dir=interrupted_report,
        )
    manifest = json.loads(
        (interrupted_report / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "interrupted"
    assert manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["completed_outputs"] == []
