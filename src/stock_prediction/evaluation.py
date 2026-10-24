"""Evaluate cross-sectional return predictions without changing model selection.

The module consumes validation predictions that already identify the
signal-time eligible population.  It computes cross-sectional metrics per
signal date first, then aggregates those observations across time.  Industry
labels are joined from the point-in-time listed-issue master on the exact
``(signal_date, code)`` key; no current classification or forward fill is
used.
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
import platform
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
import pyarrow as pa
import pyarrow.parquet as pq

from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "cross-sectional-prediction-evaluation-v1"
PREDICTION_REQUIRED_COLUMNS = (
    "fold_id",
    "validation_year",
    "signal_date",
    "code",
    "model",
    "target_column",
    "actual_target",
    "target_observed",
    "prediction",
)
PREDICTION_KEY_COLUMNS = (
    "fold_id",
    "signal_date",
    "code",
    "model",
    "target_column",
)
DAILY_GROUP_COLUMNS = (
    "fold_id",
    "validation_year",
    "signal_date",
    "model",
    "target_column",
)
INDUSTRY_CANONICAL_COLUMNS = (
    "signal_date",
    "code",
    "industry_code",
    "industry_name",
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed evaluation settings chosen before inspecting a final period.

    The default industry fields are the J-Quants 33-sector classification.
    Set ``industry_code_column`` and ``industry_name_column`` to ``S17`` and
    ``S17Nm`` to run the same evaluation at the 17-sector level.
    """

    top_n: int = 10
    industry_date_column: str = "Date"
    industry_security_column: str = "Code"
    industry_code_column: str = "S33"
    industry_name_column: str = "S33Nm"

    def __post_init__(self) -> None:
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        names = (
            self.industry_date_column,
            self.industry_security_column,
            self.industry_code_column,
            self.industry_name_column,
        )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("industry source column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("industry source column names must be distinct")


@dataclass
class EvaluationResult:
    """Notebook-friendly evaluated rows, metrics, and audit report."""

    predictions_with_industry: pd.DataFrame
    daily_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    yearly_metrics: pd.DataFrame
    industry_daily_metrics: pd.DataFrame
    industry_metrics: pd.DataFrame
    industry_yearly_metrics: pd.DataFrame
    report: dict[str, Any]


def load_config(path: Path | str) -> EvaluationConfig:
    """Read a strict JSON evaluation configuration."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(EvaluationConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return EvaluationConfig(**values)


def _require_string_values(series: pd.Series, name: str) -> None:
    if not all(isinstance(value, str) for value in series.dropna()):
        raise SchemaError(f"{name} must contain strings, not coerced numeric values")


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        parsed = series.astype("datetime64[ns]")
    else:
        _require_string_values(series, name)
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        parsed = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or parsed.ne(parsed.dt.normalize()).any():
        raise SchemaError(f"{name} must contain valid non-null midnight dates")
    return parsed.astype("datetime64[ns]")


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not predictions.columns.is_unique:
        raise SchemaError("predictions contains duplicate column names")
    missing = set(PREDICTION_REQUIRED_COLUMNS) - set(predictions.columns)
    if missing:
        raise SchemaError(f"predictions is missing required columns: {sorted(missing)}")
    output = predictions.loc[:, list(PREDICTION_REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    if output.empty:
        raise SchemaError("predictions must contain at least one eligible row")
    output["signal_date"] = _parse_dates(output["signal_date"], "predictions.signal_date")

    for column in ("fold_id", "code", "model", "target_column"):
        _require_string_values(output[column], f"predictions.{column}")
        output[column] = output[column].astype("string")
        if output[column].isna().any() or output[column].str.len().eq(0).any():
            raise SchemaError(f"predictions.{column} must contain non-empty strings")
    if not output["code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError("predictions.code must contain five-character uppercase security codes")

    if not is_numeric_dtype(output["validation_year"].dtype) or is_bool_dtype(
        output["validation_year"].dtype
    ):
        raise SchemaError("predictions.validation_year must be an integer year")
    years = pd.to_numeric(output["validation_year"], errors="coerce")
    if years.isna().any() or not years.mod(1).eq(0).all():
        raise SchemaError("predictions.validation_year must be an integer year")
    output["validation_year"] = years.astype("int64")
    if not output["validation_year"].eq(output["signal_date"].dt.year).all():
        raise SchemaError("predictions.validation_year must equal signal_date year")

    if not is_bool_dtype(output["target_observed"].dtype) or output["target_observed"].isna().any():
        raise SchemaError("predictions.target_observed must be non-null boolean")
    output["target_observed"] = output["target_observed"].astype(bool)
    for column in ("actual_target", "prediction"):
        if not is_numeric_dtype(output[column].dtype) or is_bool_dtype(output[column].dtype):
            raise SchemaError(f"predictions.{column} must be numeric with null allowed")
        output[column] = output[column].astype("float64")
        invalid = output[column].notna() & ~np.isfinite(output[column])
        if invalid.any():
            raise SchemaError(f"predictions.{column} contains non-finite non-null values")
    if not output["target_observed"].eq(output["actual_target"].notna()).all():
        count = int(output["target_observed"].ne(output["actual_target"].notna()).sum())
        raise SchemaError(
            f"predictions.target_observed disagrees with actual_target availability on {count} rows"
        )
    if output.duplicated(list(PREDICTION_KEY_COLUMNS), keep=False).any():
        count = int(output.duplicated(list(PREDICTION_KEY_COLUMNS), keep=False).sum())
        raise SchemaError(f"predictions contains {count} duplicate prediction-key rows")

    expected_models = {
        target: frozenset(group["model"].astype(str).unique())
        for target, group in output.groupby("target_column", sort=False, observed=True)
    }
    comparison_columns = ("fold_id", "validation_year", "signal_date", "target_column")
    for key, group in output.groupby(list(comparison_columns), sort=False, observed=True):
        target = str(key[-1])
        present_models = frozenset(group["model"].astype(str).unique())
        if present_models != expected_models[target]:
            raise SchemaError(f"model set differs across evaluation cross-sections at {key}")
        model_parts = [
            part.sort_values("code", kind="stable").reset_index(drop=True)
            for _, part in group.groupby("model", sort=False, observed=True)
        ]
        code_sets = [frozenset(part["code"].astype(str)) for part in model_parts]
        if any(codes != code_sets[0] for codes in code_sets[1:]):
            raise SchemaError(f"eligible code set differs across models at {key}")
        reference = model_parts[0]
        for part in model_parts[1:]:
            observed_equal = part["target_observed"].equals(reference["target_observed"])
            actual_equal = np.array_equal(
                part["actual_target"].to_numpy(dtype=np.float64),
                reference["actual_target"].to_numpy(dtype=np.float64),
                equal_nan=True,
            )
            if not observed_equal or not actual_equal:
                raise SchemaError(f"actual target differs across models at {key}")

    return output.sort_values(
        ["validation_year", "signal_date", "target_column", "model", "code"], kind="stable"
    ).reset_index(drop=True)


def _normalize_industries(
    listed_info: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    if not listed_info.columns.is_unique:
        raise SchemaError("listed_info contains duplicate column names")
    source_columns = (
        config.industry_date_column,
        config.industry_security_column,
        config.industry_code_column,
        config.industry_name_column,
    )
    missing = set(source_columns) - set(listed_info.columns)
    if missing:
        raise SchemaError(f"listed_info is missing industry columns: {sorted(missing)}")
    output = listed_info.loc[:, list(source_columns)].copy().reset_index(drop=True)
    output.columns = list(INDUSTRY_CANONICAL_COLUMNS)
    output["signal_date"] = _parse_dates(output["signal_date"], "listed_info industry date")
    for column in ("code", "industry_code", "industry_name"):
        _require_string_values(output[column], f"listed_info.{column}")
        output[column] = output[column].astype("string")
    if output["code"].isna().any() or not output["code"].str.fullmatch(
        r"[0-9A-Z]{5}", na=False
    ).all():
        raise SchemaError("listed_info security codes must be non-null five-character uppercase codes")
    partial_industry = output["industry_code"].isna() ^ output["industry_name"].isna()
    if partial_industry.any():
        raise SchemaError("listed_info industry code and name must be either both present or both null")
    present = output["industry_code"].notna()
    if (
        output.loc[present, "industry_code"].str.len().eq(0).any()
        or output.loc[present, "industry_name"].str.len().eq(0).any()
    ):
        raise SchemaError("listed_info industry code and name must not be empty strings")
    if output.duplicated(["signal_date", "code"], keep=False).any():
        count = int(output.duplicated(["signal_date", "code"], keep=False).sum())
        raise SchemaError(f"listed_info contains {count} duplicate signal_date/code rows")
    return output.sort_values(["signal_date", "code"], kind="stable").reset_index(drop=True)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _cross_section_record(
    group: pd.DataFrame,
    *,
    top_n: int,
    include_industry_coverage: bool,
) -> dict[str, Any]:
    predicted = group["prediction"].notna().to_numpy(dtype=bool)
    observed = group["target_observed"].to_numpy(dtype=bool)
    scored = group.loc[predicted & observed].copy()
    eligible_rows = int(len(group))
    if len(scored):
        actual = scored["actual_target"].to_numpy(dtype=np.float64)
        estimate = scored["prediction"].to_numpy(dtype=np.float64)
        pearson_ic = _safe_correlation(estimate, actual)
        estimate_rank = pd.Series(estimate).rank(method="average").to_numpy(dtype=np.float64)
        actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
        spearman_ic = _safe_correlation(estimate_rank, actual_rank)
    else:
        pearson_ic = spearman_ic = float("nan")

    # Selection uses predictions alone.  Missing outcomes never cause a lower
    # ranked security to be pulled into the selected group.  A selected-group
    # return is only reported when every selected target is observed.
    ordered = group.loc[predicted].sort_values(
        ["prediction", "code"], ascending=[False, True], kind="stable"
    )
    if len(ordered):
        selection_count = min(top_n, len(ordered))
        top = ordered.head(selection_count)
        bottom = ordered.tail(selection_count)
        top_observed_rows = int(top["target_observed"].sum())
        bottom_observed_rows = int(bottom["target_observed"].sum())
        top_n_mean = (
            float(top["actual_target"].mean())
            if top_observed_rows == selection_count
            else float("nan")
        )
        bottom_n_mean = (
            float(bottom["actual_target"].mean())
            if bottom_observed_rows == selection_count
            else float("nan")
        )
        spread = (
            top_n_mean - bottom_n_mean
            if np.isfinite(top_n_mean) and np.isfinite(bottom_n_mean)
            else float("nan")
        )
    else:
        selection_count = top_observed_rows = bottom_observed_rows = 0
        top_n_mean = bottom_n_mean = spread = float("nan")
    result: dict[str, Any] = {
        "eligible_rows": eligible_rows,
        "target_observed_rows": int(observed.sum()),
        "prediction_rows": int(predicted.sum()),
        "scored_rows": int((predicted & observed).sum()),
        "target_coverage": float(observed.mean()) if eligible_rows else float("nan"),
        "prediction_coverage": float(predicted.mean()) if eligible_rows else float("nan"),
        "scoring_coverage": float((predicted & observed).mean()) if eligible_rows else float("nan"),
        "pearson_ic": pearson_ic,
        "spearman_ic": spearman_ic,
        "top_n_requested": top_n,
        "selection_count": selection_count,
        "top_n_complete": selection_count == top_n,
        "top_n_target_observed_rows": top_observed_rows,
        "bottom_n_target_observed_rows": bottom_observed_rows,
        "top_n_target_coverage": (
            top_observed_rows / selection_count if selection_count else float("nan")
        ),
        "bottom_n_target_coverage": (
            bottom_observed_rows / selection_count if selection_count else float("nan")
        ),
        "top_n_mean_target": top_n_mean,
        "bottom_n_mean_target": bottom_n_mean,
        "top_bottom_spread": spread,
    }
    if include_industry_coverage:
        industry_observed = group["industry_observed"].to_numpy(dtype=bool)
        result.update(
            {
                "industry_observed_rows": int(industry_observed.sum()),
                "industry_coverage": (
                    float(industry_observed.mean()) if eligible_rows else float("nan")
                ),
            }
        )
    return result


def _daily_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    top_n: int,
    include_industry_coverage: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=True, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                **_cross_section_record(
                    group,
                    top_n=top_n,
                    include_industry_coverage=include_industry_coverage,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _mean(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _median(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.median()) if len(clean) else float("nan")


def _sample_standard_deviation(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.std(ddof=1)) if len(clean) >= 2 else float("nan")


def _aggregate_daily_metrics(
    daily: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    include_industry_coverage: bool,
) -> pd.DataFrame:
    value_columns = [
        "folds",
        "signal_dates",
        "signal_date_min",
        "signal_date_max",
        "eligible_rows",
        "target_observed_rows",
        "prediction_rows",
        "scored_rows",
        "target_coverage",
        "prediction_coverage",
        "scoring_coverage",
        "spearman_ic_mean",
        "spearman_ic_median",
        "spearman_ic_standard_deviation",
        "spearman_ic_valid_dates",
        "pearson_ic_mean",
        "pearson_ic_median",
        "pearson_ic_standard_deviation",
        "pearson_ic_valid_dates",
        "top_n_requested",
        "top_n_complete_dates",
        "top_n_fully_observed_dates",
        "bottom_n_fully_observed_dates",
        "top_n_valid_dates",
        "selection_count_min",
        "selection_count_max",
        "selected_rows",
        "top_n_target_observed_rows",
        "bottom_n_target_observed_rows",
        "top_n_target_coverage",
        "bottom_n_target_coverage",
        "top_n_mean_target_mean",
        "top_n_mean_target_standard_deviation",
        "bottom_n_mean_target_mean",
        "top_bottom_spread_mean",
        "top_bottom_spread_standard_deviation",
    ]
    if include_industry_coverage:
        value_columns.extend(("industry_observed_rows", "industry_coverage"))
    if daily.empty:
        return pd.DataFrame(columns=[*group_columns, *value_columns])
    records: list[dict[str, Any]] = []
    grouped: Any
    if group_columns:
        grouped = daily.groupby(list(group_columns), sort=True, observed=True)
    else:
        grouped = [((), daily)]
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        eligible_rows = int(group["eligible_rows"].sum())
        target_rows = int(group["target_observed_rows"].sum())
        prediction_rows = int(group["prediction_rows"].sum())
        scored_rows = int(group["scored_rows"].sum())
        record: dict[str, Any] = {
            **dict(zip(group_columns, keys, strict=True)),
            "folds": int(group["fold_id"].nunique()),
            "signal_dates": int(group["signal_date"].nunique()),
            "signal_date_min": group["signal_date"].min(),
            "signal_date_max": group["signal_date"].max(),
            "eligible_rows": eligible_rows,
            "target_observed_rows": target_rows,
            "prediction_rows": prediction_rows,
            "scored_rows": scored_rows,
            "target_coverage": target_rows / eligible_rows if eligible_rows else float("nan"),
            "prediction_coverage": (
                prediction_rows / eligible_rows if eligible_rows else float("nan")
            ),
            "scoring_coverage": scored_rows / eligible_rows if eligible_rows else float("nan"),
            "spearman_ic_mean": _mean(group["spearman_ic"]),
            "spearman_ic_median": _median(group["spearman_ic"]),
            "spearman_ic_standard_deviation": _sample_standard_deviation(
                group["spearman_ic"]
            ),
            "spearman_ic_valid_dates": int(group["spearman_ic"].notna().sum()),
            "pearson_ic_mean": _mean(group["pearson_ic"]),
            "pearson_ic_median": _median(group["pearson_ic"]),
            "pearson_ic_standard_deviation": _sample_standard_deviation(group["pearson_ic"]),
            "pearson_ic_valid_dates": int(group["pearson_ic"].notna().sum()),
            "top_n_requested": int(group["top_n_requested"].iloc[0]),
            "top_n_complete_dates": int(group["top_n_complete"].sum()),
            "top_n_fully_observed_dates": int(group["top_n_target_coverage"].eq(1.0).sum()),
            "bottom_n_fully_observed_dates": int(
                group["bottom_n_target_coverage"].eq(1.0).sum()
            ),
            "top_n_valid_dates": int(group["top_n_mean_target"].notna().sum()),
            "selection_count_min": int(group["selection_count"].min()),
            "selection_count_max": int(group["selection_count"].max()),
            "selected_rows": int(group["selection_count"].sum()),
            "top_n_target_observed_rows": int(group["top_n_target_observed_rows"].sum()),
            "bottom_n_target_observed_rows": int(
                group["bottom_n_target_observed_rows"].sum()
            ),
            "top_n_target_coverage": (
                float(group["top_n_target_observed_rows"].sum())
                / float(group["selection_count"].sum())
                if group["selection_count"].sum()
                else float("nan")
            ),
            "bottom_n_target_coverage": (
                float(group["bottom_n_target_observed_rows"].sum())
                / float(group["selection_count"].sum())
                if group["selection_count"].sum()
                else float("nan")
            ),
            "top_n_mean_target_mean": _mean(group["top_n_mean_target"]),
            "top_n_mean_target_standard_deviation": _sample_standard_deviation(
                group["top_n_mean_target"]
            ),
            "bottom_n_mean_target_mean": _mean(group["bottom_n_mean_target"]),
            "top_bottom_spread_mean": _mean(group["top_bottom_spread"]),
            "top_bottom_spread_standard_deviation": _sample_standard_deviation(
                group["top_bottom_spread"]
            ),
        }
        if include_industry_coverage:
            industry_rows = int(group["industry_observed_rows"].sum())
            record.update(
                {
                    "industry_observed_rows": industry_rows,
                    "industry_coverage": (
                        industry_rows / eligible_rows if eligible_rows else float("nan")
                    ),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def evaluate_predictions(
    predictions: pd.DataFrame,
    listed_info: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Evaluate predictions and exact-date point-in-time industry labels.

    Parameters are DataFrames so the function can be used directly in a
    notebook.  Inputs are copied and never modified.  ``predictions`` must
    contain one row per eligible security, model, and signal date.  The
    ``listed_info`` source columns are configured by :class:`EvaluationConfig`.
    """

    config = config or EvaluationConfig()
    normalized = _normalize_predictions(predictions)
    return _evaluate_normalized_predictions(normalized, listed_info, config=config)


def _evaluate_normalized_predictions(
    normalized: pd.DataFrame,
    listed_info: pd.DataFrame,
    *,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate an already validated prediction frame for the saved runner."""

    industries = _normalize_industries(listed_info, config)
    relevant_dates = normalized["signal_date"].unique()
    industries = industries.loc[industries["signal_date"].isin(relevant_dates)].copy()
    enriched = normalized.merge(
        industries,
        on=["signal_date", "code"],
        how="left",
        validate="many_to_one",
    )
    enriched["industry_observed"] = (
        enriched["industry_code"].notna() & enriched["industry_name"].notna()
    )
    enriched = enriched.sort_values(
        ["validation_year", "signal_date", "target_column", "model", "code"], kind="stable"
    ).reset_index(drop=True)

    daily = _daily_metrics(
        enriched,
        group_columns=DAILY_GROUP_COLUMNS,
        top_n=config.top_n,
        include_industry_coverage=True,
    )
    summary = _aggregate_daily_metrics(
        daily,
        group_columns=("model", "target_column"),
        include_industry_coverage=True,
    )
    yearly = _aggregate_daily_metrics(
        daily,
        group_columns=("validation_year", "model", "target_column"),
        include_industry_coverage=True,
    )

    classified = enriched.loc[enriched["industry_observed"]].copy()
    industry_daily_group_columns = (
        *DAILY_GROUP_COLUMNS,
        "industry_code",
        "industry_name",
    )
    if classified.empty:
        industry_daily = pd.DataFrame(
            columns=[
                *industry_daily_group_columns,
                "eligible_rows",
                "target_observed_rows",
                "prediction_rows",
                "scored_rows",
                "target_coverage",
                "prediction_coverage",
                "scoring_coverage",
                "pearson_ic",
                "spearman_ic",
                "top_n_requested",
                "selection_count",
                "top_n_complete",
                "top_n_target_observed_rows",
                "bottom_n_target_observed_rows",
                "top_n_target_coverage",
                "bottom_n_target_coverage",
                "top_n_mean_target",
                "bottom_n_mean_target",
                "top_bottom_spread",
            ]
        )
        industry = _aggregate_daily_metrics(
            industry_daily,
            group_columns=("model", "target_column", "industry_code", "industry_name"),
            include_industry_coverage=False,
        )
        industry_yearly = _aggregate_daily_metrics(
            industry_daily,
            group_columns=(
                "validation_year",
                "model",
                "target_column",
                "industry_code",
                "industry_name",
            ),
            include_industry_coverage=False,
        )
    else:
        industry_daily = _daily_metrics(
            classified,
            group_columns=industry_daily_group_columns,
            top_n=config.top_n,
            include_industry_coverage=False,
        )
        industry = _aggregate_daily_metrics(
            industry_daily,
            group_columns=("model", "target_column", "industry_code", "industry_name"),
            include_industry_coverage=False,
        )
        industry_yearly = _aggregate_daily_metrics(
            industry_daily,
            group_columns=(
                "validation_year",
                "model",
                "target_column",
                "industry_code",
                "industry_name",
            ),
            include_industry_coverage=False,
        )

    unique_candidates = enriched.drop_duplicates(
        ["fold_id", "signal_date", "code", "target_column"]
    )
    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "input": {
            "prediction_rows": int(len(normalized)),
            "models": sorted(normalized["model"].astype(str).unique().tolist()),
            "targets": sorted(normalized["target_column"].astype(str).unique().tolist()),
            "folds": int(normalized["fold_id"].nunique()),
            "signal_dates": int(normalized["signal_date"].nunique()),
            "date_min": normalized["signal_date"].min().strftime("%Y-%m-%d"),
            "date_max": normalized["signal_date"].max().strftime("%Y-%m-%d"),
            "securities": int(normalized["code"].nunique()),
            "target_missing_rows": int((~normalized["target_observed"]).sum()),
            "prediction_missing_rows": int(normalized["prediction"].isna().sum()),
            "duplicate_prediction_keys": 0,
        },
        "industry_join": {
            "classification": {
                "code_column": config.industry_code_column,
                "name_column": config.industry_name_column,
            },
            "join_keys": ["signal_date", "code"],
            "forward_filled": False,
            "candidate_rows": int(len(unique_candidates)),
            "classified_candidate_rows": int(unique_candidates["industry_observed"].sum()),
            "missing_candidate_rows": int((~unique_candidates["industry_observed"]).sum()),
            "candidate_coverage": float(unique_candidates["industry_observed"].mean()),
        },
        "outputs": {
            "predictions_with_industry_rows": int(len(enriched)),
            "daily_metric_rows": int(len(daily)),
            "summary_metric_rows": int(len(summary)),
            "yearly_metric_rows": int(len(yearly)),
            "industry_daily_metric_rows": int(len(industry_daily)),
            "industry_metric_rows": int(len(industry)),
            "industry_yearly_metric_rows": int(len(industry_yearly)),
        },
        "limitations": [
            "Metrics are predictive evaluations of close-to-close targets, not executable backtest returns.",
            "Industry metrics exclude unclassified exact-date candidates and report that loss separately.",
            "Top/bottom selection is fixed from predictions; missing targets are never replaced by lower-ranked rows.",
            "Selected-group returns are null unless every selected target is observed; selected coverage is reported.",
            "Top and bottom groups may overlap when a cross-section has fewer than twice top_n predicted rows.",
            "No confidence interval treating security rows as independent observations is produced.",
        ],
        "random_seed": None,
    }
    return EvaluationResult(
        predictions_with_industry=enriched,
        daily_metrics=daily,
        summary_metrics=summary,
        yearly_metrics=yearly,
        industry_daily_metrics=industry_daily,
        industry_metrics=industry,
        industry_yearly_metrics=industry_yearly,
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
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(PREDICTION_REQUIRED_COLUMNS))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype={
                "fold_id": "string",
                "code": "string",
                "model": "string",
                "target_column": "string",
            },
        )
    raise SchemaError("predictions must be a .parquet or .csv file")


def _industry_partitions(industry_dir: Path, years: Sequence[int]) -> list[Path]:
    if not industry_dir.is_dir():
        raise FileNotFoundError(f"Industry directory does not exist: {industry_dir}")
    paths = {int(path.stem): path for path in industry_dir.glob("*.parquet") if re.fullmatch(r"\d{4}", path.stem)}
    missing = set(years) - set(paths)
    if missing:
        raise FileNotFoundError(f"Industry partitions are missing for prediction years: {sorted(missing)}")
    return [paths[year] for year in sorted(set(years))]


def _inspect_industry_partition(path: Path, config: EvaluationConfig) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    required = (
        config.industry_date_column,
        config.industry_security_column,
        config.industry_code_column,
        config.industry_name_column,
    )
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise SchemaError(f"{path.name} is missing industry columns: {sorted(missing)}")
    return {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "columns_read": list(required),
        "sha256": _sha256(path),
    }


def _read_relevant_industries(
    paths: Sequence[Path],
    prediction_dates: pd.Series,
    config: EvaluationConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = (
        config.industry_date_column,
        config.industry_security_column,
        config.industry_code_column,
        config.industry_name_column,
    )
    date_index = pd.DatetimeIndex(prediction_dates).normalize().sort_values()
    for path in paths:
        dates_for_year = date_index[date_index.year == int(path.stem)]
        date_field = pq.read_schema(path).field(config.industry_date_column)
        if pa.types.is_timestamp(date_field.type):
            filter_values: list[Any] = dates_for_year.to_pydatetime().tolist()
        elif pa.types.is_date(date_field.type):
            filter_values = [value.date() for value in dates_for_year]
        else:
            raise SchemaError(
                f"{path.name}.{config.industry_date_column} must be a Parquet date or timestamp"
            )
        frame = pq.read_table(
            path,
            columns=list(columns),
            filters=[(config.industry_date_column, "in", filter_values)],
        ).to_pandas()
        parsed = _parse_dates(frame[config.industry_date_column], f"{path.name} industry date")
        if not parsed.dt.year.eq(int(path.stem)).all():
            raise SchemaError(f"{path.name} contains industry dates from another year")
        frame[config.industry_date_column] = parsed
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_new_outputs(
    predictions_path: Path,
    industry_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> None:
    if output_dir.resolve() == report_dir.resolve():
        raise ValueError("output_dir and report_dir must differ")
    for target in (output_dir, report_dir):
        if target.exists():
            raise FileExistsError(f"Output path already exists: {target}")
        if _is_relative_to(target, industry_dir):
            raise ValueError("Evaluation outputs must not be inside the industry input directory")
        if _is_relative_to(target, predictions_path.parent):
            raise ValueError("Evaluation outputs must not be inside the predictions input directory")
        if target.resolve() == predictions_path.resolve():
            raise ValueError("Evaluation output must not overwrite predictions")


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def run_evaluation(
    *,
    predictions_path: Path | str,
    industry_dir: Path | str,
    output_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Read saved predictions and point-in-time industries, then persist metrics."""

    config = config or EvaluationConfig()
    predictions_path = Path(predictions_path)
    industry_dir = Path(industry_dir)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "artifacts" / f"evaluation_{run_id}"
    )
    report_dir = (
        Path(report_dir)
        if report_dir is not None
        else PROJECT_ROOT / "reports" / f"evaluation_{run_id}"
    )
    _ensure_new_outputs(predictions_path, industry_dir, output_dir, report_dir)

    predictions = _normalize_predictions(_read_predictions(predictions_path))
    years = sorted(predictions["validation_year"].unique().tolist())
    industry_paths = _industry_partitions(industry_dir, years)
    industry_inputs = [_inspect_industry_partition(path, config) for path in industry_paths]
    prediction_input = {
        "path": str(predictions_path),
        "rows": int(len(predictions)),
        "columns": list(PREDICTION_REQUIRED_COLUMNS),
        "sha256": _sha256(predictions_path),
    }

    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest_path = report_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "prediction_input": prediction_input,
        "industry_dir": str(industry_dir),
        "industry_inputs": industry_inputs,
        "industry_years_loaded": years,
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "completed_outputs": [],
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }
    _write_json(manifest_path, _json_safe(manifest))
    try:
        listed_info = _read_relevant_industries(
            industry_paths,
            predictions["signal_date"].drop_duplicates(),
            config,
        )
        result = _evaluate_normalized_predictions(predictions, listed_info, config=config)
        manifest.update(result.report)
        outputs: tuple[tuple[str, pd.DataFrame, Path], ...] = (
            (
                "predictions_with_industry",
                result.predictions_with_industry,
                output_dir / "predictions_with_industry.parquet",
            ),
            ("daily_metrics", result.daily_metrics, output_dir / "daily_metrics.parquet"),
            ("summary_metrics", result.summary_metrics, output_dir / "summary_metrics.parquet"),
            ("yearly_metrics", result.yearly_metrics, output_dir / "yearly_metrics.parquet"),
            (
                "industry_daily_metrics",
                result.industry_daily_metrics,
                output_dir / "industry_daily_metrics.parquet",
            ),
            ("industry_metrics", result.industry_metrics, output_dir / "industry_metrics.parquet"),
            (
                "industry_yearly_metrics",
                result.industry_yearly_metrics,
                output_dir / "industry_yearly_metrics.parquet",
            ),
        )
        for name, value, path in outputs:
            value.to_parquet(path, index=False)
            manifest["completed_outputs"].append(
                {"name": name, "path": str(path), "rows": int(len(value)), "sha256": _sha256(path)}
            )
            _write_json(manifest_path, _json_safe(manifest))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, _json_safe(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for saved prediction evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--industry-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Strict JSON evaluation configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config(args.config) if args.config else EvaluationConfig()
    manifest = run_evaluation(
        predictions_path=args.predictions,
        industry_dir=args.industry_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "prediction_rows": manifest["input"]["prediction_rows"],
                "daily_metric_rows": manifest["outputs"]["daily_metric_rows"],
                "yearly_metric_rows": manifest["outputs"]["yearly_metric_rows"],
                "industry_metric_rows": manifest["outputs"]["industry_metric_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
