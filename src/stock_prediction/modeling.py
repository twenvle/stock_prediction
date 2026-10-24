"""Fit leakage-aware baseline models on purged development folds.

The module compares a training-target mean, a direct momentum signal, and an
ordinary least-squares linear model.  Median imputation and standardization
statistics are fit independently inside each training fold.  The 2025-2026
isolation period is never used for fitting, transformation, or evaluation.
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

from stock_prediction.descriptors import FEATURE_COLUMNS
from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError
from stock_prediction.splits import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    SplitConfig,
    SplitResult,
    build_walk_forward_splits,
    load_config as load_split_config,
)


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "walk-forward-baseline-modeling-v1"
MODEL_NAMES = ("constant", "momentum", "linear")
TARGET_SAMPLE_COLUMNS = {
    "target_return": "is_target_return_sample",
    "target_excess_return": "is_target_excess_return_sample",
}
KEY_COLUMNS = ("signal_date", "code", "exit_date")


@dataclass(frozen=True)
class ModelingConfig:
    """Development-only baseline settings fixed before final evaluation.

    ``target_column`` selects one already-created weekly target; it does not
    transform or redefine the target.  ``feature_columns`` order is preserved
    in the fitted preprocessing statistics and linear coefficients.
    """

    target_column: str = "target_return"
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    momentum_feature: str = "R_20"
    top_n: int = 10

    def __post_init__(self) -> None:
        if self.target_column not in TARGET_SAMPLE_COLUMNS:
            raise ValueError(f"target_column must be one of {sorted(TARGET_SAMPLE_COLUMNS)}")
        feature_columns = tuple(self.feature_columns)
        object.__setattr__(self, "feature_columns", feature_columns)
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if any(not isinstance(name, str) or not name for name in feature_columns):
            raise ValueError("feature_columns must contain non-empty strings")
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if self.momentum_feature not in feature_columns:
            raise ValueError("momentum_feature must be included in feature_columns")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")


@dataclass(frozen=True)
class FittedPreprocessor:
    """Fold-local median imputation and population-standardization state."""

    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: Sequence[str]) -> FittedPreprocessor:
        """Fit statistics from one training fold and no other rows."""

        names = tuple(feature_names)
        values = frame.loc[:, list(names)].to_numpy(dtype=np.float64, copy=True)
        if not len(values):
            raise SchemaError("Cannot fit preprocessing on an empty training sample")
        all_missing = np.isnan(values).all(axis=0)
        if all_missing.any():
            missing_names = [name for name, missing in zip(names, all_missing, strict=True) if missing]
            raise SchemaError(f"Training fold has all-missing features: {missing_names}")
        medians = np.nanmedian(values, axis=0)
        imputed = np.where(np.isnan(values), medians, values)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0)
        scales = np.where(scales <= np.finfo(np.float64).eps, 1.0, scales)
        return cls(names, medians, means, scales)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the frozen training-fold statistics without refitting."""

        values = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64, copy=True)
        imputed = np.where(np.isnan(values), self.medians, values)
        transformed = (imputed - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise SchemaError("Fold preprocessing produced non-finite values")
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "standard_deviation_ddof": 0,
        }


@dataclass(frozen=True)
class ConstantRegressor:
    """Predict the training-fold target mean."""

    value: float

    @classmethod
    def fit(cls, target: np.ndarray) -> ConstantRegressor:
        if not len(target) or not np.isfinite(target).all():
            raise SchemaError("Constant model requires a non-empty finite training target")
        return cls(float(target.mean()))

    def predict(self, rows: int) -> np.ndarray:
        return np.full(rows, self.value, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"training_target_mean": self.value}


@dataclass(frozen=True)
class MomentumRegressor:
    """Use one raw momentum feature, imputing only with its train-fold median."""

    feature_name: str
    training_median: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_name: str) -> MomentumRegressor:
        values = frame[feature_name].to_numpy(dtype=np.float64, copy=True)
        if not len(values) or np.isnan(values).all():
            raise SchemaError(f"Momentum feature {feature_name!r} is all missing in a training fold")
        return cls(feature_name, float(np.nanmedian(values)))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_name].to_numpy(dtype=np.float64, copy=True)
        prediction = np.where(np.isnan(values), self.training_median, values)
        if not np.isfinite(prediction).all():
            raise SchemaError("Momentum model produced non-finite predictions")
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "missing_value_strategy": "training_fold_median",
            "training_median": self.training_median,
        }


@dataclass(frozen=True)
class LinearRegressor:
    """Ordinary least squares on fold-locally preprocessed numeric features."""

    preprocessor: FittedPreprocessor
    intercept: float
    coefficients: np.ndarray
    design_rank: int
    singular_values: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        target: np.ndarray,
        feature_names: Sequence[str],
    ) -> LinearRegressor:
        preprocessor = FittedPreprocessor.fit(frame, feature_names)
        transformed = preprocessor.transform(frame)
        design = np.column_stack((np.ones(len(transformed), dtype=np.float64), transformed))
        solution, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
        if not np.isfinite(solution).all():
            raise SchemaError("Linear least-squares fitting produced non-finite coefficients")
        return cls(
            preprocessor=preprocessor,
            intercept=float(solution[0]),
            coefficients=solution[1:].astype(np.float64, copy=False),
            design_rank=int(rank),
            singular_values=singular_values.astype(np.float64, copy=False),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        prediction = self.intercept + transformed @ self.coefficients
        if not np.isfinite(prediction).all():
            raise SchemaError("Linear model produced non-finite predictions")
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": "ordinary_least_squares",
            "intercept": self.intercept,
            "coefficients": dict(
                zip(self.preprocessor.feature_names, self.coefficients.tolist(), strict=True)
            ),
            "design_rank": self.design_rank,
            "design_columns": 1 + len(self.coefficients),
            "singular_values": self.singular_values.tolist(),
            "preprocessor": self.preprocessor.to_dict(),
        }


@dataclass(frozen=True)
class FittedFoldModels:
    """The three fitted baselines for one development fold."""

    constant: ConstantRegressor
    momentum: MomentumRegressor
    linear: LinearRegressor

    def to_dict(self) -> dict[str, Any]:
        return {
            "constant": self.constant.to_dict(),
            "momentum": self.momentum.to_dict(),
            "linear": self.linear.to_dict(),
        }


@dataclass
class ModelingResult:
    """Notebook-friendly predictions, metrics, fitted state, and audit report."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    daily_metrics: pd.DataFrame
    fitted_models: dict[str, FittedFoldModels]
    split_result: SplitResult
    report: dict[str, Any]


def load_config(path: Path | str) -> ModelingConfig:
    """Read a strict JSON modeling configuration."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(ModelingConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if "feature_columns" in values:
        if not isinstance(values["feature_columns"], list):
            raise ValueError("feature_columns must be a JSON array")
        values["feature_columns"] = tuple(values["feature_columns"])
    return ModelingConfig(**values)


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        parsed = series.astype("datetime64[ns]")
    else:
        if not all(isinstance(value, str) for value in series.dropna()):
            raise SchemaError(f"{name} must contain ISO dates or timezone-naive datetimes")
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        parsed = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or parsed.ne(parsed.dt.normalize()).any():
        raise SchemaError(f"{name} must contain valid non-null midnight dates")
    return parsed.astype("datetime64[ns]")


def _required_columns(config: ModelingConfig) -> tuple[str, ...]:
    return (
        *KEY_COLUMNS,
        "is_signal_eligible",
        TARGET_SAMPLE_COLUMNS[config.target_column],
        config.target_column,
        *config.feature_columns,
    )


def _normalize_development_rows(
    dataset: pd.DataFrame,
    development_positions: np.ndarray,
    config: ModelingConfig,
) -> pd.DataFrame:
    if not dataset.columns.is_unique:
        raise SchemaError("weekly dataset column names must be unique")
    required = _required_columns(config)
    missing = set(required) - set(dataset.columns)
    if missing:
        raise SchemaError(f"weekly dataset is missing required modeling columns: {sorted(missing)}")
    if len(np.unique(development_positions)) != len(development_positions):
        raise SchemaError("development positions must be unique")
    if len(development_positions) and (
        development_positions.min() < 0 or development_positions.max() >= len(dataset)
    ):
        raise SchemaError("development positions are outside the weekly dataset")
    output = dataset.iloc[development_positions].loc[:, list(required)].copy()
    output.index = pd.Index(development_positions.astype(np.int64), name="source_position")
    output["signal_date"] = _parse_dates(output["signal_date"], "weekly dataset.signal_date")
    output["exit_date"] = _parse_dates(output["exit_date"], "weekly dataset.exit_date")
    if not all(isinstance(value, str) for value in output["code"]):
        raise SchemaError("weekly dataset.code must contain strings")
    output["code"] = output["code"].astype("string")
    if not output["code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError("weekly dataset.code must contain non-null five-character uppercase codes")
    if output.duplicated(["signal_date", "code"]).any():
        count = int(output.duplicated(["signal_date", "code"], keep=False).sum())
        raise SchemaError(f"development rows contain {count} duplicate signal_date/code rows")
    if not output["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END, inclusive="both").all():
        raise SchemaError("modeling development rows must stay within 2016-2024")

    boolean_columns = ("is_signal_eligible", TARGET_SAMPLE_COLUMNS[config.target_column])
    for name in boolean_columns:
        if not is_bool_dtype(output[name].dtype) or output[name].isna().any():
            raise SchemaError(f"weekly dataset.{name} must be non-null boolean")
        output[name] = output[name].astype(bool)

    numeric_columns = (config.target_column, *config.feature_columns)
    for name in numeric_columns:
        if not is_numeric_dtype(output[name].dtype) or is_bool_dtype(output[name].dtype):
            raise SchemaError(f"weekly dataset.{name} must be numeric with null allowed")
        output[name] = output[name].astype("float64")
        invalid = output[name].notna() & ~np.isfinite(output[name])
        if invalid.any():
            raise SchemaError(f"weekly dataset.{name} contains non-finite non-null values")

    sample_column = TARGET_SAMPLE_COLUMNS[config.target_column]
    expected_sample = output["is_signal_eligible"] & output[config.target_column].notna()
    if not output[sample_column].eq(expected_sample).all():
        count = int(output[sample_column].ne(expected_sample).sum())
        raise SchemaError(
            f"weekly dataset.{sample_column} disagrees with eligibility and target availability on {count} rows"
        )
    return output


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _nanmean(values: pd.Series) -> float:
    finite = values.dropna()
    return float(finite.mean()) if len(finite) else float("nan")


def _daily_model_metrics(predictions: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for signal_date, group in predictions.groupby("signal_date", sort=True, observed=True):
        predicted = np.isfinite(group["prediction"].to_numpy(dtype=np.float64))
        observed = group["target_observed"].to_numpy(dtype=bool)
        scored = group.loc[predicted & observed].copy()
        actual = scored["actual_target"].to_numpy(dtype=np.float64)
        estimate = scored["prediction"].to_numpy(dtype=np.float64)
        if len(scored):
            errors = estimate - actual
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            direction = float(np.mean(np.sign(estimate) == np.sign(actual)))
            pearson = _safe_correlation(estimate, actual)
            estimate_rank = pd.Series(estimate).rank(method="average").to_numpy(dtype=np.float64)
            actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
            spearman = _safe_correlation(estimate_rank, actual_rank)
            ordered = scored.sort_values(
                ["prediction", "code"], ascending=[False, True], kind="stable"
            )
            count = min(top_n, len(ordered))
            top_return = float(ordered.head(count)["actual_target"].mean())
            bottom_return = float(ordered.tail(count)["actual_target"].mean())
            spread = top_return - bottom_return
        else:
            mae = rmse = direction = pearson = spearman = float("nan")
            count = 0
            top_return = bottom_return = spread = float("nan")
        eligible_rows = int(len(group))
        records.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "eligible_rows": eligible_rows,
                "target_observed_rows": int(observed.sum()),
                "prediction_rows": int(predicted.sum()),
                "scored_rows": int(len(scored)),
                "target_coverage": float(observed.mean()) if eligible_rows else float("nan"),
                "prediction_coverage": float(predicted.mean()) if eligible_rows else float("nan"),
                "mae": mae,
                "rmse": rmse,
                "direction_accuracy": direction,
                "pearson": pearson,
                "spearman": spearman,
                "selection_count": count,
                "top_n_mean_target": top_return,
                "bottom_n_mean_target": bottom_return,
                "top_bottom_spread": spread,
            }
        )
    return pd.DataFrame.from_records(records)


def _fold_model_metrics(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    train_rows: int,
    train_signal_dates: int,
) -> dict[str, Any]:
    observed = predictions["target_observed"].to_numpy(dtype=bool)
    predicted = np.isfinite(predictions["prediction"].to_numpy(dtype=np.float64))
    scored = predictions.loc[observed & predicted]
    actual = scored["actual_target"].to_numpy(dtype=np.float64)
    estimate = scored["prediction"].to_numpy(dtype=np.float64)
    errors = estimate - actual
    eligible_rows = int(len(predictions))
    return {
        "train_rows": train_rows,
        "train_signal_dates": train_signal_dates,
        "validation_signal_dates": int(predictions["signal_date"].nunique()),
        "validation_eligible_rows": eligible_rows,
        "target_observed_rows": int(observed.sum()),
        "prediction_rows": int(predicted.sum()),
        "scored_rows": int(len(scored)),
        "target_coverage": float(observed.mean()) if eligible_rows else float("nan"),
        "prediction_coverage": float(predicted.mean()) if eligible_rows else float("nan"),
        "mae": float(np.mean(np.abs(errors))) if len(scored) else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(scored) else float("nan"),
        "direction_accuracy": (
            float(np.mean(np.sign(estimate) == np.sign(actual))) if len(scored) else float("nan")
        ),
        "daily_pearson_mean": _nanmean(daily["pearson"]),
        "daily_pearson_valid_dates": int(daily["pearson"].notna().sum()),
        "daily_spearman_mean": _nanmean(daily["spearman"]),
        "daily_spearman_valid_dates": int(daily["spearman"].notna().sum()),
        "daily_top_n_mean_target": _nanmean(daily["top_n_mean_target"]),
        "daily_bottom_n_mean_target": _nanmean(daily["bottom_n_mean_target"]),
        "daily_top_bottom_spread_mean": _nanmean(daily["top_bottom_spread"]),
    }


def evaluate_walk_forward_models(
    dataset: pd.DataFrame,
    split_result: SplitResult,
    *,
    config: ModelingConfig | None = None,
) -> ModelingResult:
    """Fit and evaluate all baselines using only each fold's training rows.

    ``split_result`` must have been built for the same row order as ``dataset``.
    Only ``development_positions`` are inspected for features, eligibility, or
    targets.  Isolated and out-of-scope rows remain unopened by this function.
    """

    config = config or ModelingConfig()
    normalized = _normalize_development_rows(dataset, split_result.development_positions, config)
    sample_column = TARGET_SAMPLE_COLUMNS[config.target_column]
    prediction_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    fold_metric_records: list[dict[str, Any]] = []
    fitted_models: dict[str, FittedFoldModels] = {}
    fold_reports: list[dict[str, Any]] = []

    development_position_set = set(split_result.development_positions.tolist())
    for fold in split_result.folds:
        train_positions = fold.train_positions.astype(np.int64, copy=False)
        validation_positions = fold.validation_positions.astype(np.int64, copy=False)
        if not set(train_positions.tolist()).issubset(development_position_set):
            raise SchemaError(f"{fold.fold_id} training positions are not all development rows")
        if not set(validation_positions.tolist()).issubset(development_position_set):
            raise SchemaError(f"{fold.fold_id} validation positions are not all development rows")
        if np.intersect1d(train_positions, validation_positions).size:
            raise SchemaError(f"{fold.fold_id} training and validation positions overlap")

        train_all = normalized.loc[train_positions]
        validation_all = normalized.loc[validation_positions]
        train = train_all.loc[train_all[sample_column]].copy()
        validation = validation_all.loc[validation_all["is_signal_eligible"]].copy()
        if train.empty:
            raise SchemaError(f"{fold.fold_id} has no eligible observed training targets")
        if validation.empty:
            raise SchemaError(f"{fold.fold_id} has no signal-eligible validation rows")
        target = train[config.target_column].to_numpy(dtype=np.float64)

        models = FittedFoldModels(
            constant=ConstantRegressor.fit(target),
            momentum=MomentumRegressor.fit(train, config.momentum_feature),
            linear=LinearRegressor.fit(train, target, config.feature_columns),
        )
        fitted_models[fold.fold_id] = models
        prediction_values = {
            "constant": models.constant.predict(len(validation)),
            "momentum": models.momentum.predict(validation),
            "linear": models.linear.predict(validation),
        }

        for model_name in MODEL_NAMES:
            model_predictions = pd.DataFrame(
                {
                    "fold_id": fold.fold_id,
                    "validation_year": fold.validation_year,
                    "signal_date": validation["signal_date"].to_numpy(),
                    "code": validation["code"].astype("string").to_numpy(),
                    "model": model_name,
                    "target_column": config.target_column,
                    "actual_target": validation[config.target_column].to_numpy(dtype=np.float64),
                    "target_observed": validation[config.target_column].notna().to_numpy(),
                    "prediction": prediction_values[model_name],
                }
            )
            daily = _daily_model_metrics(model_predictions, top_n=config.top_n)
            daily.insert(0, "target_column", config.target_column)
            daily.insert(0, "model", model_name)
            daily.insert(0, "validation_year", fold.validation_year)
            daily.insert(0, "fold_id", fold.fold_id)
            metrics = _fold_model_metrics(
                model_predictions,
                daily,
                train_rows=int(len(train)),
                train_signal_dates=int(train["signal_date"].nunique()),
            )
            fold_metric_records.append(
                {
                    "fold_id": fold.fold_id,
                    "validation_year": fold.validation_year,
                    "model": model_name,
                    "target_column": config.target_column,
                    **metrics,
                }
            )
            prediction_frames.append(model_predictions)
            daily_frames.append(daily)

        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "validation_year": fold.validation_year,
                "train_rows_before_sample_filter": int(len(train_all)),
                "train_sample_rows": int(len(train)),
                "train_signal_dates": int(train["signal_date"].nunique()),
                "train_signal_date_min": _date_string(train["signal_date"].min()),
                "train_signal_date_max": _date_string(train["signal_date"].max()),
                "validation_rows_before_eligibility_filter": int(len(validation_all)),
                "validation_eligible_rows": int(len(validation)),
                "validation_target_observed_rows": int(validation[config.target_column].notna().sum()),
                "validation_signal_dates": int(validation["signal_date"].nunique()),
                "validation_signal_date_min": _date_string(validation["signal_date"].min()),
                "validation_signal_date_max": _date_string(validation["signal_date"].max()),
                "purged_rows": int(len(fold.purged_positions)),
                "linear_design_rank": models.linear.design_rank,
                "linear_design_columns": 1 + len(models.linear.coefficients),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["validation_year", "signal_date", "code", "model"], kind="stable"
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame.from_records(fold_metric_records).sort_values(
        ["validation_year", "model"], kind="stable"
    ).reset_index(drop=True)
    daily_metrics = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["validation_year", "signal_date", "model"], kind="stable"
    ).reset_index(drop=True)
    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "models": list(MODEL_NAMES),
        "development_period": {
            "start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
            "end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
        },
        "development_input": {
            "rows": int(len(normalized)),
            "signal_dates": int(normalized["signal_date"].nunique()),
            "codes": int(normalized["code"].nunique()),
            "signal_date_min": _date_string(normalized["signal_date"].min()),
            "signal_date_max": _date_string(normalized["signal_date"].max()),
            "duplicate_primary_keys": 0,
            "target_missing_rows": int(normalized[config.target_column].isna().sum()),
            "feature_missing_rates": {
                name: float(normalized[name].isna().mean()) for name in config.feature_columns
            },
        },
        "isolation": {
            "rows_excluded_from_modeling": int(len(split_result.isolated_positions)),
            "targets_or_features_inspected": False,
        },
        "folds": fold_reports,
        "outputs": {
            "prediction_rows": int(len(predictions)),
            "fold_metric_rows": int(len(fold_metrics)),
            "daily_metric_rows": int(len(daily_metrics)),
        },
        "random_seed": None,
    }
    return ModelingResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        daily_metrics=daily_metrics,
        fitted_models=fitted_models,
        split_result=split_result,
        report=report,
    )


def build_walk_forward_models(
    dataset: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    *,
    config: ModelingConfig | None = None,
    split_config: SplitConfig | None = None,
) -> ModelingResult:
    """Notebook API: create purged splits, then fit development baselines."""

    split_result = build_walk_forward_splits(dataset, trading_calendar, config=split_config)
    return evaluate_walk_forward_models(dataset, split_result, config=config)


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_calendar(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        if "Date" in frame.columns:
            field = pq.read_schema(path).field("Date")
            if pa.types.is_date(field.type):
                frame["Date"] = pd.to_datetime(frame["Date"])
        return frame
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"Date": "string", "HolDiv": "string"})
    raise SchemaError("calendar must be a .csv or .parquet file")


def _development_partitions(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        path
        for path in input_dir.glob("*.parquet")
        if re.fullmatch(r"\d{4}", path.stem)
        and DEVELOPMENT_START.year <= int(path.stem) <= DEVELOPMENT_END.year
    )
    if not paths:
        raise FileNotFoundError(f"No 2016-2024 annual YYYY.parquet files found in {input_dir}")
    return paths


def _inspect_partition(path: Path, required_columns: Sequence[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    missing = set(required_columns) - set(columns)
    if missing:
        raise SchemaError(f"{path.name} is missing required modeling columns: {sorted(missing)}")
    return {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "columns": columns,
        "sha256": _sha256(path),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_new_outputs(
    input_dir: Path,
    calendar_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> None:
    if output_dir.resolve() == report_dir.resolve():
        raise ValueError("output_dir and report_dir must differ")
    for target in (output_dir, report_dir):
        if target.exists():
            raise FileExistsError(f"Output path already exists: {target}")
        if _is_relative_to(target, input_dir):
            raise ValueError("Modeling outputs must not be created inside the weekly dataset input directory")
        if target.resolve() == calendar_path.resolve():
            raise ValueError("Modeling outputs must not overwrite the trading calendar")


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def run_modeling(
    *,
    input_dir: Path | str,
    calendar_path: Path | str,
    output_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    config: ModelingConfig | None = None,
    split_config: SplitConfig | None = None,
) -> dict[str, Any]:
    """CLI/save API: read development partitions and persist model results."""

    config = config or ModelingConfig()
    split_config = split_config or SplitConfig()
    input_dir = Path(input_dir)
    calendar_path = Path(calendar_path)
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "artifacts" / f"modeling_{run_id}"
    report_dir = Path(report_dir) if report_dir is not None else PROJECT_ROOT / "reports" / f"modeling_{run_id}"
    _ensure_new_outputs(input_dir, calendar_path, output_dir, report_dir)
    paths = _development_partitions(input_dir)
    required = _required_columns(config)
    inputs = [_inspect_partition(path, required) for path in paths]

    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest_path = report_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "split_config": asdict(split_config),
        "input_dir": str(input_dir),
        "development_inputs": inputs,
        "final_period_partitions_loaded": False,
        "calendar_file": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "completed_outputs": [],
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }
    _write_json(manifest_path, _json_safe(manifest))
    try:
        frames: list[pd.DataFrame] = []
        for path in paths:
            frame = pq.read_table(path, columns=list(required)).to_pandas()
            parsed_dates = _parse_dates(frame["signal_date"], f"{path.name}.signal_date")
            if not parsed_dates.dt.year.eq(int(path.stem)).all():
                raise SchemaError(f"{path.name} contains signal dates from another year")
            frame["signal_date"] = parsed_dates
            frames.append(frame)
        dataset = pd.concat(frames, ignore_index=True)
        calendar = _read_calendar(calendar_path)
        result = build_walk_forward_models(
            dataset,
            calendar,
            config=config,
            split_config=split_config,
        )
        manifest.update(result.report)
        model_parameters = {
            "schema_id": SCHEMA_ID,
            "target_column": config.target_column,
            "folds": {
                fold_id: models.to_dict() for fold_id, models in result.fitted_models.items()
            },
        }
        outputs: tuple[tuple[str, Any, Path], ...] = (
            ("predictions", result.predictions, output_dir / "predictions.parquet"),
            ("fold_metrics", result.fold_metrics, output_dir / "fold_metrics.parquet"),
            ("daily_metrics", result.daily_metrics, output_dir / "daily_metrics.parquet"),
            ("model_parameters", model_parameters, output_dir / "model_parameters.json"),
        )
        for name, value, path in outputs:
            if isinstance(value, pd.DataFrame):
                value.to_parquet(path, index=False)
                rows = int(len(value))
            else:
                _write_json(path, _json_safe(value))
                rows = len(value.get("folds", {}))
            manifest["completed_outputs"].append(
                {"name": name, "path": str(path), "rows": rows, "sha256": _sha256(path)}
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
    """CLI entry point for development-only baseline modeling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Annual weekly dataset directory")
    parser.add_argument("--calendar", type=Path, required=True, help="Official J-Quants Date/HolDiv calendar")
    parser.add_argument("--config", type=Path, help="Strict JSON modeling configuration")
    parser.add_argument("--split-config", type=Path, help="Strict JSON split configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config(args.config) if args.config else ModelingConfig()
    split_config = load_split_config(args.split_config) if args.split_config else SplitConfig()
    manifest = run_modeling(
        input_dir=args.input_dir,
        calendar_path=args.calendar,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
        split_config=split_config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "target_column": manifest["config"]["target_column"],
                "prediction_rows": manifest["outputs"]["prediction_rows"],
                "fold_metric_rows": manifest["outputs"]["fold_metric_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
