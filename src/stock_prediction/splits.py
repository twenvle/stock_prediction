"""Create purged walk-forward splits and isolate the final evaluation period.

Development is fixed to 2016-2024 by the research protocol.  Expanding
training folds validate one calendar year at a time from 2020 through 2024.
The selected 2025-2026 interval is returned separately and is never included
in a development fold.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import pyarrow as pa
import pyarrow.parquet as pq

from stock_prediction.preprocessing import PROJECT_ROOT, SchemaError, validate_trading_calendar


LOGGER = logging.getLogger(__name__)
SCHEMA_ID = "purged-walk-forward-splits-v1"
DEVELOPMENT_START = pd.Timestamp("2016-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
VALIDATION_YEARS = tuple(range(2020, 2025))
ISOLATION_EARLIEST = pd.Timestamp("2025-01-01")
ISOLATION_LATEST = pd.Timestamp("2026-12-31")
REQUIRED_COLUMNS = ("signal_date", "code", "exit_date")


@dataclass(frozen=True)
class SplitConfig:
    """Settings which may be fixed before opening the final period.

    The isolation bounds are inclusive and refer to a complete label interval:
    both ``signal_date`` and ``exit_date`` must fall within the bounds.  The
    purge width is measured in official TSE trading sessions.
    """

    isolation_start_date: str = "2025-01-01"
    isolation_end_date: str = "2026-12-31"
    purge_trading_sessions: int = 4

    def __post_init__(self) -> None:
        start = _parse_config_date(self.isolation_start_date, "isolation_start_date")
        end = _parse_config_date(self.isolation_end_date, "isolation_end_date")
        if start < ISOLATION_EARLIEST or end > ISOLATION_LATEST:
            raise ValueError("The selectable isolation period must stay within 2025-01-01 through 2026-12-31")
        if start > end:
            raise ValueError("isolation_start_date must not follow isolation_end_date")
        value = self.purge_trading_sessions
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("purge_trading_sessions must be a non-negative integer")


@dataclass(frozen=True)
class WalkForwardFold:
    """Positional row indices for one expanding train/validation fold."""

    fold_id: str
    validation_year: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    purged_positions: np.ndarray


@dataclass
class SplitResult:
    """Notebook-friendly split indices, persisted definitions, and audit data."""

    folds: tuple[WalkForwardFold, ...]
    development_positions: np.ndarray
    isolated_positions: np.ndarray
    out_of_scope_positions: np.ndarray
    fold_definitions: pd.DataFrame
    isolated_definitions: pd.DataFrame
    out_of_scope_definitions: pd.DataFrame
    report: dict[str, Any]


def _parse_config_date(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{name} must be an ISO date string in YYYY-MM-DD form")
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed) or parsed.strftime("%Y-%m-%d") != value:
        raise ValueError(f"{name} must be a valid ISO date")
    return pd.Timestamp(parsed)


def load_config(path: Path | str) -> SplitConfig:
    """Read a strict JSON split configuration."""

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    known = {field.name for field in fields(SplitConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return SplitConfig(**values)


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name} must contain timezone-naive Japanese session dates")
    if is_datetime64_any_dtype(series.dtype):
        parsed = series.astype("datetime64[ns]")
    else:
        non_null = series.dropna()
        if not all(isinstance(value, str) for value in non_null):
            raise SchemaError(f"{name} must contain ISO dates or timezone-naive datetimes")
        valid = series.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        parsed = pd.to_datetime(series.where(valid), format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or parsed.ne(parsed.dt.normalize()).any():
        raise SchemaError(f"{name} must contain valid non-null midnight dates")
    return parsed.astype("datetime64[ns]")


def _normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if not dataset.columns.is_unique:
        raise SchemaError("weekly dataset column names must be unique")
    missing = set(REQUIRED_COLUMNS) - set(dataset.columns)
    if missing:
        raise SchemaError(f"weekly dataset is missing required columns: {sorted(missing)}")
    output = dataset.loc[:, list(REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    output["signal_date"] = _parse_dates(output["signal_date"], "weekly dataset.signal_date")
    output["exit_date"] = _parse_dates(output["exit_date"], "weekly dataset.exit_date")
    if not all(isinstance(value, str) for value in output["code"]):
        raise SchemaError("weekly dataset.code must contain strings")
    output["code"] = output["code"].astype("string")
    if not output["code"].str.fullmatch(r"[0-9A-Z]{5}", na=False).all():
        raise SchemaError("weekly dataset.code must contain non-null five-character uppercase codes")
    if output.duplicated(["signal_date", "code"]).any():
        count = int(output.duplicated(["signal_date", "code"], keep=False).sum())
        raise SchemaError(f"weekly dataset contains {count} duplicate signal_date/code rows")
    if (output["exit_date"] < output["signal_date"]).any():
        raise SchemaError("weekly dataset.exit_date must not precede signal_date")
    exit_counts = output.groupby("signal_date", sort=False)["exit_date"].nunique(dropna=False)
    if exit_counts.gt(1).any():
        dates = [value.strftime("%Y-%m-%d") for value in exit_counts.index[exit_counts.gt(1)][:5]]
        raise SchemaError(f"Every signal_date must map to one exit_date; conflicts include {dates}")
    return output


def _date_string(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _positions(mask: pd.Series | np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64, copy=False)


def _unique_periods(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> pd.DataFrame:
    return (
        frame.loc[np.asarray(mask, dtype=bool), ["signal_date", "exit_date"]]
        .drop_duplicates()
        .sort_values("signal_date", kind="stable")
        .reset_index(drop=True)
    )


def _period_summary(frame: pd.DataFrame, positions: np.ndarray) -> dict[str, Any]:
    part = frame.iloc[positions]
    return {
        "rows": int(len(part)),
        "signal_dates": int(part["signal_date"].nunique()),
        "codes": int(part["code"].nunique()),
        "signal_date_min": _date_string(part["signal_date"].min()),
        "signal_date_max": _date_string(part["signal_date"].max()),
        "exit_date_max": _date_string(part["exit_date"].max()),
    }


def _out_of_scope_reason(signal: pd.Timestamp, exit_date: pd.Timestamp, config: SplitConfig) -> str:
    isolation_start = pd.Timestamp(config.isolation_start_date)
    isolation_end = pd.Timestamp(config.isolation_end_date)
    if signal < DEVELOPMENT_START:
        return "before_development"
    if signal <= DEVELOPMENT_END and exit_date > DEVELOPMENT_END:
        return "label_crosses_development_end"
    if signal < isolation_start:
        return "before_selected_isolation"
    if signal <= isolation_end and exit_date > isolation_end:
        return "label_crosses_isolation_end"
    if signal > isolation_end:
        return "after_selected_isolation"
    return "outside_configured_periods"


def build_walk_forward_splits(
    dataset: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Build expanding annual folds without exposing the isolated rows.

    Returned arrays are zero-based positional indices for ``dataset.iloc``.
    Rows are assigned by complete signal-date groups; target availability and
    eligibility flags are deliberately not used by the splitter.
    """

    config = config or SplitConfig()
    normalized = _normalize_dataset(dataset)
    if normalized.empty:
        raise SchemaError("weekly dataset must contain at least one row")
    calendar = validate_trading_calendar(trading_calendar)
    sessions = pd.DatetimeIndex(calendar.loc[calendar["is_session"], "date"])
    if sessions.empty:
        raise SchemaError("trading calendar contains no TSE sessions")
    if calendar["date"].min() > normalized["signal_date"].min() or calendar["date"].max() < normalized["exit_date"].max():
        raise SchemaError("trading calendar must cover every input signal_date through exit_date")
    session_lookup = pd.Series(np.arange(len(sessions), dtype=np.int64), index=sessions)
    signal_indices = normalized["signal_date"].map(session_lookup)
    exit_indices = normalized["exit_date"].map(session_lookup)
    if signal_indices.isna().any() or exit_indices.isna().any():
        raise SchemaError("Every signal_date and exit_date must be an official TSE trading session")
    signal_indices = signal_indices.astype("int64")
    exit_indices = exit_indices.astype("int64")
    label_horizons = exit_indices - signal_indices

    isolation_start = pd.Timestamp(config.isolation_start_date)
    isolation_end = pd.Timestamp(config.isolation_end_date)
    development_mask = (
        normalized["signal_date"].between(DEVELOPMENT_START, DEVELOPMENT_END, inclusive="both")
        & normalized["exit_date"].le(DEVELOPMENT_END)
    )
    isolation_mask = (
        normalized["signal_date"].between(isolation_start, isolation_end, inclusive="both")
        & normalized["exit_date"].le(isolation_end)
    )
    if (development_mask & isolation_mask).any():
        raise AssertionError("Development and isolation masks unexpectedly overlap")
    out_of_scope_mask = ~(development_mask | isolation_mask)
    development_positions = _positions(development_mask)
    isolated_positions = _positions(isolation_mask)
    out_of_scope_positions = _positions(out_of_scope_mask)
    if not len(development_positions):
        raise SchemaError("weekly dataset contains no complete labels in the 2016-2024 development period")
    maximum_horizon = int(label_horizons.loc[development_mask].max())
    if config.purge_trading_sessions < maximum_horizon:
        raise ValueError(
            "purge_trading_sessions must be at least the maximum development label horizon "
            f"({maximum_horizon} TSE sessions)"
        )

    fold_objects: list[WalkForwardFold] = []
    fold_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for validation_year in VALIDATION_YEARS:
        validation_mask = development_mask & normalized["signal_date"].dt.year.eq(validation_year)
        validation_positions = _positions(validation_mask)
        if not len(validation_positions):
            raise SchemaError(f"development data contains no validation rows for {validation_year}")
        validation_start = normalized.loc[validation_mask, "signal_date"].min()
        validation_session_index = int(session_lookup.loc[validation_start])
        cutoff_index = validation_session_index - config.purge_trading_sessions
        if cutoff_index < 0:
            raise SchemaError(f"trading calendar lacks purge history before the {validation_year} validation period")
        purge_cutoff = sessions[cutoff_index]
        train_candidate_mask = development_mask & normalized["signal_date"].lt(validation_start)
        train_mask = (
            train_candidate_mask
            & normalized["signal_date"].le(purge_cutoff)
            & normalized["exit_date"].lt(validation_start)
        )
        purged_mask = train_candidate_mask & ~train_mask
        train_positions = _positions(train_mask)
        purged_positions = _positions(purged_mask)
        if not len(train_positions):
            raise SchemaError(f"purging leaves no training rows for the {validation_year} fold")
        if normalized.iloc[train_positions]["exit_date"].max() >= validation_start:
            raise AssertionError("A training label reaches the validation period after purging")
        maximum_train_signal_index = int(signal_indices.iloc[train_positions].max())
        if validation_session_index - maximum_train_signal_index < config.purge_trading_sessions:
            raise AssertionError("Training signals do not satisfy the configured purge width")
        fold_id = f"validation_{validation_year}"
        fold_objects.append(
            WalkForwardFold(
                fold_id=fold_id,
                validation_year=validation_year,
                train_positions=train_positions,
                validation_positions=validation_positions,
                purged_positions=purged_positions,
            )
        )
        for role, mask in (("train", train_mask), ("validation", validation_mask), ("purged", purged_mask)):
            periods = _unique_periods(normalized, mask)
            periods.insert(0, "role", role)
            periods.insert(0, "validation_year", validation_year)
            periods.insert(0, "fold_id", fold_id)
            fold_frames.append(periods)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "validation_year": validation_year,
                "validation_start": _date_string(validation_start),
                "purge_cutoff_signal_date": _date_string(purge_cutoff),
                "train": _period_summary(normalized, train_positions),
                "validation": _period_summary(normalized, validation_positions),
                "purged": _period_summary(normalized, purged_positions),
            }
        )

    fold_definitions = pd.concat(fold_frames, ignore_index=True)
    fold_definitions["validation_year"] = fold_definitions["validation_year"].astype("int64")
    isolated_definitions = _unique_periods(normalized, isolation_mask)
    out_of_scope_definitions = _unique_periods(normalized, out_of_scope_mask)
    if not out_of_scope_definitions.empty:
        out_of_scope_definitions["reason"] = [
            _out_of_scope_reason(row.signal_date, row.exit_date, config)
            for row in out_of_scope_definitions.itertuples(index=False)
        ]
    else:
        out_of_scope_definitions["reason"] = pd.Series(dtype="string")
    out_of_scope_definitions["reason"] = out_of_scope_definitions["reason"].astype("string")
    reason_counts = {
        str(key): int(value)
        for key, value in out_of_scope_definitions["reason"].value_counts().sort_index().items()
    }

    report = {
        "schema_id": SCHEMA_ID,
        "config": asdict(config),
        "development_period": {
            "start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
            "end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
        },
        "validation_years": list(VALIDATION_YEARS),
        "input": {
            "rows": int(len(normalized)),
            "signal_dates": int(normalized["signal_date"].nunique()),
            "codes": int(normalized["code"].nunique()),
            "signal_date_min": _date_string(normalized["signal_date"].min()),
            "signal_date_max": _date_string(normalized["signal_date"].max()),
            "exit_date_max": _date_string(normalized["exit_date"].max()),
            "duplicate_primary_keys": 0,
            "required_null_counts": {column: 0 for column in REQUIRED_COLUMNS},
        },
        "calendar": {
            "date_min": _date_string(calendar["date"].min()),
            "date_max": _date_string(calendar["date"].max()),
            "rows": int(len(calendar)),
            "sessions": int(calendar["is_session"].sum()),
        },
        "maximum_development_label_horizon_sessions": maximum_horizon,
        "development": _period_summary(normalized, development_positions),
        "isolation": _period_summary(normalized, isolated_positions),
        "out_of_scope": {
            **_period_summary(normalized, out_of_scope_positions),
            "reason_signal_date_counts": reason_counts,
        },
        "folds": fold_reports,
        "random_seed": None,
    }
    return SplitResult(
        folds=tuple(fold_objects),
        development_positions=development_positions,
        isolated_positions=isolated_positions,
        out_of_scope_positions=out_of_scope_positions,
        fold_definitions=fold_definitions,
        isolated_definitions=isolated_definitions,
        out_of_scope_definitions=out_of_scope_definitions,
        report=report,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


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


def _annual_partitions(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = sorted(path for path in input_dir.glob("*.parquet") if re.fullmatch(r"\d{4}", path.stem))
    if not paths:
        raise FileNotFoundError(f"No annual YYYY.parquet files found in {input_dir}")
    return paths


def _inspect_partition(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    missing = set(REQUIRED_COLUMNS) - set(columns)
    if missing:
        raise SchemaError(f"{path.name} is missing required columns: {sorted(missing)}")
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


def _ensure_new_outputs(input_dir: Path, calendar_path: Path, output_dir: Path, report_dir: Path) -> None:
    if output_dir.resolve() == report_dir.resolve():
        raise ValueError("output_dir and report_dir must differ")
    for target in (output_dir, report_dir):
        if target.exists():
            raise FileExistsError(f"Output path already exists: {target}")
        if _is_relative_to(target, input_dir):
            raise ValueError("Split outputs must not be created inside the weekly dataset input directory")
        if target.resolve() == calendar_path.resolve():
            raise ValueError("Split outputs must not overwrite the trading calendar")


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def run_splits(
    *,
    input_dir: Path | str,
    calendar_path: Path | str,
    output_dir: Path | str | None = None,
    report_dir: Path | str | None = None,
    config: SplitConfig | None = None,
) -> dict[str, Any]:
    """Read annual weekly keys and save auditable split definitions."""

    config = config or SplitConfig()
    input_dir = Path(input_dir)
    calendar_path = Path(calendar_path)
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Calendar does not exist: {calendar_path}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "data" / "processed" / f"splits_{run_id}"
    report_dir = Path(report_dir) if report_dir is not None else PROJECT_ROOT / "reports" / f"splits_{run_id}"
    _ensure_new_outputs(input_dir, calendar_path, output_dir, report_dir)
    paths = _annual_partitions(input_dir)
    inputs = [_inspect_partition(path) for path in paths]
    key_frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pq.read_table(path, columns=list(REQUIRED_COLUMNS)).to_pandas()
        parsed_signal_dates = _parse_dates(frame["signal_date"], f"{path.name}.signal_date")
        if not parsed_signal_dates.dt.year.eq(int(path.stem)).all():
            raise SchemaError(f"{path.name} contains signal dates from another year")
        frame["signal_date"] = parsed_signal_dates
        key_frames.append(frame)
    dataset = pd.concat(key_frames, ignore_index=True)
    calendar = _read_calendar(calendar_path)
    result = build_walk_forward_splits(dataset, calendar, config=config)

    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **result.report,
        "input_dir": str(input_dir),
        "inputs": inputs,
        "calendar_file": {"path": str(calendar_path), "sha256": _sha256(calendar_path)},
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "completed_outputs": [],
        "code_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }
    manifest_path = report_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs = (
        ("folds", result.fold_definitions, output_dir / "folds.parquet"),
        ("isolated", result.isolated_definitions, output_dir / "isolated.parquet"),
        ("out_of_scope", result.out_of_scope_definitions, output_dir / "out_of_scope.parquet"),
    )
    try:
        for name, frame, path in outputs:
            frame.to_parquet(path, index=False)
            manifest["completed_outputs"].append(
                {"name": name, "path": str(path), "rows": int(len(frame)), "sha256": _sha256(path)}
            )
            _write_json(manifest_path, manifest)
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
    """CLI entry point for purged walk-forward split generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Annual weekly dataset directory")
    parser.add_argument("--calendar", type=Path, required=True, help="Official J-Quants Date/HolDiv calendar")
    parser.add_argument("--config", type=Path, help="Strict JSON split configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config(args.config) if args.config else SplitConfig()
    manifest = run_splits(
        input_dir=args.input_dir,
        calendar_path=args.calendar,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "development_rows": manifest["development"]["rows"],
                "isolated_rows": manifest["isolation"]["rows"],
                "out_of_scope_rows": manifest["out_of_scope"]["rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
