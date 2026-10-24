"""Re-run the auditable development pipeline from immutable raw snapshots.

The orchestration starts from already-downloaded J-Quants snapshots.  Network
acquisition remains a separate step because a later API response is not the
same immutable input.  CLI and Notebook callers share :func:`run_experiment`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .backtest import BacktestConfig, load_config as load_backtest_config, run_backtest
from .descriptors import DescriptorConfig, load_config as load_descriptor_config, run_descriptors
from .earnings_events import (
    EarningsEventConfig,
    load_config as load_earnings_config,
    run_earnings_events,
)
from .evaluation import EvaluationConfig, load_config as load_evaluation_config, run_evaluation
from .modeling import MODEL_NAMES, ModelingConfig, load_config as load_modeling_config, run_modeling
from .preprocessing import (
    PreprocessingConfig,
    load_config as load_preprocessing_config,
    run_preprocessing,
)
from .splits import SplitConfig, load_config as load_split_config, run_splits
from .universe import UniverseConfig, load_config as load_universe_config, run_universe
from .weekly_dataset import (
    WeeklyDatasetConfig,
    load_config as load_weekly_config,
    run_weekly_dataset,
    run_weekly_schedule,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "stock-prediction-experiment-v1"
CONFIG_SCHEMA_VERSION = 1
STAGE_ORDER = (
    "preprocessing",
    "topix_normalization",
    "descriptors",
    "weekly_schedule",
    "universe",
    "earnings_events",
    "weekly_dataset",
    "splits",
    "modeling",
    "evaluation",
    "backtest",
)
STAGE_CONFIG_NAMES = (
    "preprocessing",
    "descriptors",
    "universe",
    "earnings_events",
    "weekly_dataset",
    "splits",
    "modeling",
    "evaluation",
    "backtest",
)
COMPLETE_STATUSES = frozenset({"complete", "completed"})
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class ExperimentInputs:
    """Immutable inputs used by all experiment stages."""

    price_snapshot_dir: str
    auxiliary_snapshot_dir: str
    delistings_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("price_snapshot_dir", "auxiliary_snapshot_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"inputs.{name} must be a non-empty path string")
        if self.delistings_path is not None and (
            not isinstance(self.delistings_path, str) or not self.delistings_path.strip()
        ):
            raise ValueError("inputs.delistings_path must be null or a non-empty path string")


@dataclass(frozen=True)
class ExperimentConfig:
    """Strict experiment settings whose resolved form is saved with every run."""

    experiment_name: str
    random_seed: int
    inputs: ExperimentInputs
    stage_configs: Mapping[str, str]
    schema_version: int = CONFIG_SCHEMA_VERSION
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= np.iinfo(np.uint32).max
        ):
            raise ValueError("random_seed must be an integer between 0 and 2**32 - 1")
        if not isinstance(self.inputs, ExperimentInputs):
            raise ValueError("inputs must be an ExperimentInputs value")
        if not isinstance(self.stage_configs, Mapping):
            raise ValueError("stage_configs must be an object")
        actual = set(self.stage_configs)
        expected = set(STAGE_CONFIG_NAMES)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(f"stage_configs keys must match exactly; missing={missing}, unknown={unknown}")
        for name, value in self.stage_configs.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"stage_configs.{name} must be a non-empty path string")


@dataclass(frozen=True)
class ExperimentPaths:
    """All derived-data, artifact, and report locations for one immutable run."""

    interim_root: Path
    processed_root: Path
    artifact_root: Path
    report_root: Path
    prices: Path
    topix: Path
    descriptors: Path
    weekly_schedule: Path
    universe: Path
    earnings_events: Path
    weekly_dataset: Path
    splits: Path
    modeling: Path
    evaluation: Path
    backtest: Path

    def as_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class ExperimentPlan:
    """Validated, side-effect-free plan returned to CLI and Notebook callers."""

    run_id: str
    project_root: Path
    config: ExperimentConfig
    inputs: Mapping[str, Path | None]
    stage_config_paths: Mapping[str, Path]
    stage_config_objects: Mapping[str, Any] = field(repr=False)
    stage_config_values: Mapping[str, Any]
    input_identifiers: Mapping[str, Any]
    paths: ExperimentPaths
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentResult:
    """Saved experiment state and report returned after a completed run."""

    run_id: str
    manifest_path: Path
    report_path: Path
    manifest: Mapping[str, Any]
    paths: ExperimentPaths


def load_config(path: Path | str) -> ExperimentConfig:
    """Load a strict experiment JSON without resolving machine-local paths."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    allowed = {"schema_version", "experiment_name", "random_seed", "inputs", "stage_configs"}
    unknown = set(payload) - allowed
    missing = allowed - set(payload)
    if unknown or missing:
        raise ValueError(
            f"Experiment configuration keys must match exactly; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    input_payload = payload["inputs"]
    if not isinstance(input_payload, dict):
        raise ValueError("inputs must be a JSON object")
    input_allowed = {"price_snapshot_dir", "auxiliary_snapshot_dir", "delistings_path"}
    input_unknown = set(input_payload) - input_allowed
    input_missing = {"price_snapshot_dir", "auxiliary_snapshot_dir"} - set(input_payload)
    if input_unknown or input_missing:
        raise ValueError(
            f"inputs keys are invalid; missing={sorted(input_missing)}, unknown={sorted(input_unknown)}"
        )
    stage_configs = payload["stage_configs"]
    if not isinstance(stage_configs, dict):
        raise ValueError("stage_configs must be a JSON object")
    return ExperimentConfig(
        schema_version=payload["schema_version"],
        experiment_name=payload["experiment_name"],
        random_seed=payload["random_seed"],
        inputs=ExperimentInputs(
            price_snapshot_dir=input_payload["price_snapshot_dir"],
            auxiliary_snapshot_dir=input_payload["auxiliary_snapshot_dir"],
            delistings_path=input_payload.get("delistings_path"),
        ),
        stage_configs=dict(stage_configs),
        source_path=config_path.resolve(),
    )


def normalize_topix(topix_raw: pd.DataFrame) -> pd.DataFrame:
    """Create the strict ``date``/``close`` derivative used by later stages.

    The input is not modified.  Coverage against the official calendar remains
    enforced independently by the descriptor and weekly-dataset stages.
    """

    if not topix_raw.columns.is_unique:
        raise ValueError("Raw TOPIX columns must be unique")
    missing = {"Date", "C"} - set(topix_raw.columns)
    if missing:
        raise ValueError(f"Raw TOPIX is missing required columns: {sorted(missing)}")
    result = topix_raw.loc[:, ["Date", "C"]].rename(columns={"Date": "date", "C": "close"}).copy()
    if isinstance(result["date"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Raw TOPIX Date must be timezone-naive")
    if pd.api.types.is_datetime64_any_dtype(result["date"].dtype):
        dates = result["date"].astype("datetime64[ns]")
    else:
        if not all(isinstance(value, str) for value in result["date"].dropna()):
            raise ValueError("Raw TOPIX Date must contain ISO strings or datetimes")
        valid = result["date"].astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        dates = pd.to_datetime(result["date"].where(valid), format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or dates.ne(dates.dt.normalize()).any():
        raise ValueError("Raw TOPIX Date must contain valid non-null midnight dates")
    result["date"] = dates.astype("datetime64[ns]")
    if result["date"].duplicated().any():
        raise ValueError("Raw TOPIX contains duplicate dates")
    if not pd.api.types.is_numeric_dtype(result["close"].dtype) or pd.api.types.is_bool_dtype(
        result["close"].dtype
    ):
        raise ValueError("Raw TOPIX C must be numeric")
    result["close"] = result["close"].astype("float64")
    invalid_close = result["close"].isna() | ~np.isfinite(result["close"]) | result["close"].le(0)
    if invalid_close.any():
        raise ValueError("Raw TOPIX C must be positive, finite, and non-null")
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 characters using letters, digits, dot, underscore, or hyphen")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if run_id.split(".", maxsplit=1)[0].upper() in reserved:
        raise ValueError("run_id must not be a reserved Windows device name")
    return run_id


def _default_run_id(config: ExperimentConfig) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", config.experiment_name.strip()).strip(".-_")
    slug = (slug or "experiment")[:32]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{slug}_{timestamp}"[:64]


def _experiment_paths(project_root: Path, run_id: str) -> ExperimentPaths:
    interim = project_root / "data" / "interim" / "experiments" / run_id
    processed = project_root / "data" / "processed" / "experiments" / run_id
    artifacts = project_root / "artifacts" / "experiments" / run_id
    reports = project_root / "reports" / "experiments" / run_id
    return ExperimentPaths(
        interim_root=interim,
        processed_root=processed,
        artifact_root=artifacts,
        report_root=reports,
        prices=interim / "prices",
        topix=interim / "topix.parquet",
        descriptors=processed / "descriptors",
        weekly_schedule=processed / "weekly_schedule.parquet",
        universe=processed / "universe",
        earnings_events=processed / "earnings_events",
        weekly_dataset=processed / "weekly_dataset",
        splits=processed / "splits",
        modeling=artifacts / "modeling",
        evaluation=artifacts / "evaluation",
        backtest=artifacts / "backtest",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _load_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} manifest must be a JSON object")
    status = value.get("status")
    if status not in COMPLETE_STATUSES:
        raise ValueError(f"{label} snapshot is not complete: status={status!r}")
    return value


def _collect_declared_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "sha256" and isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                hashes.append(item.lower())
            else:
                hashes.extend(_collect_declared_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_collect_declared_hashes(item))
    return hashes


def _snapshot_identifier(directory: Path, label: str) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = _load_manifest(manifest_path, label)
    declared_hashes = sorted(_collect_declared_hashes(manifest))
    declared_digest = hashlib.sha256(
        json.dumps(declared_hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(directory),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "snapshot_id": manifest.get("snapshot_id"),
        "status": manifest.get("status"),
        "provider": manifest.get("provider"),
        "api_version": manifest.get("api_version"),
        "request": manifest.get("request"),
        "totals": manifest.get("totals"),
        "declared_sha256_count": len(declared_hashes),
        "declared_sha256_digest": declared_digest,
    }


def _load_stage_configs(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaders = {
        "preprocessing": load_preprocessing_config,
        "descriptors": load_descriptor_config,
        "universe": load_universe_config,
        "earnings_events": load_earnings_config,
        "weekly_dataset": load_weekly_config,
        "splits": load_split_config,
        "modeling": load_modeling_config,
        "evaluation": load_evaluation_config,
        "backtest": load_backtest_config,
    }
    objects: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for name in STAGE_CONFIG_NAMES:
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Stage configuration does not exist: {path}")
        obj = loaders[name](path)
        objects[name] = obj
        values[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "values": asdict(obj),
        }
    return objects, values


def plan_experiment(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> ExperimentPlan:
    """Validate inputs/configuration and return an execution plan without writes."""

    root = Path(project_root).resolve()
    selected_run_id = _validate_run_id(run_id or _default_run_id(config))
    paths = _experiment_paths(root, selected_run_id)
    existing_roots = [
        path
        for path in (paths.interim_root, paths.processed_root, paths.artifact_root, paths.report_root)
        if path.exists()
    ]
    if existing_roots:
        raise FileExistsError(f"Experiment outputs already exist: {[str(path) for path in existing_roots]}")

    price_snapshot = _resolve_path(config.inputs.price_snapshot_dir, root)
    auxiliary_snapshot = _resolve_path(config.inputs.auxiliary_snapshot_dir, root)
    delistings = (
        None
        if config.inputs.delistings_path is None
        else _resolve_path(config.inputs.delistings_path, root)
    )
    price_required = {
        "prices": price_snapshot / "prices",
        "calendar": price_snapshot / "trading_calendar.parquet",
    }
    auxiliary_required = {
        "topix": auxiliary_snapshot / "topix.parquet",
        "listed_info": auxiliary_snapshot / "listed_info",
        "earnings_schedule": auxiliary_snapshot / "earnings_schedule",
        "financial_results": auxiliary_snapshot / "financial_results",
    }
    for label, path in {**price_required, **auxiliary_required}.items():
        expected_directory = label in {"prices", "listed_info", "earnings_schedule", "financial_results"}
        if expected_directory and not path.is_dir():
            raise FileNotFoundError(f"Required input directory does not exist: {path}")
        if not expected_directory and not path.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {path}")
        if expected_directory and not any(
            candidate.is_file() and re.fullmatch(r"\d{4}\.parquet", candidate.name)
            for candidate in path.iterdir()
        ):
            raise FileNotFoundError(f"Required input directory has no annual YYYY.parquet files: {path}")
    if delistings is not None and not delistings.is_file():
        raise FileNotFoundError(f"Delistings input does not exist: {delistings}")

    output_roots = (paths.interim_root, paths.processed_root, paths.artifact_root, paths.report_root)
    protected_inputs = (price_snapshot, auxiliary_snapshot, *(value for value in (delistings,) if value is not None))
    for output_root in output_roots:
        for input_path in protected_inputs:
            if _is_relative_to(output_root, input_path) or _is_relative_to(input_path, output_root):
                raise ValueError(
                    f"Experiment output and input paths must not overlap: output={output_root}, input={input_path}"
                )

    stage_config_paths = {
        name: _resolve_path(config.stage_configs[name], root) for name in STAGE_CONFIG_NAMES
    }
    stage_objects, stage_values = _load_stage_configs(stage_config_paths)
    modeling_config: ModelingConfig = stage_objects["modeling"]
    backtest_config: BacktestConfig = stage_objects["backtest"]
    if modeling_config.target_column != backtest_config.target_column:
        raise ValueError(
            "modeling.target_column and backtest.target_column must match so the selected predictions exist"
        )
    if backtest_config.model_name not in MODEL_NAMES:
        raise ValueError(
            f"backtest.model_name must be generated by modeling; expected one of {list(MODEL_NAMES)}"
        )

    warnings: list[str] = [
        "This experiment evaluates development-period predictions only; it does not run or unlock the 2025-2026 final period.",
        "J-Quants acquisition is intentionally outside the experiment run; immutable snapshot manifests identify the inputs.",
    ]
    evaluation_config: EvaluationConfig = stage_objects["evaluation"]
    if evaluation_config.top_n != backtest_config.top_n:
        warnings.append("evaluation.top_n and backtest.top_n differ; their selected groups are not directly comparable.")
    if modeling_config.top_n != evaluation_config.top_n:
        warnings.append("modeling.top_n and evaluation.top_n differ; built-in and independent evaluation summaries differ.")
    if delistings is None:
        warnings.append(
            "No confirmed delisting settlement table is supplied; the backtest will stop if a confirmed event must be settled."
        )

    input_identifiers: dict[str, Any] = {
        "price_snapshot": _snapshot_identifier(price_snapshot, "price"),
        "auxiliary_snapshot": _snapshot_identifier(auxiliary_snapshot, "auxiliary"),
        "calendar": {
            "path": str(price_required["calendar"]),
            "sha256": _sha256(price_required["calendar"]),
        },
        "topix_raw": {
            "path": str(auxiliary_required["topix"]),
            "sha256": _sha256(auxiliary_required["topix"]),
        },
        "delistings": (
            None if delistings is None else {"path": str(delistings), "sha256": _sha256(delistings)}
        ),
    }
    inputs: dict[str, Path | None] = {
        "price_snapshot": price_snapshot,
        "prices": price_required["prices"],
        "calendar": price_required["calendar"],
        "auxiliary_snapshot": auxiliary_snapshot,
        "topix_raw": auxiliary_required["topix"],
        "listed_info": auxiliary_required["listed_info"],
        "earnings_schedule": auxiliary_required["earnings_schedule"],
        "financial_results": auxiliary_required["financial_results"],
        "delistings": delistings,
    }
    plan = ExperimentPlan(
        run_id=selected_run_id,
        project_root=root,
        config=config,
        inputs=inputs,
        stage_config_paths=stage_config_paths,
        stage_config_objects=stage_objects,
        stage_config_values=stage_values,
        input_identifiers=input_identifiers,
        paths=paths,
        warnings=tuple(warnings),
    )
    # Objects are validated above and intentionally not retained in the public plan.
    return plan


def _versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("stock-prediction", "pandas", "numpy", "pyarrow", "requests"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def _source_hashes(project_root: Path) -> dict[str, str]:
    modules = (
        "experiment.py",
        "preprocessing.py",
        "descriptors.py",
        "universe.py",
        "earnings_events.py",
        "weekly_dataset.py",
        "splits.py",
        "modeling.py",
        "evaluation.py",
        "backtest.py",
    )
    source_dir = Path(__file__).parent
    values = {name: _sha256(source_dir / name) for name in modules}
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        values["pyproject.toml"] = _sha256(pyproject)
    return values


def _resolved_config(plan: ExperimentPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.config.schema_version,
        "experiment_name": plan.config.experiment_name,
        "random_seed": plan.config.random_seed,
        "run_id": plan.run_id,
        "project_root": str(plan.project_root),
        "inputs": {name: None if path is None else str(path) for name, path in plan.inputs.items()},
        "stage_configs": plan.stage_config_values,
        "outputs": plan.paths.as_dict(),
        "stage_order": list(STAGE_ORDER),
    }


def _stage_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "output_rows",
        "quarantined_rows",
        "eligible_rows",
        "signal_eligible_rows",
        "target_return_sample_rows",
        "target_excess_return_sample_rows",
        "main_result_complete",
        "outputs",
        "isolation",
    )
    return {key: manifest[key] for key in keys if key in manifest}


def _run_topix_normalization(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(input_path)
    missing = {"Date", "C"} - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Raw TOPIX is missing required columns: {sorted(missing)}")
    raw = pq.read_table(input_path, columns=["Date", "C"]).to_pandas()
    normalized = normalize_topix(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)
    report = {
        "schema_id": "normalized-topix-v1",
        "status": "complete",
        "input": {"path": str(input_path), "sha256": _sha256(input_path), "rows": int(len(raw))},
        "output": {"path": str(output_path), "sha256": _sha256(output_path), "rows": int(len(normalized))},
        "date_min": None if normalized.empty else normalized["date"].min().strftime("%Y-%m-%d"),
        "date_max": None if normalized.empty else normalized["date"].max().strftime("%Y-%m-%d"),
        "duplicate_dates": 0,
        "missing_close_rows": 0,
    }
    _write_json(report_path, report)
    return report


def _stage_report_dir(plan: ExperimentPlan, name: str) -> Path:
    return plan.paths.report_root / "stages" / name


def _stage_manifest_path(plan: ExperimentPlan, name: str) -> Path:
    if name in {"topix_normalization", "weekly_schedule"}:
        return plan.paths.report_root / "stages" / f"{name}.json"
    return _stage_report_dir(plan, name) / "manifest.json"


def _stage_call(plan: ExperimentPlan, name: str, configs: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = plan.inputs
    paths = plan.paths
    reports = lambda stage: _stage_report_dir(plan, stage)
    if name == "preprocessing":
        return run_preprocessing(
            raw_dir=inputs["prices"],
            output_dir=paths.prices,
            report_dir=reports(name),
            config=configs[name],
            calendar_path=inputs["calendar"],
        )
    if name == "topix_normalization":
        return _run_topix_normalization(
            inputs["topix_raw"], paths.topix, _stage_manifest_path(plan, name)  # type: ignore[arg-type]
        )
    if name == "descriptors":
        return run_descriptors(
            prices_dir=paths.prices,
            calendar_path=inputs["calendar"],
            topix_path=paths.topix,
            output_dir=paths.descriptors,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "weekly_schedule":
        return {
            "status": "complete",
            **run_weekly_schedule(
                calendar_path=inputs["calendar"],
                output_path=paths.weekly_schedule,
                report_path=_stage_manifest_path(plan, name),
                config=configs["weekly_dataset"],
            ),
        }
    if name == "universe":
        return run_universe(
            listed_info_dir=inputs["listed_info"],
            prices_dir=paths.prices,
            calendar_path=inputs["calendar"],
            schedule_path=paths.weekly_schedule,
            output_dir=paths.universe,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "earnings_events":
        return run_earnings_events(
            candidates_dir=paths.universe,
            earnings_schedule_dir=inputs["earnings_schedule"],
            financial_results_dir=inputs["financial_results"],
            calendar_path=inputs["calendar"],
            output_dir=paths.earnings_events,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "weekly_dataset":
        return run_weekly_dataset(
            prices_dir=paths.prices,
            descriptors_dir=paths.descriptors,
            universe_dir=paths.universe,
            earnings_dir=paths.earnings_events,
            calendar_path=inputs["calendar"],
            topix_path=paths.topix,
            output_dir=paths.weekly_dataset,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "splits":
        return run_splits(
            input_dir=paths.weekly_dataset,
            calendar_path=inputs["calendar"],
            output_dir=paths.splits,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "modeling":
        return run_modeling(
            input_dir=paths.weekly_dataset,
            calendar_path=inputs["calendar"],
            output_dir=paths.modeling,
            report_dir=reports(name),
            config=configs[name],
            split_config=configs["splits"],
        )
    if name == "evaluation":
        return run_evaluation(
            predictions_path=paths.modeling / "predictions.parquet",
            industry_dir=inputs["listed_info"],
            output_dir=paths.evaluation,
            report_dir=reports(name),
            config=configs[name],
        )
    if name == "backtest":
        return run_backtest(
            predictions_path=paths.modeling / "predictions.parquet",
            candidates_dir=paths.weekly_dataset,
            prices_dir=paths.prices,
            calendar_path=inputs["calendar"],
            delistings_path=inputs["delistings"],
            output_dir=paths.backtest,
            report_dir=reports(name),
            config=configs[name],
        )
    raise AssertionError(f"Unknown stage: {name}")


def _markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 30) -> str:
    selected = [name for name in columns if name in frame.columns]
    if frame.empty or not selected:
        return "該当する保存結果はありません。"
    visible = frame.loc[:, selected].head(limit)
    lines = [
        "| " + " | ".join(selected) + " |",
        "| " + " | ".join("---" for _ in selected) + " |",
    ]
    for row in visible.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    if len(frame) > limit:
        lines.append(f"\n先頭{limit}行のみ表示（全{len(frame)}行）。")
    return "\n".join(lines)


def _read_result_table(path: Path, columns: Sequence[str]) -> str:
    if not path.is_file():
        return "工程が未完了のため、結果表はありません。"
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # Report generation must preserve the primary failure record.
        return f"結果表を読み取れませんでした: `{type(exc).__name__}: {exc}`"
    return _markdown_table(frame, columns)


def render_experiment_report(manifest: Mapping[str, Any]) -> str:
    """Render a compact Japanese Markdown report from saved experiment state."""

    outputs = manifest.get("outputs", {})
    stages = manifest.get("stages", [])
    input_ids = manifest.get("input_identifiers", {})
    lines = [
        f"# 実験レポート: {manifest.get('experiment_name', 'unknown')}",
        "",
        f"- run ID: `{manifest.get('run_id', 'unknown')}`",
        f"- 状態: `{manifest.get('status', 'unknown')}`",
        f"- 開始（UTC）: `{manifest.get('started_at_utc', '—')}`",
        f"- 終了（UTC）: `{manifest.get('finished_at_utc', '—')}`",
        f"- 乱数seed: `{manifest.get('random_seed', '—')}`",
        "",
        "> このレポートは2016〜2024年の開発期間におけるモデル比較・予測評価・売買シミュレーションです。"
        "2025〜2026年の隔離期間を用いる最終バックテストではなく、将来の利益を保証しません。",
        "",
        "## 入力識別子",
        "",
    ]
    for name in ("price_snapshot", "auxiliary_snapshot"):
        identifier = input_ids.get(name, {})
        lines.extend(
            [
                f"### {name}",
                "",
                f"- path: `{identifier.get('path', '—')}`",
                f"- snapshot ID: `{identifier.get('snapshot_id', '—')}`",
                f"- manifest SHA256: `{identifier.get('manifest_sha256', '—')}`",
                f"- manifest内の宣言済みSHA256数: `{identifier.get('declared_sha256_count', '—')}`",
                f"- 取得範囲・条件: `{json.dumps(identifier.get('request'), ensure_ascii=False, sort_keys=True)}`",
                f"- manifest集計: `{json.dumps(identifier.get('totals'), ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(["## 設定識別子", "", "| 工程 | 設定ファイル | SHA256 |", "| --- | --- | --- |"])
    for name, value in manifest.get("stage_configs", {}).items():
        lines.append(
            f"| {_markdown_value(name)} | `{_markdown_value(value.get('path'))}` | "
            f"`{_markdown_value(value.get('sha256'))}` |"
        )
    lines.extend(["", "## 工程", "", "| 工程 | 状態 | 開始（UTC） | 終了（UTC） |", "| --- | --- | --- | --- |"])
    for stage in stages:
        lines.append(
            "| "
            + " | ".join(
                _markdown_value(stage.get(key))
                for key in ("name", "status", "started_at_utc", "finished_at_utc")
            )
            + " |"
        )
    if manifest.get("error"):
        error = manifest["error"]
        lines.extend(
            [
                "",
                "## 停止理由",
                "",
                f"`{error.get('type', 'Error')}: {error.get('message', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 開発期間の予測評価",
            "",
            _read_result_table(
                Path(outputs.get("evaluation", "")) / "summary_metrics.parquet",
                (
                    "model",
                    "target_column",
                    "signal_dates",
                    "eligible_rows",
                    "target_coverage",
                    "prediction_coverage",
                    "scoring_coverage",
                    "spearman_ic_mean",
                    "top_n_mean_target_mean",
                    "top_bottom_spread_mean",
                ),
            ),
            "",
            "## 開発期間の売買シミュレーション",
            "",
            _read_result_table(
                Path(outputs.get("backtest", "")) / "summary_metrics.parquet",
                (
                    "model",
                    "target_column",
                    "date_min",
                    "date_max",
                    "initial_cash_yen",
                    "ending_equity_yen",
                    "cumulative_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio_zero_risk_free",
                    "maximum_drawdown",
                    "entry_fill_coverage",
                    "total_transaction_cost_yen",
                    "main_result_complete",
                ),
            ),
            "",
            "## 警告・制限",
            "",
        ]
    )
    for warning in manifest.get("warnings", []):
        lines.append(f"- {warning}")
    lines.extend(
        [
            "- 入力rawスナップショットは変更せず、派生データと成果物を別ディレクトリへ保存しています。",
            "- 個別工程の件数、欠損、coverage、除外理由、入力SHA256は各工程のmanifestを確認してください。",
            "- バックテストは板・注文優先順位を再現せず、配当と税金を含まない価格リターンの研究用シミュレーションです。",
            "",
            "## 保存先",
            "",
        ]
    )
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def run_experiment(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> ExperimentResult:
    """Run every implemented research stage and persist progress on interruption.

    Existing run directories are never resumed or overwritten.  A stage failure
    stops all dependent stages, while the top-level and stage-level manifests
    already written remain available for diagnosis.
    """

    plan = plan_experiment(config, run_id=run_id, project_root=project_root)
    configs = plan.stage_config_objects
    plan.paths.report_root.mkdir(parents=True, exist_ok=False)
    manifest_path = plan.paths.report_root / "manifest.json"
    report_path = plan.paths.report_root / "experiment_report.md"
    resolved_config_path = plan.paths.report_root / "config.resolved.json"
    _write_json(resolved_config_path, _resolved_config(plan))
    source_config = None
    if config.source_path is not None:
        source_config = {"path": str(config.source_path), "sha256": _sha256(config.source_path)}
    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "running",
        "experiment_name": config.experiment_name,
        "run_id": plan.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "random_seed": config.random_seed,
        "randomized_stages": [],
        "development_period_only": True,
        "final_period_used_for_model_selection": False,
        "final_period_evaluated": False,
        "final_period_prepared_rows": None,
        "source_config": source_config,
        "resolved_config": {"path": str(resolved_config_path), "sha256": _sha256(resolved_config_path)},
        "input_identifiers": plan.input_identifiers,
        "stage_configs": plan.stage_config_values,
        "stage_order": list(STAGE_ORDER),
        "stages": [],
        "outputs": {
            "interim": str(plan.paths.interim_root),
            "processed": str(plan.paths.processed_root),
            "artifacts": str(plan.paths.artifact_root),
            "reports": str(plan.paths.report_root),
            "modeling": str(plan.paths.modeling),
            "evaluation": str(plan.paths.evaluation),
            "backtest": str(plan.paths.backtest),
        },
        "warnings": list(plan.warnings),
        "code_sha256": _source_hashes(plan.project_root),
        "versions": _versions(),
    }
    _write_json(manifest_path, manifest)
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    incomplete = False
    try:
        for name in STAGE_ORDER:
            stage: dict[str, Any] = {
                "name": name,
                "status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "finished_at_utc": None,
                "manifest_path": str(_stage_manifest_path(plan, name)),
            }
            manifest["stages"].append(stage)
            _write_json(manifest_path, manifest)
            LOGGER.info("experiment %s: starting %s", plan.run_id, name)
            try:
                stage_manifest = dict(_stage_call(plan, name, configs))
            except BaseException as exc:
                stage["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                stage["error_type"] = type(exc).__name__
                stage["error_message"] = str(exc)
                stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                stage_manifest_path = _stage_manifest_path(plan, name)
                if stage_manifest_path.is_file():
                    stage["manifest_sha256"] = _sha256(stage_manifest_path)
                raise
            stage_status = str(stage_manifest.get("status", "complete"))
            if stage_status not in COMPLETE_STATUSES:
                if name == "backtest" and stage_status == "incomplete_open_positions":
                    incomplete = True
                else:
                    stage["status"] = stage_status
                    stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                    raise RuntimeError(f"Stage {name} returned non-complete status {stage_status!r}")
            stage["status"] = stage_status
            stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            stage["manifest_sha256"] = _sha256(_stage_manifest_path(plan, name))
            stage["summary"] = _stage_summary(stage_manifest)
            if name == "splits":
                isolation = stage_manifest.get("isolation", {})
                manifest["final_period_prepared_rows"] = isolation.get("rows")
            _write_json(manifest_path, manifest)
            LOGGER.info("experiment %s: finished %s (%s)", plan.run_id, name, stage_status)
        manifest["status"] = "incomplete" if incomplete else "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
            _write_text(report_path, render_experiment_report(manifest))
            manifest["experiment_report"] = {"path": str(report_path), "sha256": _sha256(report_path)}
            _write_json(manifest_path, manifest)
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)
    return ExperimentResult(
        run_id=plan.run_id,
        manifest_path=manifest_path,
        report_path=report_path,
        manifest=manifest,
        paths=plan.paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for validation or an immutable full development run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Strict experiment JSON")
    parser.add_argument("--run-id", help="Optional immutable output identifier")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs/configs and print the plan without creating outputs",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.validate_only:
            plan = plan_experiment(config, run_id=args.run_id)
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "run_id": plan.run_id,
                        "stage_order": list(STAGE_ORDER),
                        "warnings": list(plan.warnings),
                        "outputs": plan.paths.as_dict(),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        result = run_experiment(config, run_id=args.run_id)
    except KeyboardInterrupt:
        LOGGER.error("Experiment interrupted; inspect the saved top-level manifest and report")
        return 130
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(
        json.dumps(
            {
                "status": result.manifest["status"],
                "run_id": result.run_id,
                "manifest": str(result.manifest_path),
                "report": str(result.report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
