"""Download immutable J-Quants V2 auxiliary-data snapshots.

This module acquires the point-in-time inputs which are intentionally absent
from :mod:`stock_prediction.data_download`: TOPIX daily bars, historical
listed-issue master snapshots, earnings-date publications, and actual
financial disclosures.  The same functions are used from notebooks and the
CLI.  Provider rows are validated before they are written and credentials are
never persisted.

Observed security-master transitions are also written as derived audit data.
They describe appearances, disappearances, and attribute changes between two
consecutive requested TSE sessions.  A same-name old/new-code pair is only a
possible code-change candidate; it is not an authoritative corporate-
action mapping and must not be treated as one without another source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import platform
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from stock_prediction.data_download import (
    API_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RATE_LIMIT_WAIT_SECONDS,
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_TIMEOUT_SECONDS,
    OPEN_HOLIDAY_DIVISIONS,
    DownloadSchemaError,
    JQuantsV2Client,
    RawDownloadError,
    resolve_api_key,
    validate_calendar,
)


LOGGER = logging.getLogger(__name__)

TOPIX_PATH = "/indices/bars/daily/topix"
LISTED_INFO_PATH = "/equities/master"
EARNINGS_SCHEDULE_PATH = "/fins/earnings-date"
FINANCIAL_RESULTS_PATH = "/fins/summary"

TOPIX_DATASET = "topix"
LISTED_INFO_DATASET = "listed_info"
EARNINGS_SCHEDULE_DATASET = "earnings_schedule"
FINANCIAL_RESULTS_DATASET = "financial_results"
DOWNLOAD_DATASETS = (
    TOPIX_DATASET,
    LISTED_INFO_DATASET,
    EARNINGS_SCHEDULE_DATASET,
    FINANCIAL_RESULTS_DATASET,
)

TOPIX_COLUMNS = ("Date", "O", "H", "L", "C")
LISTED_INFO_COLUMNS = (
    "Date",
    "Code",
    "CoName",
    "CoNameEn",
    "S17",
    "S17Nm",
    "S33",
    "S33Nm",
    "ScaleCat",
    "Mkt",
    "MktNm",
    "Mrgn",
    "MrgnNm",
    "ProdCat",
)
LISTED_INFO_REQUIRED_COLUMNS = tuple(column for column in LISTED_INFO_COLUMNS if column != "ProdCat")
EARNINGS_SCHEDULE_COLUMNS = (
    "PubDate",
    "SchDate",
    "FQName",
    "FYE",
    "Code",
    "CoName",
    "CoNameEn",
)

# Current J-Quants V2 /fins/summary schema.  Identity/publication columns are
# required.  Known financial fields are optional so an older subscription
# response which omits all-empty fields remains readable, while unknown fields
# still stop the download for an explicit schema review.  The API returns every
# field as a JSON string, including financial values, and represents missing
# values with the empty string.  Raw value fields therefore remain strings;
# numeric conversion belongs to a downstream, explicitly audited transform.
FINANCIAL_RESULTS_COLUMNS = (
    "DiscDate",
    "DiscTime",
    "Code",
    "DiscNo",
    "DocType",
    "CurPerType",
    "CurPerSt",
    "CurPerEn",
    "CurFYSt",
    "CurFYEn",
    "NxtFYSt",
    "NxtFYEn",
    "Sales",
    "OP",
    "OdP",
    "NP",
    "EPS",
    "DEPS",
    "TA",
    "Eq",
    "EqAR",
    "BPS",
    "CFO",
    "CFI",
    "CFF",
    "CashEq",
    "Div1Q",
    "Div2Q",
    "Div3Q",
    "DivFY",
    "DivAnn",
    "DivUnit",
    "DivTotalAnn",
    "PayoutRatioAnn",
    "FDiv1Q",
    "FDiv2Q",
    "FDiv3Q",
    "FDivFY",
    "FDivAnn",
    "FDivUnit",
    "FDivTotalAnn",
    "FPayoutRatioAnn",
    "NxFDiv1Q",
    "NxFDiv2Q",
    "NxFDiv3Q",
    "NxFDivFY",
    "NxFDivAnn",
    "NxFDivUnit",
    "NxFPayoutRatioAnn",
    "FSales2Q",
    "FOP2Q",
    "FOdP2Q",
    "FNP2Q",
    "FEPS2Q",
    "NxFSales2Q",
    "NxFOP2Q",
    "NxFOdP2Q",
    "NxFNp2Q",
    "NxFEPS2Q",
    "FSales",
    "FOP",
    "FOdP",
    "FNP",
    "FEPS",
    "NxFSales",
    "NxFOP",
    "NxFOdP",
    "NxFNp",
    "NxFEPS",
    "MatChgSub",
    "SigChgInC",
    "ChgByASRev",
    "ChgNoASRev",
    "ChgAcEst",
    "RetroRst",
    "ShOutFY",
    "TrShFY",
    "AvgSh",
    "NCSales",
    "NCOP",
    "NCOdP",
    "NCNP",
    "NCEPS",
    "NCTA",
    "NCEq",
    "NCEqAR",
    "NCBPS",
    "FNCSales2Q",
    "FNCOP2Q",
    "FNCOdP2Q",
    "FNCNP2Q",
    "FNCEPS2Q",
    "NxFNCSales2Q",
    "NxFNCOP2Q",
    "NxFNCOdP2Q",
    "NxFNCNP2Q",
    "NxFNCEPS2Q",
    "FNCSales",
    "FNCOP",
    "FNCOdP",
    "FNCNP",
    "FNCEPS",
    "NxFNCSales",
    "NxFNCOP",
    "NxFNCOdP",
    "NxFNCNP",
    "NxFNCEPS",
    "ShEq",
    "NCShEq",
    "ROE",
    "NCROE",
)
FINANCIAL_RESULTS_REQUIRED_COLUMNS = ("DiscDate", "DiscTime", "Code", "DiscNo", "DocType")
FINANCIAL_DATE_COLUMNS = (
    "DiscDate",
    "CurPerSt",
    "CurPerEn",
    "CurFYSt",
    "CurFYEn",
    "NxtFYSt",
    "NxtFYEn",
)
FINANCIAL_STRING_COLUMNS = frozenset(
    column
    for column in FINANCIAL_RESULTS_COLUMNS
    if column not in set(FINANCIAL_DATE_COLUMNS)
)

MASTER_TRANSITION_COLUMNS = (
    "ObservedDate",
    "PreviousObservedDate",
    "Code",
    "ChangeType",
    "ChangedFields",
    "BeforeRecord",
    "AfterRecord",
)
CODE_CHANGE_CANDIDATE_COLUMNS = (
    "ObservedDate",
    "PreviousObservedDate",
    "OldCode",
    "NewCode",
    "CoName",
    "Evidence",
)


class AuxiliaryDataClient(Protocol):
    """Interface shared by the live client and deterministic tests/notebooks."""

    def get_topix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return TOPIX daily bars for the inclusive range."""

    def get_listed_info(self, as_of_date: str) -> pd.DataFrame:
        """Return the listed-issue master as of one TSE session."""

    def get_earnings_schedule(self, publication_date: str) -> pd.DataFrame:
        """Return schedule rows published on one calendar date."""

    def get_financial_results(self, disclosure_date: str) -> pd.DataFrame:
        """Return financial-summary disclosures on one calendar date."""


@dataclass(frozen=True)
class AuxiliaryDownloadConfig:
    """Settings for one immutable auxiliary-data snapshot.

    ``start_date`` and ``end_date`` are inclusive Japanese market dates.
    Listed-info queries use TSE sessions from the supplied official calendar;
    earnings schedule and financial-result queries use every calendar date so
    a non-session publication cannot be silently omitted.
    """

    start_date: str
    end_date: str
    datasets: tuple[str, ...] = DOWNLOAD_DATASETS
    output_root: Path | str = Path("data/raw")
    output_dir: Path | str | None = None
    requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    rate_limit_wait_seconds: float = DEFAULT_RATE_LIMIT_WAIT_SECONDS

    def __post_init__(self) -> None:
        start = _parse_iso_date(self.start_date, "start_date")
        end = _parse_iso_date(self.end_date, "end_date")
        if start > end:
            raise ValueError("start_date must not be later than end_date")
        if isinstance(self.datasets, str) or not isinstance(self.datasets, tuple):
            raise ValueError("datasets must be a tuple of dataset names")
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        unknown = set(self.datasets) - set(DOWNLOAD_DATASETS)
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets must not contain duplicates")
        for name in ("requests_per_minute", "timeout_seconds", "rate_limit_wait_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")


@dataclass(frozen=True)
class AuxiliaryDownloadResult:
    """Paths and final manifest produced by a completed download."""

    output_dir: Path
    calendar_path: Path
    data_paths: dict[str, tuple[Path, ...]]
    manifest: dict[str, Any]


class AuxiliaryJQuantsV2Client(JQuantsV2Client):
    """J-Quants V2 client methods for auxiliary research inputs."""

    def get_topix(self, start_date: str, end_date: str) -> pd.DataFrame:
        records = self._get_paginated(TOPIX_PATH, {"from": start_date, "to": end_date})
        return pd.DataFrame.from_records(records)

    def get_listed_info(self, as_of_date: str) -> pd.DataFrame:
        records = self._get_paginated(LISTED_INFO_PATH, {"date": as_of_date})
        return pd.DataFrame.from_records(records)

    def get_earnings_schedule(self, publication_date: str) -> pd.DataFrame:
        records = self._get_paginated(EARNINGS_SCHEDULE_PATH, {"date": publication_date})
        return pd.DataFrame.from_records(records)

    def get_financial_results(self, disclosure_date: str) -> pd.DataFrame:
        records = self._get_paginated(FINANCIAL_RESULTS_PATH, {"date": disclosure_date})
        return pd.DataFrame.from_records(records)


def load_trading_calendar(path: Path | str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load a CSV/Parquet J-Quants calendar and select the requested range.

    A source file may cover a wider range.  Its complete schema, date validity,
    and uniqueness are checked before selecting the inclusive requested dates.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Trading calendar does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source, dtype={"HolDiv": "string"})
    else:
        raise ValueError("Trading calendar must be a .parquet or .csv file")
    if set(frame.columns) != {"Date", "HolDiv"}:
        raise DownloadSchemaError("Trading calendar columns must be exactly Date and HolDiv")
    parsed = _parse_date_series(frame["Date"], "calendar.Date")
    if parsed.isna().any() or parsed.duplicated().any():
        raise DownloadSchemaError("Trading calendar dates must be non-null and unique")
    selected = frame.loc[
        parsed.between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
        ["Date", "HolDiv"],
    ].copy()
    return validate_calendar(selected, start_date, end_date)


def validate_topix(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    expected_sessions: Sequence[pd.Timestamp] | pd.Series,
) -> pd.DataFrame:
    """Validate raw TOPIX OHLC rows and exact requested-session coverage."""

    expected = pd.DatetimeIndex(expected_sessions).normalize().sort_values()
    output = _validate_schema(
        frame,
        TOPIX_COLUMNS,
        (),
        "TOPIX",
        allow_empty=len(expected) == 0,
    )
    if output.empty:
        return _empty_frame(TOPIX_COLUMNS, date_columns={"Date"}, numeric_columns={"O", "H", "L", "C"})
    output["Date"] = _parse_date_series(output["Date"], "TOPIX.Date")
    for column in ("O", "H", "L", "C"):
        _coerce_numeric(output, column, f"TOPIX.{column}")
        if output[column].isna().any() or (output[column] <= 0).any():
            raise DownloadSchemaError(f"TOPIX.{column} must be positive and non-null")
    if output.duplicated("Date", keep=False).any():
        raise DownloadSchemaError("TOPIX response contains duplicate dates")
    outside = ~output["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    if outside.any():
        raise DownloadSchemaError("TOPIX response contains dates outside the requested range")
    if ((output["L"] > output[["O", "C"]].min(axis=1)) | (output["H"] < output[["O", "C"]].max(axis=1))).any():
        raise DownloadSchemaError("TOPIX response violates Low <= Open/Close <= High")
    actual = pd.DatetimeIndex(output["Date"]).sort_values()
    if not actual.equals(expected):
        missing = expected.difference(actual).strftime("%Y-%m-%d").tolist()
        unexpected = actual.difference(expected).strftime("%Y-%m-%d").tolist()
        raise DownloadSchemaError(
            "TOPIX dates must exactly match requested TSE sessions; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}",
        )
    output.sort_values("Date", inplace=True, ignore_index=True)
    return output


def validate_listed_info(frame: pd.DataFrame, requested_date: str) -> pd.DataFrame:
    """Validate one point-in-time listed-issue master response."""

    optional = tuple(column for column in LISTED_INFO_COLUMNS if column not in LISTED_INFO_REQUIRED_COLUMNS)
    output = _validate_schema(
        frame,
        LISTED_INFO_REQUIRED_COLUMNS,
        optional,
        "listed_info",
        allow_empty=False,
    )
    output["Date"] = _parse_date_series(output["Date"], "listed_info.Date")
    if output["Date"].isna().any() or not output["Date"].eq(pd.Timestamp(requested_date)).all():
        raise DownloadSchemaError(f"Listed-info response does not match requested date {requested_date}")
    for column in output.columns:
        if column != "Date":
            _coerce_string(output, column, f"listed_info.{column}")
    _validate_codes(output["Code"], "listed_info.Code")
    if output.duplicated(["Date", "Code"], keep=False).any():
        raise DownloadSchemaError("Listed-info response contains duplicate Date/Code keys")
    output.sort_values(["Code", "Date"], inplace=True, ignore_index=True)
    return output


def validate_earnings_schedule(frame: pd.DataFrame, publication_date: str) -> pd.DataFrame:
    """Validate schedule revisions published on one calendar date."""

    output = _validate_schema(
        frame,
        EARNINGS_SCHEDULE_COLUMNS,
        (),
        "earnings_schedule",
        allow_empty=True,
    )
    if output.empty:
        return _empty_frame(EARNINGS_SCHEDULE_COLUMNS, date_columns={"PubDate", "SchDate"})
    output["PubDate"] = _parse_date_series(output["PubDate"], "earnings_schedule.PubDate")
    output["SchDate"] = _parse_nullable_date_series(
        output["SchDate"],
        "earnings_schedule.SchDate",
    )
    if output["PubDate"].isna().any() or not output["PubDate"].eq(pd.Timestamp(publication_date)).all():
        raise DownloadSchemaError(
            f"Earnings-schedule response does not match publication date {publication_date}",
        )
    for column in set(EARNINGS_SCHEDULE_COLUMNS) - {"PubDate", "SchDate"}:
        _coerce_string(output, column, f"earnings_schedule.{column}")
    _validate_codes(output["Code"], "earnings_schedule.Code")
    if output.duplicated(keep=False).any():
        raise DownloadSchemaError("Earnings-schedule response contains exact duplicate rows")
    output.sort_values(["PubDate", "Code", "SchDate"], inplace=True, ignore_index=True)
    return output


def validate_financial_results(frame: pd.DataFrame, disclosure_date: str) -> pd.DataFrame:
    """Validate financial-summary rows disclosed on one calendar date."""

    optional = tuple(
        column for column in FINANCIAL_RESULTS_COLUMNS if column not in FINANCIAL_RESULTS_REQUIRED_COLUMNS
    )
    output = _validate_schema(
        frame,
        FINANCIAL_RESULTS_REQUIRED_COLUMNS,
        optional,
        "financial_results",
        allow_empty=True,
    )
    if output.empty:
        return _empty_financial_results()
    for column in set(FINANCIAL_DATE_COLUMNS).intersection(output.columns):
        output[column] = _parse_nullable_date_series(output[column], f"financial_results.{column}")
    if output["DiscDate"].isna().any() or not output["DiscDate"].eq(pd.Timestamp(disclosure_date)).all():
        raise DownloadSchemaError(
            f"Financial-results response does not match disclosure date {disclosure_date}",
        )
    for column in FINANCIAL_STRING_COLUMNS.intersection(output.columns):
        _coerce_string(output, column, f"financial_results.{column}")
    _validate_codes(output["Code"], "financial_results.Code")
    times = output["DiscTime"].dropna()
    if not times.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?").all():
        raise DownloadSchemaError("financial_results.DiscTime must be an HH:MM or HH:MM:SS string")
    if output.duplicated(keep=False).any():
        raise DownloadSchemaError("Financial-results response contains exact duplicate rows")
    if output.duplicated(["DiscDate", "DiscNo"], keep=False).any():
        raise DownloadSchemaError("Financial-results response contains duplicate DiscDate/DiscNo keys")
    output.sort_values(["DiscDate", "DiscTime", "Code", "DiscNo"], inplace=True, ignore_index=True)
    return output


def derive_master_transitions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare consecutive validated master snapshots.

    Returns observed transitions and conservative possible-code-change
    candidates.  The candidates require a unique exact Japanese company-name
    match among codes disappearing and appearing on the same boundary.  They
    are explicitly non-authoritative because the endpoint has no stable issuer
    identifier linking the two codes.
    """

    previous_date = _single_master_date(previous, "previous")
    current_date = _single_master_date(current, "current")
    if previous_date >= current_date:
        raise ValueError("current master date must be later than previous master date")
    previous_by_code = previous.set_index("Code", drop=False)
    current_by_code = current.set_index("Code", drop=False)
    previous_codes = set(previous_by_code.index.astype(str))
    current_codes = set(current_by_code.index.astype(str))
    removed_codes = sorted(previous_codes - current_codes)
    appeared_codes = sorted(current_codes - previous_codes)
    common_codes = sorted(previous_codes & current_codes)
    rows: list[dict[str, Any]] = []

    for code in removed_codes:
        before = previous_by_code.loc[code]
        rows.append(
            _transition_record(current_date, previous_date, code, "disappeared", (), before, None),
        )
    for code in appeared_codes:
        after = current_by_code.loc[code]
        rows.append(
            _transition_record(current_date, previous_date, code, "appeared", (), None, after),
        )
    comparable_columns = [column for column in LISTED_INFO_COLUMNS if column not in {"Date", "Code"}]
    comparable_columns = [
        column for column in comparable_columns if column in previous.columns and column in current.columns
    ]
    for code in common_codes:
        before = previous_by_code.loc[code]
        after = current_by_code.loc[code]
        changed = tuple(
            column for column in comparable_columns if not _scalars_equal(before[column], after[column])
        )
        if changed:
            rows.append(
                _transition_record(
                    current_date,
                    previous_date,
                    code,
                    "attributes_changed",
                    changed,
                    before,
                    after,
                ),
            )

    transitions = pd.DataFrame.from_records(rows, columns=MASTER_TRANSITION_COLUMNS)
    transitions = _cast_transition_frame(transitions)

    candidate_rows: list[dict[str, Any]] = []
    removed_by_name = _unique_codes_by_company_name(previous_by_code, removed_codes)
    appeared_by_name = _unique_codes_by_company_name(current_by_code, appeared_codes)
    for company_name in sorted(set(removed_by_name) & set(appeared_by_name)):
        old_codes = removed_by_name[company_name]
        new_codes = appeared_by_name[company_name]
        if len(old_codes) == 1 and len(new_codes) == 1:
            candidate_rows.append(
                {
                    "ObservedDate": current_date,
                    "PreviousObservedDate": previous_date,
                    "OldCode": old_codes[0],
                    "NewCode": new_codes[0],
                    "CoName": company_name,
                    "Evidence": "unique_exact_company_name",
                },
            )
    candidates = pd.DataFrame.from_records(candidate_rows, columns=CODE_CHANGE_CANDIDATE_COLUMNS)
    candidates = _cast_code_candidate_frame(candidates)
    return transitions, candidates


def download_auxiliary_data(
    config: AuxiliaryDownloadConfig,
    *,
    trading_calendar: pd.DataFrame,
    api_key: str | None = None,
    client: AuxiliaryDataClient | None = None,
    calendar_source_path: Path | str | None = None,
) -> AuxiliaryDownloadResult:
    """Download and save one immutable auxiliary-data snapshot.

    The supplied calendar may cover a wider period when this function is used
    from a notebook; only the configured inclusive range is stored.  On a
    normal exception or ``KeyboardInterrupt``, completed partitions, the
    current partial partition, and failure progress are persisted before the
    exception is re-raised.
    """

    calendar = _select_and_validate_calendar(
        trading_calendar,
        config.start_date,
        config.end_date,
    )
    secret = ""
    owned_client: AuxiliaryJQuantsV2Client | None = None
    if client is None:
        secret = resolve_api_key(api_key)
        owned_client = AuxiliaryJQuantsV2Client(
            secret,
            requests_per_minute=float(config.requests_per_minute),
            timeout_seconds=float(config.timeout_seconds),
            max_retries=config.max_retries,
            rate_limit_wait_seconds=float(config.rate_limit_wait_seconds),
        )
        client = owned_client
    elif api_key:
        secret = api_key

    try:
        output_dir = _create_output_directory(config)
    except BaseException:
        if owned_client is not None:
            owned_client.close()
        raise
    manifest_path = output_dir / "manifest.json"
    started_at = _utc_now()
    manifest = _initial_manifest(config, output_dir, started_at, calendar_source_path)
    _write_json(manifest_path, manifest)
    calendar_path = output_dir / "trading_calendar.parquet"
    data_paths: dict[str, list[Path]] = {dataset: [] for dataset in config.datasets}
    current_context: dict[str, Any] | None = None

    try:
        _write_parquet_once(calendar, calendar_path)
        manifest["calendar"] = {
            **_frame_stats(calendar, date_column="Date", key_columns=("Date",)),
            "path": calendar_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(calendar_path),
            "tse_trading_days": int(calendar["HolDiv"].isin(OPEN_HOLIDAY_DIVISIONS).sum()),
        }
        _touch_manifest(manifest_path, manifest)

        trading_dates = [
            value.date().isoformat()
            for value in calendar.loc[calendar["HolDiv"].isin(OPEN_HOLIDAY_DIVISIONS), "Date"]
        ]
        calendar_dates = [value.date().isoformat() for value in calendar["Date"]]

        if TOPIX_DATASET in config.datasets:
            current_context = {"dataset": TOPIX_DATASET, "year": None, "frames": [], "dates": []}
            dataset_manifest = manifest["datasets"][TOPIX_DATASET]
            dataset_manifest["status"] = "running"
            dataset_manifest["requested_dates"] = len(trading_dates)
            _touch_manifest(manifest_path, manifest)
            raw_topix = client.get_topix(config.start_date, config.end_date)
            sessions = calendar.loc[calendar["HolDiv"].isin(OPEN_HOLIDAY_DIVISIONS), "Date"]
            topix = validate_topix(raw_topix, config.start_date, config.end_date, sessions)
            topix_path = output_dir / "topix.parquet"
            _write_parquet_once(topix, topix_path)
            data_paths[TOPIX_DATASET].append(topix_path)
            dataset_manifest.update(
                {
                    "status": "completed",
                    "completed_dates": len(topix),
                    "rows": len(topix),
                    "path": topix_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(topix_path),
                    "stats": _frame_stats(topix, date_column="Date", key_columns=("Date",)),
                },
            )
            current_context = None
            _touch_manifest(manifest_path, manifest)

        if LISTED_INFO_DATASET in config.datasets:
            current_context = _download_listed_info(
                client,
                trading_dates,
                output_dir,
                manifest_path,
                manifest,
                data_paths[LISTED_INFO_DATASET],
            )
            current_context = None

        for dataset, query_dates, getter, validator, date_column, key_columns in (
            (
                EARNINGS_SCHEDULE_DATASET,
                calendar_dates,
                client.get_earnings_schedule,
                validate_earnings_schedule,
                "PubDate",
                ("PubDate", "Code", "FQName", "SchDate"),
            ),
            (
                FINANCIAL_RESULTS_DATASET,
                calendar_dates,
                client.get_financial_results,
                validate_financial_results,
                "DiscDate",
                ("DiscDate", "DiscNo"),
            ),
        ):
            if dataset not in config.datasets:
                continue
            current_context = _download_dated_dataset(
                dataset=dataset,
                query_dates=query_dates,
                getter=getter,
                validator=validator,
                output_dir=output_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                paths=data_paths[dataset],
                date_column=date_column,
                key_columns=key_columns,
            )
            current_context = None

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_iso(_utc_now())
        _touch_manifest(manifest_path, manifest)
        return AuxiliaryDownloadResult(
            output_dir=output_dir,
            calendar_path=calendar_path,
            data_paths={name: tuple(paths) for name, paths in data_paths.items()},
            manifest=manifest,
        )
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = _redact(str(exc), secret)
        failure_context = getattr(exc, "_auxiliary_context", current_context)
        if failure_context is not None:
            dataset = str(failure_context["dataset"])
            manifest["datasets"][dataset]["status"] = manifest["status"]
            _persist_partial_context(failure_context, output_dir, manifest)
        manifest["completed_at_utc"] = _utc_iso(_utc_now())
        _touch_manifest(manifest_path, manifest)
        raise
    finally:
        if owned_client is not None:
            owned_client.close()


def _download_listed_info(
    client: AuxiliaryDataClient,
    trading_dates: Sequence[str],
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    paths: list[Path],
) -> dict[str, Any] | None:
    dataset_manifest = manifest["datasets"][LISTED_INFO_DATASET]
    dataset_manifest["status"] = "running"
    dataset_manifest["requested_dates"] = len(trading_dates)
    _touch_manifest(manifest_path, manifest)
    previous: pd.DataFrame | None = None
    for year, dates in _dates_by_year(trading_dates).items():
        context: dict[str, Any] = {
            "dataset": LISTED_INFO_DATASET,
            "year": year,
            "frames": [],
            "dates": [],
            "transition_frames": [],
            "candidate_frames": [],
        }
        manifest["current_partition"] = {
            "dataset": LISTED_INFO_DATASET,
            "year": year,
            "requested_dates": len(dates),
            "completed_dates": 0,
            "last_completed_date": None,
        }
        _touch_manifest(manifest_path, manifest)
        try:
            for position, query_date in enumerate(dates, start=1):
                current = validate_listed_info(client.get_listed_info(query_date), query_date)
                context["frames"].append(current)
                context["dates"].append(query_date)
                if previous is not None:
                    transitions, candidates = derive_master_transitions(previous, current)
                    context["transition_frames"].append(transitions)
                    context["candidate_frames"].append(candidates)
                previous = current
                manifest["current_partition"]["completed_dates"] = position
                manifest["current_partition"]["last_completed_date"] = query_date
                dataset_manifest["completed_dates"] += 1
                _touch_manifest(manifest_path, manifest)
            listed = _combine_frames(context["frames"], LISTED_INFO_COLUMNS, ("Code", "Date"))
            path = output_dir / LISTED_INFO_DATASET / f"{year}.parquet"
            _write_parquet_once(listed, path)
            paths.append(path)
            transitions = _combine_frames(
                context["transition_frames"],
                MASTER_TRANSITION_COLUMNS,
                ("ObservedDate", "Code", "ChangeType"),
            )
            transition_path = output_dir / "master_transitions" / f"{year}.parquet"
            _write_parquet_once(transitions, transition_path)
            candidates = _combine_frames(
                context["candidate_frames"],
                CODE_CHANGE_CANDIDATE_COLUMNS,
                ("ObservedDate", "OldCode", "NewCode"),
            )
            candidate_path = output_dir / "code_change_candidates" / f"{year}.parquet"
            _write_parquet_once(candidates, candidate_path)
            partition = {
                "year": year,
                "requested_dates": len(dates),
                "completed_dates": len(dates),
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "stats": _frame_stats(listed, date_column="Date", key_columns=("Date", "Code")),
                "master_transitions": {
                    "path": transition_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(transition_path),
                    "rows": len(transitions),
                    "change_type_counts": _value_counts(transitions, "ChangeType"),
                },
                "code_change_candidates": {
                    "path": candidate_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(candidate_path),
                    "rows": len(candidates),
                    "authoritative": False,
                },
            }
            dataset_manifest["completed_partitions"].append(partition)
            dataset_manifest["rows"] += len(listed)
            manifest["current_partition"] = None
            _touch_manifest(manifest_path, manifest)
        except BaseException as exc:
            setattr(exc, "_auxiliary_context", context)
            raise
    dataset_manifest["status"] = "completed"
    _touch_manifest(manifest_path, manifest)
    return None


def _download_dated_dataset(
    *,
    dataset: str,
    query_dates: Sequence[str],
    getter: Callable[[str], pd.DataFrame],
    validator: Callable[[pd.DataFrame, str], pd.DataFrame],
    output_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    paths: list[Path],
    date_column: str,
    key_columns: Sequence[str],
) -> dict[str, Any] | None:
    dataset_manifest = manifest["datasets"][dataset]
    dataset_manifest["status"] = "running"
    dataset_manifest["requested_dates"] = len(query_dates)
    _touch_manifest(manifest_path, manifest)
    for year, dates in _dates_by_year(query_dates).items():
        context: dict[str, Any] = {
            "dataset": dataset,
            "year": year,
            "frames": [],
            "dates": [],
        }
        manifest["current_partition"] = {
            "dataset": dataset,
            "year": year,
            "requested_dates": len(dates),
            "completed_dates": 0,
            "last_completed_date": None,
        }
        _touch_manifest(manifest_path, manifest)
        try:
            for position, query_date in enumerate(dates, start=1):
                validated = validator(getter(query_date), query_date)
                context["frames"].append(validated)
                context["dates"].append(query_date)
                manifest["current_partition"]["completed_dates"] = position
                manifest["current_partition"]["last_completed_date"] = query_date
                dataset_manifest["completed_dates"] += 1
                _touch_manifest(manifest_path, manifest)
            columns = _dataset_columns(dataset)
            combined = _combine_frames(context["frames"], columns, key_columns)
            path = output_dir / dataset / f"{year}.parquet"
            _write_parquet_once(combined, path)
            paths.append(path)
            partition = {
                "year": year,
                "requested_dates": len(dates),
                "completed_dates": len(dates),
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "stats": _frame_stats(combined, date_column=date_column, key_columns=key_columns),
            }
            dataset_manifest["completed_partitions"].append(partition)
            dataset_manifest["rows"] += len(combined)
            manifest["current_partition"] = None
            _touch_manifest(manifest_path, manifest)
        except BaseException as exc:
            setattr(exc, "_auxiliary_context", context)
            raise
    dataset_manifest["status"] = "completed"
    _touch_manifest(manifest_path, manifest)
    return None


def _persist_partial_context(
    context: Mapping[str, Any],
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    dataset = str(context["dataset"])
    year = context.get("year")
    frames = list(context.get("frames", []))
    dates = list(context.get("dates", []))
    if year is None or not dates:
        return
    try:
        columns = _dataset_columns(dataset)
        key_columns = {
            LISTED_INFO_DATASET: ("Date", "Code"),
            EARNINGS_SCHEDULE_DATASET: ("PubDate", "Code", "FQName", "SchDate"),
            FINANCIAL_RESULTS_DATASET: ("DiscDate", "DiscNo"),
        }[dataset]
        partial = _combine_frames(frames, columns, key_columns)
        partial_path = output_dir / "partial" / dataset / f"{year}.parquet"
        _write_parquet_once(partial, partial_path)
        entry: dict[str, Any] = {
            "year": year,
            "completed_dates": len(dates),
            "last_completed_date": dates[-1],
            "path": partial_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(partial_path),
            "rows": len(partial),
        }
        if dataset == LISTED_INFO_DATASET:
            transitions = _combine_frames(
                list(context.get("transition_frames", [])),
                MASTER_TRANSITION_COLUMNS,
                ("ObservedDate", "Code", "ChangeType"),
            )
            transition_path = output_dir / "partial" / "master_transitions" / f"{year}.parquet"
            _write_parquet_once(transitions, transition_path)
            entry["master_transitions_path"] = transition_path.relative_to(output_dir).as_posix()
            candidates = _combine_frames(
                list(context.get("candidate_frames", [])),
                CODE_CHANGE_CANDIDATE_COLUMNS,
                ("ObservedDate", "OldCode", "NewCode"),
            )
            candidate_path = output_dir / "partial" / "code_change_candidates" / f"{year}.parquet"
            _write_parquet_once(candidates, candidate_path)
            entry["code_change_candidates_path"] = candidate_path.relative_to(output_dir).as_posix()
        manifest["partial_partition"] = entry
    except Exception as partial_exc:
        manifest["partial_write_error"] = str(partial_exc)


def _initial_manifest(
    config: AuxiliaryDownloadConfig,
    output_dir: Path,
    started_at: datetime,
    calendar_source_path: Path | str | None,
) -> dict[str, Any]:
    source_path = None if calendar_source_path is None else Path(calendar_source_path)
    source_record: dict[str, Any] | None = None
    if source_path is not None:
        resolved = source_path.expanduser().resolve()
        source_record = {
            "path": str(resolved),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
        }
    return {
        "manifest_schema_version": 1,
        "snapshot_id": output_dir.name,
        "status": "running",
        "started_at_utc": _utc_iso(started_at),
        "updated_at_utc": _utc_iso(started_at),
        "completed_at_utc": None,
        "provider": "J-Quants API",
        "api_version": "v2",
        "api_base_url": API_BASE_URL,
        "endpoints": {
            TOPIX_DATASET: TOPIX_PATH,
            LISTED_INFO_DATASET: LISTED_INFO_PATH,
            EARNINGS_SCHEDULE_DATASET: EARNINGS_SCHEDULE_PATH,
            FINANCIAL_RESULTS_DATASET: FINANCIAL_RESULTS_PATH,
        },
        "request": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "datasets": list(config.datasets),
            "requests_per_minute": float(config.requests_per_minute),
            "timeout_seconds": float(config.timeout_seconds),
            "max_retries": config.max_retries,
            "rate_limit_wait_seconds": float(config.rate_limit_wait_seconds),
        },
        "time_semantics": {
            "topix": "Date is the market observation date; Close is not known before that session close.",
            "listed_info": "Date is the requested point-in-time master date.",
            "earnings_schedule": "PubDate is availability/publication date; SchDate is the scheduled event date.",
            "financial_results": "DiscDate and DiscTime are disclosure date/time in Japan local time.",
        },
        "master_transition_limitations": {
            "first_requested_session_is_baseline": True,
            "appearance_or_disappearance_is_not_an_authoritative_listing_event": True,
            "code_change_candidates_are_authoritative": False,
            "candidate_rule": "unique exact CoName among disappeared and appeared codes on one boundary",
        },
        "credential_persisted": False,
        "calendar_source": source_record,
        "calendar": None,
        "datasets": {
            dataset: {
                "status": "pending",
                "requested_dates": 0,
                "completed_dates": 0,
                "rows": 0,
                "completed_partitions": [],
            }
            for dataset in config.datasets
        },
        "current_partition": None,
        "partial_partition": None,
        "software": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "pyarrow": _package_version("pyarrow"),
            "requests": _package_version("requests"),
            "module_sha256": _sha256(Path(__file__)),
        },
    }


def _validate_schema(
    frame: pd.DataFrame,
    required: Sequence[str],
    optional: Sequence[str],
    name: str,
    *,
    allow_empty: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DownloadSchemaError(f"{name} response must be a pandas DataFrame")
    if frame.empty and len(frame.columns) == 0:
        if not allow_empty:
            raise DownloadSchemaError(f"{name} returned no rows")
        return pd.DataFrame(columns=required)
    actual = set(frame.columns)
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - actual
    unexpected = actual - allowed
    if missing or unexpected:
        raise DownloadSchemaError(
            f"Unexpected {name} schema; missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )
    if frame.empty and not allow_empty:
        raise DownloadSchemaError(f"{name} returned no rows")
    ordered = [column for column in (*required, *optional) if column in frame.columns]
    return frame.loc[:, ordered].copy(deep=True)


def _select_and_validate_calendar(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DownloadSchemaError("Trading calendar must be a pandas DataFrame")
    if set(frame.columns) != {"Date", "HolDiv"}:
        raise DownloadSchemaError("Trading calendar columns must be exactly Date and HolDiv")
    parsed = _parse_date_series(frame["Date"], "calendar.Date")
    if parsed.isna().any() or parsed.duplicated().any():
        raise DownloadSchemaError("Trading calendar dates must be non-null and unique")
    selected = frame.loc[
        parsed.between(pd.Timestamp(start_date), pd.Timestamp(end_date)),
        ["Date", "HolDiv"],
    ].copy()
    return validate_calendar(selected, start_date, end_date)


def _parse_date_series(series: pd.Series, name: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DownloadSchemaError(f"{name} contains invalid dates") from exc
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    if not is_datetime64_any_dtype(parsed.dtype) or not parsed.dt.normalize().equals(parsed):
        raise DownloadSchemaError(f"{name} must contain midnight dates")
    return parsed.astype("datetime64[ns]")


def _parse_nullable_date_series(series: pd.Series, name: str) -> pd.Series:
    normalized = series.replace("", pd.NA)
    if isinstance(normalized.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    try:
        parsed = pd.to_datetime(normalized, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DownloadSchemaError(f"{name} contains invalid dates") from exc
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise DownloadSchemaError(f"{name} must be timezone-naive dates")
    if not parsed.dropna().dt.normalize().equals(parsed.dropna()):
        raise DownloadSchemaError(f"{name} must not contain times")
    return parsed.astype("datetime64[ns]")


def _coerce_string(frame: pd.DataFrame, column: str, name: str) -> None:
    if not all(isinstance(value, str) for value in frame[column].dropna()):
        raise DownloadSchemaError(f"{name} must contain strings without numeric coercion")
    frame[column] = frame[column].astype("string")


def _coerce_numeric(frame: pd.DataFrame, column: str, name: str) -> None:
    values = frame[column].dropna()
    if not values.empty and not is_numeric_dtype(values.dtype):
        raise DownloadSchemaError(f"{name} must contain numbers or nulls")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise DownloadSchemaError(f"{name} must not contain booleans")
    frame[column] = frame[column].astype("float64")
    finite = np.isfinite(frame[column].dropna().to_numpy(dtype="float64"))
    if not finite.all():
        raise DownloadSchemaError(f"{name} must contain finite numbers or nulls")


def _validate_codes(series: pd.Series, name: str) -> None:
    if series.isna().any() or not series.str.fullmatch(r"[0-9A-Z]{5}").all():
        raise DownloadSchemaError(f"{name} must contain non-null five-character uppercase codes")


def _empty_frame(
    columns: Sequence[str],
    *,
    date_columns: set[str] | frozenset[str] = frozenset(),
    numeric_columns: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for column in columns:
        if column in date_columns:
            values[column] = pd.Series(dtype="datetime64[ns]")
        elif column in numeric_columns:
            values[column] = pd.Series(dtype="float64")
        else:
            values[column] = pd.Series(dtype="string")
    return pd.DataFrame(values)


def _empty_financial_results() -> pd.DataFrame:
    return _empty_frame(
        FINANCIAL_RESULTS_COLUMNS,
        date_columns=set(FINANCIAL_DATE_COLUMNS),
    )


def _dataset_columns(dataset: str) -> Sequence[str]:
    return {
        LISTED_INFO_DATASET: LISTED_INFO_COLUMNS,
        EARNINGS_SCHEDULE_DATASET: EARNINGS_SCHEDULE_COLUMNS,
        FINANCIAL_RESULTS_DATASET: FINANCIAL_RESULTS_COLUMNS,
    }[dataset]


def _combine_frames(
    frames: Sequence[pd.DataFrame],
    preferred_columns: Sequence[str],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        if tuple(preferred_columns) == FINANCIAL_RESULTS_COLUMNS:
            return _empty_financial_results()
        date_columns = {
            "Date",
            "PubDate",
            "SchDate",
            "ObservedDate",
            "PreviousObservedDate",
        }
        return _empty_frame(preferred_columns, date_columns=date_columns.intersection(preferred_columns))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    ordered = [column for column in preferred_columns if column in combined.columns]
    for column in preferred_columns:
        if column not in combined.columns:
            if column in set(FINANCIAL_DATE_COLUMNS) | {
                "Date",
                "PubDate",
                "SchDate",
                "ObservedDate",
                "PreviousObservedDate",
            }:
                combined[column] = pd.Series(pd.NaT, index=combined.index, dtype="datetime64[ns]")
            else:
                combined[column] = pd.Series(pd.NA, index=combined.index, dtype="string")
            ordered.append(column)
    combined = combined.loc[:, ordered]
    usable_sort = [column for column in sort_columns if column in combined.columns]
    if usable_sort:
        combined.sort_values(usable_sort, inplace=True, ignore_index=True, na_position="last")
    return combined


def _single_master_date(frame: pd.DataFrame, label: str) -> pd.Timestamp:
    if frame.empty or "Date" not in frame or frame["Date"].nunique(dropna=False) != 1:
        raise ValueError(f"{label} master must contain exactly one non-empty Date")
    value = frame["Date"].iloc[0]
    if pd.isna(value):
        raise ValueError(f"{label} master Date must not be null")
    return pd.Timestamp(value)


def _transition_record(
    current_date: pd.Timestamp,
    previous_date: pd.Timestamp,
    code: str,
    change_type: str,
    changed_fields: Sequence[str],
    before: pd.Series | None,
    after: pd.Series | None,
) -> dict[str, Any]:
    return {
        "ObservedDate": current_date,
        "PreviousObservedDate": previous_date,
        "Code": code,
        "ChangeType": change_type,
        "ChangedFields": json.dumps(list(changed_fields), ensure_ascii=False, separators=(",", ":")),
        "BeforeRecord": _record_json(before),
        "AfterRecord": _record_json(after),
    }


def _record_json(record: pd.Series | None) -> str | None:
    if record is None:
        return None
    values: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            values[str(key)] = None
        elif isinstance(value, (pd.Timestamp, datetime, date)):
            values[str(key)] = pd.Timestamp(value).date().isoformat()
        elif isinstance(value, np.generic):
            values[str(key)] = value.item()
        else:
            values[str(key)] = value
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scalars_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left == right)


def _unique_codes_by_company_name(
    indexed: pd.DataFrame,
    codes: Sequence[str],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    if "CoName" not in indexed.columns:
        return output
    for code in codes:
        name = indexed.loc[code, "CoName"]
        if isinstance(name, str) and name.strip():
            output.setdefault(name, []).append(code)
    return output


def _cast_transition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(
            MASTER_TRANSITION_COLUMNS,
            date_columns={"ObservedDate", "PreviousObservedDate"},
        )
    for column in ("ObservedDate", "PreviousObservedDate"):
        frame[column] = pd.to_datetime(frame[column])
    for column in set(MASTER_TRANSITION_COLUMNS) - {"ObservedDate", "PreviousObservedDate"}:
        frame[column] = frame[column].astype("string")
    frame.sort_values(["ObservedDate", "Code", "ChangeType"], inplace=True, ignore_index=True)
    return frame


def _cast_code_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(
            CODE_CHANGE_CANDIDATE_COLUMNS,
            date_columns={"ObservedDate", "PreviousObservedDate"},
        )
    for column in ("ObservedDate", "PreviousObservedDate"):
        frame[column] = pd.to_datetime(frame[column])
    for column in set(CODE_CHANGE_CANDIDATE_COLUMNS) - {"ObservedDate", "PreviousObservedDate"}:
        frame[column] = frame[column].astype("string")
    frame.sort_values(["ObservedDate", "OldCode", "NewCode"], inplace=True, ignore_index=True)
    return frame


def _frame_stats(
    frame: pd.DataFrame,
    *,
    date_column: str,
    key_columns: Sequence[str],
) -> dict[str, Any]:
    missing = frame.isna()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        missing[column] = missing[column] | frame[column].eq("")
    return {
        "rows": len(frame),
        "securities": int(frame["Code"].nunique(dropna=True)) if "Code" in frame else None,
        "date_min": _date_or_none(frame[date_column].min()) if date_column in frame and not frame.empty else None,
        "date_max": _date_or_none(frame[date_column].max()) if date_column in frame and not frame.empty else None,
        "duplicate_keys": int(frame.duplicated(list(key_columns), keep=False).sum()) if not frame.empty else 0,
        "columns": list(frame.columns),
        "missing_rate": {column: float(value) for column, value in missing.mean().items()},
    }


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}


def _dates_by_year(values: Sequence[str]) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    for value in values:
        output.setdefault(date.fromisoformat(value).year, []).append(value)
    return output


def _create_output_directory(config: AuxiliaryDownloadConfig) -> Path:
    output_root = Path(config.output_root).expanduser().resolve()
    if config.output_dir is None:
        name = f"jquants_v2_auxiliary_{_utc_now().strftime('%Y%m%dT%H%M%S%fZ')}"
        output_dir = output_root / name
    else:
        supplied = Path(config.output_dir).expanduser()
        output_dir = supplied.resolve() if supplied.is_absolute() else (output_root / supplied).resolve()
    if output_dir == output_root or not output_dir.is_relative_to(output_root):
        raise ValueError("output_dir must be a new directory below output_root")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_parquet_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite raw data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _touch_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _utc_iso(_utc_now())
    _write_json(path, manifest)


def _parse_iso_date(value: str, name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid calendar date") from exc


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_or_none(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datasets(value: str) -> tuple[str, ...]:
    datasets = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(datasets) - set(DOWNLOAD_DATASETS)
    if not datasets or unknown or len(set(datasets)) != len(datasets):
        allowed = ",".join(DOWNLOAD_DATASETS)
        raise argparse.ArgumentTypeError(f"datasets must be unique names from: {allowed}")
    return datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="Inclusive Japan date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive Japan date (YYYY-MM-DD).")
    parser.add_argument(
        "--calendar",
        type=Path,
        required=True,
        help="J-Quants Date/HolDiv calendar CSV or Parquet covering the requested range.",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=DOWNLOAD_DATASETS,
        help=f"Comma-separated subset; default: {','.join(DOWNLOAD_DATASETS)}.",
    )
    parser.add_argument("--output-root", default="data/raw", help="Raw-data root (default: data/raw).")
    parser.add_argument(
        "--output-dir",
        help="Optional new directory below output-root; it must not already exist.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Rate limit including pagination requests (default: 5).",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        help="Wait before retrying HTTP 429 (default: 65).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prompt for the API key without echo when needed."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = AuxiliaryDownloadConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            datasets=args.datasets,
            output_root=args.output_root,
            output_dir=args.output_dir,
            requests_per_minute=args.requests_per_minute,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            rate_limit_wait_seconds=args.rate_limit_wait_seconds,
        )
        calendar = load_trading_calendar(args.calendar, config.start_date, config.end_date)
        api_key = resolve_api_key(prompt=True)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        result = download_auxiliary_data(
            config,
            trading_calendar=calendar,
            api_key=api_key,
            calendar_source_path=args.calendar,
        )
    except (ValueError, RawDownloadError, OSError) as exc:
        parser.error(str(exc))
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
