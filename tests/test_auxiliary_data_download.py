"""Tests for point-in-time J-Quants auxiliary-data acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stock_prediction.auxiliary_data_download import (
    AuxiliaryDownloadConfig,
    AuxiliaryJQuantsV2Client,
    CODE_CHANGE_CANDIDATE_COLUMNS,
    DOWNLOAD_DATASETS,
    EARNINGS_SCHEDULE_COLUMNS,
    FINANCIAL_RESULTS_COLUMNS,
    LISTED_INFO_COLUMNS,
    MASTER_TRANSITION_COLUMNS,
    download_auxiliary_data,
    derive_master_transitions,
    load_trading_calendar,
    validate_earnings_schedule,
    validate_financial_results,
    validate_listed_info,
    validate_topix,
)
from stock_prediction.data_download import DownloadSchemaError


def make_calendar(start: str, end: str, open_dates: set[str]) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "HolDiv": ["1" if value.date().isoformat() in open_dates else "0" for value in dates],
        },
    )


def make_topix(dates: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": dates,
            "O": [100.0 + position for position, _ in enumerate(dates)],
            "H": [102.0 + position for position, _ in enumerate(dates)],
            "L": [99.0 + position for position, _ in enumerate(dates)],
            "C": [101.0 + position for position, _ in enumerate(dates)],
        },
    )


def make_listed_info(
    as_of_date: str,
    records: tuple[tuple[str, str, str], ...] = (
        ("10000", "会社A", "0111"),
        ("20000", "会社B", "0112"),
    ),
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for code, name, market in records:
        rows.append(
            {
                "Date": as_of_date,
                "Code": code,
                "CoName": name,
                "CoNameEn": f"{name} EN",
                "S17": "10",
                "S17Nm": "情報通信・サービスその他",
                "S33": "5250",
                "S33Nm": "情報・通信業",
                "ScaleCat": "TOPIX Small 2",
                "Mkt": market,
                "MktNm": "プライム" if market == "0111" else "スタンダード",
                "Mrgn": "1",
                "MrgnNm": "貸借",
                "ProdCat": "1",
            },
        )
    return pd.DataFrame.from_records(rows, columns=LISTED_INFO_COLUMNS)


def make_schedule(publication_date: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "PubDate": publication_date,
                "SchDate": "2025-02-07",
                "FQName": "第3四半期",
                "FYE": "2025-03",
                "Code": "10000",
                "CoName": "会社A",
                "CoNameEn": "Company A",
            },
        ],
        columns=EARNINGS_SCHEDULE_COLUMNS,
    )


def make_financial_result(disclosure_date: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "DiscDate": disclosure_date,
                "DiscTime": "15:00:00",
                "Code": "10000",
                "DiscNo": "202501020001",
                "DocType": "FinancialStatements_3Q",
                "CurPerType": "3Q",
                "CurPerSt": "2024-04-01",
                "CurPerEn": "2024-12-31",
                "Sales": "1234",
                "OP": "100",
                "NxFNCEPS": "",
            },
        ],
    )


class FakeAuxiliaryClient:
    def __init__(
        self,
        *,
        topix: pd.DataFrame,
        listed: dict[str, pd.DataFrame | BaseException],
        schedules: dict[str, pd.DataFrame | BaseException],
        results: dict[str, pd.DataFrame | BaseException],
    ) -> None:
        self.topix = topix
        self.listed = listed
        self.schedules = schedules
        self.results = results
        self.calls: dict[str, list[Any]] = {
            "topix": [],
            "listed_info": [],
            "earnings_schedule": [],
            "financial_results": [],
        }

    def get_topix(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls["topix"].append((start_date, end_date))
        return self.topix.copy(deep=True)

    def get_listed_info(self, as_of_date: str) -> pd.DataFrame:
        self.calls["listed_info"].append(as_of_date)
        return _copy_or_raise(self.listed[as_of_date])

    def get_earnings_schedule(self, publication_date: str) -> pd.DataFrame:
        self.calls["earnings_schedule"].append(publication_date)
        return _copy_or_raise(self.schedules[publication_date])

    def get_financial_results(self, disclosure_date: str) -> pd.DataFrame:
        self.calls["financial_results"].append(disclosure_date)
        return _copy_or_raise(self.results[disclosure_date])


def _copy_or_raise(value: pd.DataFrame | BaseException) -> pd.DataFrame:
    if isinstance(value, BaseException):
        raise value
    return value.copy(deep=True)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.responses = [FakeResponse(payload) for payload in payloads]
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        pass


class NoopRateLimiter:
    def wait(self) -> None:
        pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_date": "2025-01-03", "end_date": "2025-01-02"},
        {"start_date": "2025-01-02", "end_date": "2025-01-03", "datasets": ()},
        {"start_date": "2025-01-02", "end_date": "2025-01-03", "datasets": ("unknown",)},
        {
            "start_date": "2025-01-02",
            "end_date": "2025-01-03",
            "datasets": ("topix", "topix"),
        },
    ],
)
def test_config_rejects_invalid_ranges_and_datasets(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AuxiliaryDownloadConfig(**kwargs)


def test_live_client_uses_documented_v2_endpoints_and_query_dates() -> None:
    session = FakeSession([{"data": []}, {"data": []}, {"data": []}, {"data": []}])
    client = AuxiliaryJQuantsV2Client(
        "secret",
        session=session,  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),  # type: ignore[arg-type]
    )
    client.get_topix("2025-01-01", "2025-01-31")
    client.get_listed_info("2025-01-06")
    client.get_earnings_schedule("2025-01-06")
    client.get_financial_results("2025-01-06")

    assert [call["url"].removeprefix("https://api.jquants.com/v2") for call in session.calls] == [
        "/indices/bars/daily/topix",
        "/equities/master",
        "/fins/earnings-date",
        "/fins/summary",
    ]
    assert [call["params"] for call in session.calls] == [
        {"from": "2025-01-01", "to": "2025-01-31"},
        {"date": "2025-01-06"},
        {"date": "2025-01-06"},
        {"date": "2025-01-06"},
    ]


def test_load_calendar_accepts_superset_but_requires_complete_requested_range(tmp_path: Path) -> None:
    source = tmp_path / "calendar.parquet"
    make_calendar("2025-01-01", "2025-01-05", {"2025-01-02", "2025-01-03"}).to_parquet(source)
    selected = load_trading_calendar(source, "2025-01-02", "2025-01-04")
    assert selected["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]

    incomplete = tmp_path / "incomplete.parquet"
    make_calendar("2025-01-01", "2025-01-02", {"2025-01-02"}).to_parquet(incomplete)
    with pytest.raises(DownloadSchemaError):
        load_trading_calendar(incomplete, "2025-01-01", "2025-01-03")


def test_topix_validation_requires_exact_sessions_and_ohlc_integrity() -> None:
    sessions = pd.to_datetime(["2025-01-02", "2025-01-03"])
    validated = validate_topix(
        make_topix(("2025-01-02", "2025-01-03")),
        "2025-01-01",
        "2025-01-03",
        sessions,
    )
    assert list(validated.columns) == list(("Date", "O", "H", "L", "C"))
    broken = make_topix(("2025-01-02", "2025-01-03"))
    broken.loc[0, "H"] = 50.0
    with pytest.raises(DownloadSchemaError):
        validate_topix(broken, "2025-01-01", "2025-01-03", sessions)


def test_topix_weekend_only_range_has_stable_empty_schema() -> None:
    validated = validate_topix(pd.DataFrame(), "2025-01-04", "2025-01-05", [])
    assert list(validated.columns) == ["Date", "O", "H", "L", "C"]
    assert pd.api.types.is_datetime64_any_dtype(validated["Date"])
    assert pd.api.types.is_float_dtype(validated["C"])


@pytest.mark.parametrize("change", ["unknown_column", "numeric_code", "duplicate", "wrong_date"])
def test_listed_info_schema_fails_early(change: str) -> None:
    frame = make_listed_info("2025-01-02")
    if change == "unknown_column":
        frame["FutureField"] = "x"
    elif change == "numeric_code":
        frame["Code"] = [10000, 20000]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[0, "Date"] = "2025-01-03"
    with pytest.raises(DownloadSchemaError):
        validate_listed_info(frame, "2025-01-02")


def test_schedule_and_results_preserve_publication_time_semantics() -> None:
    schedule = validate_earnings_schedule(make_schedule("2025-01-02"), "2025-01-02")
    result = validate_financial_results(make_financial_result("2025-01-02"), "2025-01-02")
    assert schedule.loc[0, "PubDate"] == pd.Timestamp("2025-01-02")
    assert schedule.loc[0, "SchDate"] == pd.Timestamp("2025-02-07")
    assert result.loc[0, "DiscDate"] == pd.Timestamp("2025-01-02")
    assert result.loc[0, "DiscTime"] == "15:00:00"
    assert result.loc[0, "Sales"] == "1234"
    assert result.loc[0, "NxFNCEPS"] == ""
    assert isinstance(result.loc[0, "Sales"], str)
    assert set(result.columns).issubset(FINANCIAL_RESULTS_COLUMNS)

    with pytest.raises(DownloadSchemaError):
        validate_earnings_schedule(make_schedule("2025-01-03"), "2025-01-02")
    bad_time = make_financial_result("2025-01-02")
    bad_time.loc[0, "DiscTime"] = "25:00"
    with pytest.raises(DownloadSchemaError):
        validate_financial_results(bad_time, "2025-01-02")

    numeric_value = make_financial_result("2025-01-02")
    numeric_value["Sales"] = pd.Series([1234], dtype="object")
    with pytest.raises(DownloadSchemaError, match="must contain strings"):
        validate_financial_results(numeric_value, "2025-01-02")


def test_empty_schedule_and_result_days_have_stable_typed_schemas() -> None:
    schedule = validate_earnings_schedule(pd.DataFrame(), "2025-01-01")
    result = validate_financial_results(pd.DataFrame(), "2025-01-01")
    assert list(schedule.columns) == list(EARNINGS_SCHEDULE_COLUMNS)
    assert list(result.columns) == list(FINANCIAL_RESULTS_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(schedule["PubDate"])
    assert pd.api.types.is_string_dtype(result["Sales"])


def test_master_transitions_do_not_claim_authoritative_delisting_or_code_change() -> None:
    previous = validate_listed_info(make_listed_info("2025-01-02"), "2025-01-02")
    current = validate_listed_info(
        make_listed_info(
            "2025-01-03",
            (
                ("10000", "会社A", "0112"),
                ("30000", "会社B", "0112"),
            ),
        ),
        "2025-01-03",
    )
    transitions, candidates = derive_master_transitions(previous, current)
    assert list(transitions.columns) == list(MASTER_TRANSITION_COLUMNS)
    assert set(transitions["ChangeType"]) == {"appeared", "disappeared", "attributes_changed"}
    assert "delisted" not in set(transitions["ChangeType"])
    assert list(candidates.columns) == list(CODE_CHANGE_CANDIDATE_COLUMNS)
    assert candidates[["OldCode", "NewCode", "Evidence"]].iloc[0].tolist() == [
        "20000",
        "30000",
        "unique_exact_company_name",
    ]


def test_download_writes_year_partitions_audit_metadata_and_all_calendar_date_queries(
    tmp_path: Path,
) -> None:
    open_dates = {"2025-01-02", "2025-01-03"}
    calendar = make_calendar("2025-01-01", "2025-01-03", open_dates)
    listed = {
        "2025-01-02": make_listed_info("2025-01-02"),
        "2025-01-03": make_listed_info(
            "2025-01-03",
            (("10000", "会社A", "0112"), ("30000", "会社B", "0112")),
        ),
    }
    schedules = {
        "2025-01-01": pd.DataFrame(),
        "2025-01-02": make_schedule("2025-01-02"),
        "2025-01-03": pd.DataFrame(),
    }
    results = {
        "2025-01-01": pd.DataFrame(),
        "2025-01-02": make_financial_result("2025-01-02"),
        "2025-01-03": pd.DataFrame(),
    }
    client = FakeAuxiliaryClient(
        topix=make_topix(("2025-01-02", "2025-01-03")),
        listed=listed,
        schedules=schedules,
        results=results,
    )
    raw_root = tmp_path / "data" / "raw"
    output = raw_root / "auxiliary"
    config = AuxiliaryDownloadConfig(
        start_date="2025-01-01",
        end_date="2025-01-03",
        output_root=raw_root,
        output_dir=output,
    )
    result = download_auxiliary_data(
        config,
        trading_calendar=calendar,
        api_key="must-not-be-saved",
        client=client,
    )

    assert result.output_dir == output.resolve()
    assert set(result.data_paths) == set(DOWNLOAD_DATASETS)
    assert len(pd.read_parquet(output / "topix.parquet")) == 2
    assert len(pd.read_parquet(output / "listed_info" / "2025.parquet")) == 4
    assert len(pd.read_parquet(output / "earnings_schedule" / "2025.parquet")) == 1
    saved_financials = pd.read_parquet(output / "financial_results" / "2025.parquet")
    assert len(saved_financials) == 1
    assert saved_financials.loc[0, "Sales"] == "1234"
    assert saved_financials.loc[0, "NxFNCEPS"] == ""
    assert len(pd.read_parquet(output / "master_transitions" / "2025.parquet")) == 3
    assert len(pd.read_parquet(output / "code_change_candidates" / "2025.parquet")) == 1
    assert client.calls["listed_info"] == ["2025-01-02", "2025-01-03"]
    assert client.calls["earnings_schedule"] == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert client.calls["financial_results"] == ["2025-01-01", "2025-01-02", "2025-01-03"]
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "completed"
    assert manifest["credential_persisted"] is False
    assert manifest["master_transition_limitations"]["code_change_candidates_are_authoritative"] is False
    assert manifest["datasets"]["listed_info"]["completed_dates"] == 2
    assert manifest["datasets"]["earnings_schedule"]["completed_dates"] == 3
    financial_stats = manifest["datasets"]["financial_results"]["completed_partitions"][0]["stats"]
    assert financial_stats["missing_rate"]["Sales"] == 0.0
    assert financial_stats["missing_rate"]["NxFNCEPS"] == 1.0
    assert "must-not-be-saved" not in manifest_text


def test_interruption_preserves_partial_master_and_transition_progress(tmp_path: Path) -> None:
    calendar = make_calendar("2025-01-02", "2025-01-03", {"2025-01-02", "2025-01-03"})
    client = FakeAuxiliaryClient(
        topix=pd.DataFrame(),
        listed={
            "2025-01-02": make_listed_info("2025-01-02"),
            "2025-01-03": KeyboardInterrupt("stop now"),
        },
        schedules={},
        results={},
    )
    raw_root = tmp_path / "raw"
    output = raw_root / "interrupted"
    config = AuxiliaryDownloadConfig(
        start_date="2025-01-02",
        end_date="2025-01-03",
        datasets=("listed_info",),
        output_root=raw_root,
        output_dir=output,
    )
    with pytest.raises(KeyboardInterrupt):
        download_auxiliary_data(config, trading_calendar=calendar, api_key="secret", client=client)

    partial = pd.read_parquet(output / "partial" / "listed_info" / "2025.parquet")
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert len(partial) == 2
    assert manifest["status"] == "interrupted"
    assert manifest["datasets"]["listed_info"]["completed_dates"] == 1
    assert manifest["partial_partition"]["last_completed_date"] == "2025-01-02"
    assert "secret" not in manifest_text


def test_output_must_be_new_and_below_raw_root(tmp_path: Path) -> None:
    calendar = make_calendar("2025-01-02", "2025-01-02", {"2025-01-02"})
    client = FakeAuxiliaryClient(
        topix=make_topix(("2025-01-02",)),
        listed={},
        schedules={},
        results={},
    )
    raw_root = tmp_path / "raw"
    outside = tmp_path / "outside"
    config = AuxiliaryDownloadConfig(
        start_date="2025-01-02",
        end_date="2025-01-02",
        datasets=("topix",),
        output_root=raw_root,
        output_dir=outside,
    )
    with pytest.raises(ValueError):
        download_auxiliary_data(config, trading_calendar=calendar, client=client)

    existing = raw_root / "existing"
    existing.mkdir(parents=True)
    config = AuxiliaryDownloadConfig(
        start_date="2025-01-02",
        end_date="2025-01-02",
        datasets=("topix",),
        output_root=raw_root,
        output_dir=existing,
    )
    with pytest.raises(FileExistsError):
        download_auxiliary_data(config, trading_calendar=calendar, client=client)
