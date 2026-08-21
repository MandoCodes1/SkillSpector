# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent behavioral corpus for dependency-source trust-boundary changes."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

DATA_DIR = Path(__file__).with_name("data")
DATA_FILES = (DATA_DIR / "sc10_findings.json", DATA_DIR / "sc10_controls.json")
STATUS_VALUES = {"fixed", "unfixed", "deferred"}
OWNER_VALUES = {"PR-1", "PR-2", "DEFERRED"}
OUTCOME_VALUES = {"finding", "inert", "limitation"}
FINDING_FIELDS = {
    "severity",
    "ecosystem",
    "surface",
    "operation",
    "scope",
    "destination",
    "destination_status",
    "file",
    "start_line",
}
ROW_FIELDS = {"id", "status", "lands_in", "expected_outcome", "files", "expected_sc10"}
PROHIBITED_FIELDS = {
    "expect",
    "expect_sc10",
    "expected_prose",
    "family",
    "generated_from",
    "index",
    "input_note",
    "kind",
    "observed_today",
    "root_cause",
}


def _load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in DATA_FILES]
    assert all(set(document) == {"schema_version", "rows"} for document in documents)
    assert all(document["schema_version"] == 1 for document in documents)
    return documents[0]["rows"], documents[1]["rows"]


FINDING_ROWS, CONTROL_ROWS = _load_rows()
ALL_ROWS = FINDING_ROWS + CONTROL_ROWS


def _row_marks(row: dict[str, Any]) -> list[pytest.MarkDecorator]:
    owner_mark = {
        "PR-1": pytest.mark.sc10_pr1,
        "PR-2": pytest.mark.sc10_pr2,
        "DEFERRED": pytest.mark.sc10_deferred,
    }[row["lands_in"]]
    marks = [owner_mark]
    if row["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {row['id']}"))
    return marks


BEHAVIOR_PARAMETERS = [pytest.param(row, id=row["id"], marks=_row_marks(row)) for row in ALL_ROWS]


def _normalized_finding(finding: Any) -> dict[str, Any]:
    evidence = finding.evidence
    normalized = {
        "severity": finding.severity,
        "ecosystem": evidence["ecosystem"],
        "surface": evidence["surface"],
        "operation": evidence["operation"],
        "scope": evidence["scope"],
        "destination": evidence["destination"],
        "destination_status": evidence["destination_status"],
        "file": finding.file,
        "start_line": finding.start_line,
    }
    end_line = getattr(finding, "end_line", None)
    if end_line is not None and end_line != finding.start_line:
        normalized["end_line"] = end_line
    return normalized


def _normalized_limitation(limitation: Any) -> dict[str, Any]:
    return {
        "reason": getattr(limitation.reason, "value", limitation.reason),
        "path": limitation.path,
        "range": {
            "start_line": limitation.start_line,
            "end_line": limitation.end_line,
        },
    }


def _multiset(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(json.dumps(record, sort_keys=True) for record in records)


def _mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested_key
            for nested_value in value.values()
            for nested_key in _mapping_keys(nested_value)
        }
    if isinstance(value, list):
        return {nested_key for item in value for nested_key in _mapping_keys(item)}
    return set()


def test_corpus_schema_and_self_checks() -> None:
    assert FINDING_ROWS, "findings corpus must not be empty"
    assert CONTROL_ROWS, "controls corpus must not be empty"

    ids = [row["id"] for row in ALL_ROWS]
    assert len(ids) == len(set(ids))
    file_inputs: list[tuple[str, str]] = []
    for row in ALL_ROWS:
        allowed_fields = ROW_FIELDS | (
            {"expected_limitation"} if "expected_limitation" in row else set()
        )
        assert set(row) == allowed_fields
        assert not (_mapping_keys(row) & PROHIBITED_FIELDS)
        assert row["status"] in STATUS_VALUES
        assert row["lands_in"] in OWNER_VALUES
        assert row["expected_outcome"] in OUTCOME_VALUES
        assert isinstance(row["files"], dict) and len(row["files"]) == 1
        path, content = next(iter(row["files"].items()))
        assert isinstance(path, str) and path
        assert isinstance(content, str)
        file_inputs.append((path, content))
        assert isinstance(row["expected_sc10"], list)
        for expected in row["expected_sc10"]:
            assert set(expected) == FINDING_FIELDS or set(expected) == FINDING_FIELDS | {"end_line"}
            assert expected["severity"] == "HIGH"
            assert expected["destination_status"] in {"resolved", "unresolved"}
            assert isinstance(expected["start_line"], int) and expected["start_line"] >= 1
            if "end_line" in expected:
                assert isinstance(expected["end_line"], int)
                assert expected["end_line"] > expected["start_line"]
        if row["expected_outcome"] == "finding":
            assert row["expected_sc10"]
            assert "expected_limitation" not in row
        elif row["expected_outcome"] == "inert":
            assert row["expected_sc10"] == []
            assert "expected_limitation" not in row
        else:
            assert row["expected_sc10"] == []
            assert set(row["expected_limitation"]) == {"reason", "path", "range"}
            assert set(row["expected_limitation"]["range"]) == {"start_line", "end_line"}
            assert row["expected_limitation"]["reason"] == "unscanned_executable_content"
            assert row["expected_limitation"]["path"] == path
            limitation_range = row["expected_limitation"]["range"]
            assert 1 <= limitation_range["start_line"] <= limitation_range["end_line"]

    assert len(file_inputs) == len(set(file_inputs))
    assert len(ALL_ROWS) == len(FINDING_ROWS) + len(CONTROL_ROWS)


@pytest.mark.parametrize("row", BEHAVIOR_PARAMETERS)
def test_dependency_source_behavior(row: dict[str, Any]) -> None:
    try:
        from skillspector.dependency_sources import analyze_dependency_sources
    except ImportError as exc:
        pytest.fail(f"real dependency-source analyzer is unavailable: {exc}")

    files = row["files"]
    analysis = analyze_dependency_sources(sorted(files), files, [])
    findings = list(getattr(analysis, "findings", analysis))
    limitations = list(getattr(analysis, "limitations", []))
    actual_sc10 = [
        _normalized_finding(finding) for finding in findings if finding.rule_id == "SC10"
    ]
    assert len(actual_sc10) == len(row["expected_sc10"])
    assert _multiset(actual_sc10) == _multiset(row["expected_sc10"])

    expected_limitations = [row["expected_limitation"]] if "expected_limitation" in row else []
    actual_limitations = [_normalized_limitation(item) for item in limitations]
    assert len(actual_limitations) == len(expected_limitations)
    assert _multiset(actual_limitations) == _multiset(expected_limitations)
