# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused end-to-end skeletons for the planned dependency-command adapters."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import DependencySourceSpan, DependencyWorkBudget
from skillspector.dependency_sources import analyze_dependency_sources
from skillspector.nested_artifacts import is_executable_content

REPRESENTATIVE_IDS = (
    "npm-flags-before-operands",
    "yarn-flags-before-operands",
    "pnpm-unmodeled",
    "pip-short-option-bundling",
    "poetry-global-options-before-subcommand",
    "cargo-has-no-command-branch",
    "uv-entirely-uncovered",
    "maven-settings-file-flag-and-unrecognised-filename",
)


def _finding_rows() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parents[1] / "nodes/analyzers/data/sc10_findings.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {row["id"]: row for row in rows}


def _gap_marks(row: dict[str, Any]) -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.sc10_pr2]
    if row["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {row['id']}"))
    return marks


def _normalized_finding(finding: Any) -> dict[str, Any]:
    evidence = finding.evidence
    result = {
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
    if finding.end_line is not None and finding.end_line != finding.start_line:
        result["end_line"] = finding.end_line
    return result


ROWS_BY_ID = _finding_rows()
MISSING_REPRESENTATIVES = set(REPRESENTATIVE_IDS) - set(ROWS_BY_ID)
assert not MISSING_REPRESENTATIVES

ADAPTER_PARAMETERS = [
    pytest.param(ROWS_BY_ID[identifier], id=identifier, marks=_gap_marks(ROWS_BY_ID[identifier]))
    for identifier in REPRESENTATIVE_IDS
]


@pytest.mark.parametrize("row", ADAPTER_PARAMETERS)
def test_dependency_command_adapter_contracts(row: dict[str, Any]) -> None:
    files = row["files"]
    raw_files = {path: content.encode("utf-8") for path, content in files.items()}
    executable_paths = frozenset(
        DependencySourceSpan(path=path, start_line=1, end_line=1).path
        for path in sorted(raw_files)
        if is_executable_content(path, raw_files[path])
    )
    analysis = analyze_dependency_sources(
        components=sorted(files),
        local_file_cache=files,
        raw_file_cache=raw_files,
        artifact_inventory=[classify_artifact(path, raw_files[path]) for path in sorted(raw_files)],
        budget=DependencyWorkBudget(),
        executable_paths=executable_paths,
    )

    actual = [
        _normalized_finding(finding) for finding in analysis.findings if finding.rule_id == "SC10"
    ]
    assert Counter(json.dumps(item, sort_keys=True) for item in actual) == Counter(
        json.dumps(item, sort_keys=True) for item in row["expected_sc10"]
    )
    assert analysis.limitations == ()
    assert row["status"] == "fixed", "unimplemented adapter contracts remain explicit red gates"
