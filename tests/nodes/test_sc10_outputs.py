# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-graph public-output contracts for direct SC10 configuration evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillspector.graph import graph

_ENFORCE_GAPS = os.getenv("SKILLSPECTOR_SC10_GAPS") == "enforce"
_SKILL = "---\nname: helper\ndescription: Formats ordinary text.\n---\n# Helper\nFormats text.\n"
_SENTINELS = ("alice", "supersecret", "querysecret", "fragmentsecret")
_NONCANONICAL_NPMRC = (
    "registry=https://alice:supersecret@packages.example.invalid/private"
    "?token=querysecret&channel=stable#fragmentsecret\n"
)
_CANONICAL_NPMRC = "registry=https://registry.npmjs.org/\n"
_EXPECTED_SC10 = {
    "rule": "SC10",
    "severity": "HIGH",
    "ecosystem": "npm",
    "surface": ".npmrc",
    "operation": "replace",
    "scope": "global",
    "destination": "https://REDACTED@packages.example.invalid/private?token=REDACTED&channel=stable",
    "destination_status": "resolved",
    "path": ".npmrc",
    "line": 1,
}


def _gap_marks(reason: str) -> list[pytest.MarkDecorator]:
    if _ENFORCE_GAPS:
        return []
    return [pytest.mark.xfail(strict=True, reason=reason)]


_DIRECT_CONFIGURATION_CASES = [
    pytest.param(
        _NONCANONICAL_NPMRC,
        _EXPECTED_SC10,
        id="credential-bearing-noncanonical-npmrc",
        marks=_gap_marks("direct configuration SC10 findings are not implemented"),
    )
]
_CANONICAL_DEFAULT_CASES = [
    pytest.param(
        _CANONICAL_NPMRC,
        id="canonical-npm-default",
        marks=_gap_marks("no real dependency-source analyzer is active yet"),
    )
]


def _write_skill(root: Path, npmrc: str) -> Path:
    (root / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    (root / ".npmrc").write_text(npmrc, encoding="utf-8")
    return root


def _scan(root: Path, output_format: str) -> dict[str, object]:
    return graph.invoke({"skill_path": str(root), "output_format": output_format, "use_llm": False})


def _normalized_sc10(result: dict[str, object]) -> list[dict[str, object]]:
    findings = result["filtered_findings"]
    assert isinstance(findings, list)
    normalized = []
    for finding in findings:
        if finding.rule_id != "SC10":
            continue
        evidence = finding.evidence
        normalized.append(
            {
                "rule": finding.rule_id,
                "severity": finding.severity,
                "ecosystem": evidence["ecosystem"],
                "surface": evidence["surface"],
                "operation": evidence["operation"],
                "scope": evidence["scope"],
                "destination": evidence["destination"],
                "destination_status": evidence["destination_status"],
                "path": finding.file,
                "line": finding.start_line,
            }
        )
    return normalized


@pytest.mark.parametrize(("npmrc", "expected"), _DIRECT_CONFIGURATION_CASES)
def test_noncanonical_npmrc_has_one_redacted_sc10_across_public_outputs(
    tmp_path: Path, npmrc: str, expected: dict[str, object]
) -> None:
    """Direct registry configuration must be a structured, redacted SC10 finding."""
    root = _write_skill(tmp_path, npmrc)
    results = {
        output_format: _scan(root, output_format)
        for output_format in (
            "terminal",
            "json",
            "markdown",
            "sarif",
        )
    }

    assert _normalized_sc10(results["json"]) == [expected]
    for result in results.values():
        completeness = result["analysis_completeness"]
        assert isinstance(completeness, dict)
        assert result["execution_successful"] is True
        assert completeness["is_complete"] is True
        assert completeness["status"] == "complete"

        serialized = result["report_body"]
        assert isinstance(serialized, str)
        assert "SC10" in serialized
        assert expected["destination"] in serialized
        assert all(sentinel not in serialized for sentinel in _SENTINELS)

    json_report = json.loads(results["json"]["report_body"])
    assert any(issue["id"] == "SC10" for issue in json_report["issues"])

    sarif_report = json.loads(results["sarif"]["report_body"])
    assert any(item["ruleId"] == "SC10" for item in sarif_report["runs"][0]["results"])


@pytest.mark.parametrize("npmrc", _CANONICAL_DEFAULT_CASES)
def test_canonical_npm_registry_is_safe_without_sc10(tmp_path: Path, npmrc: str) -> None:
    """The default npm registry remains a complete SAFE result once SC10 is active."""
    result = _scan(_write_skill(tmp_path, npmrc), "json")
    completeness = result["analysis_completeness"]
    assert isinstance(completeness, dict)
    assert any(
        status["analyzer_id"] == "dependency_sources"
        for status in completeness["analyzer_statuses"]
    )
    assert _normalized_sc10(result) == []
    assert result["risk_recommendation"] == "SAFE"
    assert result["execution_successful"] is True
    assert completeness["is_complete"] is True
    assert completeness["status"] == "complete"
