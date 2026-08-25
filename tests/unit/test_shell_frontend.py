# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for shell semantics that stay explicit limitations."""

from __future__ import annotations

import os
from typing import Any

import pytest

from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import (
    DependencySourceLimitationReason,
    DependencyWorkBudget,
)
from skillspector.dependency_sources import analyze_dependency_sources


def _gap_marks(case: dict[str, Any]) -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.sc10_pr2]
    if case["status"] != "fixed" and os.getenv("SKILLSPECTOR_SC10_GAPS") != "enforce":
        marks.append(pytest.mark.xfail(strict=True, reason=f"SC10 gap: {case['id']}"))
    return marks


UNSUPPORTED_CASES = [
    {
        "id": "xargs-manager-construction-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\n"
            "printf '%s\\n' 'config set registry https://packages.example.invalid' "
            "| xargs npm\n"
        ),
    },
    {
        "id": "env-s-split-string-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\nenv -S 'npm config set registry https://packages.example.invalid'\n"
        ),
    },
    {
        "id": "data-to-shell-pipeline-limitation",
        "status": "unfixed",
        "source": (
            "#!/bin/bash\n"
            "printf '%s\\n' 'npm config set registry https://packages.example.invalid' "
            "| sh\n"
        ),
    },
]

UNSUPPORTED_SEMANTICS = [
    pytest.param(case, id=case["id"], marks=_gap_marks(case)) for case in UNSUPPORTED_CASES
]


@pytest.mark.parametrize("case", UNSUPPORTED_SEMANTICS)
def test_unsupported_shell_semantics_are_localized_limitations(case: dict[str, Any]) -> None:
    path = "scripts/setup.sh"
    source = case["source"]
    raw = source.encode("utf-8")
    analysis = analyze_dependency_sources(
        components=[path],
        local_file_cache={path: source},
        raw_file_cache={path: raw},
        artifact_inventory=[classify_artifact(path, raw)],
        budget=DependencyWorkBudget(),
        executable_paths=frozenset({path}),
    )

    assert [finding for finding in analysis.findings if finding.rule_id == "SC10"] == []
    assert [
        (
            limitation.reason,
            limitation.path,
            limitation.start_line,
            limitation.end_line,
        )
        for limitation in analysis.limitations
    ] == [(DependencySourceLimitationReason.PARSE_INCOMPLETE, path, 2, 2)]
    assert case["status"] == "fixed", "unimplemented shell contracts remain explicit red gates"
