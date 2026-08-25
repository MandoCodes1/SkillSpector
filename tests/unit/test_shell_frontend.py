# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for shell semantics that stay explicit limitations."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

import skillspector.dependency_source_types as dependency_types
import skillspector.shell_frontend as shell_frontend
from skillspector.artifacts import classify_artifact
from skillspector.dependency_source_types import (
    DependencySourceLimitationReason,
    DependencyWorkBudget,
)
from skillspector.dependency_sources import analyze_dependency_sources


def _extract(
    path: str,
    raw: bytes,
    *,
    executable_paths: frozenset[str] = frozenset(),
    budget: DependencyWorkBudget | None = None,
) -> Any:
    return shell_frontend.extract_shell_units(
        path,
        raw,
        executable_paths=executable_paths,
        budget=budget or DependencyWorkBudget(),
    )


class _ObservedExecutablePaths(frozenset[str]):
    """Immutable inventory that records accidental whole-set iteration."""

    iteration_count: int
    membership_count: int

    def __new__(cls, values: Iterable[str]) -> _ObservedExecutablePaths:
        instance = super().__new__(cls, values)
        instance.iteration_count = 0
        instance.membership_count = 0
        return instance

    def __iter__(self) -> Iterator[str]:
        self.iteration_count += 1
        return super().__iter__()

    def __contains__(self, value: object) -> bool:
        self.membership_count += 1
        return super().__contains__(value)


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


@pytest.mark.parametrize(
    ("path", "raw", "executable_paths", "dialect", "provenance"),
    [
        (
            "scripts/setup.bash",
            b"printf ok\n",
            frozenset(),
            "bash",
            "file_suffix",
        ),
        (
            "scripts/setup.sh",
            b"printf ok\n",
            frozenset(),
            "sh",
            "file_suffix",
        ),
        (
            "scripts/setup.txt",
            b"#!/usr/bin/env dash\nprintf ok\n",
            frozenset(),
            "dash",
            "shebang",
        ),
        (
            "bundle.zip!/bin/setup",
            b"#!/bin/bash\nprintf ok\n",
            frozenset({"bundle.zip!/bin/setup"}),
            "bash",
            "shebang",
        ),
        (
            "bin/setup",
            b"#!/usr/bin/env -S bash -eu\nprintf ok\n",
            frozenset({"bin/setup"}),
            "bash",
            "shebang",
        ),
    ],
)
def test_standalone_unit_extraction_uses_only_supported_suffixes_and_shebangs(
    path: str,
    raw: bytes,
    executable_paths: frozenset[str],
    dialect: str,
    provenance: str,
) -> None:
    result = _extract(path, raw, executable_paths=executable_paths)

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.raw_bytes == raw
    assert unit.dialect.value == dialect
    assert unit.kind is dependency_types.ShellUnitKind.STANDALONE
    assert unit.provenance.value == provenance
    last_line_start = raw.rfind(b"\n", 0, max(0, len(raw) - 1)) + 1
    assert unit.origin_span == dependency_types.SourceSpan(
        path,
        0,
        len(raw),
        1,
        max(1, raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)),
        start_column=0,
        end_column=len(raw) - last_line_start,
    )
    assert result.issues == ()


def test_executable_only_without_supported_dialect_is_applicable_but_not_bash() -> None:
    path = "bundle.zip!/bin/setup"
    result = _extract(
        path,
        b"printf ok\n",
        executable_paths=frozenset({path}),
    )

    assert result.units == ()
    assert [(issue.reason, issue.outcome, issue.span.path) for issue in result.issues] == [
        (
            dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS,
            dependency_types.ShellWorkOutcome.PARTIAL,
            path,
        )
    ]


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        ("setup.zsh", b"printf ok\n"),
        ("setup.envrc", b"printf ok\n"),
        ("setup.ksh", b"printf ok\n"),
        ("Dockerfile", b"FROM scratch\nRUN printf ok\n"),
        ("build/Makefile", b"all:\n\tprintf ok\n"),
    ],
)
def test_out_of_gate_executable_dialects_remain_typed_limitations(
    path: str,
    raw: bytes,
) -> None:
    result = _extract(path, raw)

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        ("Dockerfile", b"FROM scratch\nCOPY . /src\n"),
        ("build/Makefile", b"all: generated.txt\n"),
    ],
)
def test_out_of_gate_container_and_make_files_without_executable_units_are_inert(
    path: str,
    raw: bytes,
) -> None:
    result = _extract(path, raw)

    assert result.units == ()
    assert result.issues == ()


def test_markdown_fences_honor_delimiter_length_indentation_and_info_token() -> None:
    raw = (
        b"heading\n"
        b"  ````BASH linenums\r\n"
        b"printf first\r\n"
        b"```\r\n"
        b"  ````\r\n"
        b"~~~shell-script\n"
        b"printf second\n"
        b"~~~~\n"
        b"```console\n"
        b"printf third\n"
        b"```\n"
    )

    result = _extract("docs/guide.md", raw)

    assert [unit.raw_bytes for unit in result.units] == [
        b"printf first\r\n```\r\n",
        b"printf second\n",
        b"printf third\n",
    ]
    assert [unit.dialect for unit in result.units] == [
        dependency_types.ShellDialect.BASH,
        dependency_types.ShellDialect.SH,
        dependency_types.ShellDialect.SH,
    ]
    assert all(
        unit.provenance is dependency_types.SiteProvenance.MARKDOWN_FENCE for unit in result.units
    )
    assert result.issues == ()


def test_markdown_fence_map_preserves_multibyte_crlf_physical_byte_columns() -> None:
    raw = "intro\r\n```bash\r\né npm\r\n```\r\n".encode()
    result = _extract("docs/guide.md", raw)
    unit = result.units[0]
    command_start = unit.raw_bytes.index(b"npm")

    mapped = unit.source_map.map_range(command_start, command_start + 3)

    physical_start = raw.index(b"npm")
    assert mapped == dependency_types.SourceSpan(
        "docs/guide.md",
        physical_start,
        physical_start + 3,
        3,
        3,
        start_column=3,
        end_column=6,
    )
    assert len(unit.source_map.entries) == 1


def test_repeated_extraction_produces_equal_opaque_unit_identities() -> None:
    raw = b"```bash\nprintf ok\n```\n"

    first = _extract("docs/guide.md", raw)
    second = _extract("docs/guide.md", raw)

    assert [unit.unit_id for unit in first.units] == [unit.unit_id for unit in second.units]


def test_markdown_does_not_infer_shell_inside_untagged_indented_or_non_shell_fences() -> None:
    raw = (
        b"````python\n"
        b"```bash\n"
        b"printf hidden\n"
        b"```\n"
        b"````\n"
        b"    ```bash\n"
        b"    printf indented\n"
        b"    ```\n"
        b"```text\n"
        b"printf text\n"
        b"```\n"
        b"```\n"
        b"#!/bin/bash\n"
        b"printf untagged\n"
        b"```\n"
    )

    result = _extract("docs/guide.md", raw)

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


def test_unclosed_relevant_markdown_fence_is_bounded_and_localized() -> None:
    raw = b"before\n~~~Sh\nprintf ok\n"

    result = _extract("docs/guide.md", raw)

    assert [unit.raw_bytes for unit in result.units] == [b"printf ok\n"]
    assert [
        (issue.reason, issue.span.start_line, issue.span.end_line) for issue in result.issues
    ] == [(dependency_types.ShellIssueReason.SYNTAX_ERROR, 2, 3)]


def test_invalid_utf8_in_relevant_shell_input_yields_only_a_sanitized_typed_issue() -> None:
    raw = b"#!/bin/bash\nprintf token-51e2\xff\n"

    result = _extract("scripts/setup", raw, executable_paths=frozenset({"scripts/setup"}))

    assert result.units == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ]
    assert "token-51e2" not in repr(result)


def test_malformed_markdown_units_consume_capacity_before_the_limit_issue() -> None:
    raw = b"".join(b"```bash\n\xff\n```\n" for _ in range(257))
    budget = DependencyWorkBudget()

    result = _extract("docs/malformed.md", raw, budget=budget)

    assert result.units == ()
    assert [issue.reason for issue in result.issues[:256]] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ] * 256
    assert result.issues[256].reason is dependency_types.ShellIssueReason.RESOURCE_LIMIT
    assert result.issues[256].outcome is dependency_types.ShellWorkOutcome.PARTIAL
    assert result.issues[256].exhaustion == dependency_types.DependencyWorkExhaustion(
        dependency_types.DependencyWorkResource.SHELL_UNITS,
        257,
        256,
    )
    assert (
        budget.for_file("docs/malformed.md").used(
            dependency_types.DependencyWorkResource.SHELL_UNITS
        )
        == 256
    )


@pytest.mark.parametrize(
    "raw",
    [b"printf a\x00b\n", b"printf first\rprintf second\fvalue"],
)
def test_nul_form_feed_and_lone_cr_are_preserved_without_invented_line_boundaries(
    raw: bytes,
) -> None:
    result = _extract("scripts/setup.sh", raw)

    assert [unit.raw_bytes for unit in result.units] == [raw]
    assert result.issues == ()


def test_shell_unit_limit_retains_exact_capacity_and_one_resource_issue() -> None:
    raw = b"".join(b"```bash\nprintf ok\n```\n" for _ in range(257))
    budget = DependencyWorkBudget()

    result = _extract("docs/many.md", raw, budget=budget)

    assert len(result.units) == 256
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    ]
    assert (
        budget.for_file("docs/many.md").used(dependency_types.DependencyWorkResource.SHELL_UNITS)
        == 256
    )


def test_extraction_requires_normalized_paths_immutable_inventory_and_canonical_bytes() -> None:
    with pytest.raises(ValueError):
        _extract("./scripts/setup.sh", b"printf ok\n")
    with pytest.raises(ValueError):
        shell_frontend.extract_shell_units(
            "scripts/setup.sh",
            b"printf ok\n",
            executable_paths={"scripts/setup.sh"},
            budget=DependencyWorkBudget(),
        )
    with pytest.raises(TypeError):
        shell_frontend.extract_shell_units(
            "scripts/setup.sh",
            bytearray(b"printf ok\n"),
            executable_paths=frozenset(),
            budget=DependencyWorkBudget(),
        )


def test_extraction_uses_large_normalized_executable_inventory_without_iteration() -> None:
    executable_path = "bundle.zip!/bin/setup"
    executable_paths = _ObservedExecutablePaths(
        executable_path if index == 0 else f"bin/tool-{index}" for index in range(50_000)
    )

    applicable = _extract(
        executable_path,
        b"printf ok\n",
        executable_paths=executable_paths,
    )
    inert = _extract(
        "docs/readme.txt",
        b"plain text\n",
        executable_paths=executable_paths,
    )

    assert [issue.reason for issue in applicable.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]
    assert inert.units == ()
    assert inert.issues == ()
    assert executable_paths.iteration_count == 0
    assert executable_paths.membership_count == 2
