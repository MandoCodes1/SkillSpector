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


def _analyze(
    raw: bytes,
    *,
    path: str = "scripts/lower.sh",
    budget: DependencyWorkBudget | None = None,
) -> tuple[Any, DependencyWorkBudget, Any]:
    active_budget = budget or DependencyWorkBudget()
    extraction = _extract(path, raw, budget=active_budget)
    assert len(extraction.units) == 1
    unit = extraction.units[0]
    return (
        shell_frontend.analyze_shell_unit(unit, budget=active_budget),
        active_budget,
        unit,
    )


def _argv_bytes(command: Any) -> tuple[bytes | None, ...]:
    return tuple(
        value.exact_bytes if value.state is dependency_types.StaticValueState.EXACT else None
        for value in command.argv
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


class _ObservedDraftList(list[Any]):
    """Counts draft iteration without relying on a wall-clock threshold."""

    iterated_items: int = 0

    def __iter__(self) -> Iterator[Any]:
        for item in super().__iter__():
            self.iterated_items += 1
            yield item


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


def test_shell_lowering_parses_once_and_returns_one_completed_work_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"printf 'two words' plain\n"
    real_parse = shell_frontend.parse_bash_source
    parse_calls = 0

    def recording_parse(source: bytes, **kwargs: Any) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", recording_parse)

    result, _budget, unit = _analyze(raw)

    assert parse_calls == 1
    assert [_argv_bytes(command) for command in result.commands] == [
        (b"printf", b"two words", b"plain")
    ]
    assert result.assignments == ()
    assert result.generated_configs == ()
    assert result.issues == ()
    assert [(work.unit_id, work.outcome) for work in result.work_items] == [
        (unit.unit_id, dependency_types.ShellWorkOutcome.COMPLETED)
    ]


@pytest.mark.parametrize(
    ("raw", "node_type", "expected_fields"),
    [
        (b"cmd\n", "program", frozenset()),
        (b">out cmd arg\n", "command", frozenset({"redirect", "name", "argument"})),
        (b"A=one\n", "variable_assignment", frozenset({"name", "value"})),
        (b"export A=one\n", "declaration_command", frozenset()),
        (b"cmd >out\n", "redirected_statement", frozenset({"body", "redirect"})),
        (b"cmd 2>out\n", "file_redirect", frozenset({"descriptor", "destination"})),
        (b"f() { cmd; }\n", "function_definition", frozenset({"name", "body"})),
        (b"if ok; then yes; fi\n", "if_statement", frozenset({"condition"})),
        (
            b"for x in a; do yes; done\n",
            "for_statement",
            frozenset({"variable", "value", "body"}),
        ),
        (
            b"while ok; do yes; done\n",
            "while_statement",
            frozenset({"condition", "body"}),
        ),
        (b"case x in a) yes;; esac\n", "case_statement", frozenset({"value"})),
        (
            b"case x in a|b) first; second;; *) other;; esac\n",
            "case_item",
            frozenset({"value", "termination"}),
        ),
        (b"one | two\n", "pipeline", frozenset()),
        (b"one && two\n", "list", frozenset()),
        (b"! one\n", "negated_command", frozenset()),
        (b"( one )\n", "subshell", frozenset()),
        (b"{ one; }\n", "compound_statement", frozenset()),
        (b"outer $(inner)\n", "command_substitution", frozenset()),
        (b"outer <(inner)\n", "process_substitution", frozenset()),
        (b"if one; then two;\n", "fi", frozenset()),
        (b"good\nif broken\nlast $(\n", "ERROR", frozenset()),
    ],
)
def test_pinned_bash_cst_node_and_field_contract(
    raw: bytes,
    node_type: str,
    expected_fields: frozenset[str],
) -> None:
    root = shell_frontend.parse_bash_source(raw).root_node
    pending = [root]
    selected = None
    while pending:
        node = pending.pop()
        if node.type == node_type and (node_type != "fi" or node.is_missing):
            selected = node
            break
        pending.extend(reversed(node.children))

    assert selected is not None
    fields = frozenset(
        field_name
        for index in range(len(selected.children))
        if (field_name := selected.field_name_for_child(index)) is not None
    )
    contract_key = "MISSING" if selected.is_missing else node_type
    assert shell_frontend._PINNED_CST_FIELDS[contract_key] == expected_fields
    assert fields == expected_fields


def test_shell_lowering_visits_all_structural_regions_and_pipeline_stages() -> None:
    raw = (
        b"first && second || third; fourth &\n"
        b"( sub )\n"
        b"{ grouped; }\n"
        b"if cond; then yes; else no; fi\n"
        b"for item in a; do loop; done\n"
        b"while check; do body; done\n"
        b'case "$x" in a) arm;; esac\n'
        b"f() { inside; }\n"
        b'outer "$(inner arg)" <(producer) >(consumer)\n'
        b"one | two | three\n"
        b"! negated\n"
    )

    result, _budget, _unit = _analyze(raw)

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"first",
        b"second",
        b"third",
        b"fourth",
        b"sub",
        b"grouped",
        b"cond",
        b"yes",
        b"no",
        b"loop",
        b"check",
        b"body",
        b"arm",
        b"inside",
        b"outer",
        b"inner",
        b"producer",
        b"consumer",
        b"one",
        b"two",
        b"three",
        b"negated",
    ]
    assert result.issues == ()
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED


def test_cst_substitution_depth_is_bounded_by_node_budget_not_nested_literal_budget() -> None:
    result, _budget, _unit = _analyze(
        b"outer $(middle $(inner $(deep)))\n",
        path="scripts/substitutions.sh",
    )

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"outer",
        b"middle",
        b"inner",
        b"deep",
    ]
    assert result.issues == ()


def test_shell_lowering_emits_assignments_and_joins_only_line_continuations() -> None:
    raw = b"NA\\\nME=va\\\nlue com\\\nmand ar\\\ng\nexport C=see BARE\nD=$dynamic next\n"

    result, _budget, _unit = _analyze(raw)

    assert [
        (
            assignment.name,
            assignment.value.state,
            assignment.value.exact_bytes,
        )
        for assignment in result.assignments
    ] == [
        ("NAME", dependency_types.StaticValueState.EXACT, b"value"),
        ("C", dependency_types.StaticValueState.EXACT, b"see"),
        ("D", dependency_types.StaticValueState.UNKNOWN, None),
    ]
    assert [_argv_bytes(command) for command in result.commands] == [
        (b"command", b"arg"),
        (b"export", b"C=see", b"BARE"),
        (b"next",),
    ]


def test_only_structurally_bare_time_is_timing_syntax() -> None:
    raw = b'time -p timed arg;\n\\time escaped;\n"time" quoted;\n/bin/time path;\n$timer dynamic\n'

    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"timed", b"arg"),
        (b"time", b"escaped"),
        (b"time", b"quoted"),
        (b"/bin/time", b"path"),
        (None, b"dynamic"),
    ]


def test_redirect_destinations_do_not_hide_later_argv_and_exact_ampersand_redirect() -> None:
    raw = (
        b">lead first arg\n"
        b"second >middle arg\n"
        b"third arg >trail\n"
        b"fourth > one two three\n"
        b"fifth > dest\\\nination arg\n"
        b"sixth &>both arg\n"
    )

    analysis, _budget, unit = _analyze(raw)
    private = shell_frontend._analyze_shell_unit(unit, budget=DependencyWorkBudget())

    assert [_argv_bytes(command) for command in analysis.commands] == [
        (b"first", b"arg"),
        (b"second", b"arg"),
        (b"third", b"arg"),
        (b"fourth", b"two", b"three"),
        (b"fifth", b"arg"),
        (b"sixth", b"arg"),
    ]
    assert analysis.issues == ()
    assert [
        fact.kind.value for command in private.program.commands for fact in command.redirects
    ] == [
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_truncate",
        "stdout_stderr_truncate",
    ]


@pytest.mark.parametrize(
    ("raw", "syntax_start"),
    [
        (b"cmd &> | next\n", 7),
        (b"cmd >out <\n", 9),
    ],
)
def test_command_owned_malformed_redirect_regions_never_retain_supported_facts(
    raw: bytes,
    syntax_start: int,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/malformed-redirect.sh", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert [command.site.argv[0].exact_bytes for command in private.program.commands] == [b"cmd"]
    assert [fact for command in private.program.commands for fact in command.redirects] == []
    assert [
        (issue.reason, issue.span.start_byte, issue.span.end_byte)
        for issue in private.public.issues
    ] == [
        (
            dependency_types.ShellIssueReason.SYNTAX_ERROR,
            syntax_start,
            syntax_start + 1,
        )
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        b"cmd &> | next\n",
        b"cmd >out <\n",
    ],
)
def test_malformed_redirect_suppression_does_not_poison_unrelated_commands(
    malformed: bytes,
) -> None:
    raw = b"safe >ok\n" + malformed
    budget = DependencyWorkBudget()
    unit = _extract("scripts/local-malformed-redirect.sh", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert [command.site.argv[0].exact_bytes for command in private.program.commands] == [
        b"safe",
        b"cmd",
    ]
    assert [
        [fact.kind.value for fact in command.redirects] for command in private.program.commands
    ] == [["stdout_truncate"], []]
    assert [issue.reason for issue in private.public.issues] == [
        dependency_types.ShellIssueReason.SYNTAX_ERROR
    ]


def test_private_same_parse_ir_retains_bounded_structure_without_repr_content() -> None:
    budget = DependencyWorkBudget()
    unit = _extract(
        "scripts/private-ir.sh",
        b"f() { A=secretvalue command secretliteral >secrettarget; }\n",
        budget=budget,
    ).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert len(private.program.functions) == 1
    function = private.program.functions[0]
    command = private.program.commands[0]
    assignment = private.program.assignments[0]
    assert command.function_id == function.function_id
    assert assignment.function_id == function.function_id
    assert assignment.prefix_for_command_start_byte is not None
    assert [site.name for site in command.prefix_assignments] == ["A"]
    assert all(argument.fragments for argument in command.arguments)
    assert [fact.kind.value for fact in command.redirects] == ["stdout_truncate"]
    assert private.program.regions
    rendered = repr(private)
    assert "secretliteral" not in rendered
    assert "secrettarget" not in rendered
    assert "secretvalue" not in rendered


@pytest.mark.parametrize(
    "raw",
    [
        b"command >first >second\n",
        b"command >first 2>&1\n",
        b"command <>readwrite\n",
        b"command <<<data >output arg\n",
        b"command >output <<EOF\nbody\nEOF\n",
        b"command <<<first <<<second arg\n",
    ],
)
def test_redirect_chains_and_malformed_adjacent_redirects_never_retain_facts(
    raw: bytes,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/redirect-chain.sh", raw, budget=budget).units[0]

    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    assert [command.site.argv[0].exact_bytes for command in private.program.commands] == [
        b"command"
    ]
    assert [fact for command in private.program.commands for fact in command.redirects] == []
    assert private.public.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_heredoc_wrapper_argument_field_remains_structurally_proven_argv() -> None:
    result, _budget, _unit = _analyze(b"command <<EOF arg\nbody\nEOF\n")

    assert [_argv_bytes(command) for command in result.commands] == [(b"command", b"arg")]
    assert result.issues == ()


def test_assignment_value_fragments_are_charged_as_retained_private_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    charges: list[int] = []
    original = dependency_types.DependencyFileBudget.charge_retained_shell_ir

    def recording_charge(
        file_budget: dependency_types.DependencyFileBudget,
        unit: dependency_types.ShellUnit,
        count: int,
    ) -> dependency_types.DependencyWorkExhaustion | None:
        charges.append(count)
        return original(file_budget, unit, count)

    monkeypatch.setattr(
        dependency_types.DependencyFileBudget,
        "charge_retained_shell_ir",
        recording_charge,
    )

    budget = DependencyWorkBudget()
    unit = _extract("scripts/assignment-ir.sh", b"A=value\n", budget=budget).units[0]
    private = shell_frontend._analyze_shell_unit(unit, budget=budget)

    fragment_count = len(private.program.assignments[0].value_fragments)
    assert fragment_count == 1
    assert 2 + fragment_count in charges


def test_redirect_association_does_not_rescan_full_command_inventory() -> None:
    command_count = 200
    raw = b"".join(f"command{index} >target\n".encode() for index in range(command_count))
    budget = DependencyWorkBudget()
    unit = _extract("scripts/many-redirects.sh", raw, budget=budget).units[0]
    file_budget = budget.for_file(unit.origin_span.path)
    lowerer = shell_frontend._ShellLowerer(unit, budget, file_budget)
    lowerer.walk(shell_frontend.parse_bash_source(raw).root_node)
    observed = _ObservedDraftList(lowerer.command_drafts)
    lowerer.command_drafts = observed

    program = lowerer.lower()

    assert len(program.commands) == command_count
    assert observed.iterated_items <= command_count * 3


def test_unretained_issue_still_marks_terminal_work_partial() -> None:
    budget = DependencyWorkBudget()
    assert (
        budget.charge_shell_issues(dependency_types.MAX_DEPENDENCY_SHELL_LOCALIZED_ISSUES - 1)
        is None
    )
    assert (
        budget.claim_reserved_shell_truncation_issue()
        is dependency_types.ShellTruncationClaimStatus.CLAIMED
    )

    result, _budget, _unit = _analyze(
        b"command bad\x00value\n",
        path="scripts/full-issue-budget.sh",
        budget=budget,
    )

    assert result.issues == ()
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_unsupported_descriptor_redirects_preserve_provable_argv() -> None:
    raw = b"first &>>out arg\nsecond 2>&1 arg\nthird 1>&2 arg\nfourth <&0 arg\n"

    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"first", b"arg"),
        (b"second", b"arg"),
        (b"third", b"arg"),
        (b"fourth", b"arg"),
    ]
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ] * 4
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_markdown_lowering_composes_multibyte_crlf_locations() -> None:
    raw = "intro\r\n```bash\r\nécho 'x;y' # hidden\r\n```\r\n".encode()
    budget = DependencyWorkBudget()
    extraction = _extract("docs/guide.md", raw, budget=budget)

    result = shell_frontend.analyze_shell_unit(extraction.units[0], budget=budget)

    command_start = raw.index("écho".encode())
    command_end = command_start + len("écho 'x;y'".encode())
    assert result.commands[0].span == dependency_types.SourceSpan(
        "docs/guide.md",
        command_start,
        command_end,
        3,
        3,
        start_column=0,
        end_column=len("écho 'x;y'".encode()),
    )
    assert _argv_bytes(result.commands[0]) == ("écho".encode(), b"x;y")
    assert result.issues == ()


def test_comments_quotes_and_non_shell_separators_never_create_raw_commands() -> None:
    quoted, _budget, _unit = _analyze(b"printf 'a;#b' plain # ignored\r\n")
    separated, _budget, _unit = _analyze(
        b"printf first\fsecond\rthird\x00four\n",
        path="scripts/separators.sh",
    )

    assert [_argv_bytes(command) for command in quoted.commands] == [(b"printf", b"a;#b", b"plain")]
    assert quoted.issues == ()
    assert [_argv_bytes(command) for command in separated.commands] == [
        (b"printf", b"first", b"second", None)
    ]
    assert [issue.reason for issue in separated.issues] == [
        dependency_types.ShellIssueReason.UNSUPPORTED_SEMANTICS
    ]


@pytest.mark.parametrize(
    "raw",
    [
        b"printf ~/file\n",
        b"printf file*\n",
        b"printf file?\n",
        b"printf file[ab]\n",
    ],
)
def test_unquoted_tilde_and_pathname_expansion_arguments_are_unknown(raw: bytes) -> None:
    result, _budget, _unit = _analyze(raw)

    assert result.commands[0].argv[0] == dependency_types.StaticValue.exact(b"printf")
    assert result.commands[0].argv[1].state is dependency_types.StaticValueState.UNKNOWN
    assert result.issues == ()


@pytest.mark.parametrize(
    "manager_name",
    [
        b"~/bin/npm",
        b"np*",
        b"np?",
        b"$manager",
        b"${manager}",
        b"npm-${channel}",
    ],
)
def test_unquoted_dynamic_manager_name_shapes_are_unknown(manager_name: bytes) -> None:
    result, _budget, _unit = _analyze(manager_name + b" install\n")

    assert result.commands[0].argv[0].state is dependency_types.StaticValueState.UNKNOWN
    assert result.commands[0].argv[1] == dependency_types.StaticValue.exact(b"install")
    assert result.issues == ()


def test_unquoted_tilde_assignment_value_is_unknown() -> None:
    result, _budget, _unit = _analyze(b"A=~/repo cmd\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN)
    ]
    assert _argv_bytes(result.commands[0]) == (b"cmd",)
    assert result.issues == ()


def test_unquoted_tilde_after_colon_in_assignment_values_is_unknown() -> None:
    result, _budget, _unit = _analyze(b"A=/bin:~/bin cmd\nexport PATH=/bin:~/bin\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN),
        ("PATH", dependency_types.StaticValueState.UNKNOWN),
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    "value",
    [
        b"~/repo",
        b"/bin:~/bin",
    ],
)
def test_continued_assignment_tilde_expansion_is_unknown(value: bytes) -> None:
    result, _budget, _unit = _analyze(b"A\\\n=" + value + b" cmd\n")

    assert [(assignment.name, assignment.value.state) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValueState.UNKNOWN)
    ]
    assert _argv_bytes(result.commands[0]) == (b"cmd",)
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_value"),
    [
        (b"A=foo\\\nbar cmd\n", dependency_types.StaticValue.exact(b"foobar")),
        (b"A=/bin:\\\n~/bin cmd\n", dependency_types.StaticValue.unknown()),
    ],
)
def test_prefix_assignment_continuation_absorbs_the_following_cst_name_fragment(
    raw: bytes,
    expected_value: dependency_types.StaticValue,
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", expected_value)
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"cmd",)]
    assert result.issues == ()


def test_export_continued_assignment_is_emitted_exactly_once() -> None:
    result, _budget, _unit = _analyze(b"export A=foo\\\nbar\n")

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foobar"))
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"export", b"A=foobar")]
    assert result.issues == ()


@pytest.mark.parametrize(
    "keyword",
    [b"declare", b"readonly", b"typeset", b"export"],
)
def test_declaration_continued_assignment_matches_full_argv_and_is_emitted_once(
    keyword: bytes,
) -> None:
    result, _budget, _unit = _analyze(keyword + b" A=foo\\\nbar\n")

    assert [_argv_bytes(command) for command in result.commands] == [(keyword, b"A=foobar")]
    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foobar"))
    ]
    assert result.issues == ()


def test_local_continued_assignment_in_function_matches_full_argv_and_is_emitted_once() -> None:
    result, _budget, _unit = _analyze(b"f() { local A=foo\\\nbar; }\n")

    assert [_argv_bytes(command) for command in result.commands] == [(b"local", b"A=foobar")]
    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foobar"))
    ]
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_argv", "expected_assignments"),
    [
        (b'printf "x"~/file\n', (b"printf", b"x~/file"), ()),
        (
            b'A="x"~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"x~/repo")),),
        ),
        (b"np\\\n~/bin install\n", (b"np~/bin", b"install"), ()),
    ],
)
def test_cross_fragment_tilde_outside_expansion_position_remains_exact(
    raw: bytes,
    expected_argv: tuple[bytes, ...],
    expected_assignments: tuple[tuple[str, dependency_types.StaticValue], ...],
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [expected_argv]
    assert (
        tuple((assignment.name, assignment.value) for assignment in result.assignments)
        == expected_assignments
    )
    assert result.issues == ()


def test_tilde_after_a_non_assignment_equals_in_the_value_remains_exact() -> None:
    result, _budget, _unit = _analyze(b"A=foo=~/repo cmd\n")

    assert [(assignment.name, assignment.value) for assignment in result.assignments] == [
        ("A", dependency_types.StaticValue.exact(b"foo=~/repo"))
    ]
    assert [_argv_bytes(command) for command in result.commands] == [(b"cmd",)]
    assert result.issues == ()


@pytest.mark.parametrize(
    ("raw", "expected_argv", "expected_assignments"),
    [
        (b'printf ""~/file\n', (b"printf", b"~/file"), ()),
        (
            b'A=""~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"~/repo")),),
        ),
        (
            b'A=/bin:""~/repo cmd\n',
            (b"cmd",),
            (("A", dependency_types.StaticValue.exact(b"/bin:~/repo")),),
        ),
    ],
)
def test_zero_length_quoted_fragments_block_tilde_expansion(
    raw: bytes,
    expected_argv: tuple[bytes, ...],
    expected_assignments: tuple[tuple[str, dependency_types.StaticValue], ...],
) -> None:
    result, _budget, _unit = _analyze(raw)

    assert [_argv_bytes(command) for command in result.commands] == [expected_argv]
    assert (
        tuple((assignment.name, assignment.value) for assignment in result.assignments)
        == expected_assignments
    )
    assert result.issues == ()


def test_quoted_and_escaped_tilde_and_pathname_characters_remain_exact() -> None:
    result, _budget, _unit = _analyze(
        b"printf \"~/file\" 'file*' file\\? file\\[ab\\] \\~/file file\\*\n"
        b"A=\\~/repo cmd\n"
        b"B=/bin:\\~/bin cmd\n"
        b'C="/bin:~/bin" cmd\n'
    )

    assert [_argv_bytes(command) for command in result.commands] == [
        (b"printf", b"~/file", b"file*", b"file?", b"file[ab]", b"~/file", b"file*"),
        (b"cmd",),
        (b"cmd",),
        (b"cmd",),
    ]
    assert [
        (assignment.name, assignment.value.exact_bytes) for assignment in result.assignments
    ] == [
        ("A", b"~/repo"),
        ("B", b"/bin:~/bin"),
        ("C", b"/bin:~/bin"),
    ]
    assert result.issues == ()


def test_missing_syntax_is_localized_without_discarding_proven_commands() -> None:
    result, _budget, _unit = _analyze(b"if condition; then body;\n")

    assert [command.argv[0].exact_bytes for command in result.commands] == [
        b"condition",
        b"body",
    ]
    assert [(issue.reason, issue.span.start_line) for issue in result.issues] == [
        (dependency_types.ShellIssueReason.SYNTAX_ERROR, 1)
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


def test_preparse_resource_denial_is_skipped_without_calling_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    extraction = _extract("scripts/limited.sh", b"printf ok\n", budget=budget)
    unit = extraction.units[0]
    file_budget = budget.for_file(unit.origin_span.path)
    assert file_budget.reserve_shell_parse(len(unit.raw_bytes)) is None
    assert file_budget.reserve_shell_parse(len(unit.raw_bytes)) is None

    def unexpected_parse(_source: bytes, **_kwargs: Any) -> Any:
        raise AssertionError("pre-parse denial must not invoke the parser")

    monkeypatch.setattr(shell_frontend, "parse_bash_source", unexpected_parse)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RESOURCE_LIMIT
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.SKIPPED


def test_exact_and_one_over_traversal_ir_and_value_budgets() -> None:
    raw = b"printf value\n"
    baseline_budget = DependencyWorkBudget()
    unit = _extract("scripts/bounds.sh", raw, budget=baseline_budget).units[0]
    baseline = shell_frontend.analyze_shell_unit(unit, budget=baseline_budget)
    assert baseline.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED
    baseline_file = baseline_budget.for_file(unit.origin_span.path)
    required = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: baseline_file.used_for_unit(
            unit, dependency_types.DependencyWorkResource.SHELL_CST_VISITS
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: baseline_budget.used(
            dependency_types.DependencyWorkResource.RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: baseline_file.used(
            dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES
        ),
    }
    limits = {
        dependency_types.DependencyWorkResource.SHELL_CST_VISITS: (
            dependency_types.DEPENDENCY_SHELL_CST_VISIT_FACTOR * len(raw)
            + dependency_types.DEPENDENCY_SHELL_CST_VISIT_BASE
        ),
        dependency_types.DependencyWorkResource.RETAINED_SHELL_IR: (
            dependency_types.MAX_DEPENDENCY_RETAINED_SHELL_IR
        ),
        dependency_types.DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES: (
            dependency_types.MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE
        ),
    }

    for resource, required_count in required.items():
        exact_budget = DependencyWorkBudget()
        exact_file = exact_budget.for_file(unit.origin_span.path)
        exact_file.register_shell_file_size(len(raw))
        precharge = limits[resource] - required_count
        if resource is dependency_types.DependencyWorkResource.SHELL_CST_VISITS:
            assert exact_file.charge_shell_cst_visits(unit, precharge) is None
        elif resource is dependency_types.DependencyWorkResource.RETAINED_SHELL_IR:
            assert exact_file.charge_retained_shell_ir(unit, precharge) is None
        else:
            assert exact_file.reserve_shell_value_bytes(precharge) is None
        exact = shell_frontend.analyze_shell_unit(unit, budget=exact_budget)
        assert exact.work_items[0].outcome is dependency_types.ShellWorkOutcome.COMPLETED

        over_budget = DependencyWorkBudget()
        over_file = over_budget.for_file(unit.origin_span.path)
        over_file.register_shell_file_size(len(raw))
        precharge += 1
        if resource is dependency_types.DependencyWorkResource.SHELL_CST_VISITS:
            assert over_file.charge_shell_cst_visits(unit, precharge) is None
        elif resource is dependency_types.DependencyWorkResource.RETAINED_SHELL_IR:
            assert over_file.charge_retained_shell_ir(unit, precharge) is None
        else:
            assert over_file.reserve_shell_value_bytes(precharge) is None
        over = shell_frontend.analyze_shell_unit(unit, budget=over_budget)
        assert over.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL
        assert dependency_types.ShellIssueReason.RESOURCE_LIMIT in {
            issue.reason for issue in over.issues
        }


def test_exact_parse_revisit_ceiling_calls_parser_twice_then_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/revisit.sh", b"printf ok\n", budget=budget).units[0]
    real_parse = shell_frontend.parse_bash_source
    calls = 0

    def recording_parse(source: bytes, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_parse(source, **kwargs)

    monkeypatch.setattr(shell_frontend, "parse_bash_source", recording_parse)

    first = shell_frontend.analyze_shell_unit(unit, budget=budget)
    second = shell_frontend.analyze_shell_unit(unit, budget=budget)
    denied = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert [first.work_items[0].outcome, second.work_items[0].outcome] == [
        dependency_types.ShellWorkOutcome.COMPLETED,
        dependency_types.ShellWorkOutcome.COMPLETED,
    ]
    assert denied.work_items[0].outcome is dependency_types.ShellWorkOutcome.SKIPPED
    assert calls == 2


def test_runtime_parser_failure_is_one_local_partial_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/runtime.sh", b"printf ok\n", budget=budget).units[0]

    def cancelled(_source: bytes, **_kwargs: Any) -> Any:
        raise shell_frontend.ShellParserError(
            outcome=shell_frontend.ShellParserOutcome.PARTIAL,
            reason=shell_frontend.ShellParserFailureReason.RUNTIME_LIMIT,
            deadline_tripped=True,
        )

    monkeypatch.setattr(shell_frontend, "parse_bash_source", cancelled)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [issue.reason for issue in result.issues] == [
        dependency_types.ShellIssueReason.RUNTIME_LIMIT
    ]
    assert result.work_items[0].outcome is dependency_types.ShellWorkOutcome.PARTIAL


@pytest.mark.parametrize(
    ("parser_outcome", "work_outcome"),
    [
        (
            shell_frontend.ShellParserOutcome.FAILED,
            dependency_types.ShellWorkOutcome.FAILED,
        ),
        (
            shell_frontend.ShellParserOutcome.PARTIAL,
            dependency_types.ShellWorkOutcome.PARTIAL,
        ),
    ],
)
def test_parser_unavailable_preserves_failed_or_preclassified_partial_outcome(
    monkeypatch: pytest.MonkeyPatch,
    parser_outcome: shell_frontend.ShellParserOutcome,
    work_outcome: dependency_types.ShellWorkOutcome,
) -> None:
    budget = DependencyWorkBudget()
    unit = _extract("scripts/unavailable.sh", b"printf ok\n", budget=budget).units[0]

    def unavailable(_source: bytes, **_kwargs: Any) -> Any:
        raise shell_frontend.ShellParserError(
            outcome=parser_outcome,
            reason=shell_frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE,
            deadline_tripped=False,
        )

    monkeypatch.setattr(shell_frontend, "parse_bash_source", unavailable)

    result = shell_frontend.analyze_shell_unit(unit, budget=budget)

    assert result.commands == ()
    assert [(issue.reason, issue.outcome) for issue in result.issues] == [
        (dependency_types.ShellIssueReason.SHELL_PARSER_UNAVAILABLE, work_outcome)
    ]
    assert result.work_items[0].outcome is work_outcome


@pytest.mark.timeout(10)
def test_deep_shell_nesting_is_walked_without_python_recursion() -> None:
    depth = 2_000
    raw = b"(" * depth + b"printf ok" + b")" * depth + b"\n"

    result, _budget, _unit = _analyze(raw, path="scripts/deep.sh")

    assert len(result.work_items) == 1
    assert result.work_items[0].outcome in {
        dependency_types.ShellWorkOutcome.COMPLETED,
        dependency_types.ShellWorkOutcome.PARTIAL,
    }
