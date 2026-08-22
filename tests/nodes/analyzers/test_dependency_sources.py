# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused black-box tests for direct dependency-source configuration files."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from skillspector.artifacts import ArtifactDisposition, ArtifactRecord, classify_artifact
from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_CONFIG_NODES,
    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
    MAX_DEPENDENCY_SOURCE_CHANGES,
    MAX_DEPENDENCY_SOURCE_RECORDS,
    DependencySourceLimitationReason,
    DependencyWorkBudget,
)


def _analyzer() -> Any:
    try:
        return importlib.import_module("skillspector.dependency_sources").analyze_dependency_sources
    except ImportError:
        pytest.fail("direct dependency-source analyzer is unavailable")


def _analyze(
    files: Mapping[str, str],
    *,
    components: Iterable[str] | None = None,
    raw_file_cache: Mapping[str, bytes] | None = None,
    local_file_cache: Mapping[str, str] | None = None,
    artifact_inventory: list[ArtifactRecord] | None = None,
    budget: DependencyWorkBudget | None = None,
) -> Any:
    raw = (
        dict(raw_file_cache)
        if raw_file_cache is not None
        else {path: content.encode("utf-8") for path, content in files.items()}
    )
    local = dict(local_file_cache) if local_file_cache is not None else dict(files)
    inventory = (
        artifact_inventory
        if artifact_inventory is not None
        else [classify_artifact(path, data) for path, data in raw.items()]
    )
    return _analyzer()(
        components=list(components) if components is not None else list(files),
        local_file_cache=local,
        raw_file_cache=raw,
        artifact_inventory=inventory,
        budget=budget or DependencyWorkBudget(),
    )


def _finding_projection(analysis: Any) -> list[dict[str, object]]:
    return [
        {
            **finding.evidence,
            "file": finding.file,
            "start_line": finding.start_line,
            "end_line": finding.end_line,
        }
        for finding in analysis.findings
    ]


def _assert_single_parse_limitation(analysis: Any, *, path: str, end_line: int) -> Any:
    assert analysis.findings == ()
    assert len(analysis.limitations) == 1
    limitation = analysis.limitations[0]
    assert limitation.reason is DependencySourceLimitationReason.PARSE_INCOMPLETE
    assert (limitation.path, limitation.start_line, limitation.end_line) == (path, 1, end_line)
    return limitation


def test_npm_uses_case_insensitive_last_values_and_code_owned_scopes() -> None:
    content = (
        "registry=https://first.example.invalid/simple\n"
        "REGISTRY = https://registry.npmjs.org/ # effective canonical default\n"
        '@Acme:Registry = "https://user:password@packages.example.invalid/team" ; note\n'
    )

    analysis = _analyze({"project/.npmrc": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "npm",
            "surface": ".npmrc",
            "operation": "replace",
            "scope": "scoped",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "project/.npmrc",
            "start_line": 3,
            "end_line": 3,
        }
    ]
    assert "Acme" not in repr(analysis)
    assert "password" not in repr(analysis)


def test_npm_keeps_semicolons_inside_urls_but_strips_whitespace_comments() -> None:
    content = "registry=https://packages.example.invalid/a;b ; explanation\n"

    analysis = _analyze({"npmrc": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "npm",
            "surface": ".npmrc",
            "operation": "replace",
            "scope": "global",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "npmrc",
            "start_line": 1,
            "end_line": 1,
        }
    ]


@pytest.mark.parametrize(
    "content",
    [
        "registry=\n",
        'registry="https://packages.example.invalid/simple\n',
        "registry='\n",
    ],
)
def test_npm_malformed_relevant_values_are_localized_limitations(content: str) -> None:
    analysis = _analyze({".npmrc": content})

    _assert_single_parse_limitation(analysis, path=".npmrc", end_line=2)


def test_pip_handles_delimiters_continuations_and_normalized_last_values() -> None:
    content = (
        "[global]\n"
        "index_url: https://first.example.invalid/simple\n"
        "INDEX-URL = https://packages.example.invalid/simple\n"
        "extra_index_url = https://a.example.invalid/simple\n"
        "    https://b.example.invalid/simple\n"
        "trusted-host = ignored.example.invalid\n"
        "[install]\n"
        "index-url: https://command.example.invalid/simple\n"
    )

    analysis = _analyze({"config/pip.conf": content})

    assert analysis.limitations == ()
    assert _finding_projection(analysis) == [
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "replace",
            "scope": "global",
            "destination": "https://packages.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 3,
            "end_line": 3,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "add",
            "scope": "global",
            "destination": "https://a.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 4,
            "end_line": 4,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "add",
            "scope": "global",
            "destination": "https://b.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 5,
            "end_line": 5,
        },
        {
            "ecosystem": "pip",
            "surface": "pip config",
            "operation": "replace",
            "scope": "command",
            "destination": "https://command.example.invalid/REDACTED_PATH",
            "destination_status": "resolved",
            "file": "config/pip.conf",
            "start_line": 8,
            "end_line": 8,
        },
    ]


def test_pip_sections_keep_exact_configparser_identity() -> None:
    content = (
        "[GLOBAL]\n"
        "index-url = https://first.example.invalid/simple\n"
        "[global]\n"
        "index_url = https://effective.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.start_line, finding.evidence["scope"], finding.evidence["destination"])
        for finding in analysis.findings
    ] == [
        (2, "command", "https://first.example.invalid/REDACTED_PATH"),
        (4, "global", "https://effective.example.invalid/REDACTED_PATH"),
    ]


def test_pip_same_indent_options_are_assignments_not_continuation_tokens() -> None:
    content = (
        "[global]\n"
        "  index-url = https://first.example.invalid/simple\n"
        "  extra-index-url = https://second.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [finding.evidence["operation"] for finding in analysis.findings] == ["replace", "add"]
    assert [finding.start_line for finding in analysis.findings] == [2, 3]


@pytest.mark.parametrize(
    ("option", "operation"),
    [("--index-url", "replace"), ("--EXTRA_INDEX_URL", "add")],
)
def test_pip_accepts_exactly_one_leading_double_dash(
    option: str,
    operation: str,
) -> None:
    analysis = _analyze(
        {"pip.conf": f"[global]\n{option}=https://packages.example.invalid/simple\n"}
    )

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    assert analysis.findings[0].evidence["operation"] == operation


@pytest.mark.parametrize("option", ["-index-url", "---index-url", "--trusted-host"])
def test_pip_rejects_invalid_dash_counts_and_unrelated_options(option: str) -> None:
    analysis = _analyze(
        {"pip.conf": f"[global]\n{option}=https://packages.example.invalid/simple\n"}
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_pip_double_dash_and_plain_spellings_share_last_value_semantics() -> None:
    content = (
        "[global]\n"
        "index-url=https://first.example.invalid/simple\n"
        "--INDEX_URL=https://effective.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [finding.start_line for finding in analysis.findings] == [3]
    assert analysis.findings[0].evidence["destination"] == (
        "https://effective.example.invalid/REDACTED_PATH"
    )


def test_pip_double_dash_value_has_an_exact_utf8_byte_span() -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    prefix = "# multibyte é\r\n[global]\r\n--INDEX_URL = "
    destination = "https://packages.example.invalid/simple"
    content = f"{prefix}{destination}\r\n"

    parsed = module._parse_file(
        "pip.conf",
        content,
        content.encode(),
        DependencyWorkBudget().for_file("pip.conf"),
    )

    assert parsed.limitations == ()
    assert len(parsed.changes) == 1
    assert (parsed.changes[0].span.start_byte, parsed.changes[0].span.end_byte) == (
        len(prefix.encode()),
        len(f"{prefix}{destination}".encode()),
    )


def test_pip_default_only_does_not_create_an_effective_concrete_source() -> None:
    analysis = _analyze(
        {"pip.conf": ("[DEFAULT]\nindex-url=https://packages.example.invalid/simple\n")}
    )

    assert analysis.findings == ()
    assert analysis.limitations == ()


def test_pip_concrete_override_suppresses_an_inherited_default() -> None:
    content = (
        "[DEFAULT]\n"
        "index-url=https://packages.example.invalid/simple\n"
        "[global]\n"
        "index-url=https://pypi.org/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("section", "scope"),
    [("global", "global"), ("install", "command")],
)
def test_pip_inherited_default_uses_concrete_scope_and_default_occurrence(
    section: str,
    scope: str,
) -> None:
    content = (
        f"[DEFAULT]\nindex-url=https://packages.example.invalid/simple\n[{section}]\ntimeout=30\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    finding = analysis.findings[0]
    assert finding.evidence["scope"] == scope
    assert finding.start_line == 2


def test_pip_default_inheritance_and_overrides_remain_independent_per_section() -> None:
    content = (
        "[DEFAULT]\n"
        "index-url=https://default.example.invalid/simple\n"
        "extra-index-url=https://extra.example.invalid/simple\n"
        "[global]\n"
        "index-url=https://pypi.org/simple\n"
        "[install]\n"
        "extra-index-url=https://install.example.invalid/simple\n"
        "[download]\n"
        "timeout=30\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.start_line, finding.evidence["operation"], finding.evidence["scope"])
        for finding in analysis.findings
    ] == [
        (2, "replace", "command"),
        (2, "replace", "command"),
        (3, "add", "global"),
        (3, "add", "command"),
        (7, "add", "command"),
    ]


def test_pip_queries_only_relevant_options_per_concrete_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    get_calls: list[tuple[str, str, bool]] = []
    items_calls: list[str] = []
    original_get = module._PipConfigParser.get
    original_items = module._PipConfigParser.items

    def counted_get(
        parser: Any,
        section: str,
        option: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        get_calls.append((section, option, kwargs.get("raw", False)))
        return original_get(parser, section, option, *args, **kwargs)

    def counted_items(parser: Any, section: str, *args: Any, **kwargs: Any) -> Any:
        items_calls.append(section)
        return original_items(parser, section, *args, **kwargs)

    monkeypatch.setattr(module._PipConfigParser, "get", counted_get)
    monkeypatch.setattr(module._PipConfigParser, "items", counted_items)
    irrelevant_defaults = "".join(f"setting-{index}=value-{index}\n" for index in range(64))
    content = (
        "[DEFAULT]\n"
        "index-url=https://default.example.invalid/simple\n"
        f"{irrelevant_defaults}"
        "[global]\n"
        "timeout=30\n"
        "[install]\n"
        "index-url=https://pypi.org/simple\n"
        "[download]\n"
        "extra-index-url=https://download.example.invalid/simple\n"
    )

    analysis = _analyze({"pip.conf": content})

    assert analysis.limitations == ()
    assert [
        (finding.evidence["operation"], finding.evidence["scope"], finding.start_line)
        for finding in analysis.findings
    ] == [
        ("replace", "global", 2),
        ("replace", "command", 2),
        ("add", "command", 72),
    ]
    assert items_calls == []
    assert get_calls == [
        (section, option, True)
        for section in ("global", "install", "download")
        for option in ("index-url", "extra-index-url")
    ]


@pytest.mark.parametrize(
    ("path", "content", "expected_start", "expected_end", "expected_line"),
    [
        (
            ".npmrc",
            "; multibyte é and lone carriage return \r stay on line one\r\n"
            "registry=https://packages.example.invalid/simple\r\n",
            len("; multibyte é and lone carriage return \r stay on line one\r\nregistry=".encode()),
            len(
                "; multibyte é and lone carriage return \r stay on line one\r\n"
                "registry=https://packages.example.invalid/simple".encode()
            ),
            2,
        ),
        (
            "pip.conf",
            "[global]\r\n"
            "# multibyte é and lone carriage return \r stay on line two\r\n"
            "extra-index-url = https://packages.example.invalid/simple\r\n",
            len(
                "[global]\r\n"
                "# multibyte é and lone carriage return \r stay on line two\r\n"
                "extra-index-url = ".encode()
            ),
            len(
                "[global]\r\n"
                "# multibyte é and lone carriage return \r stay on line two\r\n"
                "extra-index-url = https://packages.example.invalid/simple".encode()
            ),
            3,
        ),
    ],
)
def test_source_spans_use_utf8_bytes_and_only_lf_physical_line_boundaries(
    path: str,
    content: str,
    expected_start: int,
    expected_end: int,
    expected_line: int,
) -> None:
    module = importlib.import_module("skillspector.dependency_sources")
    raw = content.encode("utf-8")

    parsed = module._parse_file(
        path,
        content,
        raw,
        DependencyWorkBudget().for_file(path),
    )

    assert parsed.limitations == ()
    assert len(parsed.changes) == 1
    span = parsed.changes[0].span
    assert (span.start_byte, span.end_byte) == (expected_start, expected_end)
    assert (span.start_line, span.end_line) == (expected_line, expected_line)


@pytest.mark.parametrize(
    ("path", "content", "expected_status", "expected_destination"),
    [
        (".npmrc", "registry=${NPM_REGISTRY}\n", "unresolved", "unresolved"),
        (".npmrc", "registry=$NPM_REGISTRY\n", "resolved", "[REDACTED_URL]"),
        ("pip.ini", "[global]\nindex-url = %(mirror)s\n", "unresolved", "unresolved"),
        (
            "pip.ini",
            "[global]\nindex-url = https://packages.example.invalid/%2F\n",
            "resolved",
            "[REDACTED_URL]",
        ),
        ("pip.ini", "[global]\nindex-url = $PIP_INDEX_URL\n", "resolved", "[REDACTED_URL]"),
    ],
)
def test_interpolation_is_limited_to_manager_native_forms(
    path: str,
    content: str,
    expected_status: str,
    expected_destination: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.limitations == ()
    assert len(analysis.findings) == 1
    assert analysis.findings[0].evidence["destination_status"] == expected_status
    assert analysis.findings[0].evidence["destination"] == expected_destination


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (".npmrc", "registry=HTTPS://REGISTRY.NPMJS.ORG\n"),
        ("pip.conf", "[global]\nindex-url = HTTPS://PYPI.ORG/simple/\n"),
    ],
)
def test_exact_canonical_defaults_ignore_only_case_and_trailing_slash(
    path: str,
    content: str,
) -> None:
    analysis = _analyze({path: content})

    assert analysis.findings == ()
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    "value",
    [
        "https://registry.npmjs.org:443/",
        "https://registry.npmjs.org:/",
        "https://registry.npmjs.org/path",
        "https://registry.npmjs.org/?",
        "https://registry.npmjs.org/?query=1",
        "https://registry.npmjs.org/#",
        "https://registry.npmjs.org/#fragment",
    ],
)
def test_npm_canonical_origin_variants_remain_noncanonical(value: str) -> None:
    analysis = _analyze({".npmrc": f"registry={value}\n"})

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


def test_dispatch_uses_only_deduplicated_component_exact_basenames() -> None:
    files = {
        "a/.npmrc": "registry=https://a.example.invalid/simple\n",
        "b/pip.ini": "[global]\nindex-url=https://b.example.invalid/simple\n",
        "ignored/.npmrc.backup": "registry=https://ignored.example.invalid/simple\n",
        "cache-only/pip.conf": "[global]\nindex-url=https://ignored.example.invalid/simple\n",
    }

    analysis = _analyze(
        files,
        components=["b/pip.ini", "a/.npmrc", "a/.npmrc", "ignored/.npmrc.backup"],
    )

    assert [finding.file for finding in analysis.findings] == ["a/.npmrc", "b/pip.ini"]
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    ("path", "content", "expected_lines"),
    [
        (
            ".npmrc",
            "@scope:registry=https://first.example.invalid/simple\n"
            "registry=https://second.example.invalid/simple\n"
            "@SCOPE:REGISTRY=https://third.example.invalid/simple\n",
            [2, 3],
        ),
        (
            "pip.conf",
            "[global]\n"
            "index-url=https://first.example.invalid/simple\n"
            "extra-index-url=https://second.example.invalid/simple\n"
            "index_url=https://third.example.invalid/simple\n",
            [3, 4],
        ),
    ],
)
def test_effective_findings_are_ordered_by_occurrence_span(
    path: str,
    content: str,
    expected_lines: list[int],
) -> None:
    analysis = _analyze({path: content})

    assert [finding.start_line for finding in analysis.findings] == expected_lines
    assert analysis.limitations == ()


@pytest.mark.parametrize(
    "mutation",
    ["missing_inventory", "partial_inventory", "missing_raw", "missing_local", "cache_mismatch"],
)
def test_authoritative_input_failures_are_content_free_limitations(mutation: str) -> None:
    path = "pip.conf"
    content = "[global]\nindex-url=https://user:secret@packages.example.invalid/simple\n"
    raw = {path: content.encode()}
    local = {path: content}
    inventory = [classify_artifact(path, raw[path])]
    if mutation == "missing_inventory":
        inventory = []
    elif mutation == "partial_inventory":
        inventory[0]["disposition"] = ArtifactDisposition.PARTIAL
        inventory[0]["reason"] = "size_limit"
    elif mutation == "missing_raw":
        raw = {}
    elif mutation == "missing_local":
        local = {}
    else:
        local[path] = "[global]\nindex-url=https://different.example.invalid/simple\n"

    analysis = _analyze(
        {path: content},
        raw_file_cache=raw,
        local_file_cache=local,
        artifact_inventory=inventory,
    )

    _assert_single_parse_limitation(
        analysis,
        path=path,
        end_line=3 if path in raw else 1,
    )
    assert "secret" not in repr(analysis)


def test_invalid_utf8_is_not_analyzed_through_replacement_text() -> None:
    path = ".npmrc"
    raw = b"registry=https://packages.example.invalid/simple\xff\n"
    inventory = [classify_artifact(path, raw)]

    analysis = _analyze(
        {path: raw.decode("utf-8", errors="replace")},
        raw_file_cache={path: raw},
        artifact_inventory=inventory,
    )

    _assert_single_parse_limitation(analysis, path=path, end_line=2)


def test_inventory_size_proves_incomplete_physical_input_before_parsing() -> None:
    path = ".npmrc"
    content = "registry=https://packages.example.invalid/simple\n"
    raw = content.encode()
    inventory = classify_artifact(path, raw)
    inventory["size_bytes"] = 1_000_001

    analysis = _analyze({path: content}, artifact_inventory=[inventory])

    limitation = _assert_single_parse_limitation(analysis, path=path, end_line=2)
    assert limitation.ledger_metrics() == {
        "observed_bytes": 1_000_001,
        "limit_bytes": 1_000_000,
    }


def test_scan_wide_config_node_exhaustion_is_reported_without_a_partial_result() -> None:
    budget = DependencyWorkBudget()
    assert budget.charge_config_nodes(MAX_DEPENDENCY_CONFIG_NODES) is None

    analysis = _analyze(
        {".npmrc": "registry=https://packages.example.invalid/simple\n"},
        budget=budget,
    )

    limitation = _assert_single_parse_limitation(analysis, path=".npmrc", end_line=2)
    assert limitation.ledger_metrics() == {
        "observed_records": MAX_DEPENDENCY_CONFIG_NODES + 1,
        "limit_records": MAX_DEPENDENCY_CONFIG_NODES,
    }


_BUDGET_LITERAL = "https://packages.example.invalid/simple"


@pytest.mark.parametrize("resource", ["retained", "records", "changes"])
def test_candidate_budget_exact_limits_still_emit_the_finding(resource: str) -> None:
    budget = DependencyWorkBudget()
    if resource == "retained":
        assert (
            budget.charge_retained_literal_bytes(
                MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(_BUDGET_LITERAL.encode())
            )
            is None
        )
    elif resource == "records":
        assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - 1) is None
    else:
        assert budget.reserve_source_changes(MAX_DEPENDENCY_SOURCE_CHANGES - 1) is None

    analysis = _analyze({".npmrc": f"registry={_BUDGET_LITERAL}\n"}, budget=budget)

    assert len(analysis.findings) == 1
    assert analysis.limitations == ()


@pytest.mark.parametrize("resource", ["retained", "records", "changes"])
def test_candidate_budget_one_over_preserves_prior_reserved_change_and_adds_limitation(
    resource: str,
) -> None:
    budget = DependencyWorkBudget()
    if resource == "retained":
        assert (
            budget.charge_retained_literal_bytes(
                MAX_DEPENDENCY_RETAINED_LITERAL_BYTES - len(_BUDGET_LITERAL.encode())
            )
            is None
        )
    elif resource == "records":
        assert budget.charge_source_records(MAX_DEPENDENCY_SOURCE_RECORDS - 1) is None
    else:
        assert budget.reserve_source_changes(MAX_DEPENDENCY_SOURCE_CHANGES - 1) is None
    content = f"registry={_BUDGET_LITERAL}\n@scope:registry={_BUDGET_LITERAL}\n"

    analysis = _analyze({".npmrc": content}, budget=budget)

    assert [finding.start_line for finding in analysis.findings] == [1]
    assert len(analysis.limitations) == 1
    limitation = analysis.limitations[0]
    assert limitation.reason is DependencySourceLimitationReason.PARSE_INCOMPLETE
    assert limitation.ledger_metrics()
    assert set(limitation.ledger_metrics()) in (
        {"observed_bytes", "limit_bytes"},
        {"observed_records", "limit_records"},
        {"observed_findings", "limit_findings"},
    )


@pytest.mark.parametrize(
    "content",
    [
        "index-url=https://packages.example.invalid/simple\n",
        "[global\nindex-url=https://packages.example.invalid/simple\n",
        "[global]\nindex-url=\n",
    ],
)
def test_malformed_pip_configs_are_localized_limitations(content: str) -> None:
    analysis = _analyze({"pip.conf": content})

    _assert_single_parse_limitation(
        analysis,
        path="pip.conf",
        end_line=max(1, content.encode().count(b"\n") + 1),
    )
