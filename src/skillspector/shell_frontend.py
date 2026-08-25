# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-pinned extraction, parser boundary, and syntax-only shell lowering.

This module intentionally contains no package-manager policy or shell state.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache
from importlib import import_module
from math import ceil, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, cast
from warnings import catch_warnings, filterwarnings

from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_FILE_BYTES,
    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
    MAX_DEPENDENCY_RETAINED_SHELL_IR,
    MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
    MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE,
    MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
    AssignmentSite,
    CommandSite,
    DependencyFileBudget,
    DependencyWorkBudget,
    DependencyWorkExhaustion,
    DependencyWorkResource,
    ShellDialect,
    ShellExtractionResult,
    ShellFrontendResult,
    ShellIssue,
    ShellIssueReason,
    ShellTruncationClaimStatus,
    ShellUnit,
    ShellUnitKind,
    ShellWorkItem,
    ShellWorkOutcome,
    SiteProvenance,
    SourceMap,
    SourceMapEntry,
    SourceSpan,
    StaticValue,
    StaticValueState,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser, Tree

EXPECTED_BASH_ABI_VERSION: Final = 15
EXPECTED_BASH_SEMANTIC_VERSION: Final = (0, 25, 1)
MAX_TREE_SITTER_READ_BYTES: Final = 4_096
_MARKDOWN_SUFFIXES: Final = frozenset({".md", ".markdown", ".mdown", ".mkd"})
_UNSUPPORTED_SHELL_SUFFIXES: Final = frozenset({".zsh", ".envrc", ".ksh"})
_SUPPORTED_FENCE_DIALECTS: Final = {
    b"bash": ShellDialect.BASH,
    b"sh": ShellDialect.SH,
    b"shell-script": ShellDialect.SH,
    b"console": ShellDialect.SH,
}
_SUPPORTED_SHEBANG_DIALECTS: Final = {
    b"bash": ShellDialect.BASH,
    b"sh": ShellDialect.SH,
    b"dash": ShellDialect.DASH,
}
_UNSUPPORTED_SHEBANG_NAMES: Final = frozenset({b"zsh", b"ksh"})


@dataclass(frozen=True, slots=True)
class _PhysicalLine:
    start: int
    content_end: int
    full_end: int
    number: int


def _physical_lines(raw: bytes) -> tuple[_PhysicalLine, ...]:
    """Split only CRLF, LF, and lone CR physical boundaries."""
    lines: list[_PhysicalLine] = []
    line_start = 0
    line_number = 1
    cursor = 0
    while cursor < len(raw):
        byte = raw[cursor]
        if byte == 13:
            full_end = cursor + (2 if cursor + 1 < len(raw) and raw[cursor + 1] == 10 else 1)
        elif byte == 10:
            full_end = cursor + 1
        else:
            cursor += 1
            continue
        lines.append(_PhysicalLine(line_start, cursor, full_end, line_number))
        line_start = full_end
        line_number += 1
        cursor = full_end
    lines.append(_PhysicalLine(line_start, len(raw), len(raw), line_number))
    return tuple(lines)


def _line_starts(lines: tuple[_PhysicalLine, ...]) -> tuple[int, ...]:
    return tuple(line.start for line in lines)


def _span_for_bytes(
    path: str,
    starts: tuple[int, ...],
    start_byte: int,
    end_byte: int,
) -> SourceSpan:
    """Build an inclusive-line span around an exact half-open byte range."""
    start_index = max(0, bisect_right(starts, start_byte) - 1)
    if end_byte > start_byte:
        end_index = max(0, bisect_right(starts, end_byte - 1) - 1)
        end_column = end_byte - starts[end_index]
    else:
        end_index = start_index
        end_column = start_byte - starts[start_index]
    return SourceSpan(
        path,
        start_byte,
        end_byte,
        start_index + 1,
        end_index + 1,
        start_column=start_byte - starts[start_index],
        end_column=end_column,
    )


def _normalized_path(path: object) -> str:
    probe = SourceSpan(
        path,  # type: ignore[arg-type]
        0,
        0,
        1,
        1,
        start_column=0,
        end_column=0,
    )
    if probe.path != path:
        raise ValueError("path must already be normalized")
    return probe.path


def _path_suffix(path: str) -> str:
    basename = path.rsplit("/", 1)[-1].casefold()
    dot = basename.rfind(".")
    return basename[dot:] if dot >= 0 else ""


def _is_unsupported_executable_path(
    path: str,
    raw: bytes,
    lines: tuple[_PhysicalLine, ...],
) -> bool:
    basename = path.rsplit("/", 1)[-1]
    suffix = _path_suffix(path)
    if suffix in _UNSUPPORTED_SHELL_SUFFIXES:
        return True
    if basename == "Dockerfile" or basename.startswith("Dockerfile."):
        for line in lines:
            content = raw[line.start : line.content_end].lstrip(b" \t")
            if content[:3].lower() == b"run" and (len(content) == 3 or content[3] in {9, 32}):
                return True
        return False
    if basename in {"Makefile", "makefile", "GNUmakefile"} or suffix == ".mk":
        return any(raw[line.start : line.content_end].startswith(b"\t") for line in lines)
    return False


def _shebang_dialect(first_line: bytes) -> tuple[ShellDialect | None, bool]:
    if not first_line.startswith(b"#!"):
        return None, False
    words = first_line[2:].strip(b" \t").split()
    if not words:
        return None, False
    executable = words[0].rsplit(b"/", 1)[-1]
    if executable == b"env":
        word_index = 1
        if word_index < len(words) and words[word_index] == b"-S":
            word_index += 1
        if word_index >= len(words):
            return None, False
        executable = words[word_index].rsplit(b"/", 1)[-1]
    dialect = _SUPPORTED_SHEBANG_DIALECTS.get(executable)
    return dialect, executable in _UNSUPPORTED_SHEBANG_NAMES


def _fence_opener(line: bytes) -> tuple[int, int, bytes] | None:
    indentation = 0
    while indentation < len(line) and line[indentation] == 32:
        indentation += 1
    if indentation > 3 or indentation >= len(line):
        return None
    delimiter = line[indentation]
    if delimiter not in {96, 126}:
        return None
    delimiter_end = indentation
    while delimiter_end < len(line) and line[delimiter_end] == delimiter:
        delimiter_end += 1
    length = delimiter_end - indentation
    if length < 3:
        return None
    info = line[delimiter_end:].strip(b" \t")
    token = info.split(maxsplit=1)[0].lower() if info else b""
    return delimiter, length, token


def _is_fence_closer(line: bytes, delimiter: int, minimum_length: int) -> bool:
    indentation = 0
    while indentation < len(line) and line[indentation] == 32:
        indentation += 1
    if indentation > 3:
        return False
    delimiter_end = indentation
    while delimiter_end < len(line) and line[delimiter_end] == delimiter:
        delimiter_end += 1
    return (
        delimiter_end - indentation >= minimum_length and line[delimiter_end:].strip(b" \t") == b""
    )


def _retain_issue(
    issues: list[ShellIssue],
    issue: ShellIssue,
    *,
    file_budget: DependencyFileBudget,
) -> bool:
    exhaustion = file_budget.charge_shell_issues(1)
    if exhaustion is None:
        issues.append(issue)
        return True
    if file_budget.claim_reserved_shell_truncation_issue() is ShellTruncationClaimStatus.CLAIMED:
        issues.append(
            ShellIssue(
                reason=ShellIssueReason.RESOURCE_LIMIT,
                outcome=ShellWorkOutcome.PARTIAL,
                span=issue.span,
                exhaustion=exhaustion,
            )
        )
    return False


def _resource_issue(span: SourceSpan, exhaustion: DependencyWorkExhaustion) -> ShellIssue:
    return ShellIssue(
        reason=ShellIssueReason.RESOURCE_LIMIT,
        outcome=ShellWorkOutcome.PARTIAL,
        span=span,
        exhaustion=exhaustion,
    )


def _reserve_shell_unit(
    file_budget: DependencyFileBudget,
    *,
    source_map_entries: int,
) -> DependencyWorkExhaustion | None:
    next_units = file_budget.used(DependencyWorkResource.SHELL_UNITS) + 1
    next_entries = (
        file_budget.used(DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES) + source_map_entries
    )
    if next_units > MAX_DEPENDENCY_SHELL_UNITS_PER_FILE:
        return DependencyWorkExhaustion(
            DependencyWorkResource.SHELL_UNITS,
            next_units,
            MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
        )
    if next_entries > MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE:
        return DependencyWorkExhaustion(
            DependencyWorkResource.SHELL_SOURCE_MAP_ENTRIES,
            next_entries,
            MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
        )
    if exhaustion := file_budget.charge_shell_units(1):
        return exhaustion
    if exhaustion := file_budget.charge_source_map_entries(source_map_entries):
        raise RuntimeError("atomic shell-unit reservation invariant failed")
    return None


def _append_unit(
    units: list[ShellUnit],
    issues: list[ShellIssue],
    *,
    path: str,
    raw: bytes,
    starts: tuple[int, ...],
    unit_start: int,
    unit_end: int,
    dialect: ShellDialect,
    kind: ShellUnitKind,
    provenance: SiteProvenance,
    file_budget: DependencyFileBudget,
    mapped: bool,
) -> bool:
    unit_raw = raw[unit_start:unit_end]
    invalid_range: tuple[int, int] | None = None
    try:
        unit_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        invalid_range = (error.start, max(error.end, error.start + 1))
    entry_count = 1 if invalid_range is None and mapped and unit_raw else 0
    span = _span_for_bytes(path, starts, unit_start, unit_end)
    if exhaustion := _reserve_shell_unit(file_budget, source_map_entries=entry_count):
        _retain_issue(issues, _resource_issue(span, exhaustion), file_budget=file_budget)
        return False
    if invalid_range is not None:
        invalid_start, invalid_end = invalid_range
        invalid_span = _span_for_bytes(
            path,
            starts,
            unit_start + invalid_start,
            unit_start + invalid_end,
        )
        _retain_issue(
            issues,
            ShellIssue(
                reason=ShellIssueReason.SYNTAX_ERROR,
                outcome=ShellWorkOutcome.PARTIAL,
                span=invalid_span,
            ),
            file_budget=file_budget,
        )
        return True
    source_map = (
        SourceMap(
            path=path,
            entries=(SourceMapEntry(0, len(unit_raw), unit_start, unit_end),) if unit_raw else (),
            child_size_bytes=len(unit_raw),
            physical_size_bytes=len(raw),
            physical_line_starts=starts,
        )
        if mapped
        else None
    )
    units.append(
        ShellUnit(
            dialect=dialect,
            kind=kind,
            provenance=provenance,
            raw_bytes=unit_raw,
            origin_span=span,
            source_map=source_map,
        )
    )
    return True


def _unsupported_issue(path: str, starts: tuple[int, ...], raw: bytes) -> ShellIssue:
    return ShellIssue(
        reason=ShellIssueReason.UNSUPPORTED_SEMANTICS,
        outcome=ShellWorkOutcome.PARTIAL,
        span=_span_for_bytes(path, starts, 0, len(raw)),
    )


def _extract_markdown_units(
    *,
    path: str,
    raw: bytes,
    lines: tuple[_PhysicalLine, ...],
    starts: tuple[int, ...],
    file_budget: DependencyFileBudget,
) -> ShellExtractionResult:
    units: list[ShellUnit] = []
    issues: list[ShellIssue] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        opener = _fence_opener(raw[line.start : line.content_end])
        if opener is None:
            line_index += 1
            continue
        delimiter, minimum_length, token = opener
        closer_index: int | None = None
        candidate_index = line_index + 1
        while candidate_index < len(lines):
            candidate = lines[candidate_index]
            if _is_fence_closer(
                raw[candidate.start : candidate.content_end],
                delimiter,
                minimum_length,
            ):
                closer_index = candidate_index
                break
            candidate_index += 1
        content_start = line.full_end
        content_end = lines[closer_index].start if closer_index is not None else len(raw)
        dialect = _SUPPORTED_FENCE_DIALECTS.get(token)
        if dialect is not None:
            retained = _append_unit(
                units,
                issues,
                path=path,
                raw=raw,
                starts=starts,
                unit_start=content_start,
                unit_end=content_end,
                dialect=dialect,
                kind=ShellUnitKind.MARKDOWN_FENCE,
                provenance=SiteProvenance.MARKDOWN_FENCE,
                file_budget=file_budget,
                mapped=True,
            )
            if not retained:
                break
            if closer_index is None:
                issue_span = _span_for_bytes(path, starts, line.start, len(raw))
                unit_id = units[-1].unit_id if units else None
                _retain_issue(
                    issues,
                    ShellIssue(
                        reason=ShellIssueReason.SYNTAX_ERROR,
                        outcome=ShellWorkOutcome.PARTIAL,
                        span=issue_span,
                        unit_id=unit_id,
                    ),
                    file_budget=file_budget,
                )
        elif not token:
            _retain_issue(
                issues,
                ShellIssue(
                    reason=ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    outcome=ShellWorkOutcome.PARTIAL,
                    span=_span_for_bytes(path, starts, line.start, content_end),
                ),
                file_budget=file_budget,
            )
        if closer_index is None:
            break
        line_index = closer_index + 1
    return ShellExtractionResult(units=tuple(units), issues=tuple(issues))


def extract_shell_units(
    path: str,
    raw_bytes: bytes,
    *,
    executable_paths: frozenset[str],
    budget: DependencyWorkBudget,
) -> ShellExtractionResult:
    """Extract bounded applicable units without interpreting shell commands."""
    normalized_path = _normalized_path(path)
    if type(raw_bytes) is not bytes:
        raise TypeError("raw_bytes must be canonical immutable bytes")
    if not isinstance(executable_paths, frozenset):
        raise ValueError("executable_paths must be an immutable set")
    if not isinstance(budget, DependencyWorkBudget):
        raise ValueError("budget must be a DependencyWorkBudget")
    file_budget = budget.for_file(normalized_path)
    file_budget.register_shell_file_size(len(raw_bytes))
    lines = _physical_lines(raw_bytes)
    starts = _line_starts(lines)
    suffix = _path_suffix(normalized_path)
    if suffix in _MARKDOWN_SUFFIXES:
        return _extract_markdown_units(
            path=normalized_path,
            raw=raw_bytes,
            lines=lines,
            starts=starts,
            file_budget=file_budget,
        )
    if _is_unsupported_executable_path(normalized_path, raw_bytes, lines):
        issues: list[ShellIssue] = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    first_line = raw_bytes[lines[0].start : lines[0].content_end]
    shebang_dialect, unsupported_shebang = _shebang_dialect(first_line)
    if unsupported_shebang:
        issues = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    dialect = shebang_dialect
    provenance = SiteProvenance.SHEBANG
    if dialect is None and suffix in {".sh", ".bash"}:
        dialect = ShellDialect.BASH if suffix == ".bash" else ShellDialect.SH
        provenance = SiteProvenance.FILE_SUFFIX
    if dialect is not None:
        units: list[ShellUnit] = []
        issues = []
        _append_unit(
            units,
            issues,
            path=normalized_path,
            raw=raw_bytes,
            starts=starts,
            unit_start=0,
            unit_end=len(raw_bytes),
            dialect=dialect,
            kind=ShellUnitKind.STANDALONE,
            provenance=provenance,
            file_budget=file_budget,
            mapped=False,
        )
        return ShellExtractionResult(units=tuple(units), issues=tuple(issues))
    if normalized_path in executable_paths:
        issues = []
        _retain_issue(
            issues,
            _unsupported_issue(normalized_path, starts, raw_bytes),
            file_budget=file_budget,
        )
        return ShellExtractionResult(issues=tuple(issues))
    return ShellExtractionResult()


class ShellParserOutcome(StrEnum):
    """Terminal classification for a parser-runtime failure."""

    FAILED = "failed"
    PARTIAL = "partial"


class ShellParserFailureReason(StrEnum):
    """Content-free parser-runtime failure reason."""

    SHELL_PARSER_UNAVAILABLE = "shell_parser_unavailable"
    RUNTIME_LIMIT = "runtime_limit"


class ShellParserError(RuntimeError):
    """A sanitized, classified parser-runtime failure."""

    def __init__(
        self,
        *,
        outcome: ShellParserOutcome,
        reason: ShellParserFailureReason,
        deadline_tripped: bool,
    ) -> None:
        self.outcome = outcome
        self.reason = reason
        self.deadline_tripped = deadline_tripped
        super().__init__(f"{outcome.value}: {reason.value}")


def _unavailable_error(*, meaningful_work: bool = False) -> ShellParserError:
    return ShellParserError(
        outcome=(ShellParserOutcome.PARTIAL if meaningful_work else ShellParserOutcome.FAILED),
        reason=ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE,
        deadline_tripped=False,
    )


@cache
def load_bash_language() -> Language:
    """Load and validate the exact Bash grammar once, on first use."""
    try:
        tree_sitter = import_module("tree_sitter")
        tree_sitter_bash = import_module("tree_sitter_bash")
        language = cast("Language", tree_sitter.Language(tree_sitter_bash.language()))
        if language.abi_version != EXPECTED_BASH_ABI_VERSION:
            raise ValueError("unexpected Bash grammar ABI")
        if language.semantic_version != EXPECTED_BASH_SEMANTIC_VERSION:
            raise ValueError("unexpected Bash grammar semantic version")
    except ShellParserError:
        raise
    except Exception:
        raise _unavailable_error() from None
    return language


def create_bash_parser(*, deadline_monotonic: float | None = None) -> Parser:
    """Create one fresh parser around the cached immutable language."""
    try:
        tree_sitter = import_module("tree_sitter")
        language = load_bash_language()
        timeout_micros = _native_timeout_micros(deadline_monotonic)
        if timeout_micros is None:
            return cast("Parser", tree_sitter.Parser(language))
        with catch_warnings():
            filterwarnings(
                "ignore",
                message=r"Use the progress_callback in parse\(\)",
                category=DeprecationWarning,
            )
            return cast(
                "Parser",
                tree_sitter.Parser(
                    language,
                    timeout_micros=timeout_micros,
                ),
            )
    except ShellParserError:
        raise
    except Exception:
        raise _unavailable_error() from None


def _bounded_reader(
    source: bytes,
    *,
    deadline_expired: Callable[[], bool],
) -> Callable[[int, Any], bytes]:
    if type(source) is not bytes:
        raise TypeError("source must be immutable bytes")

    def read(byte_offset: int, _position: Any) -> bytes:
        if deadline_expired():
            return b""
        if byte_offset < 0 or byte_offset >= len(source):
            return b""
        return source[byte_offset : byte_offset + MAX_TREE_SITTER_READ_BYTES]

    return read


def _native_timeout_micros(deadline_monotonic: float | None) -> int | None:
    if deadline_monotonic is None:
        return None
    if isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float)):
        raise TypeError("deadline_monotonic must be a finite monotonic timestamp")
    if not isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    return max(1, ceil((deadline_monotonic - monotonic()) * 1_000_000))


def parse_bash_source(
    source: bytes,
    *,
    deadline_monotonic: float | None = None,
    meaningful_work: bool = False,
) -> Tree:
    """Parse immutable bytes through a bounded reader with classified failures."""
    deadline_tripped = False

    def deadline_expired() -> bool:
        nonlocal deadline_tripped
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            deadline_tripped = True
        return deadline_tripped

    try:
        reader = _bounded_reader(source, deadline_expired=deadline_expired)
        parser = create_bash_parser(deadline_monotonic=deadline_monotonic)
        tree = parser.parse(reader)
    except ShellParserError as error:
        if meaningful_work and error.reason is ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE:
            raise _unavailable_error(meaningful_work=True) from None
        raise
    except Exception:
        if deadline_expired():
            raise ShellParserError(
                outcome=ShellParserOutcome.PARTIAL,
                reason=ShellParserFailureReason.RUNTIME_LIMIT,
                deadline_tripped=True,
            ) from None
        raise _unavailable_error(meaningful_work=meaningful_work) from None

    if deadline_expired():
        raise ShellParserError(
            outcome=ShellParserOutcome.PARTIAL,
            reason=ShellParserFailureReason.RUNTIME_LIMIT,
            deadline_tripped=True,
        )
    if tree is None:
        raise _unavailable_error(meaningful_work=meaningful_work)
    return tree


class _ExecutionRegionKind(StrEnum):
    """Code-owned structural regions retained for later same-parse analysis."""

    PROGRAM = "program"
    LIST = "list"
    PIPELINE = "pipeline"
    IF = "if"
    ELIF = "elif"
    ELSE = "else"
    FOR = "for"
    C_STYLE_FOR = "c_style_for"
    WHILE = "while"
    UNTIL = "until"
    DO = "do"
    CASE = "case"
    CASE_ITEM = "case_item"
    FUNCTION = "function"
    COMPOUND = "compound"
    SUBSHELL = "subshell"
    COMMAND_SUBSTITUTION = "command_substitution"
    PROCESS_SUBSTITUTION = "process_substitution"
    NEGATION = "negation"
    REDIRECTED = "redirected"


class _RedirectKind(StrEnum):
    """Narrow syntax-proven redirect facts; deliberately not an FD model."""

    STDOUT_TRUNCATE = "stdout_truncate"
    STDOUT_CLOBBER = "stdout_clobber"
    STDOUT_APPEND = "stdout_append"
    STDOUT_STDERR_TRUNCATE = "stdout_stderr_truncate"


_REGION_NODE_TYPES: Final[dict[str, _ExecutionRegionKind]] = {
    "program": _ExecutionRegionKind.PROGRAM,
    "list": _ExecutionRegionKind.LIST,
    "pipeline": _ExecutionRegionKind.PIPELINE,
    "if_statement": _ExecutionRegionKind.IF,
    "elif_clause": _ExecutionRegionKind.ELIF,
    "else_clause": _ExecutionRegionKind.ELSE,
    "for_statement": _ExecutionRegionKind.FOR,
    "c_style_for_statement": _ExecutionRegionKind.C_STYLE_FOR,
    "while_statement": _ExecutionRegionKind.WHILE,
    "until_statement": _ExecutionRegionKind.UNTIL,
    "do_group": _ExecutionRegionKind.DO,
    "case_statement": _ExecutionRegionKind.CASE,
    "case_item": _ExecutionRegionKind.CASE_ITEM,
    "function_definition": _ExecutionRegionKind.FUNCTION,
    "compound_statement": _ExecutionRegionKind.COMPOUND,
    "subshell": _ExecutionRegionKind.SUBSHELL,
    "command_substitution": _ExecutionRegionKind.COMMAND_SUBSTITUTION,
    "process_substitution": _ExecutionRegionKind.PROCESS_SUBSTITUTION,
    "negated_command": _ExecutionRegionKind.NEGATION,
    "redirected_statement": _ExecutionRegionKind.REDIRECTED,
}
_SUBSTITUTION_NODE_TYPES: Final = frozenset({"command_substitution", "process_substitution"})
_DYNAMIC_VALUE_NODE_TYPES: Final = frozenset(
    {
        "simple_expansion",
        "expansion",
        "parameter_expansion",
        "arithmetic_expansion",
        "command_substitution",
        "process_substitution",
    }
)
_DECLARATION_KEYWORDS: Final = frozenset({"declare", "export", "local", "readonly", "typeset"})
_DECLARATION_KEYWORD_BYTES: Final[frozenset[bytes]] = frozenset(
    keyword.encode("ascii") for keyword in _DECLARATION_KEYWORDS
)
_SUPPORTED_REDIRECT_OPERATORS: Final = frozenset({">", ">|", ">>", "&>"})
_COMPOUND_REDIRECT_BODIES: Final = frozenset(
    {
        "compound_statement",
        "subshell",
        "if_statement",
        "for_statement",
        "c_style_for_statement",
        "while_statement",
        "until_statement",
        "case_statement",
        "function_definition",
    }
)
_PINNED_CST_FIELDS: Final[dict[str, frozenset[str]]] = {
    "program": frozenset(),
    "command": frozenset({"name", "argument", "redirect"}),
    "variable_assignment": frozenset({"name", "value"}),
    "declaration_command": frozenset(),
    "redirected_statement": frozenset({"body", "redirect"}),
    "file_redirect": frozenset({"descriptor", "destination"}),
    "function_definition": frozenset({"name", "body"}),
    "if_statement": frozenset({"condition"}),
    "for_statement": frozenset({"variable", "value", "body"}),
    "while_statement": frozenset({"condition", "body"}),
    "case_statement": frozenset({"value"}),
    "case_item": frozenset({"value", "termination"}),
    "pipeline": frozenset(),
    "list": frozenset(),
    "negated_command": frozenset(),
    "subshell": frozenset(),
    "compound_statement": frozenset(),
    "command_substitution": frozenset(),
    "process_substitution": frozenset(),
    "heredoc_redirect": frozenset({"argument"}),
    "ERROR": frozenset(),
    "MISSING": frozenset(),
}


@dataclass(frozen=True, slots=True)
class _ValueFragment:
    """A physical source fragment mapped into one retained logical value."""

    span: SourceSpan
    value_start_byte: int
    value_end_byte: int
    exact: bool


@dataclass(frozen=True, slots=True)
class _ArgumentIR:
    value: StaticValue
    span: SourceSpan
    fragments: tuple[_ValueFragment, ...]
    local_start_byte: int = field(repr=False)
    local_end_byte: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ExecutionRegion:
    region_id: int
    order: int
    kind: _ExecutionRegionKind
    span: SourceSpan
    parent_region_id: int | None
    function_id: int | None


@dataclass(frozen=True, slots=True)
class _FunctionContext:
    function_id: int
    name: StaticValue
    span: SourceSpan
    fragments: tuple[_ValueFragment, ...]
    parent_function_id: int | None


@dataclass(frozen=True, slots=True)
class _RedirectFact:
    kind: _RedirectKind
    target: _ArgumentIR
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _AssignmentIR:
    site: AssignmentSite
    order: int
    region_id: int | None
    function_id: int | None
    prefix_for_command_start_byte: int | None
    value_fragments: tuple[_ValueFragment, ...]


@dataclass(frozen=True, slots=True)
class _CommandIR:
    site: CommandSite
    order: int
    region_id: int | None
    function_id: int | None
    arguments: tuple[_ArgumentIR, ...]
    prefix_assignments: tuple[AssignmentSite, ...]
    redirects: tuple[_RedirectFact, ...]


@dataclass(frozen=True, slots=True)
class _ShellProgramIR:
    regions: tuple[_ExecutionRegion, ...] = ()
    functions: tuple[_FunctionContext, ...] = ()
    commands: tuple[_CommandIR, ...] = ()
    assignments: tuple[_AssignmentIR, ...] = ()


@dataclass(frozen=True, slots=True)
class _ShellAnalysisResult:
    public: ShellFrontendResult
    program: _ShellProgramIR


@dataclass(frozen=True, slots=True)
class _FoldedValue:
    value: StaticValue
    fragments: tuple[tuple[int, int, int, int, bool], ...]
    unquoted_tilde_offsets: tuple[int, ...] = ()
    unquoted_assignment_delimiter_offsets: tuple[int, ...] = ()
    quoted_empty_offsets: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _NodeGroup:
    nodes: tuple[Node, ...]
    start_byte: int
    end_byte: int
    raw_syntax: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _CommandDraft:
    node: Node = field(repr=False)
    region_id: int | None
    function_id: int | None
    substitution_depth: int
    non_file_redirect_count: int


@dataclass(frozen=True, slots=True)
class _AssignmentDraft:
    node: Node = field(repr=False)
    region_id: int | None
    function_id: int | None
    prefix_for_command_start_byte: int | None
    declaration_command_start_byte: int | None


@dataclass(frozen=True, slots=True)
class _FunctionDraft:
    node: Node = field(repr=False)
    function_id: int
    parent_function_id: int | None


@dataclass(frozen=True, slots=True)
class _RedirectOwner:
    statement_start_byte: int
    body_start_byte: int
    body_end_byte: int
    body_type: str
    substitution_depth: int
    has_error_child: bool
    redirect_count: int


@dataclass(frozen=True, slots=True)
class _RedirectDraft:
    node: Node = field(repr=False)
    command_start_byte: int | None
    statement_start_byte: int | None


@dataclass(frozen=True, slots=True)
class _WrapperArgumentsDraft:
    nodes: tuple[Node, ...] = field(repr=False)
    statement_start_byte: int


@dataclass(frozen=True, slots=True)
class _WalkFrame:
    node: Node = field(repr=False)
    parent_type: str | None
    parent_start_byte: int | None
    region_id: int | None
    function_id: int | None
    substitution_depth: int
    command_owner_start_byte: int | None
    redirect_owner_start_byte: int | None


def _node_key(node: Node) -> tuple[int, int, str]:
    return node.start_byte, node.end_byte, node.type


def _shell_work_item(unit: ShellUnit, outcome: ShellWorkOutcome) -> ShellWorkItem:
    return ShellWorkItem(
        unit_id=unit.unit_id,
        dialect=unit.dialect,
        kind=unit.kind,
        provenance=unit.provenance,
        span=unit.origin_span,
        outcome=outcome,
    )


class _ShellLowerer:
    """One bounded, iterative CST walk plus typed syntax-only projection."""

    def __init__(
        self,
        unit: ShellUnit,
        budget: DependencyWorkBudget,
        file_budget: DependencyFileBudget,
    ) -> None:
        self.unit = unit
        self.raw = unit.raw_bytes
        self.budget = budget
        self.file_budget = file_budget
        self.local_lines = _physical_lines(self.raw)
        self.local_line_starts = _line_starts(self.local_lines)
        self.issues: list[ShellIssue] = []
        self.regions: list[_ExecutionRegion] = []
        self.function_drafts: list[_FunctionDraft] = []
        self.command_drafts: list[_CommandDraft] = []
        self.assignment_drafts: list[_AssignmentDraft] = []
        self.redirect_owners: dict[int, _RedirectOwner] = {}
        self.redirect_drafts: list[_RedirectDraft] = []
        self.wrapper_argument_drafts: list[_WrapperArgumentsDraft] = []
        self.syntax_error_command_starts: set[int] = set()
        self.syntax_error_redirect_starts: set[int] = set()
        self.halted = False
        self.partial = False

    def _identity_span(self, start_byte: int, end_byte: int) -> SourceSpan:
        local = _span_for_bytes(
            self.unit.origin_span.path,
            self.local_line_starts,
            start_byte,
            end_byte,
        )
        origin_column = self.unit.origin_span.start_column or 0
        start_column = local.start_column
        end_column = local.end_column
        if local.start_line == 1 and start_column is not None:
            start_column += origin_column
        if local.end_line == 1 and end_column is not None:
            end_column += origin_column
        return SourceSpan(
            local.path,
            self.unit.origin_span.start_byte + local.start_byte,
            self.unit.origin_span.start_byte + local.end_byte,
            self.unit.origin_span.start_line + local.start_line - 1,
            self.unit.origin_span.start_line + local.end_line - 1,
            start_column=start_column,
            end_column=end_column,
        )

    def _span(self, start_byte: int, end_byte: int) -> SourceSpan | None:
        if end_byte <= start_byte:
            return None
        if self.unit.source_map is None:
            return self._identity_span(start_byte, end_byte)
        return self.unit.source_map.map_range(start_byte, end_byte)

    def _point_span(self, byte_offset: int) -> SourceSpan:
        bounded = min(max(byte_offset, 0), len(self.raw))
        if self.unit.source_map is None:
            return self._identity_span(bounded, bounded)
        if bounded < len(self.raw):
            following = self.unit.source_map.map_range(bounded, bounded + 1)
            if following is not None:
                return replace(
                    following, end_byte=following.start_byte, end_column=following.start_column
                )
        if bounded > 0:
            preceding = self.unit.source_map.map_range(bounded - 1, bounded)
            if preceding is not None:
                return SourceSpan(
                    preceding.path,
                    preceding.end_byte,
                    preceding.end_byte,
                    preceding.end_line,
                    preceding.end_line,
                    start_column=preceding.end_column,
                    end_column=preceding.end_column,
                )
        return replace(
            self.unit.origin_span,
            end_byte=self.unit.origin_span.start_byte,
            end_line=self.unit.origin_span.start_line,
            end_column=self.unit.origin_span.start_column,
        )

    def _node_span(self, node: Node) -> SourceSpan | None:
        return self._span(node.start_byte, node.end_byte)

    def _issue(
        self,
        reason: ShellIssueReason,
        span: SourceSpan,
        *,
        outcome: ShellWorkOutcome = ShellWorkOutcome.PARTIAL,
        exhaustion: DependencyWorkExhaustion | None = None,
    ) -> None:
        self.partial = True
        _retain_issue(
            self.issues,
            ShellIssue(
                reason=reason,
                outcome=outcome,
                span=span,
                unit_id=self.unit.unit_id,
                exhaustion=exhaustion,
            ),
            file_budget=self.file_budget,
        )

    def _resource(self, exhaustion: DependencyWorkExhaustion, span: SourceSpan) -> None:
        self._issue(ShellIssueReason.RESOURCE_LIMIT, span, exhaustion=exhaustion)
        self.halted = True

    def _reserve(self, count: int, *, value_bytes: int = 0, span: SourceSpan) -> bool:
        next_ir = self.budget.used(DependencyWorkResource.RETAINED_SHELL_IR) + count
        if next_ir > MAX_DEPENDENCY_RETAINED_SHELL_IR:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.RETAINED_SHELL_IR,
                    next_ir,
                    MAX_DEPENDENCY_RETAINED_SHELL_IR,
                ),
                span,
            )
            return False
        next_file_values = (
            self.file_budget.used(DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES) + value_bytes
        )
        if next_file_values > MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.SHELL_RETAINED_VALUE_BYTES,
                    next_file_values,
                    MAX_DEPENDENCY_SHELL_VALUE_BYTES_PER_FILE,
                ),
                span,
            )
            return False
        next_global_values = (
            self.budget.used(DependencyWorkResource.RETAINED_LITERAL_BYTES) + value_bytes
        )
        if next_global_values > MAX_DEPENDENCY_RETAINED_LITERAL_BYTES:
            self._resource(
                DependencyWorkExhaustion(
                    DependencyWorkResource.RETAINED_LITERAL_BYTES,
                    next_global_values,
                    MAX_DEPENDENCY_RETAINED_LITERAL_BYTES,
                ),
                span,
            )
            return False
        if self.file_budget.charge_retained_shell_ir(self.unit, count) is not None:
            raise RuntimeError("atomic retained shell IR reservation invariant failed")
        if self.file_budget.reserve_shell_value_bytes(value_bytes) is not None:
            raise RuntimeError("atomic retained shell value reservation invariant failed")
        return True

    def _mapped_fragments(
        self,
        fragments: tuple[tuple[int, int, int, int, bool], ...],
        *,
        fallback_start: int,
    ) -> tuple[_ValueFragment, ...] | None:
        mapped: list[_ValueFragment] = []
        for source_start, source_end, value_start, value_end, exact in fragments:
            span = self._span(source_start, source_end)
            if span is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(source_start if source_start >= 0 else fallback_start),
                )
                return None
            mapped.append(_ValueFragment(span, value_start, value_end, exact))
        return tuple(mapped)

    @staticmethod
    def _decode_unquoted(
        raw: bytes,
    ) -> tuple[bytes | None, bool, tuple[int, ...], tuple[int, ...]]:
        decoded = bytearray()
        has_pathname_expansion = False
        tilde_offsets: list[int] = []
        assignment_delimiter_offsets: list[int] = []
        index = 0
        while index < len(raw):
            byte = raw[index]
            if byte == 92:
                if index + 1 >= len(raw):
                    return None, False, (), ()
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                decoded.append(following)
                index += 2
                continue
            if byte in {0, 10, 12, 13}:
                return None, False, (), ()
            if byte in {42, 63, 91}:
                has_pathname_expansion = True
            elif byte == 126:
                tilde_offsets.append(len(decoded))
            elif byte == 58:
                assignment_delimiter_offsets.append(len(decoded))
            decoded.append(byte)
            index += 1
        return (
            bytes(decoded),
            has_pathname_expansion,
            tuple(tilde_offsets),
            tuple(assignment_delimiter_offsets),
        )

    @staticmethod
    def _decode_double_quoted(raw: bytes) -> bytes | None:
        decoded = bytearray()
        index = 0
        while index < len(raw):
            byte = raw[index]
            if byte == 92 and index + 1 < len(raw):
                following = raw[index + 1]
                if following == 10:
                    index += 2
                    continue
                if following in {36, 96, 34, 92}:
                    decoded.append(following)
                    index += 2
                    continue
            if byte in {0, 10, 13}:
                return None
            decoded.append(byte)
            index += 1
        return bytes(decoded)

    @staticmethod
    def _unknown_fold(node: Node) -> _FoldedValue:
        return _FoldedValue(
            StaticValue.unknown(),
            ((node.start_byte, node.end_byte, 0, 0, False),),
        )

    @staticmethod
    def _apply_tilde_context(
        folded: _FoldedValue,
        *,
        assignment_value: bool,
    ) -> _FoldedValue:
        if folded.value.state is not StaticValueState.EXACT:
            return folded
        dynamic = False
        delimiters = folded.unquoted_assignment_delimiter_offsets
        quoted_empty_offsets = folded.quoted_empty_offsets
        for offset in folded.unquoted_tilde_offsets:
            quote_index = bisect_left(quoted_empty_offsets, offset)
            if (
                quote_index < len(quoted_empty_offsets)
                and quoted_empty_offsets[quote_index] == offset
            ):
                continue
            if offset == 0:
                dynamic = True
                break
            if assignment_value:
                delimiter_index = bisect_left(delimiters, offset - 1)
                if delimiter_index < len(delimiters) and delimiters[delimiter_index] == offset - 1:
                    dynamic = True
                    break
        if not dynamic:
            return folded
        return _FoldedValue(
            StaticValue.unknown(),
            tuple(
                (source_start, source_end, 0, 0, False)
                for source_start, source_end, _value_start, _value_end, _exact in folded.fragments
            ),
        )

    def _fold_node(self, root: Node) -> _FoldedValue:
        results: dict[tuple[int, int, str], _FoldedValue] = {}
        stack: list[tuple[Node, bool]] = [(root, False)]
        while stack:
            node, visited = stack.pop()
            key = _node_key(node)
            if not visited:
                if node.type in _DYNAMIC_VALUE_NODE_TYPES:
                    results[key] = self._unknown_fold(node)
                    continue
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(node.children))
                continue
            node_type = node.type
            raw = self.raw[node.start_byte : node.end_byte]
            if node_type in _DYNAMIC_VALUE_NODE_TYPES:
                results[key] = self._unknown_fold(node)
                continue
            if node_type == "raw_string":
                value = raw[1:-1] if len(raw) >= 2 else b""
                results[key] = _FoldedValue(
                    StaticValue.exact(value),
                    (
                        (
                            node.start_byte + 1,
                            max(node.start_byte + 1, node.end_byte - 1),
                            0,
                            len(value),
                            True,
                        ),
                    )
                    if value
                    else (),
                    quoted_empty_offsets=(0,) if not value else (),
                )
                continue
            if node_type in {"word", "variable_name", "number"}:
                (
                    decoded_unquoted,
                    has_pathname_expansion,
                    tilde_offsets,
                    assignment_delimiter_offsets,
                ) = self._decode_unquoted(raw)
                if decoded_unquoted is None:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(node) or self._point_span(node.start_byte),
                    )
                    results[key] = self._unknown_fold(node)
                elif node_type == "word" and has_pathname_expansion:
                    results[key] = self._unknown_fold(node)
                else:
                    results[key] = _FoldedValue(
                        StaticValue.exact(decoded_unquoted),
                        ((node.start_byte, node.end_byte, 0, len(decoded_unquoted), True),)
                        if raw
                        else (),
                        tilde_offsets if node_type == "word" else (),
                        assignment_delimiter_offsets if node_type == "word" else (),
                    )
                continue
            if node_type == "string_content":
                decoded_quoted = self._decode_double_quoted(raw)
                if decoded_quoted is None:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(node) or self._point_span(node.start_byte),
                    )
                    results[key] = self._unknown_fold(node)
                else:
                    results[key] = _FoldedValue(
                        StaticValue.exact(decoded_quoted),
                        ((node.start_byte, node.end_byte, 0, len(decoded_quoted), True),)
                        if raw
                        else (),
                    )
                continue
            if node_type == "command_name":
                named = node.named_children
                results[key] = (
                    results[_node_key(named[0])] if len(named) == 1 else self._unknown_fold(node)
                )
                continue
            if node_type == "variable_assignment":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node is None:
                    results[key] = self._unknown_fold(node)
                    continue
                name_fold = results[_node_key(name_node)]
                value_fold = (
                    results[_node_key(value_node)]
                    if value_node is not None
                    else _FoldedValue(StaticValue.exact(b""), ())
                )
                if (
                    name_fold.value.state is StaticValueState.EXACT
                    and value_fold.value.state is StaticValueState.EXACT
                ):
                    name_bytes = cast(bytes, name_fold.value.exact_bytes)
                    value_bytes = cast(bytes, value_fold.value.exact_bytes)
                    combined = name_bytes + b"=" + value_bytes
                    value_offset = len(name_bytes) + 1
                    results[key] = self._apply_tilde_context(
                        _FoldedValue(
                            StaticValue.exact(combined),
                            ((node.start_byte, node.end_byte, 0, len(combined), True),),
                            tuple(
                                value_offset + offset
                                for offset in value_fold.unquoted_tilde_offsets
                            ),
                            (
                                len(name_bytes),
                                *(
                                    value_offset + offset
                                    for offset in value_fold.unquoted_assignment_delimiter_offsets
                                ),
                            ),
                            tuple(
                                value_offset + offset for offset in value_fold.quoted_empty_offsets
                            ),
                        ),
                        assignment_value=True,
                    )
                else:
                    results[key] = self._unknown_fold(node)
                continue
            if node_type in {"string", "concatenation"}:
                parts = [
                    results[_node_key(child)]
                    for child in node.named_children
                    if child.type not in {'"', "'"}
                ]
                exact = all(part.value.state is StaticValueState.EXACT for part in parts)
                output = bytearray()
                combined_fragments: list[tuple[int, int, int, int, bool]] = []
                combined_tilde_offsets: list[int] = []
                combined_assignment_delimiter_offsets: list[int] = []
                combined_quoted_empty_offsets: list[int] = []
                output_cursor = 0
                for part in parts:
                    part_bytes = part.value.exact_bytes if exact else None
                    for (
                        source_start,
                        source_end,
                        value_start,
                        value_end,
                        fragment_exact,
                    ) in part.fragments:
                        combined_fragments.append(
                            (
                                source_start,
                                source_end,
                                output_cursor + value_start if exact else 0,
                                output_cursor + value_end if exact else 0,
                                fragment_exact and exact,
                            )
                        )
                    if part_bytes is not None:
                        combined_tilde_offsets.extend(
                            output_cursor + offset for offset in part.unquoted_tilde_offsets
                        )
                        combined_assignment_delimiter_offsets.extend(
                            output_cursor + offset
                            for offset in part.unquoted_assignment_delimiter_offsets
                        )
                        combined_quoted_empty_offsets.extend(
                            output_cursor + offset for offset in part.quoted_empty_offsets
                        )
                        output.extend(part_bytes)
                        output_cursor += len(part_bytes)
                if node_type == "string" and exact and not output:
                    combined_quoted_empty_offsets.append(0)
                results[key] = _FoldedValue(
                    StaticValue.exact(bytes(output)) if exact else StaticValue.unknown(),
                    tuple(combined_fragments) or ((node.start_byte, node.end_byte, 0, 0, exact),),
                    tuple(combined_tilde_offsets),
                    tuple(combined_assignment_delimiter_offsets),
                    tuple(combined_quoted_empty_offsets),
                )
                continue
            if not node.is_named:
                results[key] = _FoldedValue(
                    StaticValue.exact(raw),
                    ((node.start_byte, node.end_byte, 0, len(raw), True),) if raw else (),
                )
                continue
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            results[key] = self._unknown_fold(node)
        return results[_node_key(root)]

    def _fold_group(
        self,
        group: _NodeGroup,
        *,
        assignment_value: bool = False,
        assignment_equals_offset: int | None = None,
    ) -> _FoldedValue:
        parts = [self._fold_node(node) for node in group.nodes]
        exact = all(part.value.state is StaticValueState.EXACT for part in parts)
        output = bytearray()
        output_cursor = 0
        fragments: list[tuple[int, int, int, int, bool]] = []
        tilde_offsets: list[int] = []
        assignment_delimiter_offsets: list[int] = []
        quoted_empty_offsets: list[int] = []
        for part in parts:
            for source_start, source_end, value_start, value_end, fragment_exact in part.fragments:
                fragments.append(
                    (
                        source_start,
                        source_end,
                        output_cursor + value_start if exact else 0,
                        output_cursor + value_end if exact else 0,
                        fragment_exact and exact,
                    )
                )
            if exact:
                value = cast(bytes, part.value.exact_bytes)
                tilde_offsets.extend(
                    output_cursor + offset for offset in part.unquoted_tilde_offsets
                )
                assignment_delimiter_offsets.extend(
                    output_cursor + offset for offset in part.unquoted_assignment_delimiter_offsets
                )
                quoted_empty_offsets.extend(
                    output_cursor + offset for offset in part.quoted_empty_offsets
                )
                output.extend(value)
                output_cursor += len(value)
        if assignment_equals_offset is not None:
            equals_index = bisect_left(assignment_delimiter_offsets, assignment_equals_offset)
            if (
                equals_index == len(assignment_delimiter_offsets)
                or assignment_delimiter_offsets[equals_index] != assignment_equals_offset
            ):
                assignment_delimiter_offsets.insert(equals_index, assignment_equals_offset)
        return self._apply_tilde_context(
            _FoldedValue(
                StaticValue.exact(bytes(output)) if exact else StaticValue.unknown(),
                tuple(fragments),
                tuple(tilde_offsets),
                tuple(assignment_delimiter_offsets),
                tuple(quoted_empty_offsets),
            ),
            assignment_value=assignment_value,
        )

    def _group_nodes(self, nodes: list[Node]) -> list[_NodeGroup]:
        groups: list[list[Node]] = []
        for node in sorted(nodes, key=lambda item: (item.start_byte, item.end_byte)):
            if groups and self.raw[groups[-1][-1].end_byte : node.start_byte] == b"\\\n":
                groups[-1].append(node)
            else:
                groups.append([node])
        return [
            _NodeGroup(
                tuple(group),
                group[0].start_byte,
                group[-1].end_byte,
                b"".join(self.raw[node.start_byte : node.end_byte] for node in group),
            )
            for group in groups
        ]

    def _argument(self, group: _NodeGroup) -> _ArgumentIR | None:
        span = self._span(group.start_byte, group.end_byte)
        if span is None:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._point_span(group.start_byte),
            )
            return None
        folded = self._fold_group(group)
        fragments = self._mapped_fragments(
            folded.fragments,
            fallback_start=group.start_byte,
        )
        if fragments is None:
            return None
        value_bytes = (
            len(cast(bytes, folded.value.exact_bytes))
            if folded.value.state is StaticValueState.EXACT
            else 0
        )
        if not self._reserve(1 + len(fragments), value_bytes=value_bytes, span=span):
            return None
        return _ArgumentIR(
            folded.value,
            span,
            fragments,
            group.start_byte,
            group.end_byte,
        )

    @staticmethod
    def _identifier(raw: bytes) -> str | None:
        if not raw or not (raw[0] == 95 or 65 <= raw[0] <= 90 or 97 <= raw[0] <= 122):
            return None
        if any(
            not (byte == 95 or 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122)
            for byte in raw[1:]
        ):
            return None
        return raw.decode("ascii")

    def _group_assignment_name(self, group: _NodeGroup) -> tuple[str, int] | None:
        equals = group.raw_syntax.find(b"=")
        if equals <= 0:
            return None
        name = self._identifier(group.raw_syntax[:equals])
        return (name, equals) if name is not None else None

    def _group_assignment_parts(
        self,
        group: _NodeGroup,
    ) -> tuple[str, StaticValue, tuple[_ValueFragment, ...]] | None:
        syntax = self._group_assignment_name(group)
        if syntax is None:
            return None
        name, equals = syntax
        folded = self._fold_group(
            group,
            assignment_value=True,
            assignment_equals_offset=equals,
        )
        if folded.value.state is StaticValueState.EXACT:
            exact = cast(bytes, folded.value.exact_bytes)
            value_offset = equals + 1
            value = StaticValue.exact(exact[value_offset:])
        else:
            value = StaticValue.unknown()
            value_offset = 0
        mapped = self._mapped_fragments(folded.fragments, fallback_start=group.start_byte)
        if mapped is None:
            return None
        if value.state is StaticValueState.EXACT:
            fragments = tuple(
                replace(
                    fragment,
                    value_start_byte=max(0, fragment.value_start_byte - value_offset),
                    value_end_byte=max(0, fragment.value_end_byte - value_offset),
                )
                for fragment in mapped
                if fragment.value_end_byte > value_offset
            )
        else:
            fragments = mapped
        return name, value, fragments

    def _assignment_from_group(
        self,
        group: _NodeGroup,
        *,
        order: int,
        region_id: int | None,
        function_id: int | None,
        prefix_for_command_start_byte: int | None,
    ) -> _AssignmentIR | None:
        parts = self._group_assignment_parts(group)
        if parts is None:
            return None
        name, value, fragments = parts
        span = self._span(group.start_byte, group.end_byte)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(group.start_byte))
            return None
        value_bytes = (
            len(cast(bytes, value.exact_bytes)) if value.state is StaticValueState.EXACT else 0
        )
        if not self._reserve(2 + len(fragments), value_bytes=value_bytes, span=span):
            return None
        site = AssignmentSite(
            unit_id=self.unit.unit_id,
            provenance=self.unit.provenance,
            span=span,
            name=name,
            value=value,
        )
        return _AssignmentIR(
            site,
            order,
            region_id,
            function_id,
            prefix_for_command_start_byte,
            fragments,
        )

    def _assignment_from_node(
        self,
        draft: _AssignmentDraft,
        *,
        order: int,
    ) -> _AssignmentIR | None:
        node = draft.node
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = self._identifier(self.raw[name_node.start_byte : name_node.end_byte])
        if name is None:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None
        value_node = node.child_by_field_name("value")
        if value_node is None:
            value = StaticValue.exact(b"")
            fragments: tuple[_ValueFragment, ...] = ()
        else:
            folded = self._apply_tilde_context(
                self._fold_node(value_node),
                assignment_value=True,
            )
            value = folded.value
            mapped = self._mapped_fragments(folded.fragments, fallback_start=value_node.start_byte)
            if mapped is None:
                return None
            fragments = mapped
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return None
        value_bytes = (
            len(cast(bytes, value.exact_bytes)) if value.state is StaticValueState.EXACT else 0
        )
        if not self._reserve(2 + len(fragments), value_bytes=value_bytes, span=span):
            return None
        return _AssignmentIR(
            AssignmentSite(
                unit_id=self.unit.unit_id,
                provenance=self.unit.provenance,
                span=span,
                name=name,
                value=value,
            ),
            order,
            draft.region_id,
            draft.function_id,
            draft.prefix_for_command_start_byte,
            fragments,
        )

    def _add_region(
        self,
        node: Node,
        kind: _ExecutionRegionKind,
        *,
        parent_region_id: int | None,
        function_id: int | None,
    ) -> int | None:
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return parent_region_id
        if not self._reserve(1, span=span):
            return parent_region_id
        region_id = len(self.regions)
        self.regions.append(
            _ExecutionRegion(
                region_id,
                region_id,
                kind,
                span,
                parent_region_id,
                function_id,
            )
        )
        return region_id

    def walk(self, root: Node) -> None:
        stack = [_WalkFrame(root, None, None, None, None, 0, None, None)]
        while stack and not self.halted:
            frame = stack.pop()
            node = frame.node
            node_span = self._node_span(node) or self._point_span(node.start_byte)
            if exhaustion := self.file_budget.charge_shell_cst_visits(self.unit, 1):
                self._resource(exhaustion, node_span)
                break
            if node.is_error:
                if frame.command_owner_start_byte is not None:
                    self.syntax_error_command_starts.add(frame.command_owner_start_byte)
                if frame.redirect_owner_start_byte is not None:
                    self.syntax_error_redirect_starts.add(frame.redirect_owner_start_byte)
                self._issue(ShellIssueReason.SYNTAX_ERROR, node_span)
                continue
            if node.is_missing:
                if frame.command_owner_start_byte is not None:
                    self.syntax_error_command_starts.add(frame.command_owner_start_byte)
                if frame.redirect_owner_start_byte is not None:
                    self.syntax_error_redirect_starts.add(frame.redirect_owner_start_byte)
                self._issue(ShellIssueReason.SYNTAX_ERROR, node_span)
                continue

            region_id = frame.region_id
            function_id = frame.function_id
            substitution_depth = frame.substitution_depth
            command_owner_start_byte = frame.command_owner_start_byte
            redirect_owner_start_byte = frame.redirect_owner_start_byte
            if node.type in {"command", "declaration_command"}:
                command_owner_start_byte = node.start_byte
            elif node.type == "redirected_statement":
                redirect_owner_start_byte = node.start_byte
            kind = _REGION_NODE_TYPES.get(node.type)
            if kind is not None:
                if node.type in _SUBSTITUTION_NODE_TYPES:
                    substitution_depth += 1
                if node.type == "function_definition":
                    new_function_id = len(self.function_drafts)
                    if not self._reserve(1, span=node_span):
                        break
                    self.function_drafts.append(_FunctionDraft(node, new_function_id, function_id))
                    function_id = new_function_id
                region_id = self._add_region(
                    node,
                    kind,
                    parent_region_id=frame.region_id,
                    function_id=function_id,
                )

            if node.type in {"command", "declaration_command"}:
                if not self._reserve(1, span=node_span):
                    break
                self.command_drafts.append(
                    _CommandDraft(
                        node,
                        region_id,
                        function_id,
                        substitution_depth,
                        sum(
                            node.field_name_for_child(index) == "redirect"
                            and child.type != "file_redirect"
                            for index, child in enumerate(node.children)
                        ),
                    )
                )
            elif node.type == "variable_assignment":
                prefix = frame.parent_start_byte if frame.parent_type == "command" else None
                declaration = (
                    frame.parent_start_byte if frame.parent_type == "declaration_command" else None
                )
                if not self._reserve(1, span=node_span):
                    break
                self.assignment_drafts.append(
                    _AssignmentDraft(node, region_id, function_id, prefix, declaration)
                )
            elif node.type == "redirected_statement":
                body = node.child_by_field_name("body")
                if body is not None:
                    if not self._reserve(1, span=node_span):
                        break
                    self.redirect_owners[node.start_byte] = _RedirectOwner(
                        node.start_byte,
                        body.start_byte,
                        body.end_byte,
                        body.type,
                        substitution_depth,
                        any(child.is_error or child.is_missing for child in node.children),
                        sum(
                            node.field_name_for_child(index) == "redirect"
                            for index in range(len(node.children))
                        ),
                    )
            elif node.type == "file_redirect":
                if not self._reserve(1, span=node_span):
                    break
                self.redirect_drafts.append(
                    _RedirectDraft(
                        node,
                        frame.parent_start_byte if frame.parent_type == "command" else None,
                        frame.parent_start_byte
                        if frame.parent_type == "redirected_statement"
                        else None,
                    )
                )
            elif node.type == "heredoc_redirect" and frame.parent_type == "redirected_statement":
                argument_nodes = tuple(
                    child
                    for index, child in enumerate(node.children)
                    if node.field_name_for_child(index) == "argument"
                )
                if argument_nodes:
                    if not self._reserve(1, span=node_span):
                        break
                    self.wrapper_argument_drafts.append(
                        _WrapperArgumentsDraft(
                            argument_nodes,
                            cast(int, frame.parent_start_byte),
                        )
                    )

            child_frames: list[_WalkFrame] = []
            previous_child: Node | None = None
            for child in node.children:
                child_command_owner = command_owner_start_byte
                child_redirect_owner = redirect_owner_start_byte
                if (
                    (child.is_error or child.is_missing)
                    and previous_child is not None
                    and previous_child.end_point.row == child.start_point.row
                ):
                    if previous_child.type in {"command", "declaration_command"}:
                        child_command_owner = previous_child.start_byte
                    elif previous_child.type == "redirected_statement":
                        child_redirect_owner = previous_child.start_byte
                child_frames.append(
                    _WalkFrame(
                        child,
                        node.type,
                        node.start_byte,
                        region_id,
                        function_id,
                        substitution_depth,
                        child_command_owner,
                        child_redirect_owner,
                    )
                )
                previous_child = child
            stack.extend(reversed(child_frames))

    def _command_groups(self, node: Node) -> list[_NodeGroup]:
        nodes: list[Node] = []
        if node.type == "declaration_command":
            if node.children:
                nodes.append(node.children[0])
            nodes.extend(node.named_children)
        else:
            for index, child in enumerate(node.children):
                if child.type == "variable_assignment" or node.field_name_for_child(index) in {
                    "name",
                    "argument",
                }:
                    nodes.append(child)
        return self._group_nodes(nodes)

    @staticmethod
    def _redirect_owner_command_start(
        owner: _RedirectOwner,
        commands_by_depth: dict[int, tuple[_CommandDraft, ...]],
        starts_by_depth: dict[int, tuple[int, ...]],
    ) -> int | None:
        if owner.body_type in _COMPOUND_REDIRECT_BODIES:
            return None
        commands = commands_by_depth.get(owner.substitution_depth, ())
        starts = starts_by_depth.get(owner.substitution_depth, ())
        left = bisect_left(starts, owner.body_start_byte)
        right = bisect_left(starts, owner.body_end_byte)
        for index in range(right - 1, left - 1, -1):
            command = commands[index]
            if command.node.end_byte <= owner.body_end_byte:
                return command.node.start_byte
        return None

    def _redirect_command_start(
        self,
        draft: _RedirectDraft,
        commands_by_depth: dict[int, tuple[_CommandDraft, ...]],
        starts_by_depth: dict[int, tuple[int, ...]],
    ) -> int | None:
        if draft.command_start_byte is not None:
            return draft.command_start_byte
        if draft.statement_start_byte is None:
            return None
        owner = self.redirect_owners.get(draft.statement_start_byte)
        if owner is None:
            return None
        return self._redirect_owner_command_start(owner, commands_by_depth, starts_by_depth)

    def _redirect_parts(
        self,
        draft: _RedirectDraft,
        *,
        allow_fact: bool,
    ) -> tuple[_RedirectFact | None, list[_ArgumentIR], bool]:
        node = draft.node
        descriptor = node.child_by_field_name("descriptor")
        destination_nodes = [
            child
            for index, child in enumerate(node.children)
            if node.field_name_for_child(index) == "destination"
        ]
        groups = self._group_nodes(destination_nodes)
        extras: list[_ArgumentIR] = []
        if not groups:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None, extras, False
        for group in groups[1:]:
            argument = self._argument(group)
            if argument is not None:
                extras.append(argument)
            if self.halted:
                return None, extras, False
        operator_node = next(
            (
                child
                for child in node.children
                if not child.is_named and child.type in _SUPPORTED_REDIRECT_OPERATORS
            ),
            None,
        )
        operator = operator_node.type if operator_node is not None else ""
        descriptor_raw = (
            self.raw[descriptor.start_byte : descriptor.end_byte] if descriptor is not None else b""
        )
        kind: _RedirectKind | None = None
        if operator == ">" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_TRUNCATE
        elif operator == ">|" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_CLOBBER
        elif operator == ">>" and descriptor_raw in {b"", b"1"}:
            kind = _RedirectKind.STDOUT_APPEND
        elif operator == "&>" and not descriptor_raw:
            kind = _RedirectKind.STDOUT_STDERR_TRUNCATE
        raw_operator = next(
            (
                child.type
                for child in node.children
                if not child.is_named and child.type not in {"\n"}
            ),
            "",
        )
        unsupported = kind is None and (
            raw_operator in {"&>>", ">&", "<&"}
            or descriptor is not None
            or operator in _SUPPORTED_REDIRECT_OPERATORS
        )
        if unsupported:
            self._issue(
                ShellIssueReason.UNSUPPORTED_SEMANTICS,
                self._node_span(node) or self._point_span(node.start_byte),
            )
            return None, extras, False
        if kind is None:
            return None, extras, True
        if not allow_fact:
            return None, extras, False
        target = self._argument(groups[0])
        if target is None:
            return None, extras, False
        span = self._node_span(node)
        if span is None:
            self._issue(ShellIssueReason.UNSUPPORTED_SEMANTICS, self._point_span(node.start_byte))
            return None, extras, False
        if not self._reserve(1, span=span):
            return None, extras, False
        return _RedirectFact(kind, target, span), extras, True

    def _functions(self) -> tuple[_FunctionContext, ...]:
        functions: list[_FunctionContext] = []
        for draft in self.function_drafts:
            name_node = draft.node.child_by_field_name("name")
            if name_node is None:
                continue
            group = _NodeGroup(
                (name_node,),
                name_node.start_byte,
                name_node.end_byte,
                self.raw[name_node.start_byte : name_node.end_byte],
            )
            argument = self._argument(group)
            if argument is None:
                if self.halted:
                    break
                continue
            if not self._reserve(1, span=argument.span):
                break
            functions.append(
                _FunctionContext(
                    draft.function_id,
                    argument.value,
                    argument.span,
                    argument.fragments,
                    draft.parent_function_id,
                )
            )
        return tuple(functions)

    def lower(self) -> _ShellProgramIR:
        if self.halted:
            return _ShellProgramIR(regions=tuple(self.regions))
        sorted_command_drafts = sorted(self.command_drafts, key=lambda item: item.node.start_byte)
        mutable_commands_by_depth: dict[int, list[_CommandDraft]] = {}
        for command in sorted_command_drafts:
            mutable_commands_by_depth.setdefault(command.substitution_depth, []).append(command)
        commands_by_depth = {
            depth: tuple(commands) for depth, commands in mutable_commands_by_depth.items()
        }
        starts_by_depth = {
            depth: tuple(command.node.start_byte for command in commands)
            for depth, commands in commands_by_depth.items()
        }
        redirects_by_command: dict[int, list[_RedirectDraft]] = {}
        for redirect in self.redirect_drafts:
            command_start = self._redirect_command_start(
                redirect,
                commands_by_depth,
                starts_by_depth,
            )
            if command_start is None:
                owner = (
                    self.redirect_owners.get(redirect.statement_start_byte)
                    if redirect.statement_start_byte is not None
                    else None
                )
                if owner is not None and owner.body_type in _COMPOUND_REDIRECT_BODIES:
                    self._issue(
                        ShellIssueReason.UNSUPPORTED_SEMANTICS,
                        self._node_span(redirect.node)
                        or self._point_span(redirect.node.start_byte),
                    )
                continue
            redirects_by_command.setdefault(command_start, []).append(redirect)

        wrapper_arguments_by_command: dict[int, list[_NodeGroup]] = {}
        for wrapper in self.wrapper_argument_drafts:
            owner = self.redirect_owners.get(wrapper.statement_start_byte)
            if owner is None:
                continue
            command_start = self._redirect_owner_command_start(
                owner,
                commands_by_depth,
                starts_by_depth,
            )
            if command_start is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(owner.body_start_byte),
                )
                continue
            wrapper_arguments_by_command.setdefault(command_start, []).extend(
                self._group_nodes(list(wrapper.nodes))
            )

        assignments: list[_AssignmentIR] = []
        assignment_keys: set[tuple[int, int]] = set()
        prefix_sites_by_command: dict[int, list[AssignmentSite]] = {}
        grouped_declaration_starts = {
            draft.node.start_byte
            for draft in sorted_command_drafts
            if draft.node.type == "declaration_command"
            and draft.node.children
            and self.raw[draft.node.children[0].start_byte : draft.node.children[0].end_byte]
            in _DECLARATION_KEYWORD_BYTES
        }
        for order, draft in enumerate(
            sorted(self.assignment_drafts, key=lambda item: item.node.start_byte)
        ):
            if (
                draft.prefix_for_command_start_byte is not None
                or draft.declaration_command_start_byte in grouped_declaration_starts
            ):
                continue
            assignment = self._assignment_from_node(draft, order=order)
            if assignment is not None:
                assignments.append(assignment)
                assignment_keys.add((draft.node.start_byte, draft.node.end_byte))
                if assignment.prefix_for_command_start_byte is not None:
                    prefix_sites_by_command.setdefault(
                        assignment.prefix_for_command_start_byte, []
                    ).append(assignment.site)
            if self.halted:
                break

        commands: list[_CommandIR] = []
        for command_draft in sorted_command_drafts:
            if self.halted:
                break
            groups = self._command_groups(command_draft.node)
            if not groups:
                continue
            prefix_groups: list[_NodeGroup] = []
            if command_draft.node.type == "command":
                while groups and self._group_assignment_name(groups[0]) is not None:
                    prefix_groups.append(groups.pop(0))
            for group in prefix_groups:
                assignment = self._assignment_from_group(
                    group,
                    order=len(assignments),
                    region_id=command_draft.region_id,
                    function_id=command_draft.function_id,
                    prefix_for_command_start_byte=command_draft.node.start_byte,
                )
                if assignment is not None:
                    assignments.append(assignment)
                    prefix_sites_by_command.setdefault(command_draft.node.start_byte, []).append(
                        assignment.site
                    )
                if self.halted:
                    break
            if self.halted or not groups:
                continue

            timed = (
                command_draft.node.type == "command"
                and groups[0].raw_syntax == b"time"
                and len(groups[0].nodes) == 1
            )
            if timed:
                groups.pop(0)
                if groups and groups[0].raw_syntax == b"-p" and len(groups[0].nodes) == 1:
                    groups.pop(0)
                if not groups:
                    continue

            name_group = groups.pop(0)
            name_argument = self._argument(name_group)
            if name_argument is None:
                continue
            arguments: list[_ArgumentIR] = [name_argument]
            for group in groups:
                argument = self._argument(group)
                if argument is not None:
                    arguments.append(argument)
                if self.halted:
                    break
            if self.halted:
                break

            for group in wrapper_arguments_by_command.get(command_draft.node.start_byte, ()):
                argument = self._argument(group)
                if argument is not None:
                    arguments.append(argument)
                if self.halted:
                    break
            if self.halted:
                break

            facts: list[_RedirectFact] = []
            span_start = name_group.start_byte if timed else command_draft.node.start_byte
            span_end = max(argument.local_end_byte for argument in arguments)
            command_redirects = sorted(
                redirects_by_command.get(command_draft.node.start_byte, ()),
                key=lambda item: item.node.start_byte,
            )
            redirect_chain = (
                len(command_redirects) + command_draft.non_file_redirect_count > 1
            ) or any(
                (
                    redirect.statement_start_byte is not None
                    and (owner := self.redirect_owners.get(redirect.statement_start_byte))
                    is not None
                    and owner.redirect_count > 1
                )
                for redirect in command_redirects
            )
            if redirect_chain:
                chain_span = (
                    self._span(
                        command_redirects[0].node.start_byte,
                        command_redirects[-1].node.end_byte,
                    )
                    if command_redirects
                    else self._node_span(command_draft.node)
                )
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    chain_span or self._point_span(command_draft.node.start_byte),
                )
            for redirect in command_redirects:
                owner = (
                    self.redirect_owners.get(redirect.statement_start_byte)
                    if redirect.statement_start_byte is not None
                    else None
                )
                allow_fact = not redirect_chain and not (
                    command_draft.node.start_byte in self.syntax_error_command_starts
                    or (
                        owner is not None
                        and (
                            owner.has_error_child
                            or owner.statement_start_byte in self.syntax_error_redirect_starts
                        )
                    )
                )
                fact, extras, _supported = self._redirect_parts(
                    redirect,
                    allow_fact=allow_fact,
                )
                if fact is not None:
                    facts.append(fact)
                arguments.extend(extras)
                span_start = min(span_start, redirect.node.start_byte)
                span_end = max(span_end, redirect.node.end_byte)
                if self.halted:
                    break
            if self.halted:
                break
            command_span = self._span(span_start, span_end)
            if command_span is None:
                self._issue(
                    ShellIssueReason.UNSUPPORTED_SEMANTICS,
                    self._point_span(span_start),
                )
                continue
            ordered_arguments = [
                arguments[0],
                *sorted(arguments[1:], key=lambda item: item.local_start_byte),
            ]
            if not self._reserve(2, span=command_span):
                break
            site = CommandSite(
                unit_id=self.unit.unit_id,
                provenance=self.unit.provenance,
                span=command_span,
                argv=tuple(argument.value for argument in ordered_arguments),
            )
            prefix_sites = tuple(prefix_sites_by_command.get(command_draft.node.start_byte, ()))
            commands.append(
                _CommandIR(
                    site,
                    len(commands),
                    command_draft.region_id,
                    command_draft.function_id,
                    tuple(ordered_arguments),
                    prefix_sites,
                    tuple(facts),
                )
            )

            if (
                site.argv[0].state is StaticValueState.EXACT
                and site.argv[0].exact_bytes in _DECLARATION_KEYWORD_BYTES
            ):
                for group in groups:
                    key = (group.start_byte, group.end_byte)
                    if key in assignment_keys:
                        continue
                    assignment = self._assignment_from_group(
                        group,
                        order=len(assignments),
                        region_id=command_draft.region_id,
                        function_id=command_draft.function_id,
                        prefix_for_command_start_byte=None,
                    )
                    if assignment is not None:
                        assignments.append(assignment)
                        assignment_keys.add(key)
                    if self.halted:
                        break

        assignments.sort(key=lambda item: item.site.span.start_byte)
        assignments = [replace(item, order=index) for index, item in enumerate(assignments)]
        commands.sort(key=lambda item: item.site.span.start_byte)
        commands = [replace(item, order=index) for index, item in enumerate(commands)]
        return _ShellProgramIR(
            regions=tuple(self.regions),
            functions=self._functions() if not self.halted else (),
            commands=tuple(commands),
            assignments=tuple(assignments),
        )


def _analyze_shell_unit(
    unit: ShellUnit,
    *,
    budget: DependencyWorkBudget,
) -> _ShellAnalysisResult:
    """Private same-parse boundary retaining bounded structure for later tasks."""
    if not isinstance(unit, ShellUnit):
        raise ValueError("unit must be a ShellUnit")
    if not isinstance(budget, DependencyWorkBudget):
        raise ValueError("budget must be a DependencyWorkBudget")
    file_budget = budget.for_file(unit.origin_span.path)
    physical_size = (
        unit.source_map.physical_size_bytes
        if unit.source_map is not None
        else unit.origin_span.end_byte
    )
    if physical_size > MAX_DEPENDENCY_FILE_BYTES:
        exhaustion = DependencyWorkExhaustion(
            DependencyWorkResource.PHYSICAL_BYTES,
            physical_size,
            MAX_DEPENDENCY_FILE_BYTES,
        )
        issues: list[ShellIssue] = []
        _retain_issue(
            issues,
            _resource_issue(unit.origin_span, exhaustion),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, ShellWorkOutcome.SKIPPED),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())
    file_budget.register_shell_file_size(physical_size)
    if parse_exhaustion := file_budget.reserve_shell_parse(len(unit.raw_bytes)):
        issues = []
        _retain_issue(
            issues,
            _resource_issue(unit.origin_span, parse_exhaustion),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, ShellWorkOutcome.SKIPPED),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())

    try:
        tree = parse_bash_source(unit.raw_bytes)
    except ShellParserError as error:
        reason = (
            ShellIssueReason.RUNTIME_LIMIT
            if error.reason is ShellParserFailureReason.RUNTIME_LIMIT
            else ShellIssueReason.SHELL_PARSER_UNAVAILABLE
        )
        outcome = (
            ShellWorkOutcome.PARTIAL
            if error.outcome is ShellParserOutcome.PARTIAL
            else ShellWorkOutcome.FAILED
        )
        issues = []
        _retain_issue(
            issues,
            ShellIssue(
                reason=reason,
                outcome=outcome,
                span=unit.origin_span,
                unit_id=unit.unit_id,
            ),
            file_budget=file_budget,
        )
        public = ShellFrontendResult(
            issues=tuple(issues),
            work_items=(_shell_work_item(unit, outcome),),
        )
        return _ShellAnalysisResult(public, _ShellProgramIR())

    lowerer = _ShellLowerer(unit, budget, file_budget)
    lowerer.walk(tree.root_node)
    program = lowerer.lower()
    outcome = ShellWorkOutcome.PARTIAL if lowerer.partial else ShellWorkOutcome.COMPLETED
    public = ShellFrontendResult(
        commands=tuple(command.site for command in program.commands),
        assignments=tuple(assignment.site for assignment in program.assignments),
        generated_configs=(),
        issues=tuple(lowerer.issues),
        work_items=(_shell_work_item(unit, outcome),),
    )
    return _ShellAnalysisResult(public, program)


def analyze_shell_unit(
    unit: ShellUnit,
    *,
    budget: DependencyWorkBudget,
) -> ShellFrontendResult:
    """Parse and lower one shell unit exactly once into bounded syntax-only sites."""
    return _analyze_shell_unit(unit, budget=budget).public
