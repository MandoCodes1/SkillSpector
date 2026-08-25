# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-pinned parser boundary and bounded shell-unit extraction.

This module intentionally contains no shell lowering or package-manager policy.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib import import_module
from math import ceil, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, cast
from warnings import catch_warnings, filterwarnings

from skillspector.dependency_source_types import (
    MAX_DEPENDENCY_SHELL_UNITS_PER_FILE,
    MAX_DEPENDENCY_SOURCE_MAP_ENTRIES_PER_FILE,
    DependencyFileBudget,
    DependencyWorkBudget,
    DependencyWorkExhaustion,
    DependencyWorkResource,
    ShellDialect,
    ShellExtractionResult,
    ShellIssue,
    ShellIssueReason,
    ShellTruncationClaimStatus,
    ShellUnit,
    ShellUnitKind,
    ShellWorkOutcome,
    SiteProvenance,
    SourceMap,
    SourceMapEntry,
    SourceSpan,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Parser, Tree

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
