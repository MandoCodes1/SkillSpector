# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only analysis of direct dependency-source configuration files."""

from __future__ import annotations

import configparser
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from skillspector.artifacts import ArtifactDisposition, ArtifactRecord, ContentKind
from skillspector.dependency_source_types import (
    DependencyEcosystem,
    DependencyFileBudget,
    DependencySourceAnalysis,
    DependencySourceLimitation,
    DependencySourceLimitationReason,
    DependencySourceOperation,
    DependencySourceParseResult,
    DependencySourceScope,
    DependencySourceSurface,
    DependencyWorkBudget,
    DependencyWorkExhaustion,
    DestinationStatus,
    SourceChange,
    SourceSpan,
    finding_from_source_change,
)
from skillspector.url_redaction import redact_url

_NPM_BASENAMES: Final = frozenset({".npmrc", "npmrc"})
_PIP_BASENAMES: Final = frozenset({"pip.conf", "pip.ini"})
_RECOGNIZED_BASENAMES: Final = _NPM_BASENAMES | _PIP_BASENAMES
_NPM_SCOPED_REGISTRY: Final = re.compile(r"^@[^:\s]+:registry$", re.IGNORECASE)
_NPM_INTERPOLATION: Final = re.compile(r"\$\{[^{}]+\}")
_PIP_INTERPOLATION: Final = re.compile(r"%\([^)]+\)s")
_PIP_ASSIGNMENT: Final = re.compile(r"^\s*([^:=\s][^:=]*?)\s*([=:])\s*(.*)$")
_PIP_SECTION: Final = re.compile(r"^\s*\[([^]]+)]\s*(?:[#;].*)?$")
_PIP_OPTIONS: Final = ("index-url", "extra-index-url")


@dataclass(frozen=True, slots=True)
class _Candidate:
    ecosystem: DependencyEcosystem
    surface: DependencySourceSurface
    operation: DependencySourceOperation
    scope: DependencySourceScope
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _ValueFragment:
    line: int
    start_byte: int
    end_byte: int


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _line_count(raw: bytes | None) -> int:
    return max(1, raw.count(b"\n") + 1) if raw is not None else 1


def _limitation(
    path: str,
    raw: bytes | None,
    exhaustion: DependencyWorkExhaustion | None = None,
) -> DependencySourceLimitation:
    metrics = exhaustion.ledger_metrics() if exhaustion is not None else {}
    return DependencySourceLimitation(
        reason=DependencySourceLimitationReason.PARSE_INCOMPLETE,
        path=path,
        start_line=1,
        end_line=_line_count(raw),
        **metrics,
    )


def _is_complete_text_record(record: ArtifactRecord, raw_size: int) -> bool:
    try:
        return (
            record.get("content_kind") == ContentKind.TEXT
            and record.get("disposition") == ArtifactDisposition.ANALYZED
            and record.get("decodable") is True
            and record.get("contains_nul") is False
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] == raw_size
        )
    except (KeyError, TypeError):
        return False


def _inventory_size(record: ArtifactRecord | None) -> int:
    if record is None:
        return 0
    size = record.get("size_bytes")
    return size if type(size) is int and size >= 0 else 0


def _physical_lines(text: str) -> list[str]:
    """Split only on LF while removing the CR that belongs to a CRLF boundary."""
    return [part[:-1] if part.endswith("\r") else part for part in text.split("\n")]


def _line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    current = 0
    parts = text.split("\n")
    for index, line in enumerate(parts):
        offsets.append(current)
        current += len(line.encode("utf-8"))
        if index < len(parts) - 1:
            current += 1
    return offsets


def _byte_range(line: str, line_offset: int, start: int, end: int) -> tuple[int, int]:
    return (
        line_offset + len(line[:start].encode("utf-8")),
        line_offset + len(line[:end].encode("utf-8")),
    )


def _strip_comment(value: str) -> str:
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if quote is None and character in {"#", ";"} and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value.rstrip()


def _normalize_literal(value: str) -> tuple[str, int, int] | None:
    left_trimmed = value.lstrip()
    left = len(value) - len(left_trimmed)
    without_comment = _strip_comment(left_trimmed)
    trimmed = without_comment.rstrip()
    if not trimmed:
        return None
    if trimmed[0] in {'"', "'"}:
        quote = trimmed[0]
        if len(trimmed) < 2 or trimmed[-1] != quote:
            return None
        literal = trimmed[1:-1]
        if not literal:
            return None
        return literal, left + 1, left + len(trimmed) - 1
    if trimmed[-1] in {'"', "'"}:
        return None
    return trimmed, left, left + len(trimmed)


def _normalize_pip_option(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("--") and not normalized.startswith("---"):
        normalized = normalized[2:]
    return normalized.casefold().replace("_", "-")


class _PipConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return _normalize_pip_option(optionstr)


def _canonical_destination(ecosystem: DependencyEcosystem, value: str) -> bool:
    if "?" in value or "#" in value:
        return False
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            return False
    except (TypeError, ValueError):
        return False
    if ecosystem is DependencyEcosystem.NPM:
        return parsed.netloc.casefold() == "registry.npmjs.org" and parsed.path in {"", "/"}
    return parsed.netloc.casefold() == "pypi.org" and parsed.path in {"/simple", "/simple/"}


def _destination(
    ecosystem: DependencyEcosystem,
    raw_destination: str,
) -> tuple[str, DestinationStatus] | None:
    if _canonical_destination(ecosystem, raw_destination):
        return None
    interpolation = (
        _NPM_INTERPOLATION if ecosystem is DependencyEcosystem.NPM else _PIP_INTERPOLATION
    )
    if interpolation.search(raw_destination):
        return "unresolved", DestinationStatus.UNRESOLVED
    return redact_url(raw_destination), DestinationStatus.RESOLVED


def _candidate_change(
    candidate: _Candidate,
    raw: bytes,
    budget: DependencyFileBudget,
) -> tuple[SourceChange | None, DependencyWorkExhaustion | None]:
    if exhaustion := budget.charge_source_records(1):
        return None, exhaustion
    literal_bytes = candidate.span.end_byte - candidate.span.start_byte
    if exhaustion := budget.charge_retained_literal_bytes(literal_bytes):
        return None, exhaustion
    raw_destination = raw[candidate.span.start_byte : candidate.span.end_byte].decode("utf-8")
    normalized = _destination(candidate.ecosystem, raw_destination)
    if normalized is None:
        return None, None
    if exhaustion := budget.reserve_source_changes():
        return None, exhaustion
    destination, status = normalized
    return (
        SourceChange(
            ecosystem=candidate.ecosystem,
            surface=candidate.surface,
            operation=candidate.operation,
            scope=candidate.scope,
            destination=destination,
            destination_status=status,
            span=candidate.span,
        ),
        None,
    )


def _changes_from_candidates(
    candidates: Sequence[_Candidate],
    *,
    path: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    changes: list[SourceChange] = []
    for candidate in candidates:
        change, exhaustion = _candidate_change(candidate, raw, budget)
        if exhaustion is not None:
            return DependencySourceParseResult(
                changes=tuple(changes),
                limitations=(_limitation(path, raw, exhaustion),),
            )
        if change is not None:
            changes.append(change)
    return DependencySourceParseResult(changes=tuple(changes))


def _parse_npm(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    effective: dict[str, _Candidate] = {}
    offsets = _line_offsets(text)
    for line_number, line in enumerate(_physical_lines(text), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if "=" not in line:
            possible_key = stripped.split(None, 1)[0].lower()
            if possible_key == "registry" or _NPM_SCOPED_REGISTRY.fullmatch(possible_key):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            continue
        key_part, value_part = line.split("=", 1)
        key = key_part.strip().lower()
        if key != "registry" and _NPM_SCOPED_REGISTRY.fullmatch(key) is None:
            continue
        if exhaustion := budget.charge_config_nodes(1):
            return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
        normalized = _normalize_literal(value_part)
        if normalized is None:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        _literal, relative_start, relative_end = normalized
        value_column = line.index("=") + 1
        start = value_column + relative_start
        end = value_column + relative_end
        start_byte, end_byte = _byte_range(line, offsets[line_number - 1], start, end)
        effective[key] = _Candidate(
            ecosystem=DependencyEcosystem.NPM,
            surface=DependencySourceSurface.NPMRC,
            operation=DependencySourceOperation.REPLACE,
            scope=(
                DependencySourceScope.GLOBAL if key == "registry" else DependencySourceScope.SCOPED
            ),
            span=SourceSpan(path, start_byte, end_byte, line_number, line_number),
        )
    return _changes_from_candidates(
        tuple(sorted(effective.values(), key=lambda candidate: candidate.span.start_byte)),
        path=path,
        raw=raw,
        budget=budget,
    )


def _pip_fragments(
    value: str,
    *,
    line: str,
    line_number: int,
    line_offset: int,
    value_column: int,
) -> list[_ValueFragment] | None:
    normalized = _normalize_literal(value)
    if normalized is None:
        return None
    literal, relative_start, relative_end = normalized
    absolute_start = value_column + relative_start
    fragments: list[_ValueFragment] = []
    for match in re.finditer(r"\S+", literal):
        token_start = absolute_start + match.start()
        token_end = absolute_start + match.end()
        start_byte, end_byte = _byte_range(line, line_offset, token_start, token_end)
        fragments.append(_ValueFragment(line_number, start_byte, end_byte))
    if not fragments or relative_end < relative_start:
        return None
    return fragments


def _pip_fragments_match_value(
    fragments: Sequence[_ValueFragment],
    configured_value: str,
    raw: bytes,
) -> bool:
    normalized = _normalize_literal(configured_value)
    if normalized is None:
        return False
    literal, _start, _end = normalized
    configured_tokens = re.findall(r"\S+", literal)
    occurrence_tokens = [
        raw[fragment.start_byte : fragment.end_byte].decode("utf-8") for fragment in fragments
    ]
    return configured_tokens == occurrence_tokens


def _parse_pip(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    lines = _physical_lines(text)
    offsets = _line_offsets(text)
    section: str | None = None
    section_seen = False
    current_key: tuple[str | None, str] | None = None
    current_fragments: list[_ValueFragment] | None = None
    current_indent: int | None = None
    occurrences: dict[tuple[str | None, str], list[_ValueFragment]] = {}

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        section_match = _PIP_SECTION.fullmatch(line)
        if section_match is not None:
            if exhaustion := budget.charge_config_nodes(1):
                return DependencySourceParseResult(
                    limitations=(_limitation(path, raw, exhaustion),)
                )
            raw_section = section_match.group(1)
            section = None if raw_section == configparser.DEFAULTSECT else raw_section
            section_seen = True
            current_key = None
            current_fragments = None
            current_indent = None
            continue
        indent = len(line) - len(line.lstrip())
        if (
            current_key is not None
            and current_fragments is not None
            and current_indent is not None
            and indent > current_indent
        ):
            fragments = _pip_fragments(
                line,
                line=line,
                line_number=line_number,
                line_offset=offsets[line_number - 1],
                value_column=0,
            )
            if fragments is None:
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            current_fragments.extend(fragments)
            occurrences[current_key] = current_fragments
            continue
        assignment = _PIP_ASSIGNMENT.fullmatch(line)
        current_key = None
        current_fragments = None
        current_indent = None
        if assignment is None:
            continue
        normalized_key = _normalize_pip_option(assignment.group(1))
        if normalized_key not in _PIP_OPTIONS:
            continue
        if not section_seen:
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        if exhaustion := budget.charge_config_nodes(1):
            return DependencySourceParseResult(limitations=(_limitation(path, raw, exhaustion),))
        value = assignment.group(3)
        fragments = _pip_fragments(
            value,
            line=line,
            line_number=line_number,
            line_offset=offsets[line_number - 1],
            value_column=assignment.start(3),
        )
        if fragments is None and value.strip():
            return DependencySourceParseResult(limitations=(_limitation(path, raw),))
        current_key = (section, normalized_key)
        current_fragments = fragments or []
        current_indent = indent
        occurrences[current_key] = current_fragments

    if any(not fragments for fragments in occurrences.values()):
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    parser = _PipConfigParser(
        interpolation=None,
        strict=False,
        delimiters=("=", ":"),
    )
    try:
        parser.read_string(text)
    except configparser.Error:
        return DependencySourceParseResult(limitations=(_limitation(path, raw),))

    candidates: list[_Candidate] = []
    for concrete_section in parser.sections():
        for normalized_key in _PIP_OPTIONS:
            configured_value = parser.get(
                concrete_section,
                normalized_key,
                raw=True,
                fallback=None,
            )
            if configured_value is None:
                continue
            fragments = occurrences.get((concrete_section, normalized_key))
            if fragments is None:
                fragments = occurrences.get((None, normalized_key))
            if fragments is None or not _pip_fragments_match_value(
                fragments,
                configured_value,
                raw,
            ):
                return DependencySourceParseResult(limitations=(_limitation(path, raw),))
            for fragment in fragments:
                candidates.append(
                    _Candidate(
                        ecosystem=DependencyEcosystem.PIP,
                        surface=DependencySourceSurface.PIP_CONFIG,
                        operation=(
                            DependencySourceOperation.REPLACE
                            if normalized_key == "index-url"
                            else DependencySourceOperation.ADD
                        ),
                        scope=(
                            DependencySourceScope.GLOBAL
                            if concrete_section == "global"
                            else DependencySourceScope.COMMAND
                        ),
                        span=SourceSpan(
                            path,
                            fragment.start_byte,
                            fragment.end_byte,
                            fragment.line,
                            fragment.line,
                        ),
                    )
                )
    candidates.sort(key=lambda candidate: candidate.span.start_byte)
    return _changes_from_candidates(candidates, path=path, raw=raw, budget=budget)


def _parse_file(
    path: str,
    text: str,
    raw: bytes,
    budget: DependencyFileBudget,
) -> DependencySourceParseResult:
    if _basename(path) in _NPM_BASENAMES:
        return _parse_npm(path, text, raw, budget)
    return _parse_pip(path, text, raw, budget)


def analyze_dependency_sources(
    *,
    components: Iterable[str],
    local_file_cache: Mapping[str, str],
    raw_file_cache: Mapping[str, bytes],
    artifact_inventory: Iterable[ArtifactRecord],
    budget: DependencyWorkBudget,
) -> DependencySourceAnalysis:
    """Analyze recognized direct config artifacts named by the component inventory."""
    inventory_by_path: dict[str, list[ArtifactRecord]] = {}
    for record in artifact_inventory:
        path = record.get("path")
        if isinstance(path, str):
            inventory_by_path.setdefault(path, []).append(record)

    changes: list[SourceChange] = []
    limitations: list[DependencySourceLimitation] = []
    for path in sorted(set(components)):
        if not isinstance(path, str) or _basename(path) not in _RECOGNIZED_BASENAMES:
            continue
        raw = raw_file_cache.get(path)
        safe_raw = raw if isinstance(raw, bytes) else None
        records = inventory_by_path.get(path, [])
        matched_record = records[0] if len(records) == 1 else None
        observed_size = max(len(safe_raw or b""), _inventory_size(matched_record))
        file_budget = budget.for_file(path)
        if exhaustion := file_budget.charge_physical_bytes(observed_size):
            limitations.append(_limitation(path, safe_raw, exhaustion))
            continue
        if (
            safe_raw is None
            or matched_record is None
            or not _is_complete_text_record(matched_record, len(safe_raw))
        ):
            limitations.append(_limitation(path, safe_raw))
            continue
        try:
            decoded = safe_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            limitations.append(_limitation(path, safe_raw))
            continue
        cached = local_file_cache.get(path)
        if not isinstance(cached, str) or cached != decoded:
            limitations.append(_limitation(path, safe_raw))
            continue
        parsed = _parse_file(path, decoded, safe_raw, file_budget)
        changes.extend(parsed.changes)
        limitations.extend(parsed.limitations)

    return DependencySourceAnalysis(
        findings=tuple(finding_from_source_change(change) for change in changes),
        limitations=tuple(limitations),
    )
