# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-pinned, bounded Tree-sitter Bash parser runtime boundary.

This module intentionally contains no shell lowering or package-manager policy.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import cache
from importlib import import_module
from math import ceil, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, cast
from warnings import catch_warnings, filterwarnings

if TYPE_CHECKING:
    from tree_sitter import Language, Parser, Tree

EXPECTED_BASH_ABI_VERSION: Final = 15
EXPECTED_BASH_SEMANTIC_VERSION: Final = (0, 25, 1)
MAX_TREE_SITTER_READ_BYTES: Final = 4_096


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
