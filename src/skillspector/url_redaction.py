# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only credential redaction for dependency-source evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import unquote_plus, urlsplit, urlunsplit

REDACTED_URL: Final = "[REDACTED_URL]"
REDACTED_REMAINDER: Final = "[REDACTED_REMAINDER]"
REDACTED_VALUE: Final = "[REDACTED_VALUE]"

MAX_REDACTION_CHARACTERS: Final = 65_536
MAX_REDACTION_CANDIDATES: Final = 1_024
MAX_REDACTION_DEPTH: Final = 16
MAX_REDACTION_NODES: Final = 10_000

_ALLOWED_SCHEMES: Final = frozenset(
    {
        "http",
        "https",
        "ssh",
        "git",
        "git+http",
        "git+https",
        "git+ssh",
        "sparse+http",
        "sparse+https",
    }
)
_CREDENTIAL_WORDS: Final = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "credential",
        "credentials",
        "key",
        "keys",
        "pass",
        "password",
        "passwd",
        "passphrase",
        "secret",
        "secrets",
        "signature",
        "signatures",
        "token",
        "tokens",
    }
)
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUERY_WORD_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")
_SCP_URL = re.compile(
    r"^(?P<user>[^@/:\\\s]+)@"
    r"(?P<host>\[[^\]\s]+\]|[^@/:\\\s]+):"
    r"(?P<path>.+)$"
)
_TEXT_CANDIDATE = re.compile(
    r"(?:(?:(?:git|sparse)\+)?(?:https?|ssh|git))://[^\s<>\"']+"
    r"|(?:[^@/:\\\s<>\"']+)@(?:\[[^\]\s]+\]|[^@/:\\\s<>\"']+):"
    r"(?:[^\s<>\"']*/[^\s<>\"']*|[^\s<>\"']*\.git(?:[?#][^\s<>\"']*)?)",
    re.IGNORECASE,
)
_TRAILING_PROSE_PUNCTUATION: Final = frozenset(".,)]}")


def _valid_bound(value: object) -> bool:
    return type(value) is int and value >= 0


def _has_ambiguous_percent_escape(value: str) -> bool:
    return _BAD_PERCENT_ESCAPE.search(value) is not None


def _query_key_is_sensitive(raw_key: str) -> bool:
    try:
        decoded = unquote_plus(raw_key, encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError):
        raise ValueError("query key is ambiguous") from None
    expanded = _CAMEL_BOUNDARY.sub("_", decoded)
    words = {word.casefold() for word in _QUERY_WORD_SEPARATOR.split(expanded) if word}
    return bool(words & _CREDENTIAL_WORDS)


def _redact_query(raw_query: str) -> str:
    if not raw_query:
        return raw_query
    redacted_parts: list[str] = []
    for raw_part in raw_query.split("&"):
        raw_key, separator, _raw_value = raw_part.partition("=")
        if separator and _query_key_is_sensitive(raw_key):
            redacted_parts.append(f"{raw_key}=REDACTED")
        else:
            redacted_parts.append(raw_part)
    return "&".join(redacted_parts)


def _redact_standard_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in _ALLOWED_SCHEMES or not parsed.netloc:
        return REDACTED_URL
    authority = parsed.netloc
    if authority.count("@") > 1 or "\\" in authority or any(char.isspace() for char in authority):
        return REDACTED_URL
    # Accessing both properties forces urllib's bracket, NFKC, and port checks.
    if not parsed.hostname:
        return REDACTED_URL
    _ = parsed.port

    host_port = authority.rsplit("@", 1)[-1]
    safe_authority = f"REDACTED@{host_port}" if "@" in authority else host_port
    safe_query = _redact_query(parsed.query)
    return urlunsplit((parsed.scheme, safe_authority, parsed.path, safe_query, ""))


def _redact_scp_url(value: str) -> str:
    without_fragment = value.split("#", 1)[0]
    match = _SCP_URL.fullmatch(without_fragment)
    if match is None or without_fragment.count("@") != 1 or "\\" in without_fragment:
        return REDACTED_URL
    raw_path, query_separator, raw_query = match.group("path").partition("?")
    if not raw_path:
        return REDACTED_URL
    safe_query = _redact_query(raw_query) if query_separator else ""
    suffix = f"?{safe_query}" if query_separator else ""
    return f"REDACTED@{match.group('host')}:{raw_path}{suffix}"


def _looks_like_scp_git(value: str) -> bool:
    without_fragment = value.split("#", 1)[0]
    if "@" not in without_fragment or ":" not in without_fragment:
        return False
    path = without_fragment.rsplit(":", 1)[-1].split("?", 1)[0]
    return "/" in path or path.casefold().endswith(".git")


def _detach_trailing_prose_punctuation(value: str) -> tuple[str, str]:
    split_at = len(value)
    while split_at > 0 and value[split_at - 1] in _TRAILING_PROSE_PUNCTUATION:
        split_at -= 1
    return value[:split_at], value[split_at:]


def redact_url(value: str, *, max_characters: int = MAX_REDACTION_CHARACTERS) -> str:
    """Return one URL-like value with credentials removed, or a fixed placeholder."""
    if not isinstance(value, str) or not _valid_bound(max_characters):
        return REDACTED_URL
    if len(value) > max_characters:
        return REDACTED_URL
    suspicious = "://" in value or _looks_like_scp_git(value)
    if not suspicious:
        return value
    if _CONTROL_CHARACTER.search(value) or _has_ambiguous_percent_escape(value):
        return REDACTED_URL
    try:
        if "://" in value:
            return _redact_standard_url(value)
        return _redact_scp_url(value)
    except (UnicodeError, ValueError):
        return REDACTED_URL
    except Exception:
        # Evidence redaction is a security boundary: unexpected parser errors fail closed.
        return REDACTED_URL


def redact_text(
    value: str,
    *,
    max_characters: int = MAX_REDACTION_CHARACTERS,
    max_candidates: int = MAX_REDACTION_CANDIDATES,
) -> str:
    """Redact bounded URL candidates while preserving ordinary text exactly."""
    if not isinstance(value, str):
        return REDACTED_REMAINDER
    if not _valid_bound(max_characters) or not _valid_bound(max_candidates):
        return REDACTED_REMAINDER

    bounded = value[:max_characters]
    was_truncated = len(value) > max_characters
    result: list[str] = []
    cursor = 0
    candidates = 0
    try:
        for match in _TEXT_CANDIDATE.finditer(bounded):
            if candidates >= max_candidates:
                result.append(bounded[cursor : match.start()])
                result.append(REDACTED_REMAINDER)
                return "".join(result)
            result.append(bounded[cursor : match.start()])
            candidate, punctuation = _detach_trailing_prose_punctuation(match.group(0))
            result.append(redact_url(candidate, max_characters=max_characters))
            result.append(punctuation)
            cursor = match.end()
            candidates += 1
    except Exception:
        result.append(REDACTED_REMAINDER)
        return "".join(result)

    result.append(bounded[cursor:])
    if was_truncated:
        result.append(REDACTED_REMAINDER)
    return "".join(result)


@dataclass(slots=True)
class _ValueWalk:
    remaining_nodes: int
    max_depth: int
    max_text_characters: int
    max_text_candidates: int
    active: set[int] = field(default_factory=set)

    def visit(self, value: object, depth: int) -> object:
        if depth > self.max_depth or self.remaining_nodes <= 0:
            return REDACTED_VALUE
        self.remaining_nodes -= 1

        if isinstance(value, str):
            return redact_text(
                value,
                max_characters=self.max_text_characters,
                max_candidates=self.max_text_candidates,
            )
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in self.active:
                return REDACTED_VALUE
            self.active.add(identity)
            try:
                if isinstance(value, Mapping):
                    result: dict[object, object] = {}
                    for key, nested in value.items():
                        if self.remaining_nodes <= 0:
                            result[key] = REDACTED_VALUE
                        else:
                            result[key] = self.visit(nested, depth + 1)
                    return result
                result_items: list[object] = []
                for nested in value:
                    if self.remaining_nodes <= 0:
                        return REDACTED_VALUE
                    result_items.append(self.visit(nested, depth + 1))
                return tuple(result_items) if isinstance(value, tuple) else result_items
            finally:
                self.active.remove(identity)
        return REDACTED_VALUE


def redact_value(
    value: object,
    *,
    max_depth: int = MAX_REDACTION_DEPTH,
    max_nodes: int = MAX_REDACTION_NODES,
    max_text_characters: int = MAX_REDACTION_CHARACTERS,
    max_text_candidates: int = MAX_REDACTION_CANDIDATES,
) -> object:
    """Recursively sanitize evidence values under explicit depth and node ceilings."""
    bounds = (max_depth, max_nodes, max_text_characters, max_text_candidates)
    if not all(_valid_bound(bound) for bound in bounds):
        return REDACTED_VALUE
    try:
        return _ValueWalk(
            remaining_nodes=max_nodes,
            max_depth=max_depth,
            max_text_characters=max_text_characters,
            max_text_candidates=max_text_candidates,
        ).visit(value, 0)
    except Exception:
        return REDACTED_VALUE
