# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, local-only credential redaction for dependency-source evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from io import StringIO
from ipaddress import IPv6Address
from typing import Final
from urllib.parse import SplitResult, unquote_plus, urlsplit

REDACTED_URL: Final = "[REDACTED_URL]"
REDACTED_REMAINDER: Final = "[REDACTED_REMAINDER]"
REDACTED_VALUE: Final = "[REDACTED_VALUE]"

# Match the repository's bounded visible-artifact ceiling so provider-bound
# content is not silently shortened before the caller can account for it.
MAX_REDACTION_CHARACTERS: Final = 16 * 1024 * 1024
MAX_REDACTION_CANDIDATES: Final = 1_024
MAX_REDACTION_DEPTH: Final = 16
MAX_REDACTION_NODES: Final = 10_000

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
        "sig",
        "signature",
        "signatures",
        "token",
        "tokens",
    }
)
_MAX_RAW_QUERY_KEY_CHARACTERS: Final = 3 * 256
_MAX_DECODED_QUERY_KEY_CHARACTERS: Final = 256
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_URI_CHARACTER = re.compile(r'[<>"`]')
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_ENCODED_UNSAFE_AUTHORITY_CHARACTER = re.compile(
    r"%(?:0[0-9A-Fa-f]|1[0-9A-Fa-f]|20|23|2[fF]|3[aAfF]|40|5[bBcCdD]|7[fF])"
)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_SCP_URL = re.compile(
    r"^(?P<user>[^@/:\\\s]+)@"
    r"(?P<host>\[[^\]\s]+\]|[^@/:\\\s]+):"
    r"(?P<path>.+)$"
)
_PROSE_OPENERS: Final = frozenset("([{<\"'`")
_PAIRED_CLOSERS: Final = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
    '"': '"',
    "'": "'",
    "`": "`",
}
_SENTENCE_PUNCTUATION: Final = frozenset(".,")


def _valid_bound(value: object) -> bool:
    return type(value) is int and value >= 0


def _has_ambiguous_percent_escape(value: str) -> bool:
    return _BAD_PERCENT_ESCAPE.search(value) is not None


def _query_key_is_sensitive(raw_key: str) -> bool:
    if len(raw_key) > _MAX_RAW_QUERY_KEY_CHARACTERS:
        raise ValueError("query key exceeds its bound")
    try:
        decoded = unquote_plus(raw_key, encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError):
        raise ValueError("query key is ambiguous") from None
    if (
        len(decoded) > _MAX_DECODED_QUERY_KEY_CHARACTERS
        or _CONTROL_CHARACTER.search(decoded)
        or _PERCENT_ESCAPE.search(decoded)
    ):
        raise ValueError("query key is ambiguous")
    folded = decoded.casefold()
    return any(term in folded for term in _CREDENTIAL_WORDS)


def _redact_query(raw_query: str) -> str:
    if not raw_query:
        return raw_query
    output = StringIO()
    cursor = 0
    for delimiter in re.finditer(r"[&;]", raw_query):
        raw_part = raw_query[cursor : delimiter.start()]
        raw_key, separator, _raw_value = raw_part.partition("=")
        if separator and _query_key_is_sensitive(raw_key):
            raw_part = f"{raw_key}=REDACTED"
        output.write(raw_part)
        output.write(delimiter.group())
        cursor = delimiter.end()
    raw_part = raw_query[cursor:]
    raw_key, separator, _raw_value = raw_part.partition("=")
    output.write(
        f"{raw_key}=REDACTED" if separator and _query_key_is_sensitive(raw_key) else raw_part
    )
    return output.getvalue()


def _valid_bracketed_ipv6(host: str) -> bool:
    if not (host.startswith("[") and host.endswith("]")):
        return False
    try:
        IPv6Address(host[1:-1])
    except ValueError:
        return False
    return True


def _valid_host_port(host_port: str) -> bool:
    if not host_port or _ENCODED_UNSAFE_AUTHORITY_CHARACTER.search(host_port):
        return False
    if host_port.startswith("["):
        close = host_port.find("]")
        if close < 0 or host_port.find("]", close + 1) >= 0:
            return False
        if not _valid_bracketed_ipv6(host_port[: close + 1]):
            return False
        suffix = host_port[close + 1 :]
        if not suffix:
            return True
        return suffix.startswith(":") and len(suffix) > 1 and suffix[1:].isdigit()
    if "[" in host_port or "]" in host_port or host_port.count(":") > 1:
        return False
    host, separator, port = host_port.partition(":")
    if not host:
        return False
    return not separator or bool(port and port.isdigit())


def _validated_safe_authority(parsed: SplitResult, authority: str) -> str | None:
    if (
        not authority
        or authority.count("@") > 1
        or "\\" in authority
        or _CONTROL_CHARACTER.search(authority)
        or any(character.isspace() for character in authority)
        or _ENCODED_UNSAFE_AUTHORITY_CHARACTER.search(authority)
    ):
        return None
    host_port = authority.rsplit("@", 1)[-1]
    if not _valid_host_port(host_port):
        return None
    try:
        hostname = parsed.hostname
        _ = parsed.port
    except (UnicodeError, ValueError):
        return None
    if not hostname:
        return None
    return f"REDACTED@{host_port}" if "@" in authority else host_port


def _redact_standard_url(value: str) -> str:
    parsed = urlsplit(value)
    if not _SCHEME.fullmatch(parsed.scheme) or not parsed.netloc:
        return REDACTED_URL
    safe_authority = _validated_safe_authority(parsed, parsed.netloc)
    if safe_authority is None:
        return REDACTED_URL
    safe_query = _redact_query(parsed.query)
    without_fragment = value.split("#", 1)[0]
    had_query_delimiter = "?" in without_fragment
    suffix = f"?{safe_query}" if had_query_delimiter else ""
    return f"{parsed.scheme}://{safe_authority}{parsed.path}{suffix}"


def _valid_scp_host(host: str) -> bool:
    if (
        not host
        or "\\" in host
        or _CONTROL_CHARACTER.search(host)
        or any(character.isspace() for character in host)
        or _ENCODED_UNSAFE_AUTHORITY_CHARACTER.search(host)
    ):
        return False
    if host.startswith("[") or host.endswith("]"):
        return _valid_bracketed_ipv6(host)
    return "[" not in host and "]" not in host and ":" not in host


def _redact_scp_url(value: str) -> str:
    without_fragment = value.split("#", 1)[0]
    match = _SCP_URL.fullmatch(without_fragment)
    if (
        match is None
        or without_fragment.count("@") != 1
        or not _valid_scp_host(match.group("host"))
    ):
        return REDACTED_URL
    raw_path, query_separator, raw_query = match.group("path").partition("?")
    if not raw_path:
        return REDACTED_URL
    safe_query = _redact_query(raw_query) if query_separator else ""
    suffix = f"?{safe_query}" if query_separator else ""
    return f"REDACTED@{match.group('host')}:{raw_path}{suffix}"


def _scp_discovery_path(candidate: str) -> str | None:
    at_sign = candidate.find("@")
    if at_sign < 0 or at_sign + 1 >= len(candidate):
        return None
    host_start = at_sign + 1
    if candidate[host_start] == "[":
        close = candidate.find("]", host_start + 1)
        if close < 0 or close + 1 >= len(candidate) or candidate[close + 1] != ":":
            return None
        separator = close + 1
    else:
        separator = candidate.find(":", host_start)
        if separator < 0:
            return None
    return candidate[separator + 1 :].split("?", 1)[0]


def _looks_like_scp_git(value: str) -> bool:
    candidates = (value.split("#", 1)[0], value.replace("#", ""))
    for candidate in candidates:
        path = _scp_discovery_path(candidate)
        if path is None:
            continue
        if "/" in path or ".git" in path.casefold():
            return True
    return False


def _hierarchical_suffix_after_authority(value: str) -> str:
    marker = value.find("://")
    if marker < 0:
        return ""
    suffix_start = len(value)
    for delimiter in "/?#":
        position = value.find(delimiter, marker + 3)
        if position >= 0:
            suffix_start = min(suffix_start, position)
    return value[suffix_start:]


def _detach_trailing_prose_punctuation(
    value: str,
    opener: str | None,
) -> tuple[str, str]:
    split_at = len(value)
    while split_at > 0 and value[split_at - 1] in _SENTENCE_PUNCTUATION:
        split_at -= 1
    if split_at > 0 and opener is not None:
        closer = value[split_at - 1]
        if _PAIRED_CLOSERS.get(closer) == opener:
            split_at -= 1
    return value[:split_at], value[split_at:]


def redact_url(value: str, *, max_characters: int = MAX_REDACTION_CHARACTERS) -> str:
    """Return one URL-like value with credentials removed, or a fixed placeholder."""
    if not isinstance(value, str) or not _valid_bound(max_characters):
        return REDACTED_URL
    if len(value) > max_characters:
        return REDACTED_URL
    is_hierarchical_uri = "://" in value
    suspicious = is_hierarchical_uri or _looks_like_scp_git(value)
    if not suspicious:
        return value
    if is_hierarchical_uri:
        first_marker = value.find("://")
        if value.find("://", first_marker + 3) >= 0 or _looks_like_scp_git(
            _hierarchical_suffix_after_authority(value)
        ):
            return REDACTED_URL
    elif "#" in value and not _looks_like_scp_git(value.split("#", 1)[0]):
        return REDACTED_URL
    if (
        _CONTROL_CHARACTER.search(value)
        or _UNSAFE_URI_CHARACTER.search(value)
        or _has_ambiguous_percent_escape(value)
    ):
        return REDACTED_URL
    try:
        if is_hierarchical_uri:
            return _redact_standard_url(value)
        return _redact_scp_url(value)
    except (UnicodeError, ValueError):
        return REDACTED_URL
    except Exception:
        # Evidence redaction is a security boundary: unexpected parser errors fail closed.
        return REDACTED_URL


class TextRedactionIncompleteReason(StrEnum):
    """Content-free reason that one bounded text redaction did not complete."""

    CHARACTER_LIMIT = "character_limit"
    CANDIDATE_LIMIT = "candidate_limit"
    INVALID_INPUT = "invalid_input"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class TextRedactionResult:
    """Sanitized text plus truthful bounded completion and usage metadata."""

    value: str
    complete: bool
    candidates: int
    reason: TextRedactionIncompleteReason | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or type(self.complete) is not bool
            or not _valid_bound(self.candidates)
            or self.candidates > MAX_REDACTION_CHARACTERS
            or (
                self.reason is not None
                and not isinstance(self.reason, TextRedactionIncompleteReason)
            )
            or self.complete is (self.reason is not None)
        ):
            raise ValueError("invalid text redaction result")


@dataclass(frozen=True, slots=True)
class _TokenRedactionResult:
    value: str
    candidates: int
    complete: bool


def _simple_token_parts(token: str) -> tuple[str, str, str, str]:
    split_at = len(token)
    while split_at > 0 and token[split_at - 1] in _SENTENCE_PUNCTUATION:
        split_at -= 1
    punctuation = token[split_at:]
    core = token[:split_at]
    if len(core) >= 2 and core[0] in _PROSE_OPENERS:
        opener = core[0]
        closer = core[-1]
        if _PAIRED_CLOSERS.get(closer) == opener:
            return opener, core[1:-1], closer, punctuation
    return "", core, "", punctuation


def _scp_signals_in_token(token: str, first_marker: int) -> int:
    if "@" not in token or ":" not in token:
        return 0
    if first_marker < 0:
        return token.count("@") if _looks_like_scp_git(token) else 0

    signals = 0
    at_before = token.find("@", 0, first_marker)
    if at_before >= 0:
        separator = token.find(":", at_before + 1, first_marker)
        assignment = token.find("=", at_before + 1, first_marker)
        if separator >= 0 and assignment < 0:
            signals += token[:first_marker].count("@")

    suffix_start = len(token)
    for delimiter in "/?#":
        position = token.find(delimiter, first_marker + 3)
        if position >= 0:
            suffix_start = min(suffix_start, position)
    suffix = token[suffix_start:]
    if _looks_like_scp_git(suffix):
        signals += suffix.count("@")
    return signals


def _simple_redact_token(token: str, *, max_candidates: int) -> _TokenRedactionResult:
    marker_count = token.count("://")
    first_marker = token.find("://")
    scp_signals = _scp_signals_in_token(token, first_marker)
    signals = marker_count + scp_signals
    if signals == 0:
        return _TokenRedactionResult(token, 0, True)
    if signals > max_candidates:
        return _TokenRedactionResult(
            REDACTED_REMAINDER,
            max_candidates,
            False,
        )

    opener, candidate, closer, punctuation = _simple_token_parts(token)
    sanitized = redact_url(candidate, max_characters=len(candidate))
    if sanitized == REDACTED_URL:
        return _TokenRedactionResult(f"{REDACTED_URL}{punctuation}", signals, True)
    return _TokenRedactionResult(
        f"{opener}{sanitized}{closer}{punctuation}",
        signals,
        True,
    )


def _redact_text_with_usage(value: str, *, max_candidates: int) -> TextRedactionResult:
    result = StringIO()
    cursor = 0
    index = 0
    candidates = 0
    try:
        while index < len(value):
            while index < len(value) and value[index].isspace():
                index += 1
            token_start = index
            while index < len(value) and not value[index].isspace():
                index += 1
            if token_start == index:
                continue
            result.write(value[cursor:token_start])
            token = _simple_redact_token(
                value[token_start:index],
                max_candidates=max_candidates - candidates,
            )
            result.write(token.value)
            if not token.complete:
                return TextRedactionResult(
                    value=result.getvalue(),
                    complete=False,
                    candidates=candidates + token.candidates,
                    reason=TextRedactionIncompleteReason.CANDIDATE_LIMIT,
                )
            candidates += token.candidates
            cursor = index
    except Exception:
        result.write(REDACTED_REMAINDER)
        return TextRedactionResult(
            value=result.getvalue(),
            complete=False,
            candidates=candidates,
            reason=TextRedactionIncompleteReason.INTERNAL_ERROR,
        )
    result.write(value[cursor:])
    return TextRedactionResult(
        value=result.getvalue(),
        complete=True,
        candidates=candidates,
        reason=None,
    )


def redact_text_result(
    value: str,
    *,
    max_characters: int = MAX_REDACTION_CHARACTERS,
    max_candidates: int = MAX_REDACTION_CANDIDATES,
) -> TextRedactionResult:
    """Return sanitized text with content-free bounded completion metadata."""
    if not isinstance(value, str):
        return TextRedactionResult(
            value=REDACTED_REMAINDER,
            complete=False,
            candidates=0,
            reason=TextRedactionIncompleteReason.INVALID_INPUT,
        )
    if value == REDACTED_REMAINDER:
        return TextRedactionResult(value=value, complete=True, candidates=0, reason=None)
    if not _valid_bound(max_characters) or not _valid_bound(max_candidates):
        return TextRedactionResult(
            value=REDACTED_REMAINDER,
            complete=False,
            candidates=0,
            reason=TextRedactionIncompleteReason.INVALID_INPUT,
        )
    if len(value) > max_characters:
        return TextRedactionResult(
            value=REDACTED_REMAINDER,
            complete=False,
            candidates=0,
            reason=TextRedactionIncompleteReason.CHARACTER_LIMIT,
        )
    if "://" not in value and ("@" not in value or ":" not in value):
        return TextRedactionResult(
            value=value,
            complete=True,
            candidates=0,
            reason=None,
        )
    return _redact_text_with_usage(value, max_candidates=max_candidates)


def redact_text(
    value: str,
    *,
    max_characters: int = MAX_REDACTION_CHARACTERS,
    max_candidates: int = MAX_REDACTION_CANDIDATES,
) -> str:
    """Redact bounded URL candidates while preserving ordinary text exactly."""
    return redact_text_result(
        value,
        max_characters=max_characters,
        max_candidates=max_candidates,
    ).value


class _AggregateRedactionExhaustedError(Exception):
    """Internal control flow for one exhausted recursive redaction budget."""


@dataclass(slots=True)
class _ValueWalk:
    remaining_nodes: int
    max_depth: int
    remaining_text_characters: int
    remaining_text_candidates: int
    active: set[int] = field(default_factory=set)

    def visit(self, value: object, depth: int) -> object:
        if depth > self.max_depth or self.remaining_nodes <= 0:
            raise _AggregateRedactionExhaustedError
        self.remaining_nodes -= 1

        if isinstance(value, str):
            if len(value) > self.remaining_text_characters:
                raise _AggregateRedactionExhaustedError
            result = redact_text_result(
                value,
                max_characters=self.remaining_text_characters,
                max_candidates=self.remaining_text_candidates,
            )
            if not result.complete:
                raise _AggregateRedactionExhaustedError
            self.remaining_text_characters -= len(value)
            self.remaining_text_candidates -= result.candidates
            return result.value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in self.active:
                return REDACTED_VALUE
            if len(value) > self.remaining_nodes:
                raise _AggregateRedactionExhaustedError
            self.active.add(identity)
            try:
                if isinstance(value, Mapping):
                    result_mapping: dict[object, object] = {}
                    for key, nested in value.items():
                        result_mapping[key] = self.visit(nested, depth + 1)
                    return result_mapping
                result_items = [self.visit(nested, depth + 1) for nested in value]
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
    """Recursively sanitize evidence values under aggregate explicit ceilings."""
    bounds = (max_depth, max_nodes, max_text_characters, max_text_candidates)
    if not all(_valid_bound(bound) for bound in bounds):
        return REDACTED_VALUE
    try:
        return _ValueWalk(
            remaining_nodes=max_nodes,
            max_depth=max_depth,
            remaining_text_characters=max_text_characters,
            remaining_text_candidates=max_text_candidates,
        ).visit(value, 0)
    except Exception:
        return REDACTED_VALUE
