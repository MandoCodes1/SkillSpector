# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for bounded dependency-source credential redaction."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import Any

import pytest


def _api() -> Any:
    """Import the real redactor while keeping the initial TDD run collectable."""
    try:
        return importlib.import_module("skillspector.url_redaction")
    except ImportError:
        pytest.fail("dependency-source URL redaction is unavailable")


def test_canonical_registry_url_has_pinned_safe_output() -> None:
    api = _api()
    raw = (
        "https://alice:supersecret@packages.example.invalid/private"
        "?token=querysecret&channel=stable#fragmentsecret"
    )

    redacted = api.redact_url(raw)

    assert redacted == (
        "https://REDACTED@packages.example.invalid/private?token=REDACTED&channel=stable"
    )
    for sentinel in ("alice", "supersecret", "querysecret", "fragmentsecret"):
        assert sentinel not in redacted


@pytest.mark.parametrize(
    "query_key",
    [
        "AUTH",
        "credential",
        "apiKey",
        "password",
        "client_secret",
        "X-Amz-Signature",
        "access_to%6ben",
        "API%5FKEY",
        "%74oken",
    ],
)
def test_credential_semantic_query_keys_are_decoded_and_redacted(query_key: str) -> None:
    api = _api()
    sentinel = "query-value-secret-98b11"

    redacted = api.redact_url(
        f"https://packages.example.invalid/simple?{query_key}={sentinel}&channel=stable"
    )

    assert sentinel not in redacted
    assert f"{query_key}=REDACTED" in redacted
    assert "channel=stable" in redacted


@pytest.mark.parametrize(
    "query_key",
    [
        "apikey",
        "APIKEY",
        "authToken",
        "AUTHTOKEN",
        "accessToken",
        "clientSecret",
        "privateKey",
        "sig",
        "%61pikey",
        "%41piKey",
    ],
)
def test_compact_and_mixed_case_credential_query_keys_are_redacted(query_key: str) -> None:
    api = _api()
    sentinel = "compact-query-secret-67fe"

    redacted = api.redact_url(f"https://packages.example.invalid/simple?{query_key}={sentinel}")

    assert redacted == (f"https://packages.example.invalid/simple?{query_key}=REDACTED")
    assert sentinel not in redacted


@pytest.mark.parametrize(
    "raw",
    [
        "ssh://ssh-user:ssh-secret-0d9f@git.example.invalid/org/repo.git#fragment-secret",
        "git+https://git-user:git-secret-a941@git.example.invalid/org/repo.git?auth=query-secret",
        "git+ssh://agent:agent-secret-c2ab@git.example.invalid/org/repo.git",
        "scp-user-secret@git.example.invalid:org/repo.git#scp-fragment-secret",
    ],
)
def test_ssh_git_and_scp_like_forms_remove_userinfo_fragments_and_secret_queries(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    for sentinel in (
        "ssh-user",
        "ssh-secret",
        "git-user",
        "git-secret",
        "query-secret",
        "agent-secret",
        "scp-user-secret",
        "fragment-secret",
        "scp-fragment-secret",
    ):
        assert sentinel not in redacted
    assert "git.example.invalid" in redacted
    assert "org/repo.git" in redacted
    assert "REDACTED" in redacted


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://user:ipv6-secret@[2001:db8::1]:8443/private",
            "https://REDACTED@[2001:db8::1]:8443/private",
        ),
        (
            "http://user:http-secret@packages.example.invalid:8080/simple",
            "http://REDACTED@packages.example.invalid:8080/simple",
        ),
        (
            "git://user:git-scheme-secret@git.example.invalid/org/repo.git",
            "git://REDACTED@git.example.invalid/org/repo.git",
        ),
    ],
)
def test_valid_ipv6_http_and_git_urls_preserve_safe_authority_and_path(
    raw: str,
    expected: str,
) -> None:
    api = _api()

    assert api.redact_url(raw) == expected


def test_cargo_sparse_url_preserves_safe_scheme_host_and_path() -> None:
    api = _api()
    raw = "sparse+https://user:sparse-secret@packages.example.invalid/index/"

    assert api.redact_url(raw) == ("sparse+https://REDACTED@packages.example.invalid/index/")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "ftp://user:ftp-secret@packages.example.invalid/private",
            "ftp://REDACTED@packages.example.invalid/private",
        ),
        (
            "custom+pkg://user:custom-secret@packages.example.invalid/private",
            "custom+pkg://REDACTED@packages.example.invalid/private",
        ),
        (
            "https://user:'apostrophe-secret@packages.example.invalid/private",
            "https://REDACTED@packages.example.invalid/private",
        ),
    ],
)
def test_generic_hierarchical_schemes_and_apostrophe_userinfo_are_sanitized(
    raw: str,
    expected: str,
) -> None:
    api = _api()

    assert api.redact_url(raw) == expected


def test_unknown_scheme_without_sensitive_components_remains_unchanged() -> None:
    api = _api()
    raw = "ftp://packages.example.invalid/private?channel=stable"

    assert api.redact_url(raw) == raw


def test_email_followed_by_colon_prose_is_not_treated_as_scp_git_syntax() -> None:
    api = _api()
    text = "Contact dev@example.invalid:today or dev@example.invalid: today."

    assert api.redact_url("dev@example.invalid:today") == "dev@example.invalid:today"
    assert api.redact_text(text) == text


@pytest.mark.parametrize(
    "raw",
    [
        "https://user:malformed-secret@[broken.example.invalid/repo",
        "https://user:port-secret@packages.example.invalid:notaport/repo",
        "https://user:percent-secret@packages.example.invalid/repo?to%ZZken=value-secret",
    ],
)
def test_malformed_suspicious_urls_fail_closed_without_throwing(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted.lower()


@pytest.mark.parametrize(
    "raw",
    [
        "https://user:range-secret@packages.example.invalid:99999/repo",
        "https://user:ipv6-secret@[2001:db8::1/repo",
        "https://user:nfkc-secret@exam\uff0fple.invalid/repo",
        "https://first:multiple-secret@second@packages.example.invalid/repo",
        "https://user:slash-secret@packages.example.invalid\\@other.invalid/repo",
        "https://user:control-secret@packages.example.invalid/repo\nInjected: value",
    ],
)
def test_ambiguous_authorities_and_urlsplit_normalization_traps_fail_closed(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted.lower()


@pytest.mark.parametrize(
    "raw",
    [
        "https://user:empty-port-secret@packages.example.invalid:/repo",
        "https://user:encoded-colon-secret@packages%3Aevil.example.invalid/repo",
        "https://user:encoded-at-secret@packages%40evil.example.invalid/repo",
        "https://user:encoded-slash-secret@packages%2Fevil.example.invalid/repo",
        "https://user:encoded-query-secret@packages%3Fevil.example.invalid/repo",
        "https://user:encoded-fragment-secret@packages%23evil.example.invalid/repo",
        "https://user:encoded-control-secret@packages%00evil.example.invalid/repo",
        "https://user:encoded-space-secret@packages%20evil.example.invalid/repo",
        "https://user:unbracketed-secret@2001:db8::1/repo",
        "https://user:bad-bracket-secret@[not-ipv6]/repo",
        "https://user:empty-host-secret@:443/repo",
        "scp-bracket-secret@[not-ipv6]:org/repo.git",
    ],
)
def test_ambiguous_authority_delimiters_ports_and_bracket_hosts_fail_closed(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted.lower()


def test_query_redaction_preserves_nonsensitive_raw_order_duplicates_blanks_flags_and_encoding() -> (
    None
):
    api = _api()
    raw = (
        "https://packages.example.invalid/simple?"
        "channel=one&token=token-secret&channel=two&blank=&flag&encoded=a%2Fb&"
        "API%5FKEY=key-secret&%74oken=decoded-secret"
    )

    redacted = api.redact_url(raw)

    assert redacted == (
        "https://packages.example.invalid/simple?"
        "channel=one&token=REDACTED&channel=two&blank=&flag&encoded=a%2Fb&"
        "API%5FKEY=REDACTED&%74oken=REDACTED"
    )


def test_query_redaction_preserves_mixed_ampersand_semicolon_delimiters() -> None:
    api = _api()
    raw = (
        "https://packages.example.invalid/simple?"
        "channel=one;apikey=semicolon-secret&flag;token=second-secret;blank=&channel=two"
    )

    redacted = api.redact_url(raw)

    assert redacted == (
        "https://packages.example.invalid/simple?"
        "channel=one;apikey=REDACTED&flag;token=REDACTED;blank=&channel=two"
    )
    assert "semicolon-secret" not in redacted
    assert "second-secret" not in redacted


@pytest.mark.parametrize("query_key", ["monkey", "compass", "tokenizer", "secretary"])
def test_query_key_substrings_that_are_not_credential_semantics_remain_unchanged(
    query_key: str,
) -> None:
    api = _api()
    raw = f"https://packages.example.invalid/simple?{query_key}=visible-value"

    assert api.redact_url(raw) == raw


def test_embedded_urls_are_redacted_without_changing_surrounding_free_text() -> None:
    api = _api()
    sentinel = "embedded-secret-741e"
    raw_url = f"https://user:{sentinel}@packages.example.invalid/private?channel=stable"
    text = f"Use registry {raw_url} for the build, then continue."

    redacted = api.redact_text(text)

    assert redacted == (
        "Use registry https://REDACTED@packages.example.invalid/private?channel=stable "
        "for the build, then continue."
    )
    assert sentinel not in redacted


def test_embedded_url_fragment_is_removed_without_swallowing_trailing_prose_punctuation() -> None:
    api = _api()
    text = (
        "Fetch (https://user:punctuation-secret@packages.example.invalid/private"
        "#fragment-secret), then continue."
    )

    assert api.redact_text(text) == (
        "Fetch (https://REDACTED@packages.example.invalid/private), then continue."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Use 'https://user:quoted-secret@packages.example.invalid/private'.",
            "Use 'https://REDACTED@packages.example.invalid/private'.",
        ),
        (
            "Use `https://user:tick-secret@packages.example.invalid/private`.",
            "Use `https://REDACTED@packages.example.invalid/private`.",
        ),
        (
            'Use "https://user:double-secret@packages.example.invalid/private".',
            'Use "https://REDACTED@packages.example.invalid/private".',
        ),
        (
            "Use https://user:'userinfo-secret@packages.example.invalid/private now.",
            "Use https://REDACTED@packages.example.invalid/private now.",
        ),
    ],
)
def test_embedded_quotes_are_preserved_while_apostrophes_inside_userinfo_are_redacted(
    text: str,
    expected: str,
) -> None:
    api = _api()

    assert api.redact_text(text) == expected


@pytest.mark.parametrize("delimiter", ['"', "`", "<"])
def test_invalid_userinfo_delimiters_cannot_leave_a_raw_secret_suffix(
    delimiter: str,
) -> None:
    api = _api()
    sentinel = "delimiter-userinfo-secret-9e41"
    text = f"Use https://user:{delimiter}{sentinel}@packages.example.invalid/private now."

    redacted = api.redact_text(text)

    assert redacted == f"Use {api.REDACTED_URL} now."
    assert sentinel not in redacted


def test_angle_bracket_prose_wrapper_is_preserved_around_a_sanitized_url() -> None:
    api = _api()
    text = "Use <https://user:angle-secret@packages.example.invalid/private>."

    assert api.redact_text(text) == ("Use <https://REDACTED@packages.example.invalid/private>.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Open (https://[2001:db8::1]:8443/index), then continue.",
            "Open (https://[2001:db8::1]:8443/index), then continue.",
        ),
        (
            "Open (https://user:ipv6-embedded-secret@[2001:db8::1]:8443/index), then.",
            "Open (https://REDACTED@[2001:db8::1]:8443/index), then.",
        ),
        (
            "See [https://user:bracket-secret@packages.example.invalid/private].",
            "See [https://REDACTED@packages.example.invalid/private].",
        ),
    ],
)
def test_embedded_ipv6_authority_brackets_and_paired_prose_closers_are_preserved(
    text: str,
    expected: str,
) -> None:
    api = _api()

    assert api.redact_text(text) == expected


def test_ordinary_no_match_text_is_byte_for_byte_unchanged() -> None:
    api = _api()
    text = "Keep punctuation, Unicode ☃, paths ./src, and email dev@example.invalid exactly."

    assert api.redact_text(text) == text


def test_default_text_bound_preserves_large_benign_provider_content() -> None:
    api = _api()
    text = "ordinary provider context\n" * 5_000

    assert len(text) > 100_000
    assert api.MAX_REDACTION_CHARACTERS == 16 * 1024 * 1024
    assert api.redact_text(text) == text


def test_text_redaction_fails_closed_when_candidate_or_character_bound_is_exhausted() -> None:
    api = _api()
    first_secret = "first-bound-secret"
    second_secret = "second-bound-secret"
    text = (
        f"https://user:{first_secret}@one.example.invalid/repo "
        f"https://user:{second_secret}@two.example.invalid/repo"
    )

    candidate_bounded = api.redact_text(text, max_candidates=1)
    character_bounded = api.redact_text(text, max_characters=24)

    assert first_secret not in candidate_bounded
    assert second_secret not in candidate_bounded
    assert api.REDACTED_REMAINDER in candidate_bounded
    assert first_secret not in character_bounded
    assert second_secret not in character_bounded
    assert api.REDACTED_REMAINDER in character_bounded


@pytest.mark.parametrize(
    ("raw", "max_characters"),
    [
        ("https://alice:123456789@example.invalid/path", 18),
        ("https://alice:password-cut@example.invalid/path", 27),
        ("https://packages.example.invalid/path?token=query-cut-secret", 52),
        ("https://packages.example.invalid/path#fragment-cut-secret", 49),
    ],
)
def test_over_bound_text_never_parses_or_returns_a_clipped_prefix(
    raw: str,
    max_characters: int,
) -> None:
    api = _api()

    redacted = api.redact_text(raw, max_characters=max_characters)

    assert redacted == api.REDACTED_REMAINDER
    assert redacted == api.redact_text(redacted, max_characters=max_characters)
    assert raw[:max_characters] not in redacted


def test_direct_url_redaction_fails_closed_when_character_bound_is_exhausted() -> None:
    api = _api()
    sentinel = "direct-bound-secret"
    raw = f"https://user:{sentinel}@packages.example.invalid/private"

    assert api.redact_url(raw, max_characters=16) == api.REDACTED_URL


def test_nested_values_are_sanitized_without_rewriting_code_owned_keys_or_container_types() -> None:
    api = _api()
    https_secret = "nested-https-secret"
    ssh_secret = "nested-ssh-secret"
    prose_secret = "nested-prose-secret"
    value = {
        "registry_url": f"https://user:{https_secret}@packages.example.invalid/private",
        "details": [
            f"ssh://user:{ssh_secret}@git.example.invalid/org/repo.git",
            (f"Mirror: https://user:{prose_secret}@mirror.example.invalid/simple", 7),
        ],
        "enabled": True,
    }

    redacted = api.redact_value(value)

    assert set(redacted) == {"registry_url", "details", "enabled"}
    assert isinstance(redacted["details"], list)
    assert isinstance(redacted["details"][1], tuple)
    assert redacted["details"][1][1] == 7
    assert redacted["enabled"] is True
    rendered = repr(redacted)
    for sentinel in (https_secret, ssh_secret, prose_secret):
        assert sentinel not in rendered
    assert "packages.example.invalid/private" in rendered
    assert "git.example.invalid/org/repo.git" in rendered


def test_nested_redaction_fails_closed_at_depth_and_item_bounds() -> None:
    api = _api()
    depth_secret = "depth-bound-secret"
    item_secret = "item-bound-secret"

    depth_bounded = api.redact_value(
        {"outer": {"inner": f"https://user:{depth_secret}@packages.example.invalid/repo"}},
        max_depth=1,
    )
    item_bounded = api.redact_value(
        {
            "first": "safe",
            "second": f"https://user:{item_secret}@packages.example.invalid/repo",
        },
        max_nodes=2,
    )

    assert depth_secret not in repr(depth_bounded)
    assert item_secret not in repr(item_bounded)
    assert api.REDACTED_VALUE in repr(depth_bounded)
    assert item_bounded == api.REDACTED_VALUE


def test_recursive_value_exact_depth_and_node_bounds_succeed_but_one_over_is_redacted() -> None:
    api = _api()
    depth_secret = "one-over-depth-secret"
    node_secret = "one-over-node-secret"
    exact_depth = {"outer": {"leaf": "safe"}}
    over_depth = {"outer": {"inner": {"leaf": f"https://user:{depth_secret}@host.invalid/x"}}}
    exact_nodes = {"leaf": "safe"}
    over_nodes = {
        "first": "safe",
        "second": f"https://user:{node_secret}@host.invalid/x",
    }

    assert api.redact_value(exact_depth, max_depth=2) == exact_depth
    depth_result = api.redact_value(over_depth, max_depth=2)
    assert depth_secret not in repr(depth_result)
    assert api.REDACTED_VALUE in repr(depth_result)
    assert api.redact_value(exact_nodes, max_nodes=2) == exact_nodes
    node_result = api.redact_value(over_nodes, max_nodes=2)
    assert node_secret not in repr(node_result)
    assert api.REDACTED_VALUE in repr(node_result)


def test_recursive_value_self_reference_terminates_fail_closed() -> None:
    api = _api()
    value: list[object] = []
    value.append(value)

    redacted = api.redact_value(value)

    assert redacted == [api.REDACTED_VALUE]


def test_recursive_text_character_budget_is_aggregate_across_sibling_values() -> None:
    api = _api()
    value = {"first": "abcd", "second": "efgh"}

    assert api.redact_value(value, max_text_characters=8) == value
    assert api.redact_value(value, max_text_characters=7) == api.REDACTED_VALUE


def test_recursive_candidate_budget_is_aggregate_across_sibling_values() -> None:
    api = _api()
    value = {
        "first": "https://user:first-aggregate-secret@one.example.invalid/repo",
        "second": "https://user:second-aggregate-secret@two.example.invalid/repo",
    }

    exact = api.redact_value(value, max_text_candidates=2)

    assert "first-aggregate-secret" not in repr(exact)
    assert "second-aggregate-secret" not in repr(exact)
    assert api.redact_value(value, max_text_candidates=1) == api.REDACTED_VALUE


class _CountingMapping(Mapping[str, str]):
    def __init__(self) -> None:
        self.iterations = 0
        self._values = {"first": "one", "second": "two", "third": "three"}

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        for key in self._values:
            self.iterations += 1
            yield key

    def __len__(self) -> int:
        return len(self._values)


def test_recursive_node_exhaustion_stops_before_iterating_an_oversized_mapping() -> None:
    api = _api()
    value = _CountingMapping()

    redacted = api.redact_value(value, max_nodes=2)

    assert redacted == api.REDACTED_VALUE
    assert value.iterations == 0


@pytest.mark.parametrize(
    "function_name",
    ["redact_url", "redact_text"],
)
def test_string_redactors_are_deterministic_and_idempotent(function_name: str) -> None:
    api = _api()
    function = getattr(api, function_name)
    raw = (
        "Prefix " if function_name == "redact_text" else ""
    ) + "https://user:idempotent-secret@packages.example.invalid/repo?token=query-secret"

    first = function(raw)

    assert function(raw) == first
    assert function(first) == first


def test_recursive_value_redaction_is_idempotent() -> None:
    api = _api()
    value = {
        "url": "https://user:value-secret@packages.example.invalid/repo",
        "items": ("plain",),
    }

    first = api.redact_value(value)

    assert api.redact_value(value) == first
    assert api.redact_value(first) == first
