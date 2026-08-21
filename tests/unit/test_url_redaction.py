# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for bounded dependency-source credential redaction."""

from __future__ import annotations

import importlib
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


def test_ordinary_no_match_text_is_byte_for_byte_unchanged() -> None:
    api = _api()
    text = "Keep punctuation, Unicode ☃, paths ./src, and email dev@example.invalid exactly."

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
    assert item_bounded == {"first": "safe", "second": api.REDACTED_VALUE}


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
