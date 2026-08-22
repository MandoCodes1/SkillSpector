# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for bounded dependency-source credential redaction."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from time import perf_counter
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
    ["ssh_key", "registry-key", "encryption.key", "x_pass", "db_sig"],
)
def test_explicitly_separated_weak_credential_words_are_redacted(query_key: str) -> None:
    api = _api()
    sentinel = "separated-weak-query-secret-e90a"

    redacted = api.redact_url(f"https://packages.example.invalid/simple?{query_key}={sentinel}")

    assert redacted.endswith(f"?{query_key}=REDACTED")
    assert sentinel not in redacted


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
    "query_key",
    [
        "authorizationtoken",
        "AUTHORIZATIONTOKEN",
        "authorizationToken",
        "authorization%54oken",
        "authenticationtoken",
        "AUTHENTICATIONTOKEN",
        "authenticationToken",
        "authentication%54oken",
        "credentialtoken",
        "CREDENTIALTOKEN",
        "credentialToken",
        "%63redentialtoken",
        "tokensecret",
        "TOKENSECRET",
        "tokenSecret",
        "%74okenSecret",
        "secretkeytoken",
        "SECRETKEYTOKEN",
        "secretKeyToken",
        "secret%4BeyToken",
        "passphrasekey",
        "PASSPHRASEKEY",
        "passphraseKey",
        "passphrase%4Bey",
        "signaturetoken",
        "SIGNATURETOKEN",
        "signatureToken",
        "signature%54oken",
        "dbpassword",
        "DBPASSWORD",
        "dbPassword",
        "db%50assword",
        "registrytoken",
        "REGISTRYTOKEN",
        "registryToken",
        "registry%54oken",
        "dbauth",
        "DBAUTH",
        "dbAuth",
        "db%41uth",
        "clientcredential",
        "CLIENTCREDENTIAL",
        "clientCredential",
        "client%43redential",
        "requestsignature",
        "REQUESTSIGNATURE",
        "requestSignature",
        "request%53ignature",
        "accesskey",
        "ACCESSKEY",
        "accessKey",
        "access%4Bey",
        "githubtoken",
        "githubTokenValue",
        "githubtokenvalue",
        "GITHUBTOKENVALUE",
        "github%54oken%56alue",
    ],
)
def test_compact_query_key_grammar_redacts_complete_credential_terms(
    query_key: str,
) -> None:
    api = _api()
    sentinel = "segmented-query-value-secret-11c4"
    raw = f"https://packages.example.invalid/simple?channel=one;{query_key}={sentinel}&channel=two"

    redacted = api.redact_url(raw)

    assert redacted == (
        f"https://packages.example.invalid/simple?channel=one;{query_key}=REDACTED&channel=two"
    )
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "//user:scheme-relative-secret@packages.example.invalid/private#private-fragment",
            "//REDACTED@packages.example.invalid/private",
        ),
        (
            "//packages.example.invalid/private?token=scheme-relative-query-secret&channel=stable",
            "//packages.example.invalid/private?token=REDACTED&channel=stable",
        ),
        (
            "//user:scheme-relative-ipv6-secret@[2001:db8::1]:8443/private",
            "//REDACTED@[2001:db8::1]:8443/private",
        ),
        (
            "//[2001:db8::1]:8443/private?channel=stable",
            "//[2001:db8::1]:8443/private?channel=stable",
        ),
    ],
)
def test_scheme_relative_urls_sanitize_authority_query_and_fragment(
    raw: str,
    expected: str,
) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == expected
    assert api.redact_url(redacted) == redacted
    assert "secret" not in redacted
    assert "fragment" not in redacted


@pytest.mark.parametrize(
    "raw",
    [
        "//user:scheme-relative-port-secret@packages.example.invalid:/private",
        "//user:scheme-relative-bracket-secret@[not-ipv6]/private",
        "//first:scheme-relative-at-secret@second@packages.example.invalid/private",
        "//user:scheme-relative-host-secret@/private",
    ],
)
def test_malformed_scheme_relative_authorities_fail_closed(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted


@pytest.mark.parametrize(
    "raw",
    [
        "https://safe.invalid/?next=//user:nested-relative-secret@evil.invalid/x",
        "https://safe.invalid//user:nested-path-secret@evil.invalid/x",
        "https://safe.invalid/?next=//evil.invalid/x?token=nested-query-secret",
    ],
)
def test_nested_scheme_relative_references_make_outer_urls_fail_closed(raw: str) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted


@pytest.mark.parametrize(
    "raw",
    [
        "//safe.invalid/path?next=https://user:nested-absolute-secret@evil.invalid/x",
        "//safe.invalid/path?next=user:nested-scp-secret@evil.invalid:repo.git",
    ],
)
def test_scheme_relative_outer_references_reject_nested_credential_candidates(
    raw: str,
) -> None:
    api = _api()

    redacted = api.redact_url(raw)

    assert redacted == api.REDACTED_URL
    assert "secret" not in redacted


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


@pytest.mark.parametrize(
    "query_key",
    [
        "monkey",
        "MONKEY",
        "Monkey",
        "monKey",
        "%6Donkey",
        "compass",
        "COMPASS",
        "Compass",
        "comPass",
        "%63ompass",
        "tokenizer",
        "TOKENIZER",
        "Tokenizer",
        "%74okenizer",
        "secretary",
        "SECRETARY",
        "Secretary",
        "%73ecretary",
        "registrytokenizer",
        "clientsecretary",
        "dbauthor",
        "accesskeyboard",
        "requestsignatory",
        "passwordless",
        "keynote",
        "privatekeyboard",
        "authorizationtokenizer",
    ],
)
def test_ambiguous_query_key_substrings_are_conservatively_redacted(
    query_key: str,
) -> None:
    api = _api()
    raw = f"https://packages.example.invalid/simple?{query_key}=visible-value"

    redacted = api.redact_url(raw)

    assert redacted.endswith(f"?{query_key}=REDACTED")
    assert "visible-value" not in redacted


def test_nested_hierarchical_uri_in_query_fails_closed_without_leaking_userinfo() -> None:
    api = _api()
    sentinel = "nested-query-uri-secret-c8b2"
    text = (
        "Use https://safe.example.invalid/path?next="
        f"https://user:{sentinel}@evil.example.invalid/repo now."
    )

    redacted = api.redact_text(text)

    assert redacted == f"Use {api.REDACTED_URL} now."
    assert sentinel not in redacted


def test_nested_scp_uri_in_query_fails_closed_without_leaking_userinfo() -> None:
    api = _api()
    sentinel = "nested-scp-query-secret-14da"
    text = (
        "Use https://safe.example.invalid/path?next="
        f"{sentinel}@evil.example.invalid:org/repo.git now."
    )

    redacted = api.redact_text(text)

    assert redacted == f"Use {api.REDACTED_URL} now."
    assert sentinel not in redacted


@pytest.mark.parametrize("query_key", ["%FFtoken", "to%00ken", "%2574oken"])
def test_ambiguous_or_control_bearing_query_keys_fail_closed(query_key: str) -> None:
    api = _api()
    sentinel = "ambiguous-key-value-secret-b712"

    redacted = api.redact_url(f"https://packages.example.invalid/simple?{query_key}={sentinel}")

    assert redacted == api.REDACTED_URL
    assert sentinel not in redacted


def test_query_key_bounds_accept_exact_decoded_and_raw_limits_and_reject_one_over() -> None:
    api = _api()
    sentinel = "bounded-key-value-secret-65e3"
    exact_decoded = ("a" * 251) + "token"
    exact_raw = ("%61" * 251) + "%74%6F%6B%65%6E"
    over_decoded = "a" + exact_decoded
    over_raw = "x" + exact_raw

    assert len(exact_decoded) == 256
    assert len(exact_raw) == 768
    for query_key in (exact_decoded, exact_raw):
        redacted = api.redact_url(f"https://packages.example.invalid/simple?{query_key}={sentinel}")
        assert redacted.endswith(f"?{query_key}=REDACTED")
        assert sentinel not in redacted
    for query_key in (over_decoded, over_raw):
        assert (
            api.redact_url(f"https://packages.example.invalid/simple?{query_key}={sentinel}")
            == api.REDACTED_URL
        )


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


@pytest.mark.parametrize(
    ("token", "sentinel"),
    [
        (
            "1https://alice:leading-digit-secret@host.invalid/path",
            "leading-digit-secret",
        ),
        (
            "-https://alice:leading-hyphen-secret@host.invalid/path",
            "leading-hyphen-secret",
        ),
        (
            "://alice:missing-scheme-secret@host.invalid/path",
            "missing-scheme-secret",
        ),
        (
            "http:://alice:double-colon-secret@host.invalid/path",
            "double-colon-secret",
        ),
        (
            "https_://alice:underscore-scheme-secret@host.invalid/path",
            "underscore-scheme-secret",
        ),
    ],
)
def test_malformed_hierarchical_tokens_fail_closed_as_one_bounded_span(
    token: str,
    sentinel: str,
) -> None:
    api = _api()

    redacted = api.redact_text(f"Use {token} now.")

    assert redacted == f"Use {api.REDACTED_URL} now."
    assert sentinel not in redacted


@pytest.mark.parametrize(
    ("text", "_previous_exact_output"),
    [
        (
            '<a href="https://host.invalid/path">link</a>',
            '<a href="https://host.invalid/path">link</a>',
        ),
        (
            '<a href="https://alice:html-secret@host.invalid/path">link</a>',
            '<a href="https://REDACTED@host.invalid/path">link</a>',
        ),
        (
            "<a href=https://host.invalid/path>link</a>",
            "<a href=https://host.invalid/path>link</a>",
        ),
        (
            "<a href=https://alice:html-unquoted-secret@host.invalid/path>link</a>",
            "<a href=https://REDACTED@host.invalid/path>link</a>",
        ),
        (
            "<a href=git@host.invalid:org/repo.git>link</a>",
            "<a href=REDACTED@host.invalid:org/repo.git>link</a>",
        ),
        (
            'x="https://host.invalid/path"; next',
            'x="https://host.invalid/path"; next',
        ),
        (
            'x="https://alice:assignment-secret@host.invalid/path"; next',
            'x="https://REDACTED@host.invalid/path"; next',
        ),
        (
            "const registry=`https://alice:tick-source-secret@host.invalid/path`; next",
            "const registry=`https://REDACTED@host.invalid/path`; next",
        ),
        (
            "[https://host.invalid/path](mailto:dev@example.invalid)",
            "[https://host.invalid/path](mailto:dev@example.invalid)",
        ),
        (
            "[https://alice:markdown-secret@host.invalid/path](mailto:dev@example.invalid)",
            "[https://REDACTED@host.invalid/path](mailto:dev@example.invalid)",
        ),
        (
            "[dev](mailto:dev@example.invalid)[site](https://host.invalid/path)",
            "[dev](mailto:dev@example.invalid)[site](https://host.invalid/path)",
        ),
        (
            "[dev@example.invalid](https://alice:reverse-markdown-secret@host.invalid/path)",
            "[dev@example.invalid](https://REDACTED@host.invalid/path)",
        ),
        (
            'x="https://host.invalid/path";y="dev@example.invalid"',
            'x="https://host.invalid/path";y="dev@example.invalid"',
        ),
        (
            'x="https://alice:code-secret@host.invalid/path";y="dev@example.invalid"',
            'x="https://REDACTED@host.invalid/path";y="dev@example.invalid"',
        ),
        (
            'const x="https://host.invalid/path"+"dev@example.invalid/path";',
            'const x="https://host.invalid/path"+"dev@example.invalid/path";',
        ),
        (
            'const x="https://alice:concat-secret@host.invalid/path"+"dev@example.invalid/path";',
            'const x="https://REDACTED@host.invalid/path"+"dev@example.invalid/path";',
        ),
    ],
)
def test_ambiguous_markup_and_source_tokens_are_masked_deterministically(
    text: str,
    _previous_exact_output: str,
) -> None:
    api = _api()
    redacted = api.redact_text(text)

    assert api.REDACTED_URL in redacted
    assert api.redact_text(redacted) == redacted
    for sentinel in (
        "html-secret",
        "html-unquoted-secret",
        "assignment-secret",
        "tick-source-secret",
        "markdown-secret",
        "reverse-markdown-secret",
        "code-secret",
        "concat-secret",
    ):
        assert sentinel not in redacted


@pytest.mark.parametrize(
    ("text", "_previous_exact_output"),
    [
        (
            '{"url":"https://host.invalid/path","enabled":true}',
            '{"url":"https://host.invalid/path","enabled":true}',
        ),
        (
            '{"url":"https://user:json-secret@host.invalid/path","enabled":true}',
            '{"url":"https://REDACTED@host.invalid/path","enabled":true}',
        ),
        (
            '["https://host.invalid/one","https://host.invalid/two"]',
            '["https://host.invalid/one","https://host.invalid/two"]',
        ),
        (
            '["https://user:first-array-secret@one.invalid/x",'
            '"https://user:second-array-secret@two.invalid/y"]',
            '["https://REDACTED@one.invalid/x","https://REDACTED@two.invalid/y"]',
        ),
    ],
)
def test_minified_json_url_tokens_are_masked_as_ambiguous_provider_context(
    text: str,
    _previous_exact_output: str,
) -> None:
    api = _api()
    redacted = api.redact_text(text)

    assert redacted == api.REDACTED_URL
    assert api.redact_text(redacted) == redacted
    assert "json-secret" not in redacted
    assert "array-secret" not in redacted


def test_paired_punctuation_survives_fragment_removal() -> None:
    api = _api()
    text = "Open [https://user:fragment-secret@host.invalid/path#private-fragment], next."

    assert api.redact_text(text) == ("Open [https://REDACTED@host.invalid/path], next.")


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

    redacted = api.redact_text(text)

    assert redacted == expected
    assert api.redact_text(redacted) == redacted
    assert "secret" not in redacted


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


@pytest.mark.parametrize(
    ("text", "_previous_exact_output"),
    [
        (
            'x="https://user:123"quoted-suffix-secret@host.invalid/path"',
            'x="[REDACTED_URL]"',
        ),
        (
            "x=`https://user:123`tick-suffix-secret@host.invalid/path`",
            "x=`[REDACTED_URL]`",
        ),
        (
            'x="https://user:123"unclosed-suffix-secret@host.invalid/path',
            'x="[REDACTED_URL]',
        ),
    ],
)
def test_apparent_wrapper_inside_incomplete_userinfo_cannot_expose_its_suffix(
    text: str,
    _previous_exact_output: str,
) -> None:
    api = _api()

    redacted = api.redact_text(text)

    assert redacted == api.REDACTED_URL
    assert "suffix-secret" not in redacted


@pytest.mark.parametrize("separator", ["+", "=", ",", ";", ":"])
def test_apparent_wrapper_cannot_use_userinfo_punctuation_to_expose_a_later_at_sign(
    separator: str,
) -> None:
    api = _api()
    text = f'x="https://user:123"quoted{separator}suffix-secret@host.invalid/path"'

    redacted = api.redact_text(text)

    assert redacted == api.REDACTED_URL
    assert "suffix-secret" not in redacted


@pytest.mark.parametrize(
    "text",
    [
        'x="https://user:123"}suffix-secret@host.invalid/path"',
        'x="https://user:123","suffix-secret@host.invalid/path"',
    ],
)
def test_apparent_wrapper_structural_shortcuts_cannot_expose_a_later_at_sign(
    text: str,
) -> None:
    api = _api()

    redacted = api.redact_text(text)

    assert api.REDACTED_URL in redacted
    assert "suffix-secret" not in redacted


def test_angle_bracket_prose_wrapper_is_preserved_around_a_sanitized_url() -> None:
    api = _api()
    text = "Use <https://user:angle-secret@packages.example.invalid/private>."

    assert api.redact_text(text) == ("Use <https://REDACTED@packages.example.invalid/private>.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Use //user:relative-prose-secret@packages.example.invalid/private now.",
            "Use //REDACTED@packages.example.invalid/private now.",
        ),
        (
            "Use (//user:relative-wrapper-secret@packages.example.invalid/private), now.",
            "Use (//REDACTED@packages.example.invalid/private), now.",
        ),
    ],
)
def test_embedded_scheme_relative_references_are_sanitized_with_simple_wrappers(
    text: str,
    expected: str,
) -> None:
    api = _api()

    redacted = api.redact_text(text)

    assert redacted == expected
    assert api.redact_text(redacted) == redacted
    assert "secret" not in redacted


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Use //packages.example.invalid/private?token=relative-query-prose-secret now.",
            "Use //packages.example.invalid/private?token=REDACTED now.",
        ),
        (
            "before\n//packages.example.invalid/private?token=relative-query-line-secret\nafter",
            "before\n//packages.example.invalid/private?token=REDACTED\nafter",
        ),
    ],
)
def test_embedded_query_only_scheme_relative_references_cross_whitespace_boundaries(
    text: str,
    expected: str,
) -> None:
    api = _api()

    redacted = api.redact_text(text)

    assert redacted == expected
    assert "secret" not in redacted


@pytest.mark.parametrize(
    "text",
    [
        "x=//user:relative-assignment-secret@packages.example.invalid/private",
        'href="//user:relative-attribute-secret@packages.example.invalid/private"',
        '<a href="//user:relative-markup-secret@packages.example.invalid/private">link</a>',
    ],
)
def test_ambiguous_scheme_relative_assignment_and_markup_tokens_are_masked(
    text: str,
) -> None:
    api = _api()

    redacted = api.redact_text(text)

    assert api.REDACTED_URL in redacted
    assert "secret" not in redacted
    assert api.redact_text(redacted) == redacted


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Use 'scp-wrapper-secret@git.example.invalid:repo.git' now.",
            "Use 'REDACTED@git.example.invalid:repo.git' now.",
        ),
        (
            "Use scp-punctuation-secret@git.example.invalid:repo.git, now.",
            "Use REDACTED@git.example.invalid:repo.git, now.",
        ),
    ],
)
def test_simple_scp_wrappers_and_punctuation_do_not_hide_dot_git_candidates(
    text: str,
    expected: str,
) -> None:
    api = _api()

    assert api.redact_text(text) == expected


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


def test_repeated_false_scp_prefixes_are_processed_in_linear_tokens() -> None:
    api = _api()
    atom = "a@h:x"
    text = ",".join([atom] * 20_000)

    started = perf_counter()
    result = api.redact_text_result(text)
    elapsed = perf_counter() - started

    assert result.value == text
    assert result.complete is True
    assert result.candidates == 0
    assert elapsed < 2.0


def test_dense_ambiguous_url_envelope_is_processed_once_and_fails_closed() -> None:
    api = _api()
    text = "[" + ",".join(f'"https://host.invalid/{index}"' for index in range(400)) + "]"

    started = perf_counter()
    result = api.redact_text_result(text, max_candidates=400)
    elapsed = perf_counter() - started

    assert result == api.TextRedactionResult(
        value=api.REDACTED_URL,
        complete=True,
        candidates=400,
        reason=None,
    )
    assert api.redact_text(result.value) == result.value
    assert elapsed < 2.0


def test_benign_16_mib_text_uses_the_constant_time_candidate_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    text = "x" * api.MAX_REDACTION_CHARACTERS

    def unexpected_scan(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("candidate scanner should not run for benign text")

    monkeypatch.setattr(api, "_redact_text_with_usage", unexpected_scan)
    started = perf_counter()
    result = api.redact_text_result(text)
    elapsed = perf_counter() - started

    assert result.value is text
    assert result.complete is True
    assert result.candidates == 0
    assert result.reason is None
    assert elapsed < 2.0


def test_nested_benign_text_uses_the_same_candidate_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    text = "nested benign provider context" * 4_000

    def unexpected_scan(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("candidate scanner should not run for benign nested text")

    monkeypatch.setattr(api, "_redact_text_with_usage", unexpected_scan)

    assert api.redact_value(
        {"body": text},
        max_nodes=2,
        max_text_characters=len(text),
    ) == {"body": text}


class _CopyCountingString(str):
    copied_characters: int

    def __new__(cls, value: str) -> _CopyCountingString:
        instance = super().__new__(cls, value)
        instance.copied_characters = 0
        return instance

    def __getitem__(self, key: int | slice) -> str:
        result = super().__getitem__(key)
        if isinstance(key, int):
            character = _CopyCountingString(result)
            character.copied_characters = self.copied_characters
            return character
        return result

    def __add__(self, other: str) -> _CopyCountingString:
        result = _CopyCountingString(super().__add__(other))
        result.copied_characters = (
            self.copied_characters + getattr(other, "copied_characters", 0) + len(self) + len(other)
        )
        return result


class _FindSpanCountingString(str):
    requested_characters: int

    def __new__(cls, value: str) -> _FindSpanCountingString:
        instance = super().__new__(cls, value)
        instance.requested_characters = 0
        return instance

    def find(
        self,
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> int:
        limit = len(self) if end is None else min(end, len(self))
        self.requested_characters += max(0, limit - start)
        return super().find(sub, start, len(self) if end is None else end)


def test_trailing_punctuation_is_detached_without_repeated_suffix_copying() -> None:
    api = _api()
    value = _CopyCountingString("https://host.invalid/path" + ("." * 1_000))

    candidate, punctuation = api._detach_trailing_prose_punctuation(value, None)

    assert candidate == "https://host.invalid/path"
    assert punctuation == "." * 1_000
    assert getattr(punctuation, "copied_characters", 0) <= len(value) * 2


def test_scheme_relative_signal_discovery_does_not_rescan_dense_suffixes() -> None:
    api = _api()
    value = _FindSpanCountingString(",".join(["x=//user@host.invalid"] * 200))

    assert api._scheme_relative_reference_signals(value) == 200
    assert value.requested_characters <= len(value) * 4


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


def test_candidate_signal_budget_has_exact_one_over_and_zero_behavior() -> None:
    api = _api()
    text = (
        "https://user:first-budget-secret@one.invalid/x "
        "https://user:second-budget-secret@two.invalid/y"
    )

    assert api.redact_text(text, max_candidates=2) == (
        "https://REDACTED@one.invalid/x https://REDACTED@two.invalid/y"
    )
    assert api.redact_text(text, max_candidates=1) == (
        f"https://REDACTED@one.invalid/x {api.REDACTED_REMAINDER}"
    )
    assert api.redact_text(text, max_candidates=0) == api.REDACTED_REMAINDER


def test_scheme_relative_candidates_have_exact_budget_usage_without_double_charging() -> None:
    api = _api()
    text = (
        "//user:first-relative-budget-secret@one.invalid/x "
        "https://user:absolute-budget-secret@two.invalid/y "
        "//three.invalid/z?token=third-relative-budget-secret"
    )

    exact = api.redact_text_result(text, max_candidates=3)
    one_over = api.redact_text_result(text, max_candidates=2)
    zero = api.redact_text_result(text, max_candidates=0)

    assert exact == api.TextRedactionResult(
        value=(
            "//REDACTED@one.invalid/x https://REDACTED@two.invalid/y "
            "//three.invalid/z?token=REDACTED"
        ),
        complete=True,
        candidates=3,
        reason=None,
    )
    assert one_over == api.TextRedactionResult(
        value=(f"//REDACTED@one.invalid/x https://REDACTED@two.invalid/y {api.REDACTED_REMAINDER}"),
        complete=False,
        candidates=2,
        reason=api.TextRedactionIncompleteReason.CANDIDATE_LIMIT,
    )
    assert zero == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=0,
        reason=api.TextRedactionIncompleteReason.CANDIDATE_LIMIT,
    )


def test_scheme_relative_ipv6_userinfo_is_one_candidate_not_an_scp_candidate() -> None:
    api = _api()
    text = "//user:relative-ipv6-budget-secret@[2001:db8::1]:8443/private"

    assert api.redact_text_result(text, max_candidates=1) == api.TextRedactionResult(
        value="//REDACTED@[2001:db8::1]:8443/private",
        complete=True,
        candidates=1,
        reason=None,
    )
    assert api.redact_text(text, max_candidates=0) == api.REDACTED_REMAINDER


def test_interior_double_slash_prefix_does_not_double_charge_a_later_relative_ipv6() -> None:
    api = _api()
    text = "a//b,x=//user:relative-ipv6-prefix-secret@[2001:db8::1]:8443/private"

    result = api.redact_text_result(text, max_candidates=1)

    assert result == api.TextRedactionResult(
        value=api.REDACTED_URL,
        complete=True,
        candidates=1,
        reason=None,
    )
    assert "secret" not in result.value


def test_non_reference_double_slashes_remain_on_the_benign_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    text = "Keep //path, src/a//b.py, and // comment text unchanged."

    def unexpected_scan(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("candidate scanner should not run without a hierarchical reference")

    monkeypatch.setattr(api, "_redact_text_with_usage", unexpected_scan)

    assert api.redact_text_result(text) == api.TextRedactionResult(
        value=text,
        complete=True,
        candidates=0,
        reason=None,
    )


def test_large_interior_double_slash_text_stays_bounded_and_off_the_candidate_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    text = " ".join(["src/a//b.py"] * 50_000)

    def unexpected_scan(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("candidate scanner should not run for interior double slashes")

    def unexpected_relative_scan(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Python relative-reference scan should not run for interior double slashes")

    monkeypatch.setattr(api, "_redact_text_with_usage", unexpected_scan)
    monkeypatch.setattr(api, "_scan_scheme_relative_references", unexpected_relative_scan)
    started = perf_counter()
    result = api.redact_text_result(text)
    elapsed = perf_counter() - started

    assert result.value is text
    assert result.complete is True
    assert result.candidates == 0
    assert elapsed < 2.0


def test_structured_text_result_distinguishes_literal_placeholder_from_exhaustion() -> None:
    api = _api()
    literal = api.redact_text_result(api.REDACTED_REMAINDER)
    literal_with_prefix = api.redact_text_result(f"prefix {api.REDACTED_REMAINDER}")
    exhausted = api.redact_text_result(
        "https://user:structured-result-secret@host.invalid/path",
        max_candidates=0,
    )
    exhausted_with_prefix = api.redact_text_result(
        "prefix https://user:structured-result-secret@host.invalid/path",
        max_candidates=0,
    )

    assert literal == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=True,
        candidates=0,
        reason=None,
    )
    assert exhausted.value == api.REDACTED_REMAINDER
    assert exhausted.complete is False
    assert exhausted.candidates == 0
    assert exhausted.reason is api.TextRedactionIncompleteReason.CANDIDATE_LIMIT
    assert literal_with_prefix.value == exhausted_with_prefix.value
    assert literal_with_prefix.complete is True
    assert exhausted_with_prefix.complete is False


def test_structured_text_result_reports_character_bound_and_retained_candidate_usage() -> None:
    api = _api()
    text = (
        "https://user:first-structured-secret@one.invalid/path "
        "https://user:second-structured-secret@two.invalid/path"
    )

    character_limited = api.redact_text_result(text, max_characters=len(text) - 1)
    candidate_limited = api.redact_text_result(text, max_candidates=1)

    assert character_limited == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=0,
        reason=api.TextRedactionIncompleteReason.CHARACTER_LIMIT,
    )
    assert candidate_limited.complete is False
    assert candidate_limited.candidates == 1
    assert candidate_limited.reason is api.TextRedactionIncompleteReason.CANDIDATE_LIMIT
    assert candidate_limited.value.endswith(api.REDACTED_REMAINDER)


def test_nested_scp_signals_share_the_hierarchical_candidate_budget() -> None:
    api = _api()
    text = "https://safe.invalid/p?next=a@b:x.git,c@d:y.git"

    assert api.redact_text(text, max_candidates=3) == api.REDACTED_URL
    assert api.redact_text(text, max_candidates=2) == api.REDACTED_REMAINDER
    assert api.redact_text(text, max_candidates=1) == api.REDACTED_REMAINDER


def test_scp_candidate_containing_a_hierarchical_marker_fails_closed_and_charges_both() -> None:
    api = _api()
    sentinel = "inverse-nested-scp-secret-0bc4"
    text = f"{sentinel}@host.invalid:org://evil.invalid/repo.git"

    assert api.redact_text(text, max_candidates=2) == api.REDACTED_URL
    assert api.redact_text(text, max_candidates=1) == api.REDACTED_REMAINDER
    assert api.redact_text(text, max_candidates=0) == api.REDACTED_REMAINDER
    assert sentinel not in api.redact_text(text)


def test_ambiguous_email_assignment_and_url_token_is_masked() -> None:
    api = _api()
    text = "owner=dev@example.invalid:url=https://host.invalid/path"

    assert api.redact_text(text) == api.REDACTED_URL


@pytest.mark.parametrize(
    "text",
    [
        "a@b:x.git,https://user:first-same-token-secret@host.invalid/x",
        (
            '["https://user:first-array-secret@one.invalid/x",'
            '"https://user:second-array-secret@two.invalid/y"]'
        ),
    ],
)
def test_structured_result_retains_same_token_candidate_usage_on_exhaustion(
    text: str,
) -> None:
    api = _api()

    result = api.redact_text_result(text, max_candidates=1)

    assert result == api.TextRedactionResult(
        value=api.REDACTED_REMAINDER,
        complete=False,
        candidates=1,
        reason=api.TextRedactionIncompleteReason.CANDIDATE_LIMIT,
    )
    assert "same-token-secret" not in result.value
    assert "array-secret" not in result.value


@pytest.mark.parametrize(
    "template",
    [
        "{sentinel}#x@git.example.invalid:org/repo.git",
        "{sentinel}@git.example.invalid#x:org/repo.git",
        "{sentinel}@git.example.invalid:org#x/repo.git",
    ],
)
def test_scp_candidate_with_an_ambiguous_raw_fragment_fails_closed(template: str) -> None:
    api = _api()
    sentinel = "scp-fragment-prefix-secret-2e70"
    text = template.format(sentinel=sentinel)

    assert api.redact_text(text) == api.REDACTED_URL
    assert api.redact_text(text, max_candidates=0) == api.REDACTED_REMAINDER
    assert sentinel not in api.redact_text(text)


def test_scp_discovery_uses_the_host_path_separator_not_the_last_colon() -> None:
    api = _api()
    sentinel = "multi-colon-scp-userinfo-secret-77a1"
    text = f"user:{sentinel}@evil.invalid:org/repo.git:x"

    assert api.redact_text(text) == api.REDACTED_URL
    assert api.redact_text(text, max_candidates=0) == api.REDACTED_REMAINDER
    assert sentinel not in api.redact_text(text)


def test_markup_scanner_output_is_idempotent() -> None:
    api = _api()
    text = (
        '{"url":"https://user:json-idempotent-secret@host.invalid/path",'
        '"mirror":"https://host.invalid/mirror"}'
    )

    first = api.redact_text(text)

    assert api.redact_text(first) == first


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
