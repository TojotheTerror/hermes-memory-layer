from pathlib import Path
import subprocess

import pytest

import hermes_memory.source_discovery as source_discovery
from hermes_memory.source_discovery import SourcePolicy, discover_sources


def _write(root: Path, relative_path: str, content: bytes = b"safe text") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_repeatable_include_and_exclude_globs_are_allowlist_first(tmp_path: Path) -> None:
    _write(tmp_path, "Operations/runbook.md")
    _write(tmp_path, "Models/guide.md")
    _write(tmp_path, "Models/private/notes.md")
    _write(tmp_path, "Personal/journal.md")

    result = discover_sources(
        tmp_path,
        SourcePolicy(
            include_patterns=("Operations/**", "Models/**"),
            exclude_patterns=("**/private/**", "**/*.tmp"),
        ),
    )

    assert result.relative_paths == ("Models/guide.md", "Operations/runbook.md")
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("Models/private/notes.md", "exclude_pattern"),
        ("Personal/journal.md", "not_in_allowlist"),
    ]


def test_apply_requires_include_patterns_or_allow_all_approval(tmp_path: Path) -> None:
    _write(tmp_path, "approved.md")

    with pytest.raises(ValueError, match="include pattern"):
        discover_sources(tmp_path, SourcePolicy(), for_apply=True)

    result = discover_sources(
        tmp_path,
        SourcePolicy(allow_all_approved=True),
        for_apply=True,
    )
    assert result.relative_paths == ("approved.md",)


def test_default_excluded_directories_are_rejected(tmp_path: Path) -> None:
    excluded_directories = (
        ".git",
        ".obsidian",
        ".trash",
        "node_modules",
        "vendor",
        ".venv",
        "build",
        "dist",
        "cache",
    )
    for directory in excluded_directories:
        _write(tmp_path, f"{directory}/nested/source.md")
    _write(tmp_path, "z-safe.md")
    _write(tmp_path, "a-safe.md")

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",)),
    )

    assert result.relative_paths == ("a-safe.md", "z-safe.md")
    assert [(item.path, item.rule) for item in result.rejected] == [
        (f"{directory}/nested/source.md", "default_excluded_directory")
        for directory in sorted(excluded_directories)
    ]


def test_repository_discovery_rejects_git_ignored_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path, ".gitignore", b"ignored.md\nignored-dir/\n")
    _write(tmp_path, "ignored.md")
    _write(tmp_path, "ignored-dir/nested.md")
    _write(tmp_path, "kept.md")

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",)),
        repository=True,
    )

    assert result.relative_paths == (".gitignore", "kept.md")
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("ignored-dir/nested.md", "git_ignored"),
        ("ignored.md", "git_ignored"),
    ]


def test_missing_git_uses_static_fallback_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "kept.md")

    def missing_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("git executable missing")

    monkeypatch.setattr(source_discovery.subprocess, "run", missing_git)

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",)),
        repository=True,
    )

    assert result.relative_paths == ("kept.md",)
    assert result.warnings == (
        "git check-ignore unavailable; static exclusions only",
    )


def test_symlink_that_escapes_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = _write(tmp_path, "outside.md")
    (root / "escape.md").symlink_to(outside)

    result = discover_sources(root, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("escape.md", "symlink_escape"),
    ]


def test_binary_file_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "image.bin", b"text prefix\x00binary payload")

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("image.bin", "binary_content"),
    ]


def test_file_larger_than_policy_limit_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "large.md", b"12345")

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",), max_file_size_bytes=4),
    )

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("large.md", "max_file_size"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    (
        "generated.pb.go",
        "api.generated.ts",
        "generated/client.py",
    ),
)
def test_generated_files_are_rejected(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "generated_file"),
    ]


@pytest.mark.parametrize(
    ("relative_path", "content"),
    (
        ("gen/api.pb.go", b"// Code generated by protoc. DO NOT EDIT.\n"),
        ("src/schema_generated.py", b"safe generated module"),
        ("src/auto-generated.ts", b"safe generated module"),
    ),
)
def test_standard_generated_suffixes_and_headers_are_rejected(
    tmp_path: Path, relative_path: str, content: bytes
) -> None:
    _write(tmp_path, relative_path, content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "generated_file"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    (
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "Gemfile.lock",
        "composer.lock",
        "uv.lock",
        "bun.lockb",
        "go.sum",
    ),
)
def test_common_lock_files_are_rejected(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "lock_file"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    (
        ".env",
        ".envrc",
        "config/.env.local",
        "config/.env.test.local",
        "config/.env-example",
        "keys/id_rsa",
        "keys/private-key",
        "keys/private_key",
        "keys/privatekey",
        "keys/ssh_host_ed25519_key",
        "config/auth.json",
        "sessions/session.json",
        "exports/token.json",
        "config/credentials.json",
        "exports/config-export.json",
        "exports/configuration_export.yaml",
        "exports/service-account.json",
        "sessions/auth-token.txt",
        "keys/deploy.ppk",
        "exports/serviceAccount.json",
        "config/refresh-token.txt",
        "config/client-auth.json",
        "sessions/browser-session.sqlite",
    ),
)
def test_secret_path_is_rejected(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "secret_path"),
    ]


@pytest.mark.parametrize(
    ("relative_path", "expected_rule"),
    (
        ("Private/corpus.md", "private_corpus_path"),
        ("Clients/acme/brief.md", "private_corpus_path"),
        ("client-corpora/acme/brief.md", "private_corpus_path"),
        ("Personal/Finance/budget.md", "sensitive_personal_path"),
        ("Personal/Tax/return.md", "sensitive_personal_path"),
        ("Personal/Medical/history.md", "sensitive_personal_path"),
        ("Personal/Household/inventory.md", "sensitive_personal_path"),
        ("Personal/Identity/passport.md", "sensitive_personal_path"),
        ("Personal/Legal/will.md", "sensitive_personal_path"),
        ("personal-finance/accounts.md", "sensitive_personal_path"),
        ("raw-chat/export.json", "raw_transcript_path"),
        ("raw-transcripts/bulk.json", "raw_transcript_path"),
        ("Raw/Transcripts/archive.json", "raw_transcript_path"),
        ("Projects/Clients/Acme/brief.md", "private_corpus_path"),
        ("Projects/Client/Acme/brief.md", "private_corpus_path"),
        ("Work/Private/Acme/notes.md", "private_corpus_path"),
        ("Archive/Finance/budget.md", "sensitive_personal_path"),
        ("Records/Taxes/return.md", "sensitive_personal_path"),
        ("Medical/history.md", "sensitive_personal_path"),
        ("Household/inventory.md", "sensitive_personal_path"),
        ("Identity/passport.md", "sensitive_personal_path"),
        ("Legal/will.md", "sensitive_personal_path"),
        ("Personal/Archive/Finance/accounts.md", "sensitive_personal_path"),
        ("exports/chat-transcript.json", "raw_transcript_path"),
        ("exports/transcript.json", "raw_transcript_path"),
        ("ChatGPT/conversations.json", "raw_transcript_path"),
    ),
)
def test_denied_corpus_path_categories_are_rejected(
    tmp_path: Path, relative_path: str, expected_rule: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, expected_rule),
    ]
    assert vars(result.rejected[0]) == {"path": relative_path, "rule": expected_rule}


@pytest.mark.parametrize(
    "relative_path",
    (
        "docs/authentication.json",
        "docs/session-guide.json",
        "docs/tokenization.json",
        "docs/generated-code.md",
        "docs/package-locking-guide.md",
        "docs/legalese.md",
        "docs/taxonomy.md",
        "docs/transcript-guide.md",
        "src/client/request.py",
        "src/identity/model.py",
        "src/transcript.py",
        "src/conversations.ts",
    ),
)
def test_safe_path_lookalikes_are_not_overblocked(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


def test_private_key_header_is_rejected_without_content_disclosure(tmp_path: Path) -> None:
    marker = "-----BEGIN FAKE PRIVATE KEY-----"
    _write(tmp_path, "notes.md", f"heading\n{marker}\n".encode())

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("notes.md", "private_key_header"),
    ]
    assert marker not in repr(result)
    assert vars(result.rejected[0]) == {
        "path": "notes.md",
        "rule": "private_key_header",
    }


@pytest.mark.parametrize(
    ("relative_path", "content"),
    (
        ("keys/backup.asc", b"-----BEGIN PGP PRIVATE KEY BLOCK-----\n"),
        ("keys/putty-export.txt", b"PuTTY-User-Key-File-3: ssh-rsa\n"),
    ),
)
def test_pgp_and_putty_private_key_signatures_are_rejected(
    tmp_path: Path, relative_path: str, content: bytes
) -> None:
    _write(tmp_path, relative_path, content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "private_key_header"),
    ]


def test_service_account_json_signature_is_rejected_without_content_disclosure(
    tmp_path: Path,
) -> None:
    signature = '"type": "service_account"'
    _write(tmp_path, "exports/gcp-sa.json", f"{{{signature}}}\n".encode())

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("exports/gcp-sa.json", "service_account_json"),
    ]
    assert signature not in repr(result)


def test_obvious_token_assignment_is_rejected_without_value_disclosure(tmp_path: Path) -> None:
    fake_value = "FAKE_TOKEN_VALUE_123456"
    _write(tmp_path, "settings.txt", f'api_token = "{fake_value}"\n'.encode())

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert [(item.path, item.rule) for item in result.rejected] == [
        ("settings.txt", "token_assignment"),
    ]
    assert fake_value not in repr(result)


def test_detector_error_rejects_only_that_file_and_discovery_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "broken.md", b"detector trigger")
    _write(tmp_path, "safe.md", b"ordinary content")
    original_detector = source_discovery._secret_content_rule

    def unreliable_detector(content: bytes) -> str | None:
        if content == b"detector trigger":
            raise RuntimeError("detector unavailable")
        return original_detector(content)

    monkeypatch.setattr(source_discovery, "_secret_content_rule", unreliable_detector)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ("safe.md",)
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("broken.md", "detection_error"),
    ]
