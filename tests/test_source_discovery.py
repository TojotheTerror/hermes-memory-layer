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


def test_pattern_normalization_removes_only_one_optional_dot_slash() -> None:
    assert source_discovery._matches(".hidden/notes.md", ("./.hidden/**",))
    assert source_discovery._matches(".hidden/notes.md", (".hidden/**",))
    assert not source_discovery._matches("hidden/notes.md", (".hidden/**",))
    assert not source_discovery._matches("hidden/notes.md", ("./.hidden/**",))
    assert not source_discovery._matches("notes.md", ("../notes.md",))


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
    assert result.warnings == ("git check-ignore unavailable; static exclusions only",)


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
def test_worktree_git_metadata_file_is_rejected_without_disclosure(
    tmp_path: Path, policy: SourcePolicy, for_apply: bool
) -> None:
    repository_root = tmp_path / "repository"
    worktree_root = tmp_path / "linked-worktree"
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    _write(repository_root, "safe.md")
    subprocess.run(["git", "-C", str(repository_root), "add", "safe.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository_root), "worktree", "add", "-qb", "linked", str(worktree_root)],
        check=True,
    )

    result = discover_sources(worktree_root, policy, for_apply=for_apply)

    assert result.relative_paths == ("safe.md",)
    assert [(item.path, item.rule) for item in result.rejected] == [
        (".git", "default_excluded_directory"),
    ]
    assert vars(result.rejected[0]) == {
        "path": ".git",
        "rule": "default_excluded_directory",
    }


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


def test_internal_symlink_target_must_match_include_allowlist(tmp_path: Path) -> None:
    target = _write(tmp_path, "unapproved/notes.md")
    (tmp_path / "approved").mkdir()
    (tmp_path / "approved/alias.md").symlink_to(target)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("approved/**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("approved/alias.md", "not_in_allowlist"),
        ("unapproved/notes.md", "not_in_allowlist"),
    ]


def test_internal_symlink_target_must_not_match_explicit_exclude(tmp_path: Path) -> None:
    target = _write(tmp_path, "blocked/notes.md")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/alias.md").symlink_to(target)

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",), exclude_patterns=("blocked/**",)),
    )

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("blocked/notes.md", "exclude_pattern"),
        ("docs/alias.md", "exclude_pattern"),
    ]


@pytest.mark.parametrize("denied_directory", ("node_modules", ".git"))
def test_internal_symlink_target_must_not_be_in_default_denied_directory(
    tmp_path: Path, denied_directory: str
) -> None:
    target_relative_path = f"{denied_directory}/metadata.md"
    target = _write(tmp_path, target_relative_path)
    (tmp_path / "zz-safe").mkdir()
    (tmp_path / "zz-safe/alias.md").symlink_to(target)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (target_relative_path, "default_excluded_directory"),
        ("zz-safe/alias.md", "default_excluded_directory"),
    ]


def test_internal_symlink_target_must_pass_secret_path_classification(tmp_path: Path) -> None:
    target = _write(tmp_path, "config/secrets.json")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/reference.json").symlink_to(target)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("config/secrets.json", "secret_path"),
        ("docs/reference.json", "secret_path"),
    ]
    assert vars(result.rejected[1]) == {"path": "docs/reference.json", "rule": "secret_path"}


def test_internal_symlink_target_must_not_be_git_ignored(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _write(tmp_path, ".gitignore", b"ignored/\n")
    target = _write(tmp_path, "ignored/notes.md")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/alias.md").symlink_to(target)

    result = discover_sources(
        tmp_path,
        SourcePolicy(include_patterns=("**",)),
        repository=True,
    )

    assert result.relative_paths == (".gitignore",)
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("docs/alias.md", "git_ignored"),
        ("ignored/notes.md", "git_ignored"),
    ]


def test_internal_symlink_is_allowed_when_alias_and_target_are_safe(tmp_path: Path) -> None:
    target = _write(tmp_path, "docs/original.md")
    (tmp_path / "docs/alias.md").symlink_to(target)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("docs/**",)))

    assert result.relative_paths == ("docs/alias.md", "docs/original.md")
    assert result.rejected == ()


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
    "relative_path",
    (
        "src/client.gen.ts",
        "proto/messages_pb2.py",
        "proto/messages_pb2_grpc.py",
        "ui/MainForm.Designer.cs",
    ),
)
def test_standard_generated_filename_variants_are_rejected(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "generated_file"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    ("src/designer.cs", "src/pb2_helpers.py", "src/generator.ts"),
)
def test_generated_filename_lookalikes_remain_allowed(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


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
    "marker",
    (
        b"// This file is automatically generated. DO NOT EDIT.\n",
        b"# Generated file; do not edit\n",
        b"// <auto-generated>\n",
        b"// <auto-generated />\n",
    ),
)
def test_standard_generated_content_markers_are_rejected(tmp_path: Path, marker: bytes) -> None:
    _write(tmp_path, "src/output.txt", marker)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert [(item.path, item.rule) for item in result.rejected] == [
        ("src/output.txt", "generated_file"),
    ]


def test_generated_content_prose_lookalike_remains_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "docs/editing.md", b"The generated file can be edited safely.\n")

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ("docs/editing.md",)
    assert result.rejected == ()


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
@pytest.mark.parametrize(
    ("relative_path", "content"),
    (
        ("src/Model.g.cs", b"safe generated output placeholder\n"),
        ("src/output.cs", b"// DO NOT EDIT. Generated by synthetic tool.\n"),
    ),
)
def test_generated_aliases_are_rejected_under_broad_approval(
    tmp_path: Path,
    policy: SourcePolicy,
    for_apply: bool,
    relative_path: str,
    content: bytes,
) -> None:
    _write(tmp_path, relative_path, content)

    result = discover_sources(tmp_path, policy, for_apply=for_apply)

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "generated_file"),
    ]
    assert vars(result.rejected[0]) == {"path": relative_path, "rule": "generated_file"}


@pytest.mark.parametrize(
    ("relative_path", "content"),
    (
        ("src/Widget.generation.cs", b"ordinary source\n"),
        ("src/g.cs", b"ordinary source\n"),
        (
            "docs/generation.md",
            b"Do not edit examples that are generated during the tutorial.\n",
        ),
    ),
)
def test_generated_alias_lookalikes_remain_allowed(
    tmp_path: Path, relative_path: str, content: bytes
) -> None:
    _write(tmp_path, relative_path, content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


def test_static_denied_directories_are_case_insensitive_under_allow_all(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "NODE_MODULES/pkg/index.js")
    _write(tmp_path, "Vendor/lib/source.py")
    _write(tmp_path, "approved.md")

    result = discover_sources(
        tmp_path,
        SourcePolicy(allow_all_approved=True),
        for_apply=True,
    )

    assert result.relative_paths == ("approved.md",)
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("NODE_MODULES/pkg/index.js", "default_excluded_directory"),
        ("Vendor/lib/source.py", "default_excluded_directory"),
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
        "checksums.txt",
        "artifacts/release.sha256",
        "artifacts/release.sha512",
        "artifacts/release.sum",
        "SHA256SUMS",
    ),
)
def test_checksum_artifacts_are_rejected(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "checksum_file"),
    ]


def test_checksum_source_module_lookalike_remains_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "src/checksum.py")

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ("src/checksum.py",)
    assert result.rejected == ()


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
@pytest.mark.parametrize(
    "relative_path",
    ("CHECKSUMS", "artifacts/release.SHA256SUMS", "artifacts/package.md5sums"),
)
def test_checksum_manifest_aliases_are_rejected_under_broad_approval(
    tmp_path: Path,
    policy: SourcePolicy,
    for_apply: bool,
    relative_path: str,
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, policy, for_apply=for_apply)

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "checksum_file"),
    ]
    assert vars(result.rejected[0]) == {"path": relative_path, "rule": "checksum_file"}


@pytest.mark.parametrize(
    "relative_path",
    ("src/checksums.py", "src/sha256sums.py", "src/release_sha256sums.py"),
)
def test_checksum_manifest_source_identifier_lookalikes_remain_allowed(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


@pytest.mark.parametrize(
    "relative_path",
    (
        "config/secrets.json",
        "config/secret.yaml",
        "config/api-keys.json",
        "settings/api_key.toml",
    ),
)
def test_obvious_secret_config_paths_are_rejected(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "secret_path"),
    ]


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
@pytest.mark.parametrize(
    "relative_path",
    ("config/prod-secrets.yaml", "settings/dev_secret_backup.json"),
)
def test_secret_stem_tokens_are_rejected_under_broad_approval(
    tmp_path: Path,
    policy: SourcePolicy,
    for_apply: bool,
    relative_path: str,
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, policy, for_apply=for_apply)

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "secret_path"),
    ]
    assert vars(result.rejected[0]) == {"path": relative_path, "rule": "secret_path"}


@pytest.mark.parametrize(
    "relative_path",
    ("config/secretary.yaml", "config/topsecret.yaml", "docs/secretariat.md"),
)
def test_secret_substrings_in_stems_remain_allowed(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


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
    "relative_path",
    ("src/client/http.py", "src/clients/http.py", "lib/clients/session.ts"),
)
def test_client_source_tree_layouts_remain_allowed(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


@pytest.mark.parametrize(
    "relative_path",
    ("Projects/Clients/Acme/brief.md", "Work/Private/Acme/notes.md"),
)
def test_plan_denied_private_corpora_survive_broad_approval(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(
        tmp_path,
        SourcePolicy(allow_all_approved=True),
        for_apply=True,
    )

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "private_corpus_path"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    (
        "chat.json",
        "exports/chats.json",
        "exports/chat-export.json",
        "exports/conversation-export.json",
        "exports/conversations_export.jsonl",
        "raw-conversations/archive.ndjson",
        "Raw/Conversations/archive.json",
        "Raw/Conversation/archive.json",
    ),
)
def test_raw_chat_and_conversation_export_aliases_are_rejected(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(
        tmp_path,
        SourcePolicy(allow_all_approved=True),
        for_apply=True,
    )

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "raw_transcript_path"),
    ]


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
@pytest.mark.parametrize(
    "relative_path",
    (
        "exports/chat-history.json",
        "exports/conversation-history.json",
        "archives/chats_history.csv",
    ),
)
def test_history_export_aliases_are_rejected_under_broad_approval(
    tmp_path: Path,
    policy: SourcePolicy,
    for_apply: bool,
    relative_path: str,
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, policy, for_apply=for_apply)

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        (relative_path, "raw_transcript_path"),
    ]
    assert vars(result.rejected[0]) == {
        "path": relative_path,
        "rule": "raw_transcript_path",
    }


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/chat.py",
        "src/conversation_history.py",
        "docs/conversation-history.md",
        "src/conversation_history.json",
    ),
)
def test_history_export_source_and_documentation_lookalikes_remain_allowed(
    tmp_path: Path, relative_path: str
) -> None:
    _write(tmp_path, relative_path)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == (relative_path,)
    assert result.rejected == ()


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
        "src/chat.py",
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


@pytest.mark.parametrize(
    "content",
    (
        b"AWS_SECRET_ACCESS_KEY=FAKE_AWS_SECRET_12345\n",
        b"github_token = 'FAKE_GITHUB_TOKEN_123'\n",
        b'PRIVATE_KEY: "FAKE_PRIVATE_KEY_12345"\n',
        b'database_password = "FAKE_DATABASE_PASSWORD"\n',
    ),
)
def test_obvious_scoped_secret_assignments_are_rejected(tmp_path: Path, content: bytes) -> None:
    _write(tmp_path, "settings.txt", content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert [(item.path, item.rule) for item in result.rejected] == [
        ("settings.txt", "token_assignment"),
    ]


@pytest.mark.parametrize(
    ("policy", "for_apply"),
    (
        (SourcePolicy(include_patterns=("**",)), False),
        (SourcePolicy(allow_all_approved=True), True),
    ),
)
@pytest.mark.parametrize(
    "content",
    (
        b"AZURE_CLIENT_SECRET=FAKE_AZURE_SECRET_12345\n",
        b"VAULT_SECRET=FAKE_VAULT_SECRET_12345\n",
        b"AZURE_TOKEN=FAKE_AZURE_TOKEN_123456\n",
        b"DATABASE_PASSWORD=FAKE_DATABASE_PASSWORD\n",
        b"GCP_PRIVATE_KEY=FAKE_GCP_PRIVATE_KEY_12345\n",
    ),
)
def test_provider_prefixed_secret_assignments_survive_no_broad_approval(
    tmp_path: Path,
    policy: SourcePolicy,
    for_apply: bool,
    content: bytes,
) -> None:
    _write(tmp_path, "settings.txt", content)

    result = discover_sources(tmp_path, policy, for_apply=for_apply)

    assert result.relative_paths == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("settings.txt", "token_assignment"),
    ]
    assert vars(result.rejected[0]) == {
        "path": "settings.txt",
        "rule": "token_assignment",
    }


@pytest.mark.parametrize(
    "content",
    (
        b"DATABASE_PASSWORD is documented in the deployment guide.\n",
        b"github_token = resolve_from_environment()\n",
        b'private_key_name = "default-signing-key"\n',
    ),
)
def test_secret_assignment_lookalikes_remain_allowed(tmp_path: Path, content: bytes) -> None:
    _write(tmp_path, "notes.txt", content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ("notes.txt",)
    assert result.rejected == ()


@pytest.mark.parametrize(
    "content",
    (
        b"AZURE_CLIENT_SECRET is documented in the deployment guide.\n",
        b"AZURE_CLIENT_SECRET = resolve_from_environment()\n",
        b'const AZURE_CLIENT_SECRET = "FAKE_AZURE_SECRET_12345";\n',
    ),
)
def test_provider_secret_assignment_prose_and_source_lookalikes_remain_allowed(
    tmp_path: Path, content: bytes
) -> None:
    _write(tmp_path, "notes.txt", content)

    result = discover_sources(tmp_path, SourcePolicy(include_patterns=("**",)))

    assert result.relative_paths == ("notes.txt",)
    assert result.rejected == ()


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
