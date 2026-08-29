from pathlib import Path
import subprocess
from typing import Any, cast

import pytest

import hermes_memory.source_discovery as source_discovery
from hermes_memory.source_discovery import (
    RepositoryDirtyError,
    RepositoryDiscoveryError,
    SourcePolicy,
    discover_repository,
    normalize_github_remote,
)


def _git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> str:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "README.md").write_text("# Synthetic repository\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "synthetic fixture",
    )
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.mark.parametrize(
    ("remote", "expected"),
    (
        ("git@github.com:owner/repo.git", "https://github.com/owner/repo"),
        ("ssh://git@github.com/owner/repo.git", "https://github.com/owner/repo"),
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
        ("https://github.com/owner/repo", "https://github.com/owner/repo"),
    ),
)
def test_github_remote_normalization_is_canonical_and_credential_free(
    remote: str, expected: str
) -> None:
    assert normalize_github_remote(remote) == expected


def test_repository_discovery_resolves_head_commit_and_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    commit = _init_repo(repo)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
        ref="HEAD",
    )

    assert result.state.revision == commit
    assert result.state.ref == "HEAD"
    assert result.state.branch == "main"


@pytest.mark.parametrize("dirty_kind", ("tracked", "untracked"))
def test_apply_rejects_dirty_tracked_and_untracked_worktrees(
    tmp_path: Path, dirty_kind: str
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    if dirty_kind == "tracked":
        (repo / "README.md").write_text("changed\n", encoding="utf-8")
    else:
        (repo / "untracked.py").write_text("print('local')\n", encoding="utf-8")

    with pytest.raises(RepositoryDirtyError, match="clean worktree"):
        discover_repository(
            repo,
            SourcePolicy(include_patterns=("**",)),
            for_apply=True,
        )


def test_clean_github_sources_have_commit_pinned_quoted_citations_and_metadata(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    relative_path = "src/hello # café.py"
    source_path = repo / relative_path
    source_path.parent.mkdir()
    source_path.write_text("first\nsecond\nthird\n", encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "add quoted path",
    )
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "remote", "add", "origin", "git@github.com:owner/repo.git")

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=(relative_path,)),
    )

    assert len(result.sources) == 1
    source = result.sources[0]
    expected_uri = f"https://github.com/owner/repo/blob/{commit}/src/hello%20%23%20caf%C3%A9.py"
    assert source.source_uri == expected_uri
    assert source.citation(2, 3) == f"{expected_uri}#L2-L3"
    assert source.content_kind == "code"
    assert source.revision == commit
    assert dict(source.metadata) == {
        "language": "python",
        "revision": commit,
        "branch": "main",
        "ref": "HEAD",
        "remote_url": "https://github.com/owner/repo",
        "relative_path": relative_path,
    }


def test_dirty_override_uses_truthful_local_citation_and_revision_marker(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository with spaces"
    commit = _init_repo(repo)
    _git(repo, "remote", "add", "origin", "https://github.com/owner/repo.git")
    (repo / "README.md").write_text("local change\n", encoding="utf-8")

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
        for_apply=True,
        allow_dirty=True,
    )

    source = result.sources[0]
    assert result.state.dirty is True
    assert source.revision == f"{commit}-dirty"
    assert source.source_uri == (repo / "README.md").resolve().as_uri()
    assert source.citation(1, 1).endswith("/README.md#L1-L1")
    assert "github.com" not in source.source_uri
    assert source.metadata["remote_url"] == "https://github.com/owner/repo"


@pytest.mark.parametrize(
    "remote",
    (
        "git@gitlab.com:owner/repo.git",
        "https://gitlab.com/owner/repo.git",
        "https://user:secret@github.com/owner/repo.git",
        "ssh://other@github.com/owner/repo.git",
        "ssh://git:secret@github.com/owner/repo.git",
        "https://github.com:8443/owner/repo.git",
        "https://github.com/owner/../repo.git",
        "https://github.com/owner%2Frepo/other.git",
    ),
)
def test_unrecognized_or_credentialed_remotes_are_not_presented_as_github(
    tmp_path: Path, remote: str
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    _git(repo, "remote", "add", "origin", remote)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
    )

    source = result.sources[0]
    assert normalize_github_remote(remote) is None
    assert source.source_uri == (repo / "README.md").resolve().as_uri()
    assert source.metadata["remote_url"] is None
    assert "secret" not in repr(result)


def test_repository_without_remote_uses_local_file_uri(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    commit = _init_repo(repo)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
    )

    source = result.sources[0]
    assert source.source_uri == (repo / "README.md").resolve().as_uri()
    assert source.revision == commit
    assert source.metadata["remote_url"] is None


def test_detached_head_has_no_branch_but_retains_ref_and_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    commit = _init_repo(repo)
    _git(repo, "checkout", "--detach", "-q", commit)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
        ref=commit,
    )

    assert result.state.branch is None
    assert result.state.ref == commit
    assert result.state.revision == commit
    assert result.sources[0].metadata["branch"] is None
    assert result.sources[0].metadata["ref"] == commit


def test_invalid_ref_fails_without_echoing_the_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    invalid_ref = "--credential=do-not-disclose"

    with pytest.raises(RepositoryDiscoveryError) as raised:
        discover_repository(
            repo,
            SourcePolicy(include_patterns=("README.md",)),
            ref=invalid_ref,
        )

    assert invalid_ref not in str(raised.value)


def test_git_commands_use_argument_lists_for_quoted_paths_and_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repository with spaces"
    _init_repo(repo)
    quoted_ref = "quote'$;name"
    _git(repo, "tag", quoted_ref)
    real_run = subprocess.run
    calls: list[tuple[list[str], object]] = []

    def recording_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((cast(list[str], args[0]), kwargs.get("shell")))
        return real_run(*args, **kwargs)

    monkeypatch.setattr("hermes_memory.source_discovery.subprocess.run", recording_run)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
        ref=quoted_ref,
    )

    assert result.state.ref == quoted_ref
    assert calls
    assert all(shell is not True for _, shell in calls)
    assert any(quoted_ref in command[-1] for command, _ in calls)


def test_repository_symlink_is_rejected_instead_of_citing_alias_content(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    (repo / "alias.md").symlink_to(repo / "README.md")

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("**",)),
    )

    assert [source.relative_path for source in result.sources] == ["README.md"]
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("alias.md", "repository_symlink"),
    ]
    assert vars(result.rejected[-1]) == {
        "path": "alias.md",
        "rule": "repository_symlink",
    }


def test_ref_that_does_not_match_checkout_is_rejected_before_citation(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    old_commit = _init_repo(repo)
    (repo / "README.md").write_text("new checkout content\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "new checkout revision",
    )

    with pytest.raises(RepositoryDiscoveryError, match="match the checked-out commit"):
        discover_repository(
            repo,
            SourcePolicy(include_patterns=("README.md",)),
            ref=old_commit,
        )


def test_repository_discovery_keeps_source_policy_order_and_safe_rejections(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (repo / ".env.production").write_text("FAKE_ONLY=true\n", encoding="utf-8")
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    (repo / "z.py").write_text("print('z')\n", encoding="utf-8")
    (repo / "ignored.py").write_text("not discovered\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", ".env.production", "a.py", "z.py")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "policy fixtures",
    )

    result = discover_repository(repo, SourcePolicy(include_patterns=("**",)))

    assert [source.relative_path for source in result.sources] == [
        ".gitignore",
        "README.md",
        "a.py",
        "z.py",
    ]
    assert [(item.path, item.rule) for item in result.rejected] == [
        (".env.production", "secret_path"),
        ("ignored.py", "git_ignored"),
    ]
    assert all(set(vars(item)) == {"path", "rule"} for item in result.rejected)


def test_repository_root_cannot_traverse_from_a_nested_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    nested = repo / "nested"
    nested.mkdir()

    with pytest.raises(RepositoryDiscoveryError, match="Git top level"):
        discover_repository(nested, SourcePolicy(include_patterns=("**",)))


@pytest.mark.parametrize("failure", ("missing", "nonzero"))
def test_git_command_failures_are_generic_and_do_not_disclose_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)

    def failed_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "missing":
            raise FileNotFoundError("git missing")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=128,
            stdout="https://user:secret@github.com/owner/repo.git",
            stderr="credential=do-not-disclose",
        )

    monkeypatch.setattr(source_discovery.subprocess, "run", failed_run)

    with pytest.raises(RepositoryDiscoveryError) as raised:
        discover_repository(repo, SourcePolicy(include_patterns=("README.md",)))

    assert str(raised.value) == "unable to inspect repository"


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    ((0, 1), (2, 1), (-1, -1)),
)
def test_line_citation_rejects_invalid_ranges(
    tmp_path: Path, start_line: int, end_line: int
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    source = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
    ).sources[0]

    with pytest.raises(ValueError, match="positive and ordered"):
        source.citation(start_line, end_line)


def test_source_read_failure_is_a_non_disclosing_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)

    def unreadable_source(*args: object, **kwargs: object) -> str:
        raise OSError("file body must not escape")

    monkeypatch.setattr(Path, "read_text", unreadable_source)

    result = discover_repository(
        repo,
        SourcePolicy(include_patterns=("README.md",)),
    )

    assert result.sources == ()
    assert [(item.path, item.rule) for item in result.rejected] == [
        ("README.md", "detection_error"),
    ]
    assert "file body" not in repr(result)
