"""Safe, deterministic source-file discovery for document ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
import re
import subprocess
from typing import Iterable, Literal


DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".obsidian",
        ".trash",
        "node_modules",
        "vendor",
        ".venv",
        "build",
        "dist",
        "cache",
        ".cache",
        "__pycache__",
    }
)


@dataclass(frozen=True)
class SourcePolicy:
    """Selection policy applied before source files may be ingested."""

    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    allow_all_approved: bool = False
    max_file_size_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_patterns", tuple(self.include_patterns))
        object.__setattr__(self, "exclude_patterns", tuple(self.exclude_patterns))
        if self.max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")


@dataclass(frozen=True)
class RejectedSource:
    """A non-sensitive rejection record containing no source content."""

    path: str
    rule: str


@dataclass(frozen=True)
class DiscoveryResult:
    """Accepted paths and safe rejection metadata in deterministic order."""

    root: Path
    sources: tuple[Path, ...]
    rejected: tuple[RejectedSource, ...]
    warnings: tuple[str, ...] = ()

    @property
    def relative_paths(self) -> tuple[str, ...]:
        return tuple(path.relative_to(self.root).as_posix() for path in self.sources)


def _matches(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.replace("\\", "/").lstrip("./")
        if fnmatchcase(path, normalized):
            return True
        if normalized.startswith("**/") and fnmatchcase(path, normalized[3:]):
            return True
    return False


def _git_ignore_status(
    root: Path, relative_path: str
) -> Literal["ignored", "not_ignored", "unavailable", "error"]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--", relative_path],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        return "unavailable"
    if completed.returncode == 0:
        return "ignored"
    if completed.returncode == 1:
        return "not_ignored"
    return "error"


def _is_secret_path(relative_path: str) -> bool:
    name = Path(relative_path).name.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }:
        return True
    if Path(name).suffix in {".key", ".pem", ".p12", ".pfx"}:
        return True
    markers = (
        "credential",
        "service-account",
        "service_account",
        "auth-token",
        "auth_token",
        "access-token",
        "access_token",
        "token-dump",
        "token_dump",
    )
    return any(marker in name for marker in markers)


def _looks_binary(content: bytes) -> bool:
    if b"\x00" in content:
        return True
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _secret_content_rule(content: bytes) -> str | None:
    if re.search(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", content, re.IGNORECASE):
        return "private_key_header"
    token_assignment = re.compile(
        rb"^[ \t]*(?:export[ \t]+)?[\"']?"
        rb"(?:api[_-]?key|(?:api|access|auth|refresh)?[_-]?token|client[_-]?secret|password)"
        rb"[\"']?[ \t]*[:=][ \t]*[\"']?[A-Za-z0-9_./+\-=]{12,}",
        re.IGNORECASE | re.MULTILINE,
    )
    if token_assignment.search(content):
        return "token_assignment"
    return None


def discover_sources(
    root: str | Path,
    policy: SourcePolicy | None = None,
    *,
    for_apply: bool = False,
    repository: bool | None = None,
) -> DiscoveryResult:
    """Discover files below *root* according to an allowlist-first policy."""

    root_path = Path(root).expanduser().resolve()
    active_policy = policy or SourcePolicy()
    if for_apply and not active_policy.include_patterns and not active_policy.allow_all_approved:
        raise ValueError(
            "an include pattern is required for apply unless allow_all_approved is set"
        )
    git_metadata = root_path / ".git"
    auto_repository = git_metadata.is_file() or (git_metadata / "HEAD").is_file()
    is_repository = repository if repository is not None else auto_repository
    git_available = is_repository
    accepted: list[Path] = []
    rejected: list[RejectedSource] = []
    warnings: list[str] = []

    candidates = (
        item for item in root_path.rglob("*") if item.is_file() or item.is_symlink()
    )
    for path in sorted(candidates, key=str):
        relative_path = path.relative_to(root_path).as_posix()
        if is_repository and relative_path.startswith(".git/"):
            continue
        if active_policy.include_patterns and not _matches(
            relative_path, active_policy.include_patterns
        ):
            rejected.append(RejectedSource(relative_path, "not_in_allowlist"))
            continue
        if _matches(relative_path, active_policy.exclude_patterns):
            rejected.append(RejectedSource(relative_path, "exclude_pattern"))
            continue
        if any(part in DEFAULT_EXCLUDED_DIRECTORIES for part in Path(relative_path).parts[:-1]):
            rejected.append(RejectedSource(relative_path, "default_excluded_directory"))
            continue
        if _is_secret_path(relative_path):
            rejected.append(RejectedSource(relative_path, "secret_path"))
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError:
                rejected.append(RejectedSource(relative_path, "detection_error"))
                continue
            if not target.is_relative_to(root_path):
                rejected.append(RejectedSource(relative_path, "symlink_escape"))
                continue
        if git_available:
            git_status = _git_ignore_status(root_path, relative_path)
            if git_status == "ignored":
                rejected.append(RejectedSource(relative_path, "git_ignored"))
                continue
            if git_status == "unavailable":
                warnings.append("git check-ignore unavailable; static exclusions only")
                git_available = False
            elif git_status == "error":
                rejected.append(RejectedSource(relative_path, "detection_error"))
                continue
        try:
            if path.stat().st_size > active_policy.max_file_size_bytes:
                rejected.append(RejectedSource(relative_path, "max_file_size"))
                continue
            content = path.read_bytes()
        except OSError:
            rejected.append(RejectedSource(relative_path, "detection_error"))
            continue
        try:
            if _looks_binary(content):
                rejected.append(RejectedSource(relative_path, "binary_content"))
                continue
            secret_rule = _secret_content_rule(content)
        except Exception:
            rejected.append(RejectedSource(relative_path, "detection_error"))
            continue
        if secret_rule:
            rejected.append(RejectedSource(relative_path, secret_rule))
            continue
        accepted.append(path)

    return DiscoveryResult(root_path, tuple(accepted), tuple(rejected), tuple(warnings))
