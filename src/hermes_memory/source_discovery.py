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

COMMON_LOCK_FILE_NAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
        "bun.lock",
        "bun.lockb",
        "go.sum",
    }
)

PERSONAL_PATH_CATEGORIES = frozenset(
    {"finance", "tax", "taxes", "medical", "household", "identity", "legal"}
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
        normalized = pattern.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
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


def _filename_tokens(name: str) -> tuple[str, ...]:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name)
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", camel_split)
    return tuple(token.lower() for token in re.split(r"[^A-Za-z0-9]+", camel_split) if token)


def _is_secret_path(relative_path: str) -> bool:
    raw_name = Path(relative_path).name
    name = raw_name.lower()
    if name.startswith(".env"):
        return True
    if name in {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "private-key",
        "private_key",
        "privatekey",
    }:
        return True
    if re.fullmatch(r"ssh_host_(?:rsa|dsa|ecdsa|ed25519)_key", name):
        return True
    if Path(name).suffix in {".key", ".pem", ".p12", ".pfx", ".ppk"}:
        return True
    if name in {"auth.json", "session.json", "token.json", "tokens.json"}:
        return True

    stem_tokens = _filename_tokens(Path(raw_name).stem)
    if any(token in {"secret", "secrets"} for token in stem_tokens) or (
        len(stem_tokens) == 2 and stem_tokens[0] == "api" and stem_tokens[1] in {"key", "keys"}
    ):
        return True

    tokens = _filename_tokens(raw_name)
    if any(token in {"credential", "credentials"} for token in tokens):
        return True
    sensitive_pairs = {
        ("service", "account"),
        ("auth", "token"),
        ("access", "token"),
        ("refresh", "token"),
        ("client", "auth"),
        ("client", "secret"),
        ("browser", "session"),
        ("token", "dump"),
        ("config", "export"),
        ("configuration", "export"),
    }
    return any(pair in sensitive_pairs for pair in zip(tokens, tokens[1:]))


def _path_category_rule(relative_path: str) -> str | None:
    path = Path(relative_path)
    name = path.name.lower()
    directories = tuple(part.lower().replace("_", "-") for part in path.parts[:-1])

    generated_stem = Path(name).stem
    if (
        "generated" in directories
        or name.startswith("generated.")
        or ".generated." in name
        or ".gen." in name
        or re.search(r"\.g\.[^.]+$", name)
        or ".designer." in name
        or re.search(r"_pb2(?:_grpc)?\.[^.]+$", name)
        or generated_stem.endswith(("_generated", "-generated"))
        or name.endswith(".pb.go")
    ):
        return "generated_file"
    if name.endswith(".lock") or name in COMMON_LOCK_FILE_NAMES:
        return "lock_file"
    checksum_suffixes = {".md5", ".sha1", ".sha224", ".sha256", ".sha384", ".sha512", ".sum"}
    if Path(name).suffix in checksum_suffixes or re.fullmatch(
        r"(?:.*\.)?(?:checksums?|(?:md5|sha(?:1|224|256|384|512))sums)(?:\.txt)?",
        name,
    ):
        return "checksum_file"
    if _is_secret_path(relative_path):
        return "secret_path"

    top = directories[0] if directories else ""
    source_tree_roots = {"src", "source", "lib", "app", "packages", "tests", "test"}
    private_categories = {
        "private",
        "client-corpus",
        "client-corpora",
        "private-corpus",
        "private-corpora",
    }
    client_directories = {"client", "clients"}
    legitimate_client_source = top in source_tree_roots and any(
        part in client_directories for part in directories
    )
    if any(part in private_categories for part in directories) or (
        any(part in client_directories for part in directories) and not legitimate_client_source
    ):
        return "private_corpus_path"

    sensitive_segments = PERSONAL_PATH_CATEGORIES.intersection(directories)
    legitimate_identity_source = sensitive_segments == {"identity"} and top in source_tree_roots
    if (sensitive_segments and not legitimate_identity_source) or any(
        part.startswith("personal-") and part.removeprefix("personal-") in PERSONAL_PATH_CATEGORIES
        for part in directories
    ):
        return "sensitive_personal_path"

    raw_categories = {
        "raw-chat",
        "raw-chats",
        "raw-transcript",
        "raw-transcripts",
        "raw-conversation",
        "raw-conversations",
    }
    transcript_tokens = _filename_tokens(path.stem)
    transcript_names = {"transcript", "transcripts", "conversation", "conversations"}
    chat_names = {"chat", "chats"}
    raw_record_names = transcript_names | chat_names
    raw_export_suffixes = {".json", ".jsonl", ".ndjson", ".csv", ".txt", ".sqlite", ".db"}
    single_record_export = len(transcript_tokens) == 1 and transcript_tokens[0] in raw_record_names
    chat_transcript_export = bool(chat_names.intersection(transcript_tokens)) and bool(
        {"transcript", "transcripts"}.intersection(transcript_tokens)
    )
    named_record_export = "export" in transcript_tokens and bool(
        raw_record_names.intersection(transcript_tokens)
    )
    directory_tokens = {token for directory in directories for token in _filename_tokens(directory)}
    export_history_directories = {
        "export",
        "exports",
        "history",
        "histories",
        "archive",
        "archives",
    }
    history_record_export = (
        bool({"history", "histories"}.intersection(transcript_tokens))
        and bool(raw_record_names.intersection(transcript_tokens))
        and bool(export_history_directories.intersection(directory_tokens))
    )
    raw_export_name = path.suffix.lower() in raw_export_suffixes and (
        single_record_export
        or chat_transcript_export
        or named_record_export
        or history_record_export
    )
    if (
        any(part in raw_categories for part in directories)
        or any(
            left == "raw" and right in raw_record_names
            for left, right in zip(directories, directories[1:])
        )
        or raw_export_name
    ):
        return "raw_transcript_path"
    return None


def _static_path_rule(relative_path: str, policy: SourcePolicy) -> str | None:
    if policy.include_patterns and not _matches(relative_path, policy.include_patterns):
        return "not_in_allowlist"
    if _matches(relative_path, policy.exclude_patterns):
        return "exclude_pattern"
    if relative_path == ".git" or any(
        part.lower() in DEFAULT_EXCLUDED_DIRECTORIES for part in Path(relative_path).parts[:-1]
    ):
        return "default_excluded_directory"
    return _path_category_rule(relative_path)


def _looks_binary(content: bytes) -> bool:
    if b"\x00" in content:
        return True
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _secret_content_rule(content: bytes) -> str | None:
    private_key_header = re.compile(
        rb"(?:-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|^PuTTY-User-Key-File-[23]:)",
        re.IGNORECASE | re.MULTILINE,
    )
    if private_key_header.search(content):
        return "private_key_header"
    if re.search(rb'"type"\s*:\s*"service_account"', content):
        return "service_account_json"
    token_assignment = re.compile(
        rb"^[ \t]*(?:export[ \t]+)?[\"']?"
        rb"(?:api[_-]?key|(?:[A-Za-z0-9]+[_-]+)*(?:secret|token|password|private[_-]+key)|"
        rb"(?:[A-Za-z0-9]+[_-]+)*secret(?:[_-]+access)?[_-]+key|"
        rb"private[_-]+key)"
        rb"[\"']?[ \t]*[:=][ \t]*"
        rb"(?:(?P<quote>[\"'])[A-Za-z0-9_./+\-=]{12,}(?P=quote)|"
        rb"[A-Za-z0-9_./+\-=]{12,})[ \t]*,?[ \t]*(?:[#;][^\r\n]*)?$",
        re.IGNORECASE | re.MULTILINE,
    )
    if token_assignment.search(content):
        return "token_assignment"
    return None


def _generated_content_rule(content: bytes) -> str | None:
    header = content[:8192]
    if re.search(
        rb"(?:@generated\b|<auto-generated\s*/?>|"
        rb"(?:\bcode generated\b|\b(?:this )?file (?:is |was )?"
        rb"(?:automatically )?generated\b|\bgenerated file\b)"
        rb"[^\r\n]*\bdo not edit\b|"
        rb"\bdo not edit\b[^\r\n]*\bgenerated by\b)",
        header,
        re.IGNORECASE,
    ):
        return "generated_file"
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

    candidates = (item for item in root_path.rglob("*") if item.is_file() or item.is_symlink())
    for path in sorted(candidates, key=str):
        relative_path = path.relative_to(root_path).as_posix()
        if is_repository and relative_path.startswith(".git/"):
            continue
        alias_path_rule = _static_path_rule(relative_path, active_policy)
        if alias_path_rule:
            rejected.append(RejectedSource(relative_path, alias_path_rule))
            continue
        target_relative_path = relative_path
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError:
                rejected.append(RejectedSource(relative_path, "detection_error"))
                continue
            if not target.is_relative_to(root_path):
                rejected.append(RejectedSource(relative_path, "symlink_escape"))
                continue
            target_relative_path = target.relative_to(root_path).as_posix()
            target_path_rule = _static_path_rule(target_relative_path, active_policy)
            if target_path_rule:
                rejected.append(RejectedSource(relative_path, target_path_rule))
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
            if path.is_symlink() and git_available:
                target_git_status = _git_ignore_status(root_path, target_relative_path)
                if target_git_status == "ignored":
                    rejected.append(RejectedSource(relative_path, "git_ignored"))
                    continue
                if target_git_status == "unavailable":
                    warnings.append("git check-ignore unavailable; static exclusions only")
                    git_available = False
                elif target_git_status == "error":
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
            if secret_rule is None:
                secret_rule = _generated_content_rule(content)
        except Exception:
            rejected.append(RejectedSource(relative_path, "detection_error"))
            continue
        if secret_rule:
            rejected.append(RejectedSource(relative_path, secret_rule))
            continue
        accepted.append(path)

    return DiscoveryResult(root_path, tuple(accepted), tuple(rejected), tuple(warnings))
