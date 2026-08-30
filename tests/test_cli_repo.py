"""Preview-first ``ingest-repo`` CLI — Click-runner behavior tests.

Every test drives the command through ``CliRunner``. Cloud dependencies are
injected via module-level factory seams on ``cli`` so nothing here constructs a
real Vertex/BigQuery client or makes a network call. The real (pure)
``plan_repository_ingestion`` runs against a temporary Git repository on disk;
Git subprocesses are local reads, never network.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from click.testing import CliRunner

from hermes_memory import cli


PY_BODY = '''"""Module docstring for a genuine source file."""


def substantial_function(value):
    """A real function with distinctive body content used for leak checks."""
    accumulator = 0
    for index in range(value):
        accumulator += index * SECRET_MARKER_UNIQUE_TOKEN
    return accumulator


class SubstantialClass:
    def method(self):
        return SECRET_MARKER_UNIQUE_TOKEN
'''.replace("SECRET_MARKER_UNIQUE_TOKEN", "7")

# A distinctive token that only ever appears inside a file body, never in the
# deterministic body-free accounting output.
BODY_MARKER = "distinctive body content used for leak checks"

DOC_BODY = """# Repository Guide

This is a substantial paragraph of genuine documentation content that clears any
minimum-length gate so discovery accepts it as a real markdown source.

## Details

A second section with more prose so the note produces at least one packed chunk.
"""


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(
    root: Path,
    files: dict[str, str],
    *,
    remote: str | None = "https://github.com/octo/example.git",
) -> str:
    """Create a committed Git repository and return its HEAD commit SHA."""
    root.mkdir(parents=True, exist_ok=True)
    _run_git(root, "init", "-q", "-b", "main")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test User")
    _run_git(root, "config", "commit.gpgsign", "false")
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    if remote is not None:
        _run_git(root, "remote", "add", "origin", remote)
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "-q", "-m", "initial commit")
    return _run_git(root, "rev-parse", "HEAD")


# --- Slice 1: command identity ----------------------------------------------


def test_command_is_registered_as_ingest_repo():
    assert "ingest-repo" in cli.main.commands


# --- Slice 2: default preview (dry-run) prints a deterministic plan table ----


def test_default_run_is_preview_and_prints_plan_table(tmp_path):
    root = tmp_path / "repo"
    sha = _init_repo(root, {"README.md": DOC_BODY, "pkg/module.py": PY_BODY})
    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-repo", "--user", "tojo", "--repo", str(root)])

    assert result.exit_code == 0, result.output
    out = result.output
    # preview framing + honest no-write banner
    assert "preview" in out.lower()
    assert "no writes" in out.lower() or "no bigquery" in out.lower()
    # deterministic accounting is present for both discovered sources
    assert "discovered=2" in out
    assert "README.md" in out
    assert "pkg/module.py" in out
    assert "chunks=" in out and "tokens=" in out and "cost_estimate=" in out
    # commit-pinned citation: the exact HEAD SHA appears in a github blob URL
    assert f"github.com/octo/example/blob/{sha}/pkg/module.py" in out
    # never prints a file body
    assert BODY_MARKER not in out


def test_preview_output_is_deterministic_across_runs(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"z.py": PY_BODY, "a.py": PY_BODY})
    runner = CliRunner()
    args = ["ingest-repo", "--user", "tojo", "--repo", str(root)]
    first = runner.invoke(cli.main, args)
    second = runner.invoke(cli.main, args)
    assert first.exit_code == 0, first.output
    assert first.output == second.output
    # sources are ordered by relative_path (a before z)
    assert first.output.index("a.py") < first.output.index("z.py")


# --- Slice 3: --json emits deterministic, body-free JSON --------------------


def test_json_flag_emits_stable_commit_pinned_body_free_json(tmp_path):
    root = tmp_path / "repo"
    sha = _init_repo(root, {"README.md": DOC_BODY, "pkg/module.py": PY_BODY})
    runner = CliRunner()
    args = [
        "ingest-repo",
        "--user",
        "tojo",
        "--agent",
        "hermes",
        "--repo",
        str(root),
        "--json",
    ]
    result = runner.invoke(cli.main, args)
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["mode"] == "preview"
    assert payload["user_id"] == "tojo"
    assert payload["agent_name"] == "hermes"
    counts = payload["counts"]
    assert counts["discovered"] == 2
    assert counts["skipped"] == 0
    assert counts["chunks"] >= 2
    assert counts["requests"] == counts["chunks"]
    assert counts["tokens"] > 0
    assert payload["cost_estimate"] > 0
    # per-source detail, ordered by relative_path, with commit-pinned citations
    paths = [s["relative_path"] for s in payload["discovered"]]
    assert paths == ["README.md", "pkg/module.py"]
    for s in payload["discovered"]:
        assert set(s) == {
            "relative_path",
            "status",
            "chunk_count",
            "token_count",
            "source_uri",
            "revision",
        }
    module = next(s for s in payload["discovered"] if s["relative_path"] == "pkg/module.py")
    assert module["source_uri"] == f"https://github.com/octo/example/blob/{sha}/pkg/module.py"
    assert module["revision"] == sha
    # never leaks a file body
    assert BODY_MARKER not in result.output


def test_json_output_is_byte_identical_across_runs(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"one.py": PY_BODY, "two.py": PY_BODY})
    runner = CliRunner()
    args = ["ingest-repo", "--user", "tojo", "--repo", str(root), "--json"]
    first = runner.invoke(cli.main, args)
    second = runner.invoke(cli.main, args)
    assert first.exit_code == 0, first.output
    assert first.output == second.output


# --- Slice 4: explicit --ref -------------------------------------------------


def test_explicit_ref_is_pinned_in_citations(tmp_path):
    root = tmp_path / "repo"
    sha = _init_repo(root, {"pkg/module.py": PY_BODY})
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-repo", "--user", "tojo", "--repo", str(root), "--ref", "main", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    module = payload["discovered"][0]
    # --ref main resolves to the same HEAD commit and pins that exact SHA
    assert module["revision"] == sha
    assert module["source_uri"] == f"https://github.com/octo/example/blob/{sha}/pkg/module.py"


# --- Slice 5: include / exclude patterns ------------------------------------


def test_include_pattern_restricts_to_allowlist(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"keep/module.py": PY_BODY, "drop/other.py": PY_BODY})
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-repo", "--user", "tojo", "--repo", str(root), "--include", "keep/*.py", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    paths = [s["relative_path"] for s in payload["discovered"]]
    assert paths == ["keep/module.py"]
    # the excluded-by-allowlist source is reported as rejected, not silently dropped
    rejected_paths = [r["path"] for r in payload["rejected"]]
    assert "drop/other.py" in rejected_paths


def test_exclude_pattern_removes_matches(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"keep/module.py": PY_BODY, "skip/gen.py": PY_BODY})
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-repo", "--user", "tojo", "--repo", str(root), "--exclude", "skip/*", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    paths = [s["relative_path"] for s in payload["discovered"]]
    assert paths == ["keep/module.py"]
    rejected_paths = [r["path"] for r in payload["rejected"]]
    assert "skip/gen.py" in rejected_paths


# --- Slice 6: language filter ------------------------------------------------


def test_language_filter_keeps_only_that_language(tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"pkg/module.py": PY_BODY, "README.md": DOC_BODY})
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-repo", "--user", "tojo", "--repo", str(root), "--language", "python", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    paths = [s["relative_path"] for s in payload["discovered"]]
    assert paths == ["pkg/module.py"]
    # the markdown file is filtered out of the allowlist (reported, not silent)
    rejected_paths = [r["path"] for r in payload["rejected"]]
    assert "README.md" in rejected_paths


# --- Slice 7: dirty-tree rejection on --apply (unless --allow-dirty) ---------


class _FakeEmbeddingResult:
    def __init__(self, values):
        self.values = values


class _FakeEmbeddingClient:
    def __init__(self, dimensions=3, model="test-embedding-model"):
        self.dimensions = dimensions
        self.model = model
        self.embed_calls: list[list[str]] = []

    def embed_many(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        return [
            _FakeEmbeddingResult(tuple(float(i + 1) for i in range(self.dimensions))) for _ in texts
        ]


class _FakeCloud:
    """Records apply-time calls; never touches a network."""

    def __init__(self):
        self.order: list[str] = []
        self.inserted: list[dict] = []
        self.finalized: list[str] = []
        self.state: dict[str, dict] = {}

    def state_reader(self, source_id, *, user_id, agent_name):
        return self.state.get(source_id)

    def insert_chunks(
        self, chunks, *, user_id, agent_name, embedding_model, embedding_dimensions, cfg
    ):
        self.order.append("insert_chunks")
        self.inserted.extend(chunks)
        return len(chunks)

    def finalize_source_revision(
        self, source_id, active_chunk_ids, *, source, user_id, agent_name, cfg
    ):
        self.order.append("finalize_source_revision")
        self.finalized.append(source_id)
        self.state[source_id] = {
            "source_id": source_id,
            "revision": source["revision"],
            "content_hash": source["content_hash"],
            "is_active": True,
        }


def _install_apply_fakes(monkeypatch, cloud, embedder):
    monkeypatch.setattr(cli, "_build_embedding_client", lambda cfg: embedder)
    monkeypatch.setattr(cli, "_build_state_reader", lambda cfg: cloud.state_reader)
    monkeypatch.setattr(cli, "_build_insert_chunks", lambda cfg: cloud.insert_chunks)
    monkeypatch.setattr(
        cli, "_build_finalize_source_revision", lambda cfg: cloud.finalize_source_revision
    )
    monkeypatch.setattr(cli, "_build_semantic_gateway", lambda cfg: None)


def test_apply_on_dirty_worktree_is_rejected(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"pkg/module.py": PY_BODY})
    # make the worktree dirty
    (root / "pkg" / "module.py").write_text(PY_BODY + "\n# edit\n", encoding="utf-8")

    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_apply_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["ingest-repo", "--user", "tojo", "--repo", str(root), "--apply"]
    )
    assert result.exit_code != 0
    assert "dirty" in result.output.lower() or "clean" in result.output.lower()
    # nothing written
    assert cloud.finalized == []
    assert cloud.inserted == []


def test_apply_allow_dirty_permits_write(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"pkg/module.py": PY_BODY})
    (root / "pkg" / "module.py").write_text(PY_BODY + "\n# edit\n", encoding="utf-8")

    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_apply_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-repo", "--user", "tojo", "--repo", str(root), "--apply", "--allow-dirty"],
    )
    assert result.exit_code == 0, result.output
    assert cloud.order == ["insert_chunks", "finalize_source_revision"]
    assert cloud.finalized
    assert embedder.embed_calls
    assert BODY_MARKER not in result.output


# --- Slice 8: --apply wires plan + apply through injected seams --------------


def test_apply_writes_through_plan_and_apply(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _init_repo(root, {"pkg/module.py": PY_BODY, "README.md": DOC_BODY})
    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_apply_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["ingest-repo", "--user", "tojo", "--repo", str(root), "--apply"]
    )
    assert result.exit_code == 0, result.output
    # both sources inserted then finalized (order is insert then finalize per source)
    assert cloud.order == [
        "insert_chunks",
        "finalize_source_revision",
        "insert_chunks",
        "finalize_source_revision",
    ]
    assert len(cloud.finalized) == 2
    assert cloud.inserted
    assert embedder.embed_calls
    assert "written=2" in result.output or "applied" in result.output.lower()
    assert BODY_MARKER not in result.output


# --- Slice 9: STRICT dry-run isolation --------------------------------------


def test_dry_run_constructs_no_client_makes_no_network(monkeypatch, tmp_path):
    """A dry run must build NO cloud client and make NO network call."""
    import socket

    root = tmp_path / "repo"
    _init_repo(root, {"pkg/module.py": PY_BODY})

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must not construct a client or call an apply seam")

    # Every apply-time seam explodes if touched.
    monkeypatch.setattr(cli, "_build_embedding_client", _boom)
    monkeypatch.setattr(cli, "_build_state_reader", _boom)
    monkeypatch.setattr(cli, "_build_insert_chunks", _boom)
    monkeypatch.setattr(cli, "_build_finalize_source_revision", _boom)
    monkeypatch.setattr(cli, "_build_semantic_gateway", _boom)

    # Client factories deeper in the stack also explode if constructed.
    import hermes_memory.bigquery_store as bqs
    import hermes_memory.config as config_mod

    monkeypatch.setattr(bqs, "_bq_client", _boom)
    monkeypatch.setattr(config_mod, "get_vertex_client", _boom)

    # Any socket use is a hard failure (Git subprocesses do not open Python sockets).
    def _no_socket(*args, **kwargs):
        raise AssertionError("dry run must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_socket)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-repo", "--user", "tojo", "--repo", str(root)])
    assert result.exit_code == 0, result.output
    assert "discovered=1" in result.output
