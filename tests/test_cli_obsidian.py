"""Preview-first ``ingest-obsidian`` CLI — Click-runner behavior tests.

Every test drives the command through ``CliRunner``. Cloud dependencies are
injected via module-level factory seams on ``cli`` so nothing here constructs a
real Vertex/BigQuery/Memory Bank client or makes a network call. The real
``plan_obsidian_ingestion`` (pure) runs against a temporary vault on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from hermes_memory import cli


NOTE_BODY = """# Daily Note

This is a substantial paragraph of genuine note content that comfortably clears
any minimum-length gate so the discovery policy accepts it as a real source.

## Section Two

A second section with more prose so the note produces at least one packed chunk
carrying real derived text ready for embedding downstream.
"""


def _write_note(root: Path, relative: str, body: str = NOTE_BODY) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- Slice 1: command identity + allowlist validation ----------------------


def test_command_is_still_named_ingest_obsidian():
    assert "ingest-obsidian" in cli.main.commands


def test_missing_allowlist_fails_clearly():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-obsidian", "--user", "tojo"])
    assert result.exit_code != 0
    assert "vault" in result.output.lower()


def test_empty_allowlist_value_fails_clearly():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", "  "])
    assert result.exit_code != 0
    assert "vault" in result.output.lower()


# --- Slice 2: default preview (dry-run) prints a deterministic plan table ----


def test_default_run_is_preview_and_prints_plan_table(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root)])

    assert result.exit_code == 0, result.output
    out = result.output
    # preview framing + honest no-write banner
    assert "preview" in out.lower()
    assert "no writes" in out.lower() or "no bigquery" in out.lower()
    # deterministic accounting is present
    assert "discovered=1" in out
    assert "notes/daily.md" in out
    assert "chunks=" in out and "tokens=" in out and "cost_estimate=" in out
    # never prints the note body
    assert "substantial paragraph" not in out


def test_preview_output_is_deterministic_across_runs(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/zeta.md")
    _write_note(root, "notes/alpha.md")
    runner = CliRunner()
    args = ["ingest-obsidian", "--user", "tojo", "--vault", str(root)]
    first = runner.invoke(cli.main, args)
    second = runner.invoke(cli.main, args)
    assert first.exit_code == 0, first.output
    assert first.output == second.output
    # sources are ordered by relative_path (alpha before zeta)
    assert first.output.index("notes/alpha.md") < first.output.index("notes/zeta.md")


# --- Slice 3: --json emits deterministic, body-free JSON --------------------


def test_json_flag_emits_stable_body_free_json(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/zeta.md")
    _write_note(root, "notes/alpha.md")
    runner = CliRunner()
    args = [
        "ingest-obsidian",
        "--user",
        "tojo",
        "--agent",
        "hermes",
        "--vault",
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
    assert counts["rejected"] == 0
    assert counts["chunks"] >= 2
    assert counts["requests"] == counts["chunks"]
    assert counts["tokens"] > 0
    assert payload["cost_estimate"] > 0
    # per-source detail, ordered by relative_path
    paths = [s["relative_path"] for s in payload["discovered"]]
    assert paths == ["notes/alpha.md", "notes/zeta.md"]
    for s in payload["discovered"]:
        assert set(s) == {"relative_path", "status", "chunk_count", "token_count"}
    # never leaks the note body
    assert "substantial paragraph" not in result.output


def test_json_output_is_byte_identical_across_runs(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/one.md")
    _write_note(root, "notes/two.md")
    runner = CliRunner()
    args = ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--json"]
    first = runner.invoke(cli.main, args)
    second = runner.invoke(cli.main, args)
    assert first.exit_code == 0, first.output
    assert first.output == second.output


# --- Slice 4: incompatible flag combinations are rejected -------------------


def test_promote_without_apply_is_rejected(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--promote-to-memory-bank"],
    )
    assert result.exit_code != 0
    assert "--apply" in result.output
    assert "promote" in result.output.lower()


def test_prune_without_apply_is_rejected(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--prune"]
    )
    assert result.exit_code != 0
    assert "--apply" in result.output
    assert "prune" in result.output.lower()


def test_prune_with_limit_is_rejected(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "ingest-obsidian",
            "--user",
            "tojo",
            "--vault",
            str(root),
            "--apply",
            "--prune",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "prune" in result.output.lower()
    assert "limit" in result.output.lower()


# --- Slice 6: --batch-chars is deprecated, warns, and maps to nothing -------


def test_batch_chars_is_deprecated_and_ignored(tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    runner = CliRunner()
    # invoke WITH --batch-chars and capture the plan output
    with_flag = runner.invoke(
        cli.main,
        ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--batch-chars", "6000"],
    )
    assert with_flag.exit_code == 0, with_flag.output
    assert "deprecat" in with_flag.output.lower()
    assert "batch-chars" in with_flag.output.lower()

    # the deprecation must NOT change semantics: the plan body is identical to a
    # run without the flag once the warning line is stripped.
    without_flag = runner.invoke(
        cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root)]
    )
    assert without_flag.exit_code == 0, without_flag.output

    def _strip_deprecation(text: str) -> str:
        kept = [line for line in text.splitlines() if "deprecat" not in line.lower()]
        return "\n".join(kept).strip()

    assert _strip_deprecation(with_flag.output) == without_flag.output.strip()


# --- Slice 5: --apply wires plan + apply through injected seams -------------


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
        self.memory_bank_calls: list[tuple] = []
        self.prune_calls: list[dict] = []

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

    def memory_bank_client(self, name, texts, scope, *, cfg):
        self.memory_bank_calls.append((name, tuple(texts), scope))

    def deactivate_missing_sources(
        self, corpus_id, seen_source_ids, *, user_id, agent_name, prune, limited, cfg
    ):
        self.prune_calls.append(
            {
                "corpus_id": corpus_id,
                "seen": tuple(seen_source_ids),
                "prune": prune,
                "limited": limited,
            }
        )


def _install_fakes(monkeypatch, cloud, embedder, *, memory_bank_name="mb://fake"):
    monkeypatch.setattr(cli, "_build_embedding_client", lambda cfg: embedder)
    monkeypatch.setattr(cli, "_build_state_reader", lambda cfg: cloud.state_reader)
    monkeypatch.setattr(cli, "_build_insert_chunks", lambda cfg: cloud.insert_chunks)
    monkeypatch.setattr(
        cli, "_build_finalize_source_revision", lambda cfg: cloud.finalize_source_revision
    )
    monkeypatch.setattr(cli, "_build_semantic_gateway", lambda cfg: None)
    monkeypatch.setattr(
        cli, "_build_memory_bank", lambda cfg: (cloud.memory_bank_client, memory_bank_name)
    )
    monkeypatch.setattr(cli, "_deactivate_missing_sources", cloud.deactivate_missing_sources)


def test_apply_writes_through_plan_and_apply(monkeypatch, tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--apply"]
    )
    assert result.exit_code == 0, result.output
    # apply order per source is insert then finalize
    assert cloud.order == ["insert_chunks", "finalize_source_revision"]
    assert cloud.inserted
    assert cloud.finalized
    assert embedder.embed_calls
    # promotion NOT performed without the promote flag
    assert cloud.memory_bank_calls == []
    # no prune without the prune flag
    assert cloud.prune_calls == []
    # report framing, body-free
    assert "applied" in result.output.lower() or "written" in result.output.lower()
    assert "substantial paragraph" not in result.output


def test_apply_promote_requires_flag_and_wires_memory_bank(monkeypatch, tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")
    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "ingest-obsidian",
            "--user",
            "tojo",
            "--vault",
            str(root),
            "--apply",
            "--promote-to-memory-bank",
        ],
    )
    assert result.exit_code == 0, result.output
    assert cloud.memory_bank_calls, "promotion must call the Memory Bank client"


def test_apply_limit_caps_written_sources(monkeypatch, tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/alpha.md")
    _write_note(root, "notes/beta.md")
    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_fakes(monkeypatch, cloud, embedder)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--apply", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    # exactly one source finalized despite two discovered
    assert len(cloud.finalized) == 1


def test_apply_prune_only_runs_with_prune_flag(monkeypatch, tmp_path):
    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")

    # without --prune: no deactivate call
    cloud = _FakeCloud()
    embedder = _FakeEmbeddingClient()
    _install_fakes(monkeypatch, cloud, embedder)
    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--apply"]
    )
    assert result.exit_code == 0, result.output
    assert cloud.prune_calls == []

    # with --prune: deactivate is called with the seen source ids and prune=True
    cloud2 = _FakeCloud()
    embedder2 = _FakeEmbeddingClient()
    _install_fakes(monkeypatch, cloud2, embedder2)
    result2 = runner.invoke(
        cli.main,
        ["ingest-obsidian", "--user", "tojo", "--vault", str(root), "--apply", "--prune"],
    )
    assert result2.exit_code == 0, result2.output
    assert len(cloud2.prune_calls) == 1
    call = cloud2.prune_calls[0]
    assert call["prune"] is True
    assert call["limited"] is False
    assert call["seen"], "prune must pass the seen source ids"


# --- Slice 7: STRICT dry-run isolation --------------------------------------


def test_dry_run_constructs_no_client_makes_no_network_writes_no_state(monkeypatch, tmp_path):
    """A dry run must build NO client, call NO network, and write NO state file."""
    import socket

    root = tmp_path / "vault"
    _write_note(root, "notes/daily.md")

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must not construct a client or call a seam")

    # Every apply-time seam explodes if touched.
    monkeypatch.setattr(cli, "_build_embedding_client", _boom)
    monkeypatch.setattr(cli, "_build_state_reader", _boom)
    monkeypatch.setattr(cli, "_build_insert_chunks", _boom)
    monkeypatch.setattr(cli, "_build_finalize_source_revision", _boom)
    monkeypatch.setattr(cli, "_build_semantic_gateway", _boom)
    monkeypatch.setattr(cli, "_build_memory_bank", _boom)
    monkeypatch.setattr(cli, "_deactivate_missing_sources", _boom)

    # Client factories deeper in the stack also explode if constructed.
    import hermes_memory.bigquery_store as bqs
    import hermes_memory.config as config_mod

    monkeypatch.setattr(bqs, "_bq_client", _boom)
    monkeypatch.setattr(config_mod, "get_vertex_client", _boom)

    # Any socket use is a hard failure.
    def _no_socket(*args, **kwargs):
        raise AssertionError("dry run must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_socket)

    # Redirect the legacy manifest/state dir into tmp and assert nothing lands.
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest-obsidian", "--user", "tojo", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert "discovered=1" in result.output

    # No manifest/state file written anywhere under the fake home.
    if fake_home.exists():
        leaked = [p for p in fake_home.rglob("*") if p.is_file()]
        assert leaked == [], f"dry run wrote state files: {leaked}"
