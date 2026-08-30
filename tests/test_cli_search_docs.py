"""``search-docs`` CLI — Click-runner behavior tests.

Every test drives the command through ``CliRunner`` and injects fakes for the
two module-level seams (``_build_query_embedding_client`` and
``_search_document_chunks``). Nothing here constructs a real Vertex/BigQuery
client or makes a network call: deeper client factories are monkeypatched to
explode and ``socket.socket`` is trapped, proving the command touches only the
injected fakes.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from hermes_memory import cli
from hermes_memory.bigquery_store import DocumentChunkSearchResult


def _make_result(
    *,
    distance,
    content,
    citation,
    source_path,
    heading_path=(),
    symbol=None,
    start_line=1,
    end_line=2,
    chunk_id="chunk-x",
    source_id="source-x",
    corpus_id="corpus-x",
) -> DocumentChunkSearchResult:
    return DocumentChunkSearchResult(
        distance=distance,
        content=content,
        contextual_content=content,
        citation=citation,
        source_path=source_path,
        heading_path=heading_path,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        chunk_id=chunk_id,
        source_id=source_id,
        corpus_id=corpus_id,
    )


class _FakeEmbeddingResult:
    def __init__(self, values):
        self.values = values


class _FakeQueryEmbeddingClient:
    """Records how many times and with what task type the query was embedded."""

    def __init__(self, task_type, dimensions=3, model="test-embedding-model"):
        self.task_type = task_type
        self.dimensions = dimensions
        self.model = model
        self.embed_calls: list[str] = []

    def embed(self, text):
        self.embed_calls.append(text)
        return _FakeEmbeddingResult(tuple(float(i + 1) for i in range(self.dimensions)))


class _FakeSearch:
    """Records the query embedding + scope/filters passed to the search."""

    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def __call__(
        self,
        query_embedding,
        *,
        user_id,
        agent_name,
        corpus_id=None,
        source_kind=None,
        content_kind=None,
        top_k=None,
        cfg=None,
    ):
        self.calls.append(
            {
                "query_embedding": tuple(query_embedding),
                "user_id": user_id,
                "agent_name": agent_name,
                "corpus_id": corpus_id,
                "source_kind": source_kind,
                "content_kind": content_kind,
                "top_k": top_k,
            }
        )
        return list(self.results)


def _install_fakes(monkeypatch, *, results, capture=None):
    """Install fake embedding + search seams and trap real clients + sockets."""
    made = {}

    def _build(cfg, *, task_type):
        client = _FakeQueryEmbeddingClient(task_type)
        made["embedder"] = client
        return client

    search = _FakeSearch(results)
    made["search"] = search
    monkeypatch.setattr(cli, "_build_query_embedding_client", _build)
    monkeypatch.setattr(cli, "_search_document_chunks", search)

    # Deeper real-client factories must never be constructed.
    import socket

    import hermes_memory.bigquery_store as bqs
    import hermes_memory.config as config_mod

    def _boom(*args, **kwargs):
        raise AssertionError("search-docs must not construct a real client")

    monkeypatch.setattr(bqs, "_bq_client", _boom)
    monkeypatch.setattr(config_mod, "get_vertex_client", _boom)

    def _no_socket(*args, **kwargs):
        raise AssertionError("search-docs must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_socket)

    if capture is not None:
        capture.update(made)
    return made


# --- Slice 1: command identity ----------------------------------------------


def test_command_is_registered_as_search_docs():
    assert "search-docs" in cli.main.commands


# --- Slice 2: default query type is docs (RETRIEVAL_QUERY) -------------------


def test_default_query_type_embeds_once_with_retrieval_query(monkeypatch):
    made = _install_fakes(
        monkeypatch,
        results=[
            _make_result(
                distance=0.1,
                content="a retrieved passage excerpt",
                citation="notes/daily.md#L1-L4",
                source_path="notes/daily.md",
            )
        ],
    )
    runner = CliRunner()
    result = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "how do I do X"])
    assert result.exit_code == 0, result.output
    # embedded exactly once, with the docs task type
    assert made["embedder"].task_type == "RETRIEVAL_QUERY"
    assert made["embedder"].embed_calls == ["how do I do X"]
    # search received that one embedding vector
    assert len(made["search"].calls) == 1
    assert made["search"].calls[0]["query_embedding"] == (1.0, 2.0, 3.0)
    # ranked excerpt + citation printed
    assert "a retrieved passage excerpt" in result.output
    assert "notes/daily.md#L1-L4" in result.output


# --- Slice 3: --query-type code uses CODE_RETRIEVAL_QUERY --------------------


def test_query_type_code_embeds_with_code_retrieval_query(monkeypatch):
    made = _install_fakes(
        monkeypatch,
        results=[
            _make_result(
                distance=0.2,
                content="def handler(): ...",
                citation="pkg/module.py#L10-L20",
                source_path="pkg/module.py",
                symbol="handler",
            )
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["search-docs", "--user", "tojo", "--query", "handler function", "--query-type", "code"],
    )
    assert result.exit_code == 0, result.output
    assert made["embedder"].task_type == "CODE_RETRIEVAL_QUERY"
    assert made["embedder"].embed_calls == ["handler function"]
    assert "pkg/module.py#L10-L20" in result.output


def test_invalid_query_type_is_rejected(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["search-docs", "--user", "tojo", "--query", "x", "--query-type", "banana"],
    )
    assert result.exit_code != 0


# --- Slice 4: ranked citation output, deterministic ordering ----------------


def test_results_are_printed_ranked_by_distance(monkeypatch):
    made = _install_fakes(
        monkeypatch,
        results=[
            _make_result(
                distance=0.05,
                content="closest match text",
                citation="a.md#L1-L2",
                source_path="a.md",
            ),
            _make_result(
                distance=0.9,
                content="farther match text",
                citation="b.md#L3-L4",
                source_path="b.md",
            ),
        ],
    )
    runner = CliRunner()
    result = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "q"])
    assert result.exit_code == 0, result.output
    # the nearer (smaller distance) citation appears before the farther one
    assert result.output.index("a.md#L1-L2") < result.output.index("b.md#L3-L4")
    assert made["search"].calls  # search was actually invoked


# --- Slice 5: --json emits deterministic, stable-ordered results ------------


def test_json_flag_emits_stable_ordered_results(monkeypatch):
    _install_fakes(
        monkeypatch,
        results=[
            _make_result(
                distance=0.05,
                content="closest match text",
                citation="a.md#L1-L2",
                source_path="a.md",
                heading_path=("Intro",),
                chunk_id="chunk-a",
                source_id="source-a",
                corpus_id="corpus-a",
            ),
            _make_result(
                distance=0.9,
                content="farther match text",
                citation="b.py#L3-L4",
                source_path="b.py",
                symbol="thing",
                chunk_id="chunk-b",
                source_id="source-b",
                corpus_id="corpus-b",
            ),
        ],
    )
    runner = CliRunner()
    result = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "q", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["user_id"] == "tojo"
    assert payload["query_type"] == "docs"
    assert payload["count"] == 2
    results = payload["results"]
    # order follows the ranked search order (nearest first)
    assert [r["citation"] for r in results] == ["a.md#L1-L2", "b.py#L3-L4"]
    first = results[0]
    assert set(first) == {
        "rank",
        "distance",
        "citation",
        "source_path",
        "heading_path",
        "symbol",
        "start_line",
        "end_line",
        "excerpt",
        "chunk_id",
        "source_id",
        "corpus_id",
    }
    assert first["rank"] == 1
    assert first["excerpt"] == "closest match text"


def test_json_output_is_byte_identical_across_runs(monkeypatch):
    def _results():
        return [
            _make_result(
                distance=0.05,
                content="closest match text",
                citation="a.md#L1-L2",
                source_path="a.md",
            )
        ]

    runner = CliRunner()
    _install_fakes(monkeypatch, results=_results())
    first = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "q", "--json"])
    _install_fakes(monkeypatch, results=_results())
    second = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "q", "--json"])
    assert first.exit_code == 0, first.output
    assert first.output == second.output


# --- Slice 6: query-type routes content_kind filter -------------------------


def test_code_query_type_filters_search_to_code_content(monkeypatch):
    made = _install_fakes(monkeypatch, results=[])
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["search-docs", "--user", "tojo", "--query", "q", "--query-type", "code"],
    )
    assert result.exit_code == 0, result.output
    assert made["search"].calls[0]["content_kind"] == "code"


# --- Slice 7: empty results reported cleanly, no crash ----------------------


def test_no_results_is_reported_without_error(monkeypatch):
    _install_fakes(monkeypatch, results=[])
    runner = CliRunner()
    result = runner.invoke(cli.main, ["search-docs", "--user", "tojo", "--query", "q"])
    assert result.exit_code == 0, result.output
    assert "no" in result.output.lower() and "result" in result.output.lower()
