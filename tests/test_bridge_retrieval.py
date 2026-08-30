"""Task 17 — citation-aware document retrieval as a SEPARATE bridge channel.

Every test injects fakes; no live Vertex/BigQuery client is ever constructed.
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from hermes_memory.bigquery_store import DocumentChunkSearchResult
from hermes_memory.config import HermesMemoryConfig
from hermes_memory.hermes_bridge import HermesBridge

_FakeEmbedding = namedtuple("_FakeEmbedding", ["values"])


def _doc_result(
    *,
    citation="obsidian://Notes/rag.md#Retrieval",
    content="Retrieval augments generation with cited source chunks.",
    source_path="Notes/rag.md",
    heading_path=("Retrieval",),
    distance=0.12,
    chunk_id="chunk-1",
) -> DocumentChunkSearchResult:
    return DocumentChunkSearchResult(
        distance=distance,
        content=content,
        contextual_content=f"[{source_path}] {content}",
        citation=citation,
        source_path=source_path,
        heading_path=heading_path,
        symbol=None,
        start_line=1,
        end_line=4,
        chunk_id=chunk_id,
        source_id="source-1",
        corpus_id="corpus-1",
    )


def _bridge(*, embedder, document_search, memory_bank_retriever=None):
    cfg = HermesMemoryConfig(project="test-project", agent_engine_id="fake-engine")
    return (
        HermesBridge(
            cfg,
            memory_bank_retriever=memory_bank_retriever
            or (lambda *a, **k: [{"fact": "memory-bank fact"}]),
            bigquery_retriever=lambda *a, **k: [],
            local_memory_reader=lambda *, limit: [],
            query_embedder=embedder,
            document_search=document_search,
        ),
        cfg,
    )


def test_document_hit_lands_in_separate_channel_with_citation_retained():
    """ONE Memory Bank fact + ONE document hit -> separate fields, citation kept."""
    doc = _doc_result()
    bridge, _ = _bridge(
        embedder=lambda text: _FakeEmbedding(values=(0.1, 0.2, 0.3)),
        document_search=lambda embedding, **kw: [doc],
    )

    ctx = bridge.retrieve_context(user_id="u", query="what is rag?", top_k=3)

    # Memory Bank recall keeps its own channel.
    assert [h["fact"] for h in ctx["memory_bank_hits"]] == ["memory-bank fact"]

    # Documents are a SEPARATE channel, not flattened into facts.
    assert "document_hits" in ctx
    assert len(ctx["document_hits"]) == 1
    hit = ctx["document_hits"][0]
    assert hit["citation"] == doc.citation
    assert hit["content"] == doc.content
    assert hit["source_path"] == doc.source_path

    # Citation must NOT be flattened into the fact / merged-fact channel.
    assert doc.citation not in ctx["prompt_context"]
    for merged in ctx["merged"]:
        assert doc.citation not in (merged.get("fact") or "")
    # The document is NOT smuggled into the fact channel as a fake 'fact'.
    assert all(m.get("fact") != doc.content for m in ctx["merged"])


def test_query_is_embedded_exactly_once_for_document_retrieval():
    calls = {"embed": 0, "search_embedding": None}

    def embedder(text):
        calls["embed"] += 1
        return _FakeEmbedding(values=(0.4, 0.5, 0.6))

    def document_search(embedding, **kw):
        calls["search_embedding"] = embedding
        return [_doc_result()]

    bridge, _ = _bridge(embedder=embedder, document_search=document_search)
    bridge.retrieve_context(user_id="u", query="q", top_k=5)

    assert calls["embed"] == 1
    # The single embedding vector is what reaches document search.
    assert calls["search_embedding"] == (0.4, 0.5, 0.6)


def test_document_search_failure_is_fail_open_and_preserves_memory_bank(capsys):
    def boom(embedding, **kw):
        raise RuntimeError("vector search timed out")

    bridge, _ = _bridge(
        embedder=lambda text: _FakeEmbedding(values=(0.1,)),
        document_search=boom,
    )

    ctx = bridge.retrieve_context(user_id="u", query="q", top_k=3)

    # The turn still returns and Memory Bank recall is preserved.
    assert [h["fact"] for h in ctx["memory_bank_hits"]] == ["memory-bank fact"]
    assert ctx["document_hits"] == []
    assert ctx["prompt_context"] == "- memory-bank fact"
    # Never print bodies/embeddings; a bounded diagnostic is acceptable.
    assert "vector search timed out" in capsys.readouterr().out


def test_embedding_failure_is_fail_open_and_preserves_memory_bank():
    def boom_embed(text):
        raise RuntimeError("embedding backend unavailable")

    searched = {"called": False}

    def document_search(embedding, **kw):
        searched["called"] = True
        return [_doc_result()]

    bridge, _ = _bridge(embedder=boom_embed, document_search=document_search)
    ctx = bridge.retrieve_context(user_id="u", query="q", top_k=3)

    assert [h["fact"] for h in ctx["memory_bank_hits"]] == ["memory-bank fact"]
    assert ctx["document_hits"] == []
    assert searched["called"] is False  # never reached search after embed failed


def test_document_top_k_and_char_limit_enforced_before_return():
    results = [
        _doc_result(
            citation=f"obsidian://Notes/n{i}.md#H{i}",
            content="X" * 100,
            source_path=f"Notes/n{i}.md",
            chunk_id=f"chunk-{i}",
            distance=0.1 * i,
        )
        for i in range(5)
    ]

    passed_top_k = {}

    def document_search(embedding, **kw):
        passed_top_k["top_k"] = kw.get("top_k")
        return results  # backend hands back everything; bridge must enforce

    bridge, _ = _bridge(
        embedder=lambda text: _FakeEmbedding(values=(0.1,)),
        document_search=document_search,
    )

    ctx = bridge.retrieve_context(
        user_id="u",
        query="q",
        top_k=8,
        document_top_k=2,
        document_context_char_limit=150,
    )

    # top_k is requested from the backend AND enforced on the returned list.
    assert passed_top_k["top_k"] == 2
    assert len(ctx["document_hits"]) <= 2

    # Total document context content is bounded by the char limit.
    total_chars = sum(len(h["content"]) for h in ctx["document_hits"])
    assert total_chars <= 150

    # Even a truncated hit retains its citation.
    for hit in ctx["document_hits"]:
        assert hit["citation"].startswith("obsidian://Notes/n")


def test_document_retrieval_can_be_disabled():
    called = {"embed": False}

    def embedder(text):
        called["embed"] = True
        return _FakeEmbedding(values=(0.1,))

    bridge, _ = _bridge(
        embedder=embedder,
        document_search=lambda embedding, **kw: [_doc_result()],
    )

    ctx = bridge.retrieve_context(user_id="u", query="q", top_k=3, document_retrieval_enabled=False)

    assert ctx["document_hits"] == []
    assert called["embed"] is False  # disabled => no embedding work at all
    # Memory Bank recall unaffected.
    assert [h["fact"] for h in ctx["memory_bank_hits"]] == ["memory-bank fact"]


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_invalid_document_top_k_disables_channel_without_breaking_turn(bad_top_k):
    bridge, _ = _bridge(
        embedder=lambda text: _FakeEmbedding(values=(0.1,)),
        document_search=lambda embedding, **kw: [_doc_result()],
    )
    ctx = bridge.retrieve_context(user_id="u", query="q", top_k=3, document_top_k=bad_top_k)
    # Fail-open: turn returns, Memory Bank preserved, docs simply empty.
    assert ctx["document_hits"] == []
    assert [h["fact"] for h in ctx["memory_bank_hits"]] == ["memory-bank fact"]
