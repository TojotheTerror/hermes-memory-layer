"""Task 17 — plugin prefetch surfaces document sections with stable citations.

The provider is exercised with an injected fake bridge; no real HermesBridge,
Vertex, or BigQuery client is constructed and no network call is made.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

gcp_memory_bank = importlib.import_module("gcp_memory_bank")
GcpMemoryBankProvider = gcp_memory_bank.GcpMemoryBankProvider


class _FakeExecutor:
    """Runs submitted work synchronously so tests stay deterministic."""

    def submit(self, fn, /, *args, **kwargs):
        class _Immediate:
            def __init__(self, value=None, error=None):
                self._value = value
                self._error = error

            def result(self, timeout=None):
                if self._error is not None:
                    raise self._error
                return self._value

        try:
            return _Immediate(value=fn(*args, **kwargs))
        except Exception as e:  # pragma: no cover - defensive
            return _Immediate(error=e)


class _FakeBridge:
    """Captures retrieve_context kwargs and returns a fixed dual-channel result."""

    memory_bank_name = "projects/p/locations/l/reasoningEngines/e"

    def __init__(self, result):
        self._result = result
        self.calls = []

    def retrieve_context(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


# Two Markdown sections, each a document hit with its own stable citation.
_DOC_RESULT = {
    "query": "how does retrieval work?",
    "scope": {"user_id": "tojo", "agent_name": "hermes"},
    "memory_bank_hits": [{"fact": "User prefers concise answers."}],
    "bigquery_hits": [],
    "local_hits": [],
    "document_hits": [
        {
            "citation": "obsidian://Vault/RAG.md#Overview",
            "content": "RAG retrieves cited chunks before generation.",
            "source_path": "Vault/RAG.md",
            "heading_path": ["Overview"],
            "chunk_id": "chunk-a",
            "origin": "document",
        },
        {
            "citation": "obsidian://Vault/RAG.md#Citations",
            "content": "Each chunk keeps a stable citation to its source.",
            "source_path": "Vault/RAG.md",
            "heading_path": ["Citations"],
            "chunk_id": "chunk-b",
            "origin": "document",
        },
    ],
    "merged": [{"fact": "User prefers concise answers.", "origin": "memory_bank"}],
    "prompt_context": "- User prefers concise answers.",
}


def _provider_with_bridge(result, config=None):
    provider = GcpMemoryBankProvider(config=config or {})
    provider._bridge = _FakeBridge(result)
    provider._executor = _FakeExecutor()
    return provider


def test_prefetch_surfaces_two_document_sections_with_stable_citations():
    provider = _provider_with_bridge(_DOC_RESULT)

    out = provider.prefetch("how does retrieval work?", session_id="s1")

    # Memory Bank fact still surfaces.
    assert "User prefers concise answers." in out

    # Both document sections appear WITH their citations retained.
    assert "obsidian://Vault/RAG.md#Overview" in out
    assert "obsidian://Vault/RAG.md#Citations" in out
    assert "RAG retrieves cited chunks before generation." in out
    assert "Each chunk keeps a stable citation to its source." in out

    # Documents are a SEPARATE section, not flattened into the fact list.
    assert "## GCP Memory Bank" in out
    assert "Documents" in out


def test_document_citations_are_stable_across_prefetch_calls():
    provider = _provider_with_bridge(_DOC_RESULT)

    first = provider.prefetch("how does retrieval work?", session_id="s1")
    second = provider.prefetch("how does retrieval work?", session_id="s1")

    assert first == second
    # Citations appear in a deterministic order matching the hit order.
    assert first.index("obsidian://Vault/RAG.md#Overview") < first.index(
        "obsidian://Vault/RAG.md#Citations"
    )


def test_prefetch_passes_document_config_to_bridge():
    provider = _provider_with_bridge(
        _DOC_RESULT,
        config={
            "document_retrieval_enabled": True,
            "document_top_k": 3,
            "document_context_char_limit": 2000,
        },
    )

    provider.prefetch("how does retrieval work?", session_id="s1")

    call = provider._bridge.calls[0]
    assert call["document_retrieval_enabled"] is True
    assert call["document_top_k"] == 3
    assert call["document_context_char_limit"] == 2000


def test_prefetch_returns_memory_bank_only_when_no_documents():
    result = dict(_DOC_RESULT, document_hits=[])
    provider = _provider_with_bridge(result)

    out = provider.prefetch("how does retrieval work?", session_id="s1")

    assert "User prefers concise answers." in out
    assert "Documents" not in out


def test_document_retrieval_disabled_when_configured_off():
    provider = _provider_with_bridge(_DOC_RESULT, config={"document_retrieval_enabled": False})

    provider.prefetch("how does retrieval work?", session_id="s1")

    call = provider._bridge.calls[0]
    assert call["document_retrieval_enabled"] is False


def test_config_schema_exposes_document_fields():
    provider = GcpMemoryBankProvider(config={})
    keys = {field["key"] for field in provider.get_config_schema()}
    assert "document_retrieval_enabled" in keys
    assert "document_top_k" in keys
    assert "document_context_char_limit" in keys


def test_save_config_round_trips_document_fields(tmp_path):
    yaml = pytest.importorskip("yaml")
    provider = GcpMemoryBankProvider(config={})
    values = {
        "user_id": "tojo",
        "document_retrieval_enabled": True,
        "document_top_k": 6,
        "document_context_char_limit": 4000,
    }
    provider.save_config(values, str(tmp_path))

    written = yaml.safe_load((tmp_path / "config.yaml").read_text())
    saved = written["memory"]["gcp_memory_bank"]
    assert saved["document_retrieval_enabled"] is True
    assert saved["document_top_k"] == 6
    assert saved["document_context_char_limit"] == 4000
