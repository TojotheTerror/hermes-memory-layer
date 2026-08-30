"""Hermes bridge — local SQLite ↔ cloud dual-retrieval + sync."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .bigquery_store import insert_memory
from .config import HermesMemoryConfig, load_config
from .memory_bank import generate_from_contents, generate_from_session, retrieve_memories


def _local_db_path() -> Path:
    return Path.home() / ".hermes" / "hermes.db"


def _memories_dir() -> Path:
    return Path.home() / ".hermes" / "memories"


def _obsidian_manifest_path() -> Path:
    return _memories_dir() / "obsidian_ingest_manifest.json"


def _load_obsidian_manifest() -> dict:
    p = _obsidian_manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_obsidian_manifest(manifest: dict) -> None:
    p = _obsidian_manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ingest_obsidian_documents(
    vault_roots,
    *,
    cfg: HermesMemoryConfig | None = None,
    user_id: str,
    agent_name: str = "hermes",
    embedding_client,
    state_reader: Callable[..., dict | None] | None = None,
    insert_chunks: Callable[..., int] | None = None,
    finalize_source_revision: Callable[..., None] | None = None,
    semantic_gateway=None,
    memory_bank_client: Callable[..., object] | None = None,
    promote_to_memory_bank: bool = False,
    memory_bank_name: str | None = None,
    manifest_loader: Callable[[], dict] | None = None,
    policy=None,
):
    """Ingest Obsidian notes as documents with BigQuery document_sources authority.

    Skip decisions are made ONLY against the BigQuery ``document_sources`` state
    (via ``state_reader``, defaulting to ``bigquery_store.get_source_state``). The
    v1 local ``obsidian_ingest_manifest.json`` is read exactly once, purely to
    report prior legacy state in the report — it never gates or skips a document
    write, and the manifest file is left untouched.
    """

    from . import bigquery_store
    from .ingestion import apply_ingestion_plan, plan_obsidian_ingestion

    cfg = cfg or load_config()

    # One-time, report-only read of the legacy v1 manifest. NEVER used to gate.
    loader = manifest_loader or _load_obsidian_manifest
    legacy_manifest = loader() or {}
    legacy_manifest_entries = len(legacy_manifest)

    if state_reader is None:

        def _default_state_reader(source_id, *, user_id, agent_name):
            return bigquery_store.get_source_state(
                source_id, user_id=user_id, agent_name=agent_name, cfg=cfg
            )

        state_reader = _default_state_reader

    insert_chunks = insert_chunks or bigquery_store.insert_chunks
    finalize_source_revision = finalize_source_revision or bigquery_store.finalize_source_revision

    plan = plan_obsidian_ingestion(
        vault_roots,
        cfg=cfg,
        user_id=user_id,
        agent_name=agent_name,
        state_reader=state_reader,
        policy=policy,
    )
    return apply_ingestion_plan(
        plan,
        cfg=cfg,
        user_id=user_id,
        agent_name=agent_name,
        embedding_client=embedding_client,
        insert_chunks=insert_chunks,
        finalize_source_revision=finalize_source_revision,
        semantic_gateway=semantic_gateway,
        memory_bank_client=memory_bank_client,
        promote_to_memory_bank=promote_to_memory_bank,
        memory_bank_name=memory_bank_name,
        legacy_manifest_entries=legacy_manifest_entries,
    )


DEFAULT_OBSIDIAN_VAULTS = [
    "~/Vaults/Fully Experimental",
    "~/Vaults/Hermes_Agent",
    "~/Vaults/Passthrough",
    "~/Documents/Thoughtseize",
]

_SKIP_DIR_MARKERS = (".obsidian", ".trash", ".git")


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def discover_obsidian_notes(vault_paths: list[str], min_chars: int = 200) -> list[dict]:
    """Walk vaults, skip config/trash dirs and sub-min-chars stub notes."""
    import hashlib

    notes: list[dict] = []
    for vp_raw in vault_paths:
        vp = Path(vp_raw).expanduser()
        if not vp.exists():
            continue
        for f in vp.rglob("*.md"):
            path_str = str(f)
            if any(
                f"/{marker}/" in path_str or path_str.endswith(f"/{marker}")
                for marker in _SKIP_DIR_MARKERS
            ):
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            body = _strip_frontmatter(raw).strip()
            if len(body) < min_chars:
                continue
            h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                rel = str(f.relative_to(vp))
            except ValueError:
                rel = f.name
            notes.append({"path": path_str, "vault": str(vp), "rel": rel, "hash": h, "body": body})
    return notes


def batch_notes(notes: list[dict], batch_chars: int = 6000) -> list[list[dict]]:
    """Group notes into batches under an approx char budget per Memory Bank generate() call."""
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for n in notes:
        entry_len = len(n["rel"]) + len(n["vault"]) + len(n["body"]) + 32
        if cur and cur_len + entry_len > batch_chars:
            batches.append(cur)
            cur = []
            cur_len = 0
        cur.append(n)
        cur_len += entry_len
    if cur:
        batches.append(cur)
    return batches


def read_curated_memory_files() -> list[dict]:
    """Read the curated MEMORY.md / USER.md files (§-separated durable facts)."""
    out: list[dict] = []
    for fname, kind in [("MEMORY.md", "memory"), ("USER.md", "user_profile")]:
        path = _memories_dir() / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for chunk in text.split("§"):
            fact = chunk.strip()
            if fact:
                out.append({"fact": fact, "source_file": fname, "kind": kind})
    return out


def read_local_memories(limit: int = 50) -> list[dict]:
    path = _local_db_path()
    if not path.exists():
        # Hermes uses journal_mode wal with different schema — try to find any db
        for cand in [Path.home() / ".hermes" / "memory.db", Path.home() / ".hermes" / "agent.db"]:
            if cand.exists():
                path = cand
                break
        else:
            return []
    try:
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # discover tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if not tables:
            return []
        # try common memory table names
        for t in ["memories", "memory", "agent_memory", "hermes_memory"]:
            if t in tables:
                cur.execute(f"SELECT * FROM {t} LIMIT {limit}")
                return [dict(r) for r in cur.fetchall()]
        # fallback: first table
        cur.execute(f"SELECT * FROM {tables[0]} LIMIT {limit}")
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[bridge] local db read failed: {e}")
        return []


def _retrieve_bigquery_memories(user_id: str, *, top_k: int, cfg: HermesMemoryConfig) -> list[dict]:
    """Retrieve structured BigQuery hits when a client is available."""
    from .bigquery_store import _bq_client, query_memories_sql

    client = _bq_client(cfg)
    if client is None:
        return []
    sql = query_memories_sql(user_id, limit=top_k, cfg=cfg)
    rows = list(client.query(sql).result())
    return [{"fact": row["fact"], "source": "bigquery", "raw": dict(row)} for row in rows]


def _default_query_embedder(cfg: HermesMemoryConfig):
    """Build a live RETRIEVAL_QUERY embedder callable, or None when offline.

    Returns a ``callable(text) -> EmbeddingResult`` or ``None``. Constructing
    the SDK client is deferred here so importing this module never touches the
    network, and unit tests inject their own embedder instead of ever reaching
    this factory.
    """
    from .config import get_vertex_client

    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        return None
    from .embeddings import VertexEmbeddingClient

    embedder = VertexEmbeddingClient(
        client=client,
        model=cfg.document_embedding_model,
        dimensions=cfg.document_embedding_dimensions,
        task_type="RETRIEVAL_QUERY",
    )
    return embedder.embed


def _default_document_search(embedding, **kwargs):
    """Default document channel search — delegates to bigquery_store."""
    from .bigquery_store import search_document_chunks

    return search_document_chunks(embedding, **kwargs)


def _document_hit_to_dict(hit) -> dict:
    """Project a DocumentChunkSearchResult onto a citation-bearing channel dict.

    Citations are ALWAYS retained. Content/embeddings are never printed.
    """
    return {
        "citation": hit.citation,
        "content": hit.content,
        "contextual_content": hit.contextual_content,
        "source_path": hit.source_path,
        "heading_path": list(hit.heading_path),
        "symbol": hit.symbol,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "corpus_id": hit.corpus_id,
        "distance": hit.distance,
        "origin": "document",
    }


class HermesBridge:
    """Offline-safe bridge between local Hermes and GCP memory layer."""

    def __init__(
        self,
        cfg: HermesMemoryConfig | None = None,
        *,
        memory_bank_retriever: Callable[..., list[dict]] = retrieve_memories,
        bigquery_retriever: Callable[..., list[dict]] = _retrieve_bigquery_memories,
        local_memory_reader: Callable[..., list[dict]] = read_local_memories,
        query_embedder: Callable[[str], Any] | None = None,
        document_search: Callable[..., list] = _default_document_search,
    ):
        self.cfg = cfg or load_config()
        self._memory_bank_retriever = memory_bank_retriever
        self._bigquery_retriever = bigquery_retriever
        self._local_memory_reader = local_memory_reader
        # Document retrieval is a SEPARATE channel. The query embedder is
        # injected (by production wiring or tests). When None the document
        # channel is simply inert — this class never constructs a live client.
        self._query_embedder = query_embedder
        self._document_search = document_search

    @property
    def memory_bank_name(self) -> str | None:
        return self.cfg.agent_engine_name

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 8,
        agent_name: str = "hermes",
        *,
        document_retrieval_enabled: bool = True,
        document_top_k: int | None = None,
        document_context_char_limit: int | None = None,
    ) -> dict:
        """Dual retrieval: Memory Bank semantic + BigQuery SQL (mock if offline).

        Document retrieval is a SEPARATE, citation-bearing channel merged by
        field (``document_hits``), never flattened into the ``fact`` channel. It
        fails open: any embedding or search error preserves Memory Bank recall
        and the turn still returns.
        """
        scope = {"user_id": user_id, "agent_name": agent_name}
        bank_hits: list[dict] = []
        if self.memory_bank_name:
            try:
                bank_hits = self._memory_bank_retriever(
                    self.memory_bank_name, scope, query, top_k=top_k, cfg=self.cfg
                )
            except Exception as e:
                print(f"[bridge] Memory Bank retrieve failed: {e}")
        else:
            # mock — don't create a real client
            bank_hits = [
                {
                    "fact": f"[mock] memory {i} for '{query}' (scope={scope})",
                    "score": 0.9 - i * 0.1,
                    "scope": scope,
                }
                for i in range(min(top_k, 3))
            ]

        # BigQuery structured hits (text match fallback if no embeddings yet)
        bq_hits: list[dict] = []
        try:
            bq_hits = self._bigquery_retriever(user_id, top_k=top_k, cfg=self.cfg)
        except Exception as e:
            print(f"[bridge] BigQuery retrieve failed: {e}")

        # Local SQLite as third source (always available)
        local_hits: list[dict] = []
        try:
            local_hits = self._local_memory_reader(limit=top_k)
        except Exception:
            print("[bridge] Local memory read failed")

        # Merge + dedupe by fact text (case-insensitive)
        seen: set[str] = set()
        merged: list[dict] = []
        for hit in bank_hits + bq_hits:
            fact = (hit.get("fact") or hit.get("text") or "")[:500]
            key = fact.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append({**hit, "origin": "memory_bank" if hit in bank_hits else "bigquery"})
        # Append local hits that add new info
        for lh in local_hits:
            fact = str(lh.get("fact") or lh.get("content") or lh.get("text") or "")[:500]
            if fact and fact.lower().strip() not in seen:
                merged.append({"fact": fact, "origin": "local", "raw": lh})

        # Document channel — SEPARATE from facts, citation-bearing, fail-open.
        document_hits = self._retrieve_documents(
            user_id,
            query,
            agent_name=agent_name,
            enabled=document_retrieval_enabled,
            document_top_k=document_top_k,
            document_context_char_limit=document_context_char_limit,
        )

        return {
            "query": query,
            "scope": scope,
            "memory_bank_hits": bank_hits,
            "bigquery_hits": bq_hits,
            "local_hits": local_hits,
            "document_hits": document_hits,
            "merged": merged[:top_k],
            "prompt_context": "\n".join(f"- {m['fact']}" for m in merged[:top_k] if m.get("fact")),
        }

    def _retrieve_documents(
        self,
        user_id: str,
        query: str,
        *,
        agent_name: str,
        enabled: bool,
        document_top_k: int | None,
        document_context_char_limit: int | None,
    ) -> list[dict]:
        """Citation-aware document channel. Embed ONCE, search, then enforce
        ``document_top_k`` and ``document_context_char_limit`` before returning.

        Fail-open contract: any embedding or search failure returns ``[]`` so
        Memory Bank recall is preserved and the turn is never blocked. Never
        prints bodies, credentials, or embeddings.
        """
        if not enabled:
            return []

        top_k = document_top_k if document_top_k is not None else self.cfg.document_top_k
        char_limit = (
            document_context_char_limit
            if document_context_char_limit is not None
            else self.cfg.document_context_char_limit
        )
        # Invalid bounds disable the channel rather than break the turn.
        if type(top_k) is not int or top_k <= 0:
            return []
        if type(char_limit) is not int or char_limit <= 0:
            return []

        embedder = self._query_embedder
        if embedder is None:
            # No embedder wired (e.g. offline or unit isolation): document
            # channel is inert. Memory Bank recall is unaffected.
            return []

        try:
            # Embed the query exactly ONCE for document retrieval.
            embedded = embedder(query)
            embedding = getattr(embedded, "values", embedded)
            raw_hits = self._document_search(
                embedding,
                user_id=user_id,
                agent_name=agent_name,
                top_k=top_k,
                cfg=self.cfg,
            )
        except Exception as e:
            # Fail-open: preserve Memory Bank recall, never surface bodies.
            print(f"[bridge] document retrieve failed: {e}")
            return []

        # Enforce top_k on the returned list (defence in depth) then the char
        # budget, always retaining each hit's citation.
        hits: list[dict] = []
        used_chars = 0
        for hit in raw_hits[:top_k]:
            projected = _document_hit_to_dict(hit)
            content_len = len(projected["content"])
            if used_chars + content_len > char_limit:
                remaining = char_limit - used_chars
                if remaining <= 0:
                    break
                projected["content"] = projected["content"][:remaining]
                projected["truncated"] = True
                used_chars += remaining
                hits.append(projected)
                break
            used_chars += content_len
            hits.append(projected)
        return hits

    def explicit_remember(
        self, user_id: str, fact: str, agent_name: str = "hermes", metadata: dict | None = None
    ) -> dict:
        """Direct fact write — Memory Bank + BigQuery mirror."""
        scope = {"user_id": user_id, "agent_name": agent_name}
        result: dict[str, Any] = {"scope": scope, "fact": fact}
        if self.memory_bank_name:
            try:
                result["memory_bank"] = generate_from_contents(
                    self.memory_bank_name, [fact], scope, cfg=self.cfg
                )
            except Exception as e:
                result["memory_bank_error"] = str(e)
        # Always mirror to BigQuery (mock if offline)
        try:
            result["bigquery"] = insert_memory(
                fact=fact, scope=scope, cfg=self.cfg, source="direct", metadata=metadata
            )
        except Exception as e:
            result["bigquery_error"] = str(e)
        return result

    def sync_session(
        self,
        session_name: str,
        user_id: str,
        events: list[dict] | None = None,
        agent_name: str = "hermes",
    ) -> dict:
        """Ship a Session's events to Memory Bank generation + BigQuery."""
        result: dict[str, Any] = {"session_name": session_name, "user_id": user_id}
        if events is not None:
            from .bigquery_store import insert_session

            try:
                result["bigquery_session"] = insert_session(
                    session_name, user_id, events, cfg=self.cfg
                )
            except Exception as e:
                result["bigquery_error"] = str(e)
        if self.memory_bank_name:
            try:
                # scope must exactly match the scope used at retrieval time (Memory Bank does exact-match, not subset)
                result["memory_bank"] = generate_from_session(
                    self.memory_bank_name,
                    session_name,
                    scope={"user_id": user_id, "agent_name": agent_name},
                    cfg=self.cfg,
                )
            except Exception as e:
                result["memory_bank_error"] = str(e)
        return result
