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
            return text[end + 4:].lstrip("\n")
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
            if any(f"/{marker}/" in path_str or path_str.endswith(f"/{marker}") for marker in _SKIP_DIR_MARKERS):
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


def _retrieve_bigquery_memories(
    user_id: str, *, top_k: int, cfg: HermesMemoryConfig
) -> list[dict]:
    """Retrieve structured BigQuery hits when a client is available."""
    from .bigquery_store import _bq_client, query_memories_sql

    client = _bq_client(cfg)
    if client is None:
        return []
    sql = query_memories_sql(user_id, limit=top_k, cfg=cfg)
    rows = list(client.query(sql).result())
    return [{"fact": row["fact"], "source": "bigquery", "raw": dict(row)} for row in rows]


class HermesBridge:
    """Offline-safe bridge between local Hermes and GCP memory layer."""

    def __init__(
        self,
        cfg: HermesMemoryConfig | None = None,
        *,
        memory_bank_retriever: Callable[..., list[dict]] = retrieve_memories,
        bigquery_retriever: Callable[..., list[dict]] = _retrieve_bigquery_memories,
        local_memory_reader: Callable[..., list[dict]] = read_local_memories,
    ):
        self.cfg = cfg or load_config()
        self._memory_bank_retriever = memory_bank_retriever
        self._bigquery_retriever = bigquery_retriever
        self._local_memory_reader = local_memory_reader

    @property
    def memory_bank_name(self) -> str | None:
        return self.cfg.agent_engine_name

    def retrieve_context(self, user_id: str, query: str, top_k: int = 8, agent_name: str = "hermes") -> dict:
        """Dual retrieval: Memory Bank semantic + BigQuery SQL (mock if offline)."""
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
                {"fact": f"[mock] memory {i} for '{query}' (scope={scope})", "score": 0.9 - i * 0.1, "scope": scope}
                for i in range(min(top_k, 3))
            ]

        # BigQuery structured hits (text match fallback if no embeddings yet)
        bq_hits: list[dict] = []
        try:
            bq_hits = self._bigquery_retriever(user_id, top_k=top_k, cfg=self.cfg)
        except Exception as e:
            print(f"[bridge] BigQuery retrieve failed: {e}")

        # Local SQLite as third source (always available)
        local_hits = self._local_memory_reader(limit=top_k)

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

        return {
            "query": query,
            "scope": scope,
            "memory_bank_hits": bank_hits,
            "bigquery_hits": bq_hits,
            "local_hits": local_hits,
            "merged": merged[:top_k],
            "prompt_context": "\n".join(f"- {m['fact']}" for m in merged[:top_k] if m.get("fact")),
        }

    def explicit_remember(self, user_id: str, fact: str, agent_name: str = "hermes", metadata: dict | None = None) -> dict:
        """Direct fact write — Memory Bank + BigQuery mirror."""
        scope = {"user_id": user_id, "agent_name": agent_name}
        result: dict[str, Any] = {"scope": scope, "fact": fact}
        if self.memory_bank_name:
            try:
                result["memory_bank"] = generate_from_contents(self.memory_bank_name, [fact], scope, cfg=self.cfg)
            except Exception as e:
                result["memory_bank_error"] = str(e)
        # Always mirror to BigQuery (mock if offline)
        try:
            result["bigquery"] = insert_memory(fact=fact, scope=scope, cfg=self.cfg, source="direct", metadata=metadata)
        except Exception as e:
            result["bigquery_error"] = str(e)
        return result

    def sync_session(self, session_name: str, user_id: str, events: list[dict] | None = None, agent_name: str = "hermes") -> dict:
        """Ship a Session's events to Memory Bank generation + BigQuery."""
        result: dict[str, Any] = {"session_name": session_name, "user_id": user_id}
        if events is not None:
            from .bigquery_store import insert_session
            try:
                result["bigquery_session"] = insert_session(session_name, user_id, events, cfg=self.cfg)
            except Exception as e:
                result["bigquery_error"] = str(e)
        if self.memory_bank_name:
            try:
                # scope must exactly match the scope used at retrieval time (Memory Bank does exact-match, not subset)
                result["memory_bank"] = generate_from_session(
                    self.memory_bank_name, session_name, scope={"user_id": user_id, "agent_name": agent_name}, cfg=self.cfg
                )
            except Exception as e:
                result["memory_bank_error"] = str(e)
        return result
