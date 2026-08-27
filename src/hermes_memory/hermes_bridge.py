"""Hermes bridge — local SQLite ↔ cloud dual-retrieval + sync."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .bigquery_store import insert_memory
from .config import HermesMemoryConfig, load_config
from .memory_bank import append_event, generate_from_contents, generate_from_session, retrieve_memories


def _local_db_path() -> Path:
    return Path.home() / ".hermes" / "hermes.db"


def _memories_dir() -> Path:
    return Path.home() / ".hermes" / "memories"


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


class HermesBridge:
    """Offline-safe bridge between local Hermes and GCP memory layer."""

    def __init__(self, cfg: HermesMemoryConfig | None = None):
        self.cfg = cfg or load_config()

    @property
    def memory_bank_name(self) -> str | None:
        return self.cfg.agent_engine_name

    def retrieve_context(self, user_id: str, query: str, top_k: int = 8, agent_name: str = "hermes") -> dict:
        """Dual retrieval: Memory Bank semantic + BigQuery SQL (mock if offline)."""
        scope = {"user_id": user_id, "agent_name": agent_name}
        bank_hits: list[dict] = []
        if self.memory_bank_name:
            try:
                bank_hits = retrieve_memories(self.memory_bank_name, scope, query, top_k=top_k, cfg=self.cfg)
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
            from .bigquery_store import _bq_client, query_memories_sql
            client = _bq_client(self.cfg)
            if client is not None:
                sql = query_memories_sql(user_id, limit=top_k, cfg=self.cfg)
                rows = list(client.query(sql).result())
                bq_hits = [{"fact": r["fact"], "source": "bigquery", "raw": dict(r)} for r in rows]
        except Exception as e:
            print(f"[bridge] BigQuery retrieve failed: {e}")

        # Local SQLite as third source (always available)
        local_hits = read_local_memories(limit=top_k)

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
