"""Memory Bank wrapper — create, generate, retrieve, purge. Mock-safe."""
from __future__ import annotations

import datetime
from typing import Any

try:
    from vertexai.types import (
        ReasoningEngineContextSpecMemoryBankConfig as MemoryBankConfig,
        ReasoningEngineContextSpecMemoryBankConfigGenerationConfig as GenerationConfig,
        ReasoningEngineContextSpecMemoryBankConfigSimilaritySearchConfig as SimilaritySearchConfig,
        ReasoningEngineContextSpecMemoryBankConfigTtlConfig as TtlConfig,
        MemoryBankCustomizationConfig as CustomizationConfig,
        MemoryBankCustomizationConfigConsolidationConfig as ConsolidationConfig,
        MemoryBankCustomizationConfigMemoryTopic as MemoryTopic,
        MemoryBankCustomizationConfigMemoryTopicManagedMemoryTopic as ManagedMemoryTopic,
        ManagedTopicEnum,
    )
    _HAS_VERTEX_TYPES = True
except ImportError:
    _HAS_VERTEX_TYPES = False

from .config import HermesMemoryConfig, get_vertex_client, load_config


def build_memory_bank_config(cfg: HermesMemoryConfig):
    if not _HAS_VERTEX_TYPES:
        return {"note": "mock — vertexai types not installed"}
    return MemoryBankConfig(
        generation_config=GenerationConfig(model=cfg.generation_model_path),
        similarity_search_config=SimilaritySearchConfig(embedding_model=cfg.embedding_model_path),
        ttl_config=TtlConfig(memory_revision_default_ttl=f"{cfg.ttl_days * 24 * 3600}s"),
        customization_configs=[
            CustomizationConfig(
                memory_topics=[
                    MemoryTopic(managed_memory_topic=ManagedMemoryTopic(managed_topic_enum=ManagedTopicEnum.USER_PERSONAL_INFO)),
                    MemoryTopic(managed_memory_topic=ManagedMemoryTopic(managed_topic_enum=ManagedTopicEnum.USER_PREFERENCES)),
                    MemoryTopic(managed_memory_topic=ManagedMemoryTopic(managed_topic_enum=ManagedTopicEnum.KEY_CONVERSATION_DETAILS)),
                    MemoryTopic(managed_memory_topic=ManagedMemoryTopic(managed_topic_enum=ManagedTopicEnum.EXPLICIT_INSTRUCTIONS)),
                ],
                generate_memories_examples=[],
                consolidation_config=ConsolidationConfig(revisions_per_candidate_count=1),
                enable_third_person_memories=False,
            )
        ],
        disable_memory_revisions=False,
    )


def create_memory_bank(cfg: HermesMemoryConfig | None = None, staging_bucket: str | None = None):
    """Create a new Memory Bank (Agent Engine). Returns resource name or mock string."""
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        return f"mock://projects/{cfg.project}/locations/{cfg.location}/reasoningEngines/mock-memory-bank"
    mb_config = build_memory_bank_config(cfg)
    engine = client.agent_engines.create(
        config={
            "context_spec": {"memory_bank_config": mb_config},
            **({"staging_bucket": staging_bucket} if staging_bucket else {}),
        }
    )
    return engine.api_resource.name


def update_memory_bank_config(agent_engine_name: str, cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        print("[mock] update_memory_bank_config skipped")
        return agent_engine_name
    mb_config = build_memory_bank_config(cfg)
    result = client.agent_engines.update(
        name=agent_engine_name,
        config={"context_spec": {"memory_bank_config": mb_config}},
    )
    return result.api_resource.name


# ---- Sessions helpers ----

def create_session(sessions_engine_name: str, user_id: str, cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        return {"name": f"{sessions_engine_name}/sessions/mock-{user_id}", "user_id": user_id, "mock": True}
    session = client.agent_engines.sessions.create(name=sessions_engine_name, user_id=user_id)
    return session


def append_event(session_name: str, text: str, role: str = "user", cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        print(f"[mock] append_event: {role}: {text[:80]}")
        return {"mock": True}
    return client.agent_engines.sessions.events.append(
        name=session_name,
        author=role,
        invocation_id="1",
        timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
        config={"content": {"role": role, "parts": [{"text": text}]}},
    )


# ---- Memory generation / retrieval ----

def generate_from_session(memory_bank_name: str, session_name: str, scope: dict | None = None, cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        print(f"[mock] generate_from_session: {session_name} -> {memory_bank_name} scope={scope}")
        return {"mock": True, "operation": "generate"}
    kwargs: dict[str, Any] = {"name": memory_bank_name, "vertex_session_source": {"session": session_name}}
    if scope:
        kwargs["scope"] = scope
    return client.agent_engines.memories.generate(**kwargs)


def generate_from_contents(memory_bank_name: str, texts: list[str], scope: dict, role: str = "user", cfg: HermesMemoryConfig | None = None):
    """Direct contents source — no Session needed. Good for explicit facts."""
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        print(f"[mock] generate_from_contents: {texts} scope={scope}")
        return {"mock": True}
    # Use dict form to avoid google.genai dependency at import time
    events = [{"content": {"role": role, "parts": [{"text": t}]}} for t in texts]
    return client.agent_engines.memories.generate(
        name=memory_bank_name,
        direct_contents_source={"events": events},
        scope=scope,
    )


def retrieve_memories(memory_bank_name: str, scope: dict, query: str, top_k: int = 8, cfg: HermesMemoryConfig | None = None) -> list[dict]:
    # mock path — must be checked before creating a real client
    if memory_bank_name.startswith("mock://"):
        return [
            {"fact": f"[mock] memory {i} for '{query}' (scope={scope})", "score": 0.9 - i * 0.1, "scope": scope}
            for i in range(min(top_k, 3))
        ]
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        return [
            {"fact": f"[mock] memory {i} for '{query}' (scope={scope})", "score": 0.9 - i * 0.1, "scope": scope}
            for i in range(min(top_k, 3))
        ]
    result = client.agent_engines.memories.retrieve(
        name=memory_bank_name,
        scope=scope,
        similarity_search_params={"search_query": query, "top_k": top_k},
    )
    # normalize to list[dict]
    memories = getattr(result, "memories", None) or getattr(result, "response", None) or result
    if hasattr(memories, "__iter__"):
        out = []
        for m in memories:
            out.append({"fact": getattr(m, "fact", str(m)), "raw": m})
        return out
    return []


def list_memories(memory_bank_name: str, scope: dict, cfg: HermesMemoryConfig | None = None) -> list[dict]:
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        return [{"fact": "[mock] all memories", "scope": scope}]
    result = client.agent_engines.memories.list(name=memory_bank_name, scope=scope)
    return list(result) if hasattr(result, "__iter__") else [result]


def purge_memories(memory_bank_name: str, filter_str: str, force: bool = False, cfg: HermesMemoryConfig | None = None):
    cfg = cfg or load_config()
    client = get_vertex_client(cfg.project, cfg.location)
    if client is None:
        print(f"[mock] purge filter={filter_str} force={force}")
        return {"mock": True, "purge_count": 0}
    return client.agent_engines.memories.purge(name=memory_bank_name, filter=filter_str, force=force)
