"""Central config — env, lazy Vertex AI client, model choices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


_DOCUMENT_OVERRIDE_MISSING = object()


@dataclass(frozen=True)
class HermesMemoryConfig:
    project: str
    location: str = "us-central1"
    bq_location: str = "US"
    bq_dataset: str = "hermes_memory"
    agent_engine_id: str | None = None
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    document_embedding_model: str = "gemini-embedding-001"
    document_embedding_dimensions: int = 768
    chunk_target_tokens: int = 600
    chunk_min_tokens: int = 250
    chunk_max_tokens: int = 900
    chunk_overlap_tokens: int = 80
    embedding_concurrency: int = 4
    document_top_k: int = 4
    document_context_char_limit: int = 8_000
    ttl_days: int = 365

    def __post_init__(self) -> None:
        integer_settings = (
            "document_embedding_dimensions",
            "chunk_min_tokens",
            "chunk_target_tokens",
            "chunk_max_tokens",
            "chunk_overlap_tokens",
            "embedding_concurrency",
            "document_top_k",
            "document_context_char_limit",
        )
        for setting in integer_settings:
            if type(getattr(self, setting)) is not int:
                raise TypeError(f"{setting} must be an integer")

        positive_settings = tuple(
            setting for setting in integer_settings if setting != "chunk_overlap_tokens"
        )
        for setting in positive_settings:
            if getattr(self, setting) <= 0:
                raise ValueError(f"{setting} must be greater than zero")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if self.chunk_min_tokens > self.chunk_target_tokens:
            raise ValueError("chunk_min_tokens must be less than or equal to chunk_target_tokens")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_target_tokens must be less than or equal to chunk_max_tokens")
        if self.chunk_overlap_tokens >= self.chunk_min_tokens:
            raise ValueError("chunk_overlap_tokens must be less than chunk_min_tokens")

    @property
    def generation_model_path(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/publishers/google/models/{self.generation_model}"

    @property
    def embedding_model_path(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/publishers/google/models/{self.embedding_model}"

    @property
    def agent_engine_name(self) -> str | None:
        if not self.agent_engine_id:
            return None
        return f"projects/{self.project}/locations/{self.location}/reasoningEngines/{self.agent_engine_id}"


def _load_document_setting(overrides, setting, environment, default):
    override = overrides.get(setting, _DOCUMENT_OVERRIDE_MISSING)
    if override is not _DOCUMENT_OVERRIDE_MISSING:
        if override is None or override == "":
            raise ValueError(f"{setting} must not be None or empty")
        return override
    return os.environ.get(environment) or default


def _load_document_int(overrides, setting, environment, default):
    override = overrides.get(setting, _DOCUMENT_OVERRIDE_MISSING)
    if override is not _DOCUMENT_OVERRIDE_MISSING:
        value = _load_document_setting(overrides, setting, environment, default)
        if type(value) is not int:
            raise TypeError(f"{setting} must be an integer")
        return value
    return int(_load_document_setting(overrides, setting, environment, default))


def load_config(**overrides) -> HermesMemoryConfig:
    return HermesMemoryConfig(
        project=overrides.get("project")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("PROJECT_ID")
        or "gen-lang-client-0810135629",
        location=overrides.get("location")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "us-central1",
        bq_location=overrides.get("bq_location") or os.environ.get("BQ_LOCATION") or "US",
        bq_dataset=overrides.get("bq_dataset") or os.environ.get("BQ_DATASET") or "hermes_memory",
        agent_engine_id=overrides.get("agent_engine_id")
        or os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
        or "8113170407277723648",
        generation_model=overrides.get("generation_model")
        or os.environ.get("MEMORY_GENERATION_MODEL")
        or "gemini-2.5-flash",
        embedding_model=overrides.get("embedding_model")
        or os.environ.get("MEMORY_EMBEDDING_MODEL")
        or "text-embedding-005",
        document_embedding_model=_load_document_setting(
            overrides,
            "document_embedding_model",
            "DOCUMENT_EMBEDDING_MODEL",
            "gemini-embedding-001",
        ),
        document_embedding_dimensions=_load_document_int(
            overrides, "document_embedding_dimensions", "DOCUMENT_EMBEDDING_DIMENSIONS", 768
        ),
        chunk_target_tokens=_load_document_int(
            overrides, "chunk_target_tokens", "DOCUMENT_CHUNK_TARGET_TOKENS", 600
        ),
        chunk_min_tokens=_load_document_int(
            overrides, "chunk_min_tokens", "DOCUMENT_CHUNK_MIN_TOKENS", 250
        ),
        chunk_max_tokens=_load_document_int(
            overrides, "chunk_max_tokens", "DOCUMENT_CHUNK_MAX_TOKENS", 900
        ),
        chunk_overlap_tokens=_load_document_int(
            overrides, "chunk_overlap_tokens", "DOCUMENT_CHUNK_OVERLAP_TOKENS", 80
        ),
        embedding_concurrency=_load_document_int(
            overrides, "embedding_concurrency", "DOCUMENT_EMBEDDING_CONCURRENCY", 4
        ),
        document_top_k=_load_document_int(overrides, "document_top_k", "DOCUMENT_TOP_K", 4),
        document_context_char_limit=_load_document_int(
            overrides, "document_context_char_limit", "DOCUMENT_CONTEXT_CHAR_LIMIT", 8_000
        ),
        ttl_days=int(overrides.get("ttl_days") or os.environ.get("MEMORY_TTL_DAYS") or 365),
    )


@lru_cache(maxsize=1)
def get_vertex_client(project: str | None = None, location: str | None = None):
    """Lazy Vertex AI client — returns None if SDK not installed or no creds (mock mode)."""
    try:
        import vertexai

        cfg = (
            load_config(project=project, location=location)
            if (project or location)
            else load_config()
        )
        client = vertexai.Client(project=cfg.project, location=cfg.location)
        return client
    except Exception as e:
        print(f"[hermes-memory] Vertex AI client unavailable (mock mode): {e}")
        return None
