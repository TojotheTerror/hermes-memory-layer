"""Central config — env, lazy Vertex AI client, model choices."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HermesMemoryConfig:
    project: str
    location: str = "us-central1"
    bq_location: str = "US"
    bq_dataset: str = "hermes_memory"
    agent_engine_id: str | None = None
    generation_model: str = "gemini-3.5-flash"
    embedding_model: str = "text-embedding-005"
    ttl_days: int = 365

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


def load_config(**overrides) -> HermesMemoryConfig:
    return HermesMemoryConfig(
        project=overrides.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID") or "gen-lang-client-0810135629",
        location=overrides.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1",
        bq_location=overrides.get("bq_location") or os.environ.get("BQ_LOCATION") or "US",
        bq_dataset=overrides.get("bq_dataset") or os.environ.get("BQ_DATASET") or "hermes_memory",
        agent_engine_id=overrides.get("agent_engine_id") or os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"),
        generation_model=overrides.get("generation_model") or os.environ.get("MEMORY_GENERATION_MODEL") or "gemini-3.5-flash",
        embedding_model=overrides.get("embedding_model") or os.environ.get("MEMORY_EMBEDDING_MODEL") or "text-embedding-005",
        ttl_days=int(overrides.get("ttl_days") or os.environ.get("MEMORY_TTL_DAYS") or 365),
    )


@lru_cache(maxsize=1)
def get_vertex_client(project: str | None = None, location: str | None = None):
    """Lazy Vertex AI client — returns None if SDK not installed or no creds (mock mode)."""
    try:
        import vertexai
        cfg = load_config(project=project, location=location) if (project or location) else load_config()
        client = vertexai.Client(project=cfg.project, location=cfg.location)
        return client
    except Exception as e:
        print(f"[hermes-memory] Vertex AI client unavailable (mock mode): {e}")
        return None
