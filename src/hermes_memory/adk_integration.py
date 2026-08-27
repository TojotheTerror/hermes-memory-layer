"""ADK integration — VertexAiMemoryBankService + BigQuery tool + callbacks."""
from __future__ import annotations

from typing import Any

from .config import HermesMemoryConfig, load_config


def build_memory_agent(project: str | None = None, location: str | None = None, agent_engine_id: str | None = None):
    """Build an ADK agent with Memory Bank + BigQuery tools.

    Returns (agent, runner) when google-adk is installed, else raises with guidance.
    """
    try:
        from google.adk.agents import Agent
        from google.adk.agents.callback_context import CallbackContext
        from google.adk.memory import VertexAiMemoryBankService
        from google.adk.tools import FunctionTool, LoadMemoryTool
        from google.adk.runners import Runner
    except ImportError as e:
        raise RuntimeError("google-adk not installed. pip install 'hermes-memory-layer[adk]'") from e

    cfg = load_config(project=project, location=location, agent_engine_id=agent_engine_id)

    # --- BigQuery tool for the agent ---
    def bigquery_query(sql: str) -> str:
        """Run a read-only BigQuery SQL query against the hermes_memory dataset."""
        # guard: only SELECT/SHOW allowed
        s = sql.strip().lower()
        if not (s.startswith("select") or s.startswith("with") or s.startswith("show")):
            return "Error: only SELECT/WITH/SHOW queries are allowed."
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=cfg.project)
            rows = list(client.query(sql).result(max_results=50))
            if not rows:
                return "No results."
            return "\n".join(str(dict(r)) for r in rows[:20])
        except Exception as e:
            return f"BigQuery error: {e}"

    bq_tool = FunctionTool(func=bigquery_query)

    # --- Callbacks ---
    async def add_session_to_memory_callback(callback_context: CallbackContext):
        try:
            await callback_context.add_session_to_memory()
        except Exception as e:
            print(f"[ADK] add_session_to_memory failed: {e}")
        return None

    # --- Memory service ---
    # The service is constructed lazily by the Runner; we pass a builder.
    def memory_service_builder():
        return VertexAiMemoryBankService(
            project=cfg.project,
            location=cfg.location,
            agent_engine_id=cfg.agent_engine_id or "",
        )

    agent = Agent(
        name="hermes_memory_agent",
        model="gemini-2.5-flash",
        instruction=(
            "You are Hermes, a helpful fleet assistant with long-term memory. "
            "Use load_memory to recall user preferences and past facts before answering. "
            "Use bigquery_query for analytical questions about memory history. "
            "Be concise and cite which memories you used."
        ),
        tools=[LoadMemoryTool(), bq_tool],
        after_agent_callback=add_session_to_memory_callback,
    )

    runner = Runner(agent=agent, memory_service_builder=memory_service_builder if cfg.agent_engine_id else None)
    return agent, runner


def build_adk_app(agent=None, project: str | None = None, location: str | None = None, agent_engine_id: str | None = None):
    """Build an AdkApp ready for `client.agent_engines.create(agent_engine=adk_app)`."""
    try:
        from vertexai.agent_engines import AdkApp
        from google.adk.memory import VertexAiMemoryBankService
    except ImportError as e:
        raise RuntimeError("Install google-cloud-aiplatform[agent_engines,adk]") from e

    cfg = load_config(project=project, location=location, agent_engine_id=agent_engine_id)
    if agent is None:
        agent, _ = build_memory_agent(project=cfg.project, location=cfg.location, agent_engine_id=cfg.agent_engine_id)

    def memory_service_builder():
        return VertexAiMemoryBankService(project=cfg.project, location=cfg.location, agent_engine_id=cfg.agent_engine_id or "")

    adk_app = AdkApp(agent=agent, memory_service_builder=memory_service_builder if cfg.agent_engine_id else None)
    return adk_app
