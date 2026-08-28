"""GCP Memory Bank plugin — MemoryProvider interface.

Auto-ingestion + auto-retrieval of durable memories via Google Cloud
Agent Platform (Vertex AI Memory Bank, semantic extraction/search) mirrored
into BigQuery (SQL analytics, audit trail, TTL). Backed by the
``hermes_memory`` package (github.com/TojotheTerror/hermes-memory-layer).

How it plugs into the Hermes turn loop (see agent/memory_provider.py):
  prefetch(query)          -- called before each LLM call; blocking, bounded.
                               Runs HermesBridge.retrieve_context() (Memory
                               Bank semantic search + BigQuery SQL) and
                               returns formatted context text to inject.
  sync_turn(user, asst)    -- called after each turn; fire-and-forget
                               background thread. Writes the turn as an
                               explicit memory (Memory Bank extraction via
                               gemini-2.5-flash + BigQuery mirror).
  on_memory_write(...)     -- mirrors built-in MEMORY.md / USER.md writes
                               into the same GCP backend.

Config (config.yaml, all optional):
  memory:
    gcp_memory_bank:
      user_id: tojo          # scope key (default: "tojo")
      agent_name: hermes     # scope key (default: "hermes")
      prefetch_timeout: 8    # seconds to block prefetch() before giving up
      top_k: 5               # memories to retrieve per turn

Requires (already configured on this machine, see ~/hermes-memory-layer):
  GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_AGENT_ENGINE_ID
  in the environment (set in ~/.bashrc and ~/.config/environment.d/), plus
  Application Default Credentials (gcloud auth application-default login).
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

_GLYPH = "\u2601\ufe0f"  # cloud, to distinguish from the generic brain glyph
_DEFAULT_PREFETCH_TIMEOUT = 8.0
_DEFAULT_TOP_K = 5
_MIN_SYNC_LEN = 8  # skip near-empty turns


def _load_plugin_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}
        provider_config = memory_config.get("gcp_memory_bank", {})
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


class GcpMemoryBankProvider(MemoryProvider):
    """Vertex AI Memory Bank + BigQuery memory provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._user_id = str(self._config.get("user_id", "tojo"))
        self._agent_name = str(self._config.get("agent_name", "hermes"))
        self._prefetch_timeout = float(self._config.get("prefetch_timeout", _DEFAULT_PREFETCH_TIMEOUT))
        self._top_k = int(self._config.get("top_k", _DEFAULT_TOP_K))

        self._bridge = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._sync_lock = threading.Lock()
        self._pending_syncs: list[threading.Thread] = []
        self._last_recall_count = 0
        self._last_recall_had_content = False

    @property
    def name(self) -> str:
        return "gcp_memory_bank"

    # -- availability ---------------------------------------------------

    def is_available(self) -> bool:
        """Config + import check only — no network calls.

        Hardened against a transient submodule-resolution miss: if
        ``hermes_memory`` imports but its ``.config`` submodule is briefly
        unresolvable (e.g. a path-ordering hiccup while two copies of the
        package are momentarily visible during boot), we invalidate the
        import caches and retry ONCE before reporting unavailable. This
        prevents a one-time boot-time race from being cached as a permanent
        False by callers that probe availability only once at startup.
        """
        try:
            import hermes_memory  # noqa: F401
        except Exception as e:
            logger.debug("gcp_memory_bank: hermes_memory not importable: %s", e)
            return False

        for attempt in (1, 2):
            try:
                from hermes_memory.config import load_config as load_hm_config

                cfg = load_hm_config()
                return bool(cfg.project)
            except ModuleNotFoundError as e:
                # Submodule not resolvable yet — likely a transient import-path
                # race. Invalidate finder caches and retry once.
                if attempt == 1:
                    logger.debug(
                        "gcp_memory_bank: hermes_memory.config miss (attempt %d), "
                        "invalidating caches and retrying: %s",
                        attempt, e,
                    )
                    import importlib
                    importlib.invalidate_caches()
                    continue
                logger.debug("gcp_memory_bank: config still unavailable after retry: %s", e)
                return False
            except Exception as e:
                logger.debug("gcp_memory_bank: config unavailable: %s", e)
                return False
        return False

    def unavailable_reason(self) -> str:
        return (
            "hermes_memory package not importable, or GOOGLE_CLOUD_PROJECT unset. "
            "Install with: pip install -e ~/hermes-memory-layer, and ensure "
            "GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION/GOOGLE_CLOUD_AGENT_ENGINE_ID "
            "are exported (see ~/.config/environment.d/hermes-memory.conf)."
        )

    # -- lifecycle --------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        agent_context = kwargs.get("agent_context", "primary")
        # Skip live writes from non-primary contexts (cron/subagent system
        # prompts should not pollute the user's Memory Bank scope).
        if agent_context not in ("primary", None, ""):
            logger.debug("gcp_memory_bank: skipping init for agent_context=%s", agent_context)
            self._bridge = None
            return
        try:
            from hermes_memory.hermes_bridge import HermesBridge

            self._bridge = HermesBridge()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="gcp-memory-bank"
            )
            logger.info(
                "gcp_memory_bank initialized: engine=%s scope={user_id:%s, agent_name:%s}",
                self._bridge.memory_bank_name, self._user_id, self._agent_name,
            )
        except Exception as e:
            logger.warning("gcp_memory_bank: initialize failed: %s", e)
            self._bridge = None

    def system_prompt_block(self) -> str:
        if not self._bridge:
            return ""
        return (
            "# GCP Memory Bank\n"
            "Active. Long-term memory (Vertex AI Memory Bank + BigQuery) is "
            "automatically recalled before each turn and written after each "
            "turn — no tool calls needed. Relevant facts appear as "
            "'## GCP Memory Bank' context when found."
        )

    # -- auto-retrieval -----------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Blocking, bounded recall — Memory Bank semantic + BigQuery SQL."""
        self._last_recall_count = 0
        self._last_recall_had_content = False
        if not self._bridge or not self._executor:
            return ""
        if is_trivial_prompt(query):
            return ""
        try:
            future = self._executor.submit(
                self._bridge.retrieve_context,
                user_id=self._user_id,
                query=query,
                top_k=self._top_k,
                agent_name=self._agent_name,
            )
            result = future.result(timeout=self._prefetch_timeout)
        except concurrent.futures.TimeoutError:
            logger.debug("gcp_memory_bank: prefetch timed out after %ss", self._prefetch_timeout)
            return ""
        except Exception as e:
            logger.debug("gcp_memory_bank: prefetch failed: %s", e)
            return ""

        merged = result.get("merged") or []
        if not merged:
            return ""
        self._last_recall_count = len(merged)
        self._last_recall_had_content = True
        lines = [f"- {m['fact']}" for m in merged if m.get("fact")]
        if not lines:
            return ""
        return "## GCP Memory Bank\n" + "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs synchronously (bounded) at turn start."""

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_recall_had_content:
            return None
        return RecallStatus(provider_label="GCP Memory Bank", count=self._last_recall_count, glyph=_GLYPH)

    # -- auto-ingestion -----------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write the completed turn to Memory Bank + BigQuery in the background."""
        if not self._bridge:
            return
        if not user_content or len(user_content.strip()) < _MIN_SYNC_LEN:
            return

        fact = f"User: {user_content[:2000].strip()}\nAssistant: {assistant_content[:2000].strip()}"

        def _sync():
            try:
                self._bridge.explicit_remember(
                    user_id=self._user_id, fact=fact, agent_name=self._agent_name
                )
            except Exception as e:
                logger.debug("gcp_memory_bank: sync_turn failed: %s", e)

        t = threading.Thread(target=_sync, daemon=True, name="gcp-memory-bank-sync")
        t.start()
        with self._sync_lock:
            self._pending_syncs = [x for x in self._pending_syncs if x.is_alive()] + [t]

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into GCP Memory Bank."""
        if not self._bridge or action not in {"add", "replace"} or not content:
            return

        def _write():
            try:
                label = "User profile" if target == "user" else "Agent memory"
                self._bridge.explicit_remember(
                    user_id=self._user_id,
                    fact=f"[{label}] {content}",
                    agent_name=self._agent_name,
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug("gcp_memory_bank: on_memory_write failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="gcp-memory-bank-memwrite")
        t.start()
        with self._sync_lock:
            self._pending_syncs = [x for x in self._pending_syncs if x.is_alive()] + [t]

    # -- tools (none — fully automatic, keep the model's toolset narrow) ---

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    # -- shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        with self._sync_lock:
            pending = list(self._pending_syncs)
            self._pending_syncs = []
        for t in pending:
            t.join(timeout=10.0)
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # -- setup wizard (informational only — config lives in shell env) -----

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "user_id",
                "description": "Memory scope user_id (matches Memory Bank scope)",
                "default": "tojo",
            },
            {
                "key": "agent_name",
                "description": "Memory scope agent_name (matches Memory Bank scope)",
                "default": "hermes",
            },
            {
                "key": "top_k",
                "description": "Memories to retrieve per turn",
                "default": "5",
            },
            {
                "key": "prefetch_timeout",
                "description": "Max seconds to block turn start on recall",
                "default": "8",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path

        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml

            from hermes_cli.config import read_user_config_raw

            existing = read_user_config_raw(config_path)
            existing.setdefault("memory", {})
            existing["memory"]["gcp_memory_bank"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.debug("gcp_memory_bank: save_config failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the GCP Memory Bank provider with the plugin system."""
    ctx.register_memory_provider(GcpMemoryBankProvider())
