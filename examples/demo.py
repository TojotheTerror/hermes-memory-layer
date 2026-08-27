"""End-to-end demo — works in mock mode (no GCP creds required)."""
from hermes_memory.config import load_config
from hermes_memory.hermes_bridge import HermesBridge

cfg = load_config()
print(f"Config: project={cfg.project} location={cfg.location} dataset={cfg.bq_dataset}")
print(f"Memory Bank: {cfg.agent_engine_name or 'mock (set GOOGLE_CLOUD_AGENT_ENGINE_ID to use real)'}")
print()

bridge = HermesBridge(cfg)

# 1. Explicit remember (mirrors to BQ + Memory Bank mock)
print("== 1. Explicit remember ==")
res = bridge.explicit_remember(user_id="tojo", fact="I run a 3-node homelab: Caladan (Linux, Ryzen 9 9900X), Sietch Tabr (server), Arcanum (Windows). Fleet topology at :9090.", metadata={"source": "demo"})
print(res.get("bigquery", res))
print()

# 2. Another fact
print("== 2. Preference remember ==")
res2 = bridge.explicit_remember(user_id="tojo", fact="I prefer event-driven reflex+reasoner monitoring — Go probes at zero token cost, LLMs only on anomalies.")
print(res2.get("bigquery", res2))
print()

# 3. Dual retrieval
print("== 3. Dual retrieval (Memory Bank + BigQuery + local) ==")
ctx = bridge.retrieve_context(user_id="tojo", query="how does fleet monitoring work?", top_k=5)
print(f"Memory Bank hits: {len(ctx['memory_bank_hits'])}")
for h in ctx["memory_bank_hits"]:
    print(f"  [MB] {h.get('fact','')[:100]}")
print(f"BigQuery hits: {len(ctx['bigquery_hits'])}")
for h in ctx["bigquery_hits"]:
    print(f"  [BQ] {h.get('fact','')[:100]}")
print(f"Local hits: {len(ctx['local_hits'])}")
print(f"Merged ({len(ctx['merged'])}):")
for m in ctx["merged"]:
    print(f"  [{m.get('origin')}] {m.get('fact','')[:120]}")
print()
print("Prompt context:")
print(ctx["prompt_context"] or "(empty — populate memories first)")
print()
print("== Demo done ==")
print("With real GCP: set GOOGLE_CLOUD_AGENT_ENGINE_ID and run hermes-memory init")
