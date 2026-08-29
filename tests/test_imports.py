def test_imports():
    import hermes_memory
    assert hermes_memory.__version__ == "0.1.0"
    from hermes_memory.config import load_config
    cfg = load_config()
    assert cfg.project
    assert cfg.bq_dataset == "hermes_memory"

def test_config_paths():
    from hermes_memory.config import load_config
    cfg = load_config(project="test-proj", location="us-central1")
    assert "test-proj" in cfg.generation_model_path
    assert "gemini-2.5-flash" in cfg.generation_model_path

def test_bigquery_ddl_present():
    from hermes_memory.bigquery_store import DDL_MEMORIES, DDL_SESSIONS
    assert "CREATE TABLE IF NOT EXISTS" in DDL_MEMORIES
    assert "embedding ARRAY<FLOAT64>" in DDL_MEMORIES
    assert "sessions" in DDL_SESSIONS

def test_bridge_mock_is_fully_isolated(monkeypatch):
    from hermes_memory.config import HermesMemoryConfig
    from hermes_memory.hermes_bridge import HermesBridge

    def fail_live_access(*args, **kwargs):
        raise AssertionError("bridge smoke test attempted live or local production access")

    monkeypatch.setattr("hermes_memory.memory_bank.get_vertex_client", fail_live_access)
    monkeypatch.setattr("hermes_memory.bigquery_store._bq_client", fail_live_access)
    monkeypatch.setattr("hermes_memory.hermes_bridge.sqlite3.connect", fail_live_access)

    calls = []

    def fake_memory_bank(memory_bank_name, scope, query, *, top_k, cfg):
        calls.append(("memory_bank", memory_bank_name, scope, query, top_k, cfg.project))
        return [{"fact": "memory-bank fact"}]

    def fake_bigquery(user_id, *, top_k, cfg):
        calls.append(("bigquery", user_id, top_k, cfg.project))
        return [{"fact": "bigquery fact", "source": "bigquery"}]

    def fake_local_memory(*, limit):
        calls.append(("local", limit))
        return [{"fact": "local fact"}]

    cfg = HermesMemoryConfig(project="test-project", agent_engine_id="fake-engine")
    bridge = HermesBridge(
        cfg,
        memory_bank_retriever=fake_memory_bank,
        bigquery_retriever=fake_bigquery,
        local_memory_reader=fake_local_memory,
    )

    ctx = bridge.retrieve_context(user_id="test_user", query="hello", top_k=3)

    assert [item["fact"] for item in ctx["merged"]] == [
        "memory-bank fact",
        "bigquery fact",
        "local fact",
    ]
    assert ctx["prompt_context"] == "- memory-bank fact\n- bigquery fact\n- local fact"
    assert calls == [
        (
            "memory_bank",
            "projects/test-project/locations/us-central1/reasoningEngines/fake-engine",
            {"user_id": "test_user", "agent_name": "hermes"},
            "hello",
            3,
            "test-project",
        ),
        ("bigquery", "test_user", 3, "test-project"),
        ("local", 3),
    ]
