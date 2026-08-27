import importlib

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
    assert "gemini-3.5-flash" in cfg.generation_model_path

def test_bigquery_ddl_present():
    from hermes_memory.bigquery_store import DDL_MEMORIES, DDL_SESSIONS, DDL_REVISIONS
    assert "CREATE TABLE IF NOT EXISTS" in DDL_MEMORIES
    assert "embedding ARRAY<FLOAT64>" in DDL_MEMORIES
    assert "sessions" in DDL_SESSIONS

def test_bridge_mock():
    from hermes_memory.hermes_bridge import HermesBridge
    b = HermesBridge()
    ctx = b.retrieve_context(user_id="test_user", query="hello", top_k=3)
    assert "merged" in ctx
    assert "prompt_context" in ctx
