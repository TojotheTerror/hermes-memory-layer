import json
import re
from pathlib import Path

import pytest

from hermes_memory import bigquery_store
from hermes_memory.config import HermesMemoryConfig


ROOT = Path(__file__).parents[1]

SOURCE_FIELDS = [
    ("source_id", "STRING", "REQUIRED"),
    ("corpus_id", "STRING", "REQUIRED"),
    ("user_id", "STRING", "REQUIRED"),
    ("agent_name", "STRING", "REQUIRED"),
    ("source_kind", "STRING", "REQUIRED"),
    ("content_kind", "STRING", "REQUIRED"),
    ("relative_path", "STRING", "REQUIRED"),
    ("source_uri", "STRING", "REQUIRED"),
    ("revision", "STRING", "REQUIRED"),
    ("content_hash", "STRING", "REQUIRED"),
    ("metadata", "JSON", "NULLABLE"),
    ("is_active", "BOOL", "REQUIRED"),
    ("first_seen_at", "TIMESTAMP", "REQUIRED"),
    ("last_seen_at", "TIMESTAMP", "REQUIRED"),
    ("updated_at", "TIMESTAMP", "REQUIRED"),
]

CHUNK_FIELDS = [
    ("chunk_id", "STRING", "REQUIRED"),
    ("source_id", "STRING", "REQUIRED"),
    ("corpus_id", "STRING", "REQUIRED"),
    ("user_id", "STRING", "REQUIRED"),
    ("agent_name", "STRING", "REQUIRED"),
    ("ordinal", "INT64", "REQUIRED"),
    ("content", "STRING", "REQUIRED"),
    ("contextual_content", "STRING", "REQUIRED"),
    ("content_hash", "STRING", "REQUIRED"),
    ("heading_path", "STRING", "REPEATED"),
    ("symbol", "STRING", "NULLABLE"),
    ("start_line", "INT64", "NULLABLE"),
    ("end_line", "INT64", "NULLABLE"),
    ("citation", "STRING", "REQUIRED"),
    ("embedding", "FLOAT64", "REPEATED"),
    ("embedding_model", "STRING", "REQUIRED"),
    ("embedding_dimensions", "INT64", "REQUIRED"),
    ("metadata", "JSON", "NULLABLE"),
    ("is_active", "BOOL", "REQUIRED"),
    ("created_at", "TIMESTAMP", "REQUIRED"),
    ("updated_at", "TIMESTAMP", "REQUIRED"),
]


def _ddl_fields(ddl: str, table_name: str) -> list[tuple[str, str, str]]:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS `[^`]+\.{table_name}` \((.*?)\)\s*CLUSTER BY",
        ddl,
        flags=re.DOTALL,
    )
    assert match, f"missing idempotent DDL for {table_name}"
    fields = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        field_match = re.match(r"(\w+)\s+(ARRAY<([^>]+)>|\w+)(\s+NOT NULL)?", line)
        assert field_match, f"unparseable field declaration: {line}"
        name, declared_type, array_type, required = field_match.groups()
        if array_type:
            fields.append((name, array_type, "REPEATED"))
        else:
            fields.append((name, declared_type, "REQUIRED" if required else "NULLABLE"))
    return fields


def _json_fields(path: Path) -> list[tuple[str, str, str]]:
    schema = json.loads(path.read_text())
    return [(field["name"], field["type"], field["mode"]) for field in schema["fields"]]


def _cluster_fields(ddl: str, table_name: str | None = None) -> list[str]:
    pattern = r"CLUSTER BY ([^;]+);"
    if table_name:
        pattern = (
            rf"CREATE TABLE IF NOT EXISTS `[^`]+\.{table_name}` \(.*?\)\s*"
            rf"CLUSTER BY ([^;]+);"
        )
    match = re.search(pattern, ddl, flags=re.DOTALL)
    assert match, f"missing clustering contract for {table_name or 'DDL'}"
    return [field.strip() for field in match.group(1).split(",")]


@pytest.mark.parametrize(
    ("table_name", "ddl_name", "expected_fields", "cluster_fields"),
    [
        (
            "document_sources",
            "DDL_DOCUMENT_SOURCES",
            SOURCE_FIELDS,
            ["user_id", "agent_name", "corpus_id", "source_kind"],
        ),
        (
            "document_chunks",
            "DDL_DOCUMENT_CHUNKS",
            CHUNK_FIELDS,
            ["user_id", "agent_name", "corpus_id", "source_id"],
        ),
    ],
)
def test_document_table_contracts_match_python_sql_and_terraform_json(
    table_name, ddl_name, expected_fields, cluster_fields
):
    python_ddl = getattr(bigquery_store, ddl_name)
    sql_ddl = (ROOT / "bigquery" / "schema.sql").read_text()
    terraform_schema = ROOT / "terraform" / "schemas" / f"{table_name}.json"

    assert _ddl_fields(python_ddl, table_name) == expected_fields
    assert _ddl_fields(sql_ddl, table_name) == expected_fields
    assert _json_fields(terraform_schema) == expected_fields
    assert _cluster_fields(python_ddl) == cluster_fields
    assert _cluster_fields(sql_ddl, table_name) == cluster_fields


def test_terraform_pins_supported_cli_and_google_provider_series():
    terraform = (ROOT / "terraform" / "main.tf").read_text()

    assert 'required_version = ">= 1.6.0, < 2.0.0"' in terraform
    assert 'version = "~> 6.0"' in terraform


def test_terraform_commits_google_provider_lock():
    lock_path = ROOT / "terraform" / ".terraform.lock.hcl"

    assert lock_path.is_file(), "terraform provider lockfile must be committed"
    lock = lock_path.read_text()
    assert 'provider "registry.terraform.io/hashicorp/google"' in lock
    assert 'version     = "6.' in lock
    assert 'constraints = "~> 6.0"' in lock


def test_terraform_defines_protected_document_tables_without_indexes():
    terraform = (ROOT / "terraform" / "main.tf").read_text()
    variables = (ROOT / "terraform" / "variables.tf").read_text()
    schema_sql = (ROOT / "bigquery" / "schema.sql").read_text()

    expected_clusters = {
        "document_sources": ["user_id", "agent_name", "corpus_id", "source_kind"],
        "document_chunks": ["user_id", "agent_name", "corpus_id", "source_id"],
    }
    for table_name, cluster_fields in expected_clusters.items():
        resource = re.search(
            rf'resource "google_bigquery_table" "{table_name}" \{{(.*?)\n\}}',
            terraform,
            flags=re.DOTALL,
        )
        assert resource, f"missing Terraform resource for {table_name}"
        assert re.search(rf'table_id\s*=\s*"{table_name}"', resource.group(1))
        assert f"schemas/{table_name}.json" in resource.group(1)
        assert "deletion_protection = true" in resource.group(1)
        assert re.search(r"lifecycle\s*\{\s*prevent_destroy\s*=\s*true\s*\}", resource.group(1))
        quoted_fields = ", ".join(f'"{field}"' for field in cluster_fields)
        assert f"clustering          = [{quoted_fields}]" in resource.group(1)

    location = re.search(r'variable "location" \{(.*?)\n\}', variables, flags=re.DOTALL)
    assert location and 'default = "US"' in location.group(1)
    assert "VECTOR INDEX" not in terraform.upper()
    assert "VECTOR INDEX" not in schema_sql.upper()
    assert "VECTOR INDEX" not in bigquery_store.DDL_DOCUMENT_SOURCES.upper()
    assert "VECTOR INDEX" not in bigquery_store.DDL_DOCUMENT_CHUNKS.upper()


@pytest.mark.parametrize(
    "table_name",
    ["memories", "sessions", "memory_revisions", "document_sources", "document_chunks"],
)
def test_terraform_passes_each_schema_fields_array_to_provider(table_name):
    terraform = (ROOT / "terraform" / "main.tf").read_text()
    resource = re.search(
        rf'resource "google_bigquery_table" "{table_name}" \{{(.*?)\n\}}',
        terraform,
        flags=re.DOTALL,
    )

    assert resource
    schema_path = f'file("${{path.module}}/schemas/{table_name}.json")'
    assert f"jsonencode(jsondecode({schema_path}).fields)" in resource.group(1)


class _FakeJob:
    def __init__(self, rows=None):
        self._rows = rows

    def result(self):
        return self._rows


class _FakeBigQueryClient:
    def __init__(self):
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        return _FakeJob()


class _StoreFakeClient:
    def __init__(self, query_results=()):
        self.queries = []
        self.query_results = list(query_results)
        self.inserts = []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        rows = self.query_results.pop(0) if self.query_results else []
        return _FakeJob(rows)

    def insert_rows_json(self, table, rows, row_ids):
        self.inserts.append((table, rows, row_ids))
        return []


class _LifecycleFakeClient(_StoreFakeClient):
    def __init__(self, sources, chunks):
        super().__init__()
        self.sources = {source["source_id"]: dict(source) for source in sources}
        self.chunks = {chunk["chunk_id"]: dict(chunk) for chunk in chunks}

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        parameters = _query_parameters(job_config)
        source = self.sources.get(parameters["source_id"])
        if sql.startswith("SELECT source_id, revision, content_hash, is_active"):
            if source is None:
                return _FakeJob([])
            return _FakeJob(
                [
                    {
                        "source_id": source["source_id"],
                        "revision": source["revision"],
                        "content_hash": source["content_hash"],
                        "is_active": source["is_active"],
                    }
                ]
            )
        if sql.startswith("BEGIN TRANSACTION"):
            expected = set(parameters["active_chunk_ids"])
            matching = [
                chunk
                for chunk in self.chunks.values()
                if chunk["source_id"] == parameters["source_id"]
                and chunk["user_id"] == parameters["user_id"]
                and chunk["agent_name"] == parameters["agent_name"]
            ]
            present = {chunk["chunk_id"] for chunk in matching} & expected
            if "ASSERT" in sql and len(present) != len(expected):
                missing = len(expected) - len(present)
                noun = "chunk" if missing == 1 else "chunks"
                raise RuntimeError(f"missing {missing} expected {noun}")
            for chunk in matching:
                chunk["is_active"] = chunk["chunk_id"] in expected
            incoming = {
                name: value
                for name, value in parameters.items()
                if name not in {"active_chunk_ids", "user_id", "agent_name"}
            }
            incoming["metadata"] = json.loads(incoming["metadata"])
            if source is None:
                source = {}
                self.sources[parameters["source_id"]] = source
            source.update(incoming)
            source.update(
                user_id=parameters["user_id"],
                agent_name=parameters["agent_name"],
                is_active=True,
            )
            return _FakeJob([])
        if sql.startswith("MERGE"):
            incoming = {
                name: value
                for name, value in parameters.items()
                if name not in {"user_id", "agent_name"}
            }
            incoming["metadata"] = json.loads(incoming["metadata"])
            if source is not None:
                if "WHEN MATCHED AND NOT target.is_active" not in sql or not source["is_active"]:
                    source.update(incoming)
                    source["is_active"] = "is_active = TRUE" in sql
            else:
                incoming.update(
                    user_id=parameters["user_id"],
                    agent_name=parameters["agent_name"],
                    is_active="PARSE_JSON(@metadata), TRUE" in sql,
                )
                self.sources[parameters["source_id"]] = incoming
            return _FakeJob([])
        raise AssertionError(f"unexpected query: {sql}")

    def active_chunk_ids(self):
        return {chunk_id for chunk_id, chunk in self.chunks.items() if chunk["is_active"]}


def _query_parameters(job_config):
    return {
        parameter.name: parameter.value if hasattr(parameter, "value") else parameter.values
        for parameter in job_config.query_parameters
    }


def _source(**overrides):
    source = {
        "source_id": "source-1",
        "corpus_id": "corpus-1",
        "source_kind": "obsidian",
        "content_kind": "markdown",
        "relative_path": "notes/it's-complicated.md",
        "source_uri": "file:///vault/notes/it's-complicated.md",
        "revision": "revision-1",
        "content_hash": "hash-1",
        "metadata": {"label": "pilot"},
    }
    source.update(overrides)
    return source


def _chunk(**overrides):
    chunk = {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "corpus_id": "corpus-1",
        "ordinal": 0,
        "text": "Chunk body",
        "contextual_text": "Heading\nChunk body",
        "content_hash": "chunk-hash-1",
        "heading_path": ("Heading",),
        "symbol": None,
        "start_line": 2,
        "end_line": 3,
        "citation": "notes/example.md#L2-L3",
        "embedding": (0.1, 0.2, 0.3),
        "embedding_model": "test-embedding-model",
        "embedding_dimensions": 3,
        "metadata": {"kind": "paragraph"},
    }
    chunk.update(overrides)
    return chunk


def test_upsert_source_skips_unchanged_source_with_parameterized_lookup(monkeypatch):
    source = _source()
    client = _StoreFakeClient(
        [
            [
                {
                    "source_id": source["source_id"],
                    "revision": source["revision"],
                    "content_hash": source["content_hash"],
                    "is_active": True,
                }
            ]
        ]
    )
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    changed = bigquery_store.upsert_source(
        source,
        user_id="user' OR TRUE --",
        agent_name="hermes",
        cfg=cfg,
    )

    assert changed is False
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert source["source_id"] not in sql
    assert "user' OR TRUE --" not in sql
    assert "@source_id" in sql and "@user_id" in sql
    assert _query_parameters(job_config) == {
        "source_id": source["source_id"],
        "user_id": "user' OR TRUE --",
        "agent_name": "hermes",
    }


def test_upsert_source_parameterizes_every_source_value(monkeypatch):
    source = _source(
        source_id="source'; DROP TABLE x --",
        corpus_id="corpus'; DROP TABLE x --",
        relative_path="notes/'quoted'.md",
        source_uri="file:///vault/notes/'quoted'.md",
    )
    client = _StoreFakeClient([[], []])
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    changed = bigquery_store.upsert_source(
        source,
        user_id="user'; DROP TABLE x --",
        agent_name="agent'; DROP TABLE x --",
        cfg=cfg,
    )

    assert changed is True
    assert len(client.queries) == 2
    sql, job_config = client.queries[1]
    for value in (
        *source.values(),
        "user'; DROP TABLE x --",
        "agent'; DROP TABLE x --",
    ):
        if isinstance(value, str):
            assert value not in sql
    parameters = _query_parameters(job_config)
    assert parameters["source_id"] == source["source_id"]
    assert parameters["corpus_id"] == source["corpus_id"]
    assert parameters["relative_path"] == source["relative_path"]
    assert parameters["source_uri"] == source["source_uri"]
    assert parameters["user_id"] == "user'; DROP TABLE x --"
    assert parameters["agent_name"] == "agent'; DROP TABLE x --"
    assert json.loads(parameters["metadata"]) == source["metadata"]


def test_interrupted_existing_source_keeps_active_revision_and_retries_changed(monkeypatch):
    old_source = _source(
        revision="revision-old",
        content_hash="hash-old",
        relative_path="notes/old.md",
        metadata={"label": "old"},
        is_active=True,
    )
    new_source = _source(
        revision="revision-new",
        content_hash="hash-new",
        relative_path="notes/new.md",
        metadata={"label": "new"},
    )
    client = _LifecycleFakeClient(
        [old_source],
        [
            {
                "chunk_id": "old-active",
                "source_id": old_source["source_id"],
                "is_active": True,
            }
        ],
    )
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    assert bigquery_store.upsert_source(new_source, user_id="user-1", agent_name="hermes", cfg=cfg)

    assert client.sources[old_source["source_id"]] == old_source
    assert client.active_chunk_ids() == {"old-active"}
    assert bigquery_store.upsert_source(new_source, user_id="user-1", agent_name="hermes", cfg=cfg)


def test_interrupted_new_source_is_not_finalized_and_retries_changed(monkeypatch):
    source = _source(revision="revision-new", content_hash="hash-new")
    client = _LifecycleFakeClient([], [])
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    assert bigquery_store.upsert_source(source, user_id="user-1", agent_name="hermes", cfg=cfg)

    assert client.sources[source["source_id"]]["is_active"] is False
    assert bigquery_store.upsert_source(source, user_id="user-1", agent_name="hermes", cfg=cfg)


def test_insert_chunks_uses_chunk_ids_as_deterministic_insert_ids(monkeypatch):
    chunks = [_chunk(), _chunk(chunk_id="chunk-2", ordinal=1)]
    client = _StoreFakeClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    inserted = bigquery_store.insert_chunks(
        chunks,
        user_id="user-1",
        agent_name="hermes",
        embedding_model="test-embedding-model",
        embedding_dimensions=3,
        cfg=cfg,
    )

    assert inserted == 2
    assert len(client.inserts) == 1
    table, rows, row_ids = client.inserts[0]
    assert table == "test-project.test_dataset.document_chunks"
    assert row_ids == ["chunk-1", "chunk-2"]
    assert [row["chunk_id"] for row in rows] == row_ids
    assert all(row["is_active"] is False for row in rows)
    assert all(row["embedding_model"] == "test-embedding-model" for row in rows)
    assert all(row["embedding_dimensions"] == 3 for row in rows)


@pytest.mark.parametrize(
    "chunk",
    [
        pytest.param(_chunk(embedding_model="wrong-model"), id="model"),
        pytest.param(_chunk(embedding_dimensions=2), id="declared-dimensions"),
        pytest.param(_chunk(embedding=(0.1, 0.2)), id="vector-length"),
    ],
)
def test_insert_chunks_rejects_embedding_contract_before_client_writes(monkeypatch, chunk):
    client = _StoreFakeClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    with pytest.raises(ValueError, match="embedding"):
        bigquery_store.insert_chunks(
            [chunk],
            user_id="user-1",
            agent_name="hermes",
            embedding_model="test-embedding-model",
            embedding_dimensions=3,
            cfg=cfg,
        )

    assert client.queries == []
    assert client.inserts == []


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="boolean"),
        pytest.param("private-vector-value", id="non-numeric"),
    ],
)
@pytest.mark.parametrize("position", range(3), ids=("first", "middle", "last"))
def test_insert_chunks_rejects_non_finite_or_non_numeric_embedding_values_before_client_writes(
    monkeypatch, invalid_value, position
):
    embedding = [0.1, 0.2, 0.3]
    embedding[position] = invalid_value
    chunk = _chunk(embedding=embedding, text="private-source-content")
    client = _StoreFakeClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    with pytest.raises(ValueError, match="embedding") as exc_info:
        bigquery_store.insert_chunks(
            [chunk],
            user_id="user-1",
            agent_name="hermes",
            embedding_model="test-embedding-model",
            embedding_dimensions=3,
            cfg=cfg,
        )

    error = str(exc_info.value)
    assert "private-source-content" not in error
    assert "private-vector-value" not in error
    assert client.queries == []
    assert client.inserts == []


def test_finalize_source_revision_atomically_activates_chunks_and_source_metadata(monkeypatch):
    old_source = _source(
        source_id="source'; --",
        revision="revision-old",
        content_hash="hash-old",
        relative_path="notes/old.md",
        metadata={"label": "old"},
        user_id="user'; --",
        agent_name="hermes",
        is_active=True,
    )
    new_source = _source(
        source_id="source'; --",
        revision="revision-new",
        content_hash="hash-new",
        relative_path="notes/new'; --.md",
        metadata={"label": "new"},
    )
    chunks = [
        {
            "chunk_id": "old-active",
            "source_id": new_source["source_id"],
            "user_id": "user'; --",
            "agent_name": "hermes",
            "is_active": True,
        },
        {
            "chunk_id": "new-1'; --",
            "source_id": new_source["source_id"],
            "user_id": "user'; --",
            "agent_name": "hermes",
            "is_active": False,
        },
    ]
    client = _LifecycleFakeClient([old_source], chunks)
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    bigquery_store.finalize_source_revision(
        new_source["source_id"],
        ["new-1'; --"],
        source=new_source,
        user_id="user'; --",
        agent_name="hermes",
        cfg=cfg,
    )

    assert client.active_chunk_ids() == {"new-1'; --"}
    persisted = client.sources[new_source["source_id"]]
    assert {key: persisted[key] for key in new_source} == new_source
    assert persisted["is_active"] is True
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "BEGIN TRANSACTION" in sql and "COMMIT TRANSACTION" in sql
    assert "document_chunks" in sql and "document_sources" in sql
    for value in (*new_source.values(), "user'; --", "new-1'; --"):
        if isinstance(value, str):
            assert value not in sql
    parameters = _query_parameters(job_config)
    assert parameters["revision"] == "revision-new"
    assert parameters["content_hash"] == "hash-new"
    assert parameters["active_chunk_ids"] == ["new-1'; --"]
    assert json.loads(parameters["metadata"]) == {"label": "new"}


def test_finalize_source_revision_preserves_source_and_chunks_when_new_set_is_incomplete(
    monkeypatch,
):
    old_source = _source(
        source_id="source'; --",
        revision="revision-old",
        content_hash="hash-old",
        metadata={"label": "old"},
        user_id="user'; --",
        agent_name="hermes",
        is_active=True,
    )
    new_source = _source(
        source_id="source'; --",
        revision="revision-new",
        content_hash="hash-new",
        metadata={"label": "new"},
    )
    chunks = [
        {
            "chunk_id": "old-active",
            "source_id": "source'; --",
            "user_id": "user'; --",
            "agent_name": "hermes",
            "is_active": True,
        },
        {
            "chunk_id": "new-present",
            "source_id": "source'; --",
            "user_id": "user'; --",
            "agent_name": "hermes",
            "is_active": False,
        },
    ]
    client = _LifecycleFakeClient([old_source], chunks)
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    with pytest.raises(RuntimeError, match="missing 1 expected chunk"):
        bigquery_store.finalize_source_revision(
            "source'; --",
            ["new-present", "new-missing'; --"],
            source=new_source,
            user_id="user'; --",
            agent_name="hermes",
            cfg=cfg,
        )

    assert client.sources[old_source["source_id"]] == old_source
    assert client.active_chunk_ids() == {"old-active"}
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "source'; --" not in sql
    assert "new-missing'; --" not in sql
    assert _query_parameters(job_config)["active_chunk_ids"] == [
        "new-present",
        "new-missing'; --",
    ]


def test_finalize_source_revision_switches_active_set_only_after_completeness_proof(monkeypatch):
    source = _source(revision="revision-2", content_hash="hash-2")
    chunks = [
        {
            "chunk_id": "old-active",
            "source_id": "source-1",
            "user_id": "user-1",
            "agent_name": "hermes",
            "is_active": True,
        },
        {
            "chunk_id": "new-1",
            "source_id": "source-1",
            "user_id": "user-1",
            "agent_name": "hermes",
            "is_active": False,
        },
        {
            "chunk_id": "new-2",
            "source_id": "source-1",
            "user_id": "user-1",
            "agent_name": "hermes",
            "is_active": False,
        },
    ]
    client = _LifecycleFakeClient([], chunks)
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    bigquery_store.finalize_source_revision(
        "source-1",
        ["new-1", "new-2"],
        source=source,
        user_id="user-1",
        agent_name="hermes",
        cfg=cfg,
    )

    assert client.active_chunk_ids() == {"new-1", "new-2"}
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "ASSERT" in sql and "SET is_active" in sql
    assert "source-1" not in sql
    assert "new-1" not in sql
    assert _query_parameters(job_config)["active_chunk_ids"] == ["new-1", "new-2"]


def test_deactivate_missing_sources_requires_explicit_prune(monkeypatch):
    client = _StoreFakeClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    with pytest.raises(ValueError, match="prune=True"):
        bigquery_store.deactivate_missing_sources(
            "corpus-1",
            ["source-1"],
            user_id="user-1",
            agent_name="hermes",
            cfg=cfg,
        )

    assert client.queries == []


def test_deactivate_missing_sources_rejects_limited_runs_before_client_writes(monkeypatch):
    client = _StoreFakeClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    with pytest.raises(ValueError, match="limited run"):
        bigquery_store.deactivate_missing_sources(
            "corpus-1",
            ["source-1"],
            user_id="user-1",
            agent_name="hermes",
            prune=True,
            limited=True,
            cfg=cfg,
        )

    assert client.queries == []


def test_deactivate_missing_sources_parameterizes_scope_and_deactivates_chunks(monkeypatch):
    client = _StoreFakeClient([[]])
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    bigquery_store.deactivate_missing_sources(
        "corpus'; --",
        ["seen-1", "seen'; --"],
        user_id="user'; --",
        agent_name="agent'; --",
        prune=True,
        cfg=cfg,
    )

    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "BEGIN TRANSACTION" in sql and "COMMIT TRANSACTION" in sql
    assert "document_chunks" in sql and "document_sources" in sql
    for value in ("corpus'; --", "seen-1", "seen'; --", "user'; --", "agent'; --"):
        assert value not in sql
    assert _query_parameters(job_config) == {
        "corpus_id": "corpus'; --",
        "seen_source_ids": ["seen-1", "seen'; --"],
        "user_id": "user'; --",
        "agent_name": "agent'; --",
    }


@pytest.mark.parametrize("dataset_id", ["123dataset", "7", "a" * 1024])
def test_ensure_tables_accepts_valid_dataset_identifiers(monkeypatch, dataset_id):
    client = _FakeBigQueryClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset=dataset_id)

    bigquery_store.ensure_tables(cfg)

    assert client.queries
    assert all(f"`test-project.{dataset_id}." in sql for sql in client.queries)


@pytest.mark.parametrize(
    "dataset_id",
    [
        pytest.param("", id="empty"),
        pytest.param("a" * 1025, id="too-long"),
        pytest.param("invalid-dataset", id="punctuation"),
        pytest.param("test dataset", id="whitespace"),
        pytest.param("project.dataset", id="qualification"),
        pytest.param("test_dataset/*comment*/", id="comment"),
        pytest.param("`test_dataset`", id="backticks"),
        pytest.param("test_dataset;", id="semicolon"),
    ],
)
def test_ensure_tables_rejects_invalid_dataset_identifiers_before_query(monkeypatch, dataset_id):
    client = _FakeBigQueryClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset=dataset_id)

    with pytest.raises(ValueError, match=r"Invalid BigQuery bq_dataset"):
        bigquery_store.ensure_tables(cfg)

    assert client.queries == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project", "test-project` DROP TABLE victims; --"),
        ("project", "test-project;DROP TABLE victims"),
        ("project", "test-project` -- comment"),
        ("project", "test project"),
        ("project", "other-project.test_dataset"),
        ("bq_dataset", "test_dataset` DROP TABLE victims; --"),
        ("bq_dataset", "test_dataset;DROP TABLE victims"),
        ("bq_dataset", "test_dataset/*comment*/"),
        ("bq_dataset", "test dataset"),
        ("bq_dataset", "other_project.test_dataset"),
    ],
)
def test_ensure_tables_rejects_adversarial_ddl_identifiers_before_query(monkeypatch, field, value):
    client = _FakeBigQueryClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = (
        HermesMemoryConfig(project=value, bq_dataset="test_dataset")
        if field == "project"
        else HermesMemoryConfig(project="test-project", bq_dataset=value)
    )

    with pytest.raises(ValueError, match=rf"Invalid BigQuery {field}"):
        bigquery_store.ensure_tables(cfg)

    assert client.queries == []


def test_ensure_tables_executes_document_ddl_idempotently(monkeypatch):
    client = _FakeBigQueryClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", bq_dataset="test_dataset")

    bigquery_store.ensure_tables(cfg)
    bigquery_store.ensure_tables(cfg)

    source_queries = [sql for sql in client.queries if ".document_sources`" in sql]
    chunk_queries = [sql for sql in client.queries if ".document_chunks`" in sql]
    assert len(source_queries) == 2
    assert len(chunk_queries) == 2
    assert all("CREATE TABLE IF NOT EXISTS" in sql for sql in source_queries + chunk_queries)
    assert all("`test-project.test_dataset." in sql for sql in source_queries + chunk_queries)
