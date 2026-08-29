import json
import re
from dataclasses import FrozenInstanceError
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
    ("source_kind", "STRING", "REQUIRED"),
    ("content_kind", "STRING", "REQUIRED"),
    ("relative_path", "STRING", "REQUIRED"),
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


def test_document_chunk_schema_denormalizes_required_source_retrieval_fields():
    # Integration contract: Task8 chunk row writers must populate these REQUIRED fields.
    required_source_fields = {
        ("source_kind", "STRING", "REQUIRED"),
        ("content_kind", "STRING", "REQUIRED"),
        ("relative_path", "STRING", "REQUIRED"),
    }
    schemas = (
        _ddl_fields(bigquery_store.DDL_DOCUMENT_CHUNKS, "document_chunks"),
        _ddl_fields((ROOT / "bigquery" / "schema.sql").read_text(), "document_chunks"),
        _json_fields(ROOT / "terraform" / "schemas" / "document_chunks.json"),
    )

    for schema in schemas:
        assert required_source_fields <= set(schema)


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
    def result(self):
        return None


class _FakeBigQueryClient:
    def __init__(self):
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        return _FakeJob()


class _FakeSearchJob:
    def __init__(self, rows=()):
        self._rows = rows

    def result(self):
        return self._rows


class _FakeSearchClient:
    def __init__(self, rows=()):
        self.rows = rows
        self.queries = []

    def query(self, sql, *, job_config):
        self.queries.append((sql, job_config))
        return _FakeSearchJob(self.rows)


def test_search_document_chunks_uses_direct_base_table_query(monkeypatch):
    client = _FakeSearchClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(
        project="test-project",
        bq_dataset="test_dataset",
        document_embedding_model="test-model",
        document_embedding_dimensions=3,
    )

    bigquery_store.search_document_chunks(
        [0.25, -0.5, 1],
        user_id="user",
        agent_name="agent",
        corpus_id="corpus",
        source_kind="repository",
        content_kind="markdown",
        top_k=7,
        cfg=cfg,
    )

    sql = client.queries[0][0]
    base_query_match = re.search(
        r"FROM VECTOR_SEARCH\(\s*"
        r"\(\s*(SELECT\b.*?\bFROM\s+"
        r"`test-project\.test_dataset\.document_chunks`\s+AS\s+chunks\s+"
        r"WHERE\b.*?)\s*\),\s*'embedding'",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert base_query_match, "first VECTOR_SEARCH argument must query the physical chunks table"
    base_query = base_query_match.group(1)
    assert len(re.findall(r"\bSELECT\b", base_query, flags=re.IGNORECASE)) == 1
    assert len(re.findall(r"\bFROM\b", base_query, flags=re.IGNORECASE)) == 1
    assert len(re.findall(r"\bWHERE\b", base_query, flags=re.IGNORECASE)) == 1
    assert not re.search(
        r"\b(?:WITH|JOIN|UNION|GROUP\s+BY|HAVING|QUALIFY|ORDER\s+BY|LIMIT)\b",
        base_query,
        flags=re.IGNORECASE,
    )
    assert "filtered_chunks" not in base_query


def test_search_document_chunks_uses_exact_prefiltered_parameterized_vector_search(monkeypatch):
    client = _FakeSearchClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(
        project="test-project",
        bq_dataset="test_dataset",
        document_embedding_model="test-model",
        document_embedding_dimensions=3,
    )

    results = bigquery_store.search_document_chunks(
        [0.25, -0.5, 1],
        user_id="user' OR TRUE --",
        agent_name="agent`); DROP TABLE x; --",
        corpus_id="corpus/*keep parameterized*/",
        source_kind="repository' UNION ALL SELECT",
        content_kind="markdown; DELETE",
        top_k=7,
        cfg=cfg,
    )

    assert results == []
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    normalized_sql = " ".join(sql.split())
    assert "WITH " not in normalized_sql.upper()
    assert "SELECT @query_embedding AS embedding" in normalized_sql
    assert "FROM VECTOR_SEARCH(" in normalized_sql
    assert "query_column_to_search => 'embedding'" in normalized_sql
    assert "distance_type => 'COSINE'" in normalized_sql
    assert "options => '{\"use_brute_force\":true}'" in normalized_sql
    assert "VECTOR INDEX" not in normalized_sql.upper()
    assert "FROM `test-project.test_dataset.document_chunks` AS chunks" in normalized_sql
    assert "document_sources" not in normalized_sql
    assert "chunks.relative_path AS source_path" in normalized_sql
    assert "chunks.is_active = TRUE" in normalized_sql
    base_query_end = normalized_sql.index("), 'embedding'")
    for predicate in (
        "chunks.user_id = @user_id",
        "chunks.agent_name = @agent_name",
        "chunks.embedding_model = @embedding_model",
        "chunks.embedding_dimensions = @embedding_dimensions",
        "chunks.corpus_id = @corpus_id",
        "chunks.source_kind = @source_kind",
        "chunks.content_kind = @content_kind",
    ):
        assert predicate in normalized_sql
        assert normalized_sql.index(predicate) < base_query_end
    for selected_field in (
        "distance",
        "base.content",
        "base.contextual_content",
        "base.citation",
        "base.source_path",
        "base.heading_path",
        "base.symbol",
        "base.start_line",
        "base.end_line",
        "base.chunk_id",
        "base.source_id",
        "base.corpus_id",
    ):
        assert selected_field in normalized_sql
    assert "ORDER BY distance ASC, base.chunk_id ASC" in normalized_sql
    for parameter_name in (
        "user_id",
        "agent_name",
        "embedding_model",
        "embedding_dimensions",
        "corpus_id",
        "source_kind",
        "content_kind",
        "top_k",
    ):
        assert f"@{parameter_name}" in normalized_sql
    assert normalized_sql.index("chunks.is_active = TRUE") < base_query_end
    parameter_values = {
        parameter.name: getattr(parameter, "value", getattr(parameter, "values", None))
        for parameter in job_config.query_parameters
    }
    assert parameter_values == {
        "query_embedding": [0.25, -0.5, 1.0],
        "user_id": "user' OR TRUE --",
        "agent_name": "agent`); DROP TABLE x; --",
        "embedding_model": "test-model",
        "embedding_dimensions": 3,
        "corpus_id": "corpus/*keep parameterized*/",
        "source_kind": "repository' UNION ALL SELECT",
        "content_kind": "markdown; DELETE",
        "top_k": 7,
    }
    for value in parameter_values.values():
        if isinstance(value, str):
            assert value not in sql


def test_search_document_chunks_omits_optional_filters_and_uses_configured_top_k(monkeypatch):
    client = _FakeSearchClient()
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(
        project="test-project",
        bq_dataset="test_dataset",
        document_embedding_dimensions=3,
        document_top_k=9,
    )

    results = bigquery_store.search_document_chunks(
        [0.1, 0.2, 0.3],
        user_id="user",
        agent_name="agent",
        cfg=cfg,
    )

    assert results == []
    sql, job_config = client.queries[0]
    parameter_values = {
        parameter.name: getattr(parameter, "value", getattr(parameter, "values", None))
        for parameter in job_config.query_parameters
    }
    assert "@corpus_id" not in sql
    assert "@source_kind" not in sql
    assert "@content_kind" not in sql
    assert "corpus_id" not in parameter_values
    assert "source_kind" not in parameter_values
    assert "content_kind" not in parameter_values
    assert parameter_values["top_k"] == 9


def test_search_document_chunks_returns_empty_when_bigquery_client_is_unavailable(
    monkeypatch, capsys
):
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: None)
    cfg = HermesMemoryConfig(project="test-project", document_embedding_dimensions=3)

    results = bigquery_store.search_document_chunks(
        [0.1, 0.2, 0.3],
        user_id="user",
        agent_name="agent",
        cfg=cfg,
    )

    assert results == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("embedding", "error"),
    [
        pytest.param([0.1, True, 0.3], TypeError, id="boolean"),
        pytest.param([0.1, "0.2", 0.3], TypeError, id="string"),
        pytest.param([0.1, float("nan"), 0.3], ValueError, id="nan"),
        pytest.param([0.1, float("inf"), 0.3], ValueError, id="positive-infinity"),
        pytest.param([0.1, float("-inf"), 0.3], ValueError, id="negative-infinity"),
        pytest.param([0.1, 0.2], ValueError, id="too-short"),
        pytest.param([0.1, 0.2, 0.3, 0.4], ValueError, id="too-long"),
    ],
)
def test_search_document_chunks_rejects_invalid_embedding_before_client(
    monkeypatch, embedding, error
):
    def fail_client_construction(cfg):
        raise AssertionError("invalid input reached client construction")

    monkeypatch.setattr(bigquery_store, "_bq_client", fail_client_construction)
    cfg = HermesMemoryConfig(
        project="test-project",
        document_embedding_dimensions=3,
    )

    with pytest.raises(error, match="query_embedding"):
        bigquery_store.search_document_chunks(
            embedding,
            user_id="user",
            agent_name="agent",
            cfg=cfg,
        )


@pytest.mark.parametrize(
    ("top_k", "error"),
    [
        pytest.param(True, TypeError, id="boolean"),
        pytest.param(1.0, TypeError, id="float"),
        pytest.param("2", TypeError, id="string"),
        pytest.param(0, ValueError, id="zero"),
        pytest.param(-1, ValueError, id="negative"),
    ],
)
def test_search_document_chunks_rejects_invalid_top_k_without_lossy_coercion(
    monkeypatch, top_k, error
):
    def fail_client_construction(cfg):
        raise AssertionError("invalid input reached client construction")

    monkeypatch.setattr(bigquery_store, "_bq_client", fail_client_construction)
    cfg = HermesMemoryConfig(project="test-project", document_embedding_dimensions=3)

    with pytest.raises(error, match="top_k"):
        bigquery_store.search_document_chunks(
            [0.1, 0.2, 0.3],
            user_id="user",
            agent_name="agent",
            top_k=top_k,
            cfg=cfg,
        )


def test_search_document_chunks_returns_immutable_citation_bearing_results(monkeypatch):
    row = {
        "distance": 0.125,
        "content": "source excerpt",
        "contextual_content": "heading context\nsource excerpt",
        "citation": "docs/guide.md:12-18",
        "source_path": "docs/guide.md",
        "heading_path": ["Guide", "Setup"],
        "symbol": "configure",
        "start_line": 12,
        "end_line": 18,
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "corpus_id": "corpus-1",
    }
    client = _FakeSearchClient([row])
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", document_embedding_dimensions=3)

    results = bigquery_store.search_document_chunks(
        [0.1, 0.2, 0.3],
        user_id="user",
        agent_name="agent",
        cfg=cfg,
    )

    assert results == [
        bigquery_store.DocumentChunkSearchResult(
            distance=0.125,
            content="source excerpt",
            contextual_content="heading context\nsource excerpt",
            citation="docs/guide.md:12-18",
            source_path="docs/guide.md",
            heading_path=("Guide", "Setup"),
            symbol="configure",
            start_line=12,
            end_line=18,
            chunk_id="chunk-1",
            source_id="source-1",
            corpus_id="corpus-1",
        )
    ]
    with pytest.raises(FrozenInstanceError):
        results[0].content = "changed"
    with pytest.raises(TypeError):
        results[0].heading_path[0] = "changed"


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda row: row.pop("citation"), id="missing-field"),
        pytest.param(lambda row: row.update(distance="0.125"), id="distance-type"),
        pytest.param(lambda row: row.update(distance=float("nan")), id="distance-nan"),
        pytest.param(lambda row: row.update(content=None), id="content-type"),
        pytest.param(lambda row: row.update(heading_path=["Guide", 7]), id="heading-type"),
        pytest.param(lambda row: row.update(symbol=7), id="symbol-type"),
        pytest.param(lambda row: row.update(start_line=True), id="line-boolean"),
    ],
)
def test_search_document_chunks_rejects_malformed_rows_without_exposing_content(
    monkeypatch, capsys, mutation
):
    sensitive_content = "PRIVATE SOURCE CONTENT must never appear in errors"
    row = {
        "distance": 0.125,
        "content": sensitive_content,
        "contextual_content": "private context",
        "citation": "docs/guide.md:12-18",
        "source_path": "docs/guide.md",
        "heading_path": ["Guide", "Setup"],
        "symbol": None,
        "start_line": 12,
        "end_line": 18,
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "corpus_id": "corpus-1",
    }
    mutation(row)
    client = _FakeSearchClient([row])
    monkeypatch.setattr(bigquery_store, "_bq_client", lambda cfg: client)
    cfg = HermesMemoryConfig(project="test-project", document_embedding_dimensions=3)

    with pytest.raises(ValueError, match="malformed document search result") as exc_info:
        bigquery_store.search_document_chunks(
            [0.1, 0.2, 0.3],
            user_id="user",
            agent_name="agent",
            cfg=cfg,
        )

    assert sensitive_content not in str(exc_info.value)
    assert sensitive_content not in capsys.readouterr().out


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
