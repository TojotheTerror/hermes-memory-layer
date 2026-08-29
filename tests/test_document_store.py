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


def _cluster_fields(ddl: str) -> list[str]:
    match = re.search(r"CLUSTER BY ([^;]+);", ddl)
    assert match
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


def test_terraform_defines_document_tables_without_indexes():
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
        assert f'schemas/{table_name}.json' in resource.group(1)
        assert "deletion_protection = false" in resource.group(1)
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
