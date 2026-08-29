from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[1]
EVALUATION_CASES = (
    {
        "query": "How are missing chunks replayed during storage recovery?",
        "relative_path": "Operations/agent-memory.md",
        "heading": "Recovery",
        "marker": "Replay only the missing chunk identifiers.",
    },
    {
        "query": "Which function constructs an empty fixture store?",
        "relative_path": "main.go",
        "symbol": "NewStore",
        "marker": "func NewStore() *InMemoryStore",
    },
    {
        "query": "How are snapshots ordered?",
        "relative_path": "main.go",
        "symbol": "(*InMemoryStore).Snapshot",
        "marker": "sort.Strings(ids)",
    },
)


def test_obsidian_fixture_covers_markdown_evaluation_boundaries():
    text = (FIXTURES / "obsidian" / "Operations" / "agent-memory.md").read_text()

    assert text.startswith("---\n")
    assert "tags:\n  - hermes\n  - agent-memory\n  - runbook" in text
    assert "## Storage\n\n### Layout" in text
    assert text.count("### Recovery") == 2
    assert "- Memory Bank facts for personalized recall." in text
    assert "```yaml\nmemory:" in text

    oversized = text.split("## Oversized Incident Narrative\n", 1)[1].split("\n## Runtime", 1)[0]
    assert len(oversized) > 4_000


def test_repo_fixture_covers_code_and_ignore_contracts():
    repo = FIXTURES / "repo"
    go_source = (repo / "main.go").read_text()
    readme = (repo / "README.md").read_text()
    ignored_environment = (repo / ".env.example").read_text().splitlines()

    assert "type Memory struct" in go_source
    assert "type Store interface" in go_source
    assert "func NewStore() *InMemoryStore" in go_source
    assert "func (s *InMemoryStore) Snapshot() []Memory" in go_source
    assert readme.count("## Recovery") == 2
    assert "```bash\ngo run ./main.go\n```" in readme
    assert ".env.example" in (repo / ".gitignore").read_text().splitlines()
    assert ignored_environment
    assert all(line.endswith("=FAKE_TEST_MARKER_ONLY") for line in ignored_environment)


def test_evaluation_cases_point_to_committed_fixture_answers():
    for case in EVALUATION_CASES:
        root = "obsidian" if case["relative_path"].startswith("Operations/") else "repo"
        fixture = FIXTURES / root / case["relative_path"]

        assert fixture.is_file(), case
        assert case["marker"] in fixture.read_text(), case


def test_load_queries_builds_deeply_immutable_contracts_from_json_compatible_yaml(tmp_path):
    import json
    from dataclasses import FrozenInstanceError

    import pytest

    from hermes_memory.evaluation import load_queries

    fixture = tmp_path / "queries.yaml"
    fixture.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "recovery-replay",
                        "query": "How are missing chunks replayed?",
                        "expected_source_path": "Operations/agent-memory.md",
                        "expected_heading": "Recovery",
                    }
                ],
            }
        )
    )

    queries = load_queries(fixture)

    assert len(queries) == 1
    assert queries[0].id == "recovery-replay"
    assert queries[0].expected_source_path == "Operations/agent-memory.md"
    assert queries[0].expected_heading == "Recovery"
    assert queries[0].expected_symbol is None
    with pytest.raises(FrozenInstanceError):
        queries[0].query = "changed"


def test_load_queries_rejects_malformed_or_unapproved_fixtures(tmp_path):
    import json

    import pytest

    from hermes_memory.evaluation import load_queries

    malformed_documents = (
        "not JSON-compatible YAML",
        json.dumps({"version": 1, "queries": []}),
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "outside-policy",
                        "query": "Where is this?",
                        "expected_source_path": "Secrets/private.md",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "duplicate",
                        "query": "First?",
                        "expected_source_path": "README.md",
                    },
                    {
                        "id": "duplicate",
                        "query": "Second?",
                        "expected_source_path": "main.go",
                    },
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "non-string-heading",
                        "query": "Which heading contains this?",
                        "expected_source_path": "README.md",
                        "expected_heading": ["Usage"],
                    }
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "unknown-field",
                        "query": "What is this?",
                        "expected_source_path": "README.md",
                        "source_body": "must never be accepted",
                    }
                ],
            }
        ),
    )

    for index, document in enumerate(malformed_documents):
        fixture = tmp_path / f"bad-{index}.yaml"
        fixture.write_text(document)
        with pytest.raises(ValueError, match="query fixture"):
            load_queries(fixture)


def test_load_queries_rejects_conflicting_duplicate_field_definitions(tmp_path):
    import pytest

    from hermes_memory.evaluation import load_queries

    fixture = tmp_path / "conflicting.yaml"
    fixture.write_text(
        """{
          "version": 1,
          "queries": [{
            "id": "conflicting-definition",
            "query": "First definition?",
            "query": "Second definition?",
            "expected_source_path": "README.md"
          }]
        }"""
    )

    with pytest.raises(ValueError, match="duplicate field 'query'"):
        load_queries(fixture)


def test_search_hit_rejects_non_string_heading_path_values():
    from typing import Any, cast

    import pytest

    from hermes_memory.evaluation import SearchHit

    with pytest.raises(TypeError, match="heading path"):
        SearchHit(
            chunk_id="bad-heading",
            source_path="README.md",
            heading_path=cast(Any, ("Usage", 7)),
            symbol=None,
            start_line=1,
            end_line=1,
            citation="README.md#L1",
            distance=0.1,
        )


def test_citation_validation_accepts_local_and_commit_pinned_github_ranges_only():
    from hermes_memory.evaluation import SearchHit, is_valid_citation

    local = SearchHit(
        chunk_id="local-1",
        source_path="Operations/agent-memory.md",
        heading_path=("Agent Memory Operations", "Storage", "Recovery"),
        symbol=None,
        start_line=30,
        end_line=38,
        citation="Operations/agent-memory.md#L30-L38",
        distance=0.1,
    )
    github = SearchHit(
        chunk_id="git-1",
        source_path="main.go",
        heading_path=(),
        symbol="NewStore",
        start_line=25,
        end_line=28,
        citation=(
            "https://github.com/example/repository/blob/"
            "0123456789abcdef0123456789abcdef01234567/main.go#L25-L28"
        ),
        distance=0.2,
    )

    assert is_valid_citation(local)
    assert is_valid_citation(github)

    invalid = (
        SearchHit(**{**local.__dict__, "citation": "Operations/agent-memory.md"}),
        SearchHit(**{**local.__dict__, "citation": "Operations/agent-memory.md#L38-L30"}),
        SearchHit(**{**local.__dict__, "citation": "Operations/agent-memory.md#L29-L38"}),
        SearchHit(
            **{
                **github.__dict__,
                "citation": github.citation.replace(
                    "0123456789abcdef0123456789abcdef01234567", "main"
                ),
            }
        ),
        SearchHit(
            **{**github.__dict__, "citation": github.citation.replace("main.go", "README.md")}
        ),
    )
    assert [is_valid_citation(hit) for hit in invalid] == [False] * len(invalid)


def test_citation_validation_rejects_arbitrary_urls_disguised_as_local_paths():
    from hermes_memory.evaluation import SearchHit, is_valid_citation

    for source_path in (
        "https://example.com/repository/file.md",
        "file:///tmp/file.md",
        "custom:repository/file.md",
    ):
        hit = SearchHit(
            chunk_id="url-1",
            source_path=source_path,
            heading_path=(),
            symbol=None,
            start_line=3,
            end_line=3,
            citation=f"{source_path}#L3",
            distance=0.1,
        )
        assert is_valid_citation(hit) is False


def test_citation_validation_requires_l_prefix_on_both_range_endpoints():
    from hermes_memory.evaluation import SearchHit, is_valid_citation

    local = SearchHit(
        chunk_id="local-range",
        source_path="main.go",
        heading_path=(),
        symbol=None,
        start_line=25,
        end_line=28,
        citation="main.go#L25-28",
        distance=0.1,
    )
    github = SearchHit(
        **{
            **local.__dict__,
            "chunk_id": "github-range",
            "citation": (
                "https://github.com/example/repository/blob/"
                "0123456789abcdef0123456789abcdef01234567/main.go#L25-28"
            ),
        }
    )

    assert is_valid_citation(local) is False
    assert is_valid_citation(github) is False


def test_evaluator_computes_deterministic_metrics_deduplicates_hits_and_passes_pilot_gates():
    from dataclasses import FrozenInstanceError

    import pytest

    from hermes_memory.evaluation import (
        EvaluationQuery,
        SearchHit,
        SearchResponse,
        evaluate_queries,
    )

    queries = (
        EvaluationQuery("q1", "Recovery?", "Operations/agent-memory.md", "Recovery"),
        EvaluationQuery("q2", "Constructor?", "main.go", expected_symbol="NewStore"),
        EvaluationQuery("q3", "Design?", "README.md"),
        EvaluationQuery("q4", "Usage?", "README.md", "Usage"),
        EvaluationQuery("q5", "Missing?", "main.go", expected_symbol="missing"),
    )

    def hit(
        chunk_id,
        path,
        *,
        distance=0.1,
        heading=(),
        symbol=None,
        start_line=1,
    ):
        return SearchHit(
            chunk_id=chunk_id,
            source_path=path,
            heading_path=heading,
            symbol=symbol,
            start_line=start_line,
            end_line=start_line,
            citation=f"{path}#L{start_line}",
            distance=distance,
        )

    responses = {
        "q1": SearchResponse(
            hits=(hit("recovery", "Operations/agent-memory.md", heading=("Recovery",)),),
            bytes_processed=10,
        ),
        "q2": SearchResponse(
            hits=(
                hit("expected", "main.go", symbol="NewStore"),
                hit("decoy", "README.md"),
                hit("expected", "main.go", symbol="NewStore"),
            ),
            bytes_processed=20,
        ),
        "q3": SearchResponse(hits=(hit("design", "README.md"),), bytes_processed=30),
        "q4": SearchResponse(
            hits=(
                hit("z-target", "README.md", heading=("Usage",)),
                hit("d", "main.go"),
                hit("b", "main.go"),
                hit("a", "main.go"),
                hit("c", "main.go"),
            ),
            bytes_processed=40,
        ),
        "q5": SearchResponse(hits=(), bytes_processed=50),
    }
    clock_values = iter((0.0, 0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 1.0, 1.0, 2.5))

    report = evaluate_queries(
        queries, lambda query: responses[query.id], clock=clock_values.__next__
    )

    assert report.query_count == 5
    assert report.recall_at_5 == 0.8
    assert report.mrr == 0.54
    assert report.citation_validity == 1.0
    assert report.latency_p50_seconds == pytest.approx(0.3)
    assert report.latency_p95_seconds == pytest.approx(1.28)
    assert report.total_bytes_processed == 150
    assert report.truncation_count == 0
    assert report.zero_truncation_rate == 1.0
    assert tuple(result.relevant_rank for result in report.results) == (1, 2, 1, 5, None)
    assert tuple(result.returned_hit_count for result in report.results) == (1, 2, 1, 5, 0)
    assert report.pilot_gate.passed is True
    assert report.pilot_gate.label == "pilot gates (not global SLAs)"
    with pytest.raises(FrozenInstanceError):
        report.recall_at_5 = 0.0


def test_evaluator_fails_closed_for_empty_input_missing_usage_and_failed_gate():
    import pytest

    from hermes_memory.evaluation import (
        EvaluationQuery,
        SearchHit,
        SearchResponse,
        evaluate_queries,
    )

    with pytest.raises(ValueError, match="at least one"):
        evaluate_queries((), lambda query: SearchResponse())

    query = EvaluationQuery("miss", "Miss?", "README.md")
    invalid_hit = SearchHit(
        chunk_id="bad-citation",
        source_path="main.go",
        heading_path=(),
        symbol=None,
        start_line=1,
        end_line=1,
        citation="main.go#L2",
        distance=0.1,
    )
    clock_values = iter((0.0, 1.6))

    report = evaluate_queries(
        (query,),
        lambda candidate: SearchResponse(hits=(invalid_hit,), truncated=True),
        clock=clock_values.__next__,
    )

    assert report.recall_at_5 == 0.0
    assert report.mrr == 0.0
    assert report.citation_validity == 0.0
    assert report.total_bytes_processed is None
    assert report.truncation_count == 1
    assert report.zero_truncation_rate == 0.0
    assert report.latency_p95_seconds == pytest.approx(1.6)
    assert report.pilot_gate.passed is False
    assert report.pilot_gate.failed_metrics == (
        "recall_at_5",
        "citation_validity",
        "truncation_count",
        "latency_p95_seconds",
    )


def test_offline_runtime_probe_matches_exact_task5_task9_api_and_is_lazy(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    from google import genai

    from hermes_memory import bigquery_store, config
    from hermes_memory.evaluation import EvaluationQuery, make_runtime_query_executor

    calls = []
    cfg = SimpleNamespace(
        project="test-project",
        location="test-location",
        document_embedding_model="test-model",
        document_embedding_dimensions=3,
    )

    class FakeEmbeddingClient:
        def __init__(
            self,
            *,
            client,
            model="gemini-embedding-001",
            dimensions=768,
            task_type="RETRIEVAL_DOCUMENT",
            concurrency=4,
            max_attempts=3,
            initial_retry_delay=1.0,
            sleep=None,
        ):
            calls.append(
                (
                    "embedding-client",
                    {
                        "client": client,
                        "model": model,
                        "dimensions": dimensions,
                        "task_type": task_type,
                        "concurrency": concurrency,
                        "max_attempts": max_attempts,
                        "initial_retry_delay": initial_retry_delay,
                        "sleep": sleep,
                    },
                )
            )

        def embed(self, text):
            calls.append(("embed", text))
            return SimpleNamespace(values=(0.1, 0.2, 0.3), truncated=False)

    def fake_search(
        values,
        *,
        user_id,
        agent_name,
        corpus_id=None,
        source_kind=None,
        content_kind=None,
        top_k=None,
        cfg=None,
    ):
        calls.append(
            (
                "search",
                values,
                {
                    "user_id": user_id,
                    "agent_name": agent_name,
                    "corpus_id": corpus_id,
                    "source_kind": source_kind,
                    "content_kind": content_kind,
                    "top_k": top_k,
                    "cfg": cfg,
                },
            )
        )
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                source_path="main.go",
                heading_path=(),
                symbol="NewStore",
                start_line=25,
                end_line=28,
                citation="main.go#L25-L28",
                distance=0.1,
                content="BODY_MARKER_SHOULD_NOT_APPEAR",
            )
        ]

    monkeypatch.setenv("HERMES_MEMORY_EVALUATION_USER_ID", "test-user")
    monkeypatch.setenv("HERMES_MEMORY_EVALUATION_AGENT_NAME", "test-agent")
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(("sdk", kwargs)) or object())
    monkeypatch.setattr(bigquery_store, "search_document_chunks", fake_search, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "hermes_memory.embeddings",
        SimpleNamespace(VertexEmbeddingClient=FakeEmbeddingClient),
    )

    execute = make_runtime_query_executor()
    assert [call[0] for call in calls] == ["sdk", "embedding-client"]
    response = execute(EvaluationQuery("constructor", "Constructor?", "main.go"))

    embedding_call = next(call for call in calls if call[0] == "embedding-client")
    assert embedding_call[1]["task_type"] == "RETRIEVAL_QUERY"
    assert [call for call in calls if call[0] == "embed"] == [("embed", "Constructor?")]
    search_calls = [call for call in calls if call[0] == "search"]
    assert len(search_calls) == 1
    search_call = search_calls[0]
    assert calls.index(("embed", "Constructor?")) < calls.index(search_call)
    assert search_call[1] == (0.1, 0.2, 0.3)
    assert search_call[2]["user_id"] == "test-user"
    assert search_call[2]["agent_name"] == "test-agent"
    assert search_call[2]["top_k"] == 5
    assert response.hits[0].chunk_id == "chunk-1"
    assert not hasattr(response.hits[0], "content")
    assert response.truncated is False


def test_runtime_query_executor_fails_clearly_when_pending_backend_is_absent(monkeypatch):
    import pytest

    from hermes_memory import evaluation
    from hermes_memory.evaluation import make_runtime_query_executor

    real_import_module = evaluation.importlib.import_module

    def import_without_pending_backend(name):
        if name == "hermes_memory.embeddings":
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setenv("HERMES_MEMORY_EVALUATION_USER_ID", "test-user")
    monkeypatch.setattr(evaluation.importlib, "import_module", import_without_pending_backend)
    with pytest.raises(RuntimeError, match="Task 5/Task 9 backend is unavailable"):
        make_runtime_query_executor()


def test_evaluate_docs_cli_uses_injected_factory_and_writes_safe_exact_json(monkeypatch, tmp_path):
    import json

    from click.testing import CliRunner

    from hermes_memory import evaluation
    from hermes_memory.cli import main

    queries_path = tmp_path / "queries.yaml"
    output_path = tmp_path / "report.json"
    queries_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "constructor",
                        "query": "Which function constructs the fixture store?",
                        "expected_source_path": "main.go",
                        "expected_symbol": "NewStore",
                    }
                ],
            }
        )
    )
    calls = []

    def fake_factory():
        calls.append("factory")

        def execute(query):
            calls.append(query.id)
            return evaluation.SearchResponse(
                hits=(
                    evaluation.SearchHit(
                        chunk_id="chunk-1",
                        source_path="main.go",
                        heading_path=(),
                        symbol="NewStore",
                        start_line=25,
                        end_line=28,
                        citation="main.go#L25-L28",
                        distance=0.1,
                    ),
                )
            )

        return execute

    monkeypatch.setattr(evaluation, "make_runtime_query_executor", fake_factory)

    result = CliRunner().invoke(
        main,
        ["evaluate-docs", "--queries", str(queries_path), "--json", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert calls == ["factory", "constructor"]
    report_text = output_path.read_text()
    report = json.loads(report_text)
    assert report_text.endswith("\n")
    assert report["schema_version"] == 1
    assert report["metrics"]["recall_at_5"] == 1.0
    assert report["metrics"]["citation_validity"] == 1.0
    assert report["pilot_gate"]["label"] == "pilot gates (not global SLAs)"
    assert report["pilot_gate"]["thresholds"] == {
        "citation_validity_required": 1.0,
        "latency_p95_seconds_max": 1.5,
        "recall_at_5_min": 0.8,
        "truncation_count_max": 0,
    }
    assert "Which function constructs" not in report_text
    assert "BODY_MARKER_SHOULD_NOT_APPEAR" not in report_text


def test_evaluate_docs_cli_writes_failed_gate_report_before_exiting_nonzero(monkeypatch, tmp_path):
    import json

    from click.testing import CliRunner

    from hermes_memory import evaluation
    from hermes_memory.cli import main

    queries_path = tmp_path / "queries.yaml"
    output_path = tmp_path / "report.json"
    queries_path.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "missing-answer",
                        "query": "Where is the missing answer?",
                        "expected_source_path": "README.md",
                    }
                ],
            }
        )
    )
    output_path.write_text("stale-report")
    replacements = []
    real_replace = evaluation.os.replace

    def recording_replace(source, target):
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(evaluation.os, "replace", recording_replace)
    monkeypatch.setattr(
        evaluation,
        "make_runtime_query_executor",
        lambda: lambda query: evaluation.SearchResponse(),
    )

    result = CliRunner().invoke(
        main,
        ["evaluate-docs", "--queries", str(queries_path), "--json", str(output_path)],
    )

    assert result.exit_code != 0
    assert result.output == ""
    assert len(replacements) == 1
    temporary_path, replaced_path = replacements[0]
    assert temporary_path.parent == output_path.parent
    assert temporary_path != output_path
    assert replaced_path == output_path
    report = json.loads(output_path.read_text())
    assert report["pilot_gate"]["passed"] is False
    assert report["pilot_gate"]["failed_metrics"] == ["recall_at_5", "citation_validity"]


def test_evaluate_docs_cli_rejects_malformed_fixture_without_touching_output(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from hermes_memory import evaluation
    from hermes_memory.cli import main

    queries_path = tmp_path / "bad.yaml"
    output_path = tmp_path / "report.json"
    queries_path.write_text('{"version": 1, "queries": []}')
    output_path.write_text("preserve-existing-output")
    monkeypatch.setattr(
        evaluation,
        "make_runtime_query_executor",
        lambda: (_ for _ in ()).throw(AssertionError("runtime factory must not run")),
    )

    result = CliRunner().invoke(
        main,
        ["evaluate-docs", "--queries", str(queries_path), "--json", str(output_path)],
    )

    assert result.exit_code != 0
    assert "query fixture" in result.output
    assert output_path.read_text() == "preserve-existing-output"


def test_committed_pilot_fixture_has_ten_to_fifteen_approved_non_sensitive_queries():
    from hermes_memory.evaluation import APPROVED_PILOT_SOURCE_PATHS, load_queries

    queries = load_queries(ROOT / "evaluation" / "queries.yaml")

    assert 10 <= len(queries) <= 15
    assert {query.expected_source_path for query in queries} <= APPROVED_PILOT_SOURCE_PATHS
    assert all("credential" not in query.query.lower() for query in queries)
    assert all("secret" not in query.query.lower() for query in queries)
    assert all(
        query.expected_symbol is None
        for query in queries
        if query.expected_source_path == "main.go"
    )

    source_text = {
        "Operations/agent-memory.md": (
            FIXTURES / "obsidian" / "Operations" / "agent-memory.md"
        ).read_text(),
        "main.go": (FIXTURES / "repo" / "main.go").read_text(),
        "README.md": (FIXTURES / "repo" / "README.md").read_text(),
    }
    for query in queries:
        if query.expected_heading is not None:
            assert query.expected_heading in source_text[query.expected_source_path]
        if query.expected_symbol is not None:
            symbol_name = query.expected_symbol.rsplit(".", 1)[-1]
            assert symbol_name in source_text[query.expected_source_path]
