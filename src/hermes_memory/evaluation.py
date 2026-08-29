"""Offline document retrieval evaluation contracts."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APPROVED_PILOT_SOURCE_PATHS = frozenset(
    {
        "Operations/agent-memory.md",
        "main.go",
        "README.md",
    }
)


@dataclass(frozen=True)
class EvaluationQuery:
    """One non-sensitive pilot query and its expected source label."""

    id: str
    query: str
    expected_source_path: str
    expected_heading: str | None = None
    expected_symbol: str | None = None


@dataclass(frozen=True)
class SearchHit:
    """The citation metadata needed for evaluation; source bodies are excluded."""

    chunk_id: str
    source_path: str
    heading_path: tuple[str, ...]
    symbol: str | None
    start_line: int
    end_line: int
    citation: str
    distance: float

    def __post_init__(self) -> None:
        if isinstance(self.heading_path, (str, bytes)):
            raise TypeError("hit heading path must be a sequence of strings")
        heading_path = tuple(self.heading_path)
        if any(type(value) is not str for value in heading_path):
            raise TypeError("hit heading path must contain only strings")
        object.__setattr__(self, "heading_path", heading_path)
        if type(self.start_line) is not int or type(self.end_line) is not int:
            raise TypeError("hit line numbers must be integers")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("hit line range must be positive and ordered")
        if type(self.distance) not in (int, float) or not math.isfinite(self.distance):
            raise ValueError("hit distance must be a finite number")
        if not self.chunk_id or not self.source_path or not self.citation:
            raise ValueError("hit identifiers, source path, and citation must be non-empty")


_LINE_RANGE = r"#L(?P<start>[1-9][0-9]*)(?:-L(?P<end>[1-9][0-9]*))?"
_LOCAL_CITATION = re.compile(rf"(?P<path>[^#]+){_LINE_RANGE}\Z")
_GITHUB_CITATION = re.compile(
    rf"https://github\.com/[A-Za-z0-9][A-Za-z0-9._-]*/"
    rf"[A-Za-z0-9][A-Za-z0-9._-]*/blob/"
    rf"(?P<commit>[0-9a-fA-F]{{40}})/(?P<path>[^#]+){_LINE_RANGE}\Z"
)


def _safe_relative_path(path: str) -> bool:
    return path in APPROVED_PILOT_SOURCE_PATHS


def is_valid_citation(hit: SearchHit) -> bool:
    """Validate exact local or commit-pinned GitHub inclusive line citations."""
    github_match = _GITHUB_CITATION.fullmatch(hit.citation)
    match = github_match or _LOCAL_CITATION.fullmatch(hit.citation)
    if match is None:
        return False
    path = match.group("path")
    if (
        path not in APPROVED_PILOT_SOURCE_PATHS
        or hit.source_path not in APPROVED_PILOT_SOURCE_PATHS
    ):
        return False
    if github_match is not None and int(github_match.group("commit"), 16) == 0:
        return False
    end = match.group("end")
    cited_start = int(match.group("start"))
    cited_end = cited_start if end is None else int(end)
    return (
        _safe_relative_path(path)
        and path == hit.source_path
        and cited_start <= cited_end
        and (cited_start, cited_end) == (hit.start_line, hit.end_line)
    )


@dataclass(frozen=True)
class SearchResponse:
    """One injected search execution result without source content."""

    hits: tuple[SearchHit, ...] = ()
    bytes_processed: int | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        if not all(isinstance(hit, SearchHit) for hit in self.hits):
            raise TypeError("search response hits must be SearchHit records")
        if self.bytes_processed is not None and (
            type(self.bytes_processed) is not int or self.bytes_processed < 0
        ):
            raise ValueError("bytes_processed must be a non-negative integer or null")
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be a boolean")


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Safe per-query measurements; query text and source bodies are omitted."""

    query_id: str
    relevant_rank: int | None
    returned_hit_count: int
    valid_citation_count: int
    latency_seconds: float
    bytes_processed: int | None
    truncated: bool


@dataclass(frozen=True)
class PilotGateVerdict:
    """Verdict for the initial pilot only, never a global service-level target."""

    passed: bool
    failed_metrics: tuple[str, ...]
    label: str = "pilot gates (not global SLAs)"

    def __post_init__(self) -> None:
        object.__setattr__(self, "failed_metrics", tuple(self.failed_metrics))


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable aggregate document retrieval evaluation report."""

    query_count: int
    recall_at_5: float
    mrr: float
    citation_validity: float
    latency_p50_seconds: float
    latency_p95_seconds: float
    total_bytes_processed: int | None
    truncation_count: int
    zero_truncation_rate: float
    results: tuple[QueryEvaluationResult, ...]
    pilot_gate: PilotGateVerdict

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, body-free JSON contract."""
        return {
            "schema_version": 1,
            "metrics": {
                "query_count": self.query_count,
                "recall_at_5": self.recall_at_5,
                "mrr": self.mrr,
                "citation_validity": self.citation_validity,
                "latency_p50_seconds": self.latency_p50_seconds,
                "latency_p95_seconds": self.latency_p95_seconds,
                "total_bytes_processed": self.total_bytes_processed,
                "truncation_count": self.truncation_count,
                "zero_truncation_rate": self.zero_truncation_rate,
            },
            "pilot_gate": {
                "label": self.pilot_gate.label,
                "passed": self.pilot_gate.passed,
                "failed_metrics": list(self.pilot_gate.failed_metrics),
                "thresholds": {
                    "recall_at_5_min": 0.80,
                    "citation_validity_required": 1.00,
                    "truncation_count_max": 0,
                    "latency_p95_seconds_max": 1.5,
                },
            },
            "results": [
                {
                    "query_id": result.query_id,
                    "relevant_rank": result.relevant_rank,
                    "returned_hit_count": result.returned_hit_count,
                    "valid_citation_count": result.valid_citation_count,
                    "latency_seconds": result.latency_seconds,
                    "bytes_processed": result.bytes_processed,
                    "truncated": result.truncated,
                }
                for result in self.results
            ],
        }


def write_report_json(report: EvaluationReport, path: str | Path) -> None:
    """Atomically replace an explicit output path with the exact JSON report."""
    output_path = Path(path)
    payload = (
        json.dumps(
            report.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            os.fchmod(temporary.fileno(), 0o600)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _deduplicate_and_rank(hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
    ordered = sorted(
        hits,
        key=lambda hit: (
            float(hit.distance),
            hit.chunk_id,
            hit.source_path,
            hit.heading_path,
            hit.symbol or "",
            hit.start_line,
            hit.end_line,
            hit.citation,
        ),
    )
    unique: list[SearchHit] = []
    seen_by_chunk_id: dict[str, SearchHit] = {}
    for hit in ordered:
        previous = seen_by_chunk_id.get(hit.chunk_id)
        if previous is None:
            unique.append(hit)
            seen_by_chunk_id[hit.chunk_id] = hit
        elif hit != previous:
            raise ValueError(f"conflicting duplicate chunk id: {hit.chunk_id}")
    return tuple(unique)


def _is_relevant(query: EvaluationQuery, hit: SearchHit) -> bool:
    return (
        hit.source_path == query.expected_source_path
        and (query.expected_heading is None or query.expected_heading in hit.heading_path)
        and (query.expected_symbol is None or query.expected_symbol == hit.symbol)
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def evaluate_queries(
    queries: Sequence[EvaluationQuery],
    execute_search: Callable[[EvaluationQuery], SearchResponse],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    """Execute an injected search callback and compute deterministic pilot metrics."""
    immutable_queries = tuple(queries)
    if not immutable_queries:
        raise ValueError("evaluation requires at least one query")
    if not all(isinstance(query, EvaluationQuery) for query in immutable_queries):
        raise TypeError("queries must contain EvaluationQuery records")

    results: list[QueryEvaluationResult] = []
    total_returned_hits = 0
    total_valid_citations = 0
    reciprocal_rank_sum = 0.0
    recalled_query_count = 0
    latencies: list[float] = []
    all_bytes_available = True
    total_bytes = 0

    for query in immutable_queries:
        started_at = clock()
        response = execute_search(query)
        finished_at = clock()
        if not isinstance(response, SearchResponse):
            raise TypeError("search callback must return SearchResponse")
        latency = finished_at - started_at
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("search callback produced an invalid measured latency")

        ranked_hits = _deduplicate_and_rank(response.hits)[:5]
        relevant_rank = next(
            (rank for rank, hit in enumerate(ranked_hits, start=1) if _is_relevant(query, hit)),
            None,
        )
        if relevant_rank is not None:
            recalled_query_count += 1
            reciprocal_rank_sum += 1.0 / relevant_rank
        valid_citations = sum(is_valid_citation(hit) for hit in ranked_hits)
        total_returned_hits += len(ranked_hits)
        total_valid_citations += valid_citations
        latencies.append(latency)
        if response.bytes_processed is None:
            all_bytes_available = False
        else:
            total_bytes += response.bytes_processed
        results.append(
            QueryEvaluationResult(
                query_id=query.id,
                relevant_rank=relevant_rank,
                returned_hit_count=len(ranked_hits),
                valid_citation_count=valid_citations,
                latency_seconds=latency,
                bytes_processed=response.bytes_processed,
                truncated=response.truncated,
            )
        )

    query_count = len(immutable_queries)
    recall_at_5 = recalled_query_count / query_count
    mrr = reciprocal_rank_sum / query_count
    citation_validity = total_valid_citations / total_returned_hits if total_returned_hits else 0.0
    truncation_count = sum(result.truncated for result in results)
    zero_truncation_rate = (query_count - truncation_count) / query_count
    latency_p50 = _percentile(latencies, 0.50)
    latency_p95 = _percentile(latencies, 0.95)

    failed_metrics = tuple(
        name
        for name, passed in (
            ("recall_at_5", recall_at_5 >= 0.80),
            ("citation_validity", citation_validity == 1.00),
            ("truncation_count", truncation_count == 0),
            ("latency_p95_seconds", latency_p95 <= 1.5),
        )
        if not passed
    )
    return EvaluationReport(
        query_count=query_count,
        recall_at_5=recall_at_5,
        mrr=mrr,
        citation_validity=citation_validity,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        total_bytes_processed=total_bytes if all_bytes_available else None,
        truncation_count=truncation_count,
        zero_truncation_rate=zero_truncation_rate,
        results=tuple(results),
        pilot_gate=PilotGateVerdict(
            passed=not failed_metrics,
            failed_metrics=failed_metrics,
        ),
    )


def _result_value(row: Any, field: str) -> Any:
    if isinstance(row, dict):
        return row[field]
    return getattr(row, field)


def make_runtime_query_executor() -> Callable[[EvaluationQuery], SearchResponse]:
    """Build the lazy Task 5/Task 9 runtime adapter used by the CLI.

    Unit tests replace this factory, so importing this module never creates a
    cloud client. Live evaluation requires an explicit user scope in the
    ``HERMES_MEMORY_EVALUATION_USER_ID`` environment variable.
    """
    user_id = os.environ.get("HERMES_MEMORY_EVALUATION_USER_ID")
    agent_name = os.environ.get("HERMES_MEMORY_EVALUATION_AGENT_NAME", "hermes")
    if not user_id:
        raise RuntimeError("document evaluation requires HERMES_MEMORY_EVALUATION_USER_ID")
    try:
        from google import genai

        bigquery_store = importlib.import_module("hermes_memory.bigquery_store")
        embeddings = importlib.import_module("hermes_memory.embeddings")
        search_document_chunks = bigquery_store.search_document_chunks
        VertexEmbeddingClient = embeddings.VertexEmbeddingClient
        from .config import load_config
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Task 5/Task 9 backend is unavailable; integrate embeddings and "
            "search_document_chunks before live evaluation"
        ) from exc

    cfg = load_config()
    sdk_client = genai.Client(
        vertexai=True,
        project=cfg.project,
        location=cfg.location,
    )
    embedder = VertexEmbeddingClient(
        client=sdk_client,
        model=cfg.document_embedding_model,
        dimensions=cfg.document_embedding_dimensions,
        task_type="RETRIEVAL_QUERY",
    )

    def execute(query: EvaluationQuery) -> SearchResponse:
        embedding = embedder.embed(query.query)
        rows = search_document_chunks(
            embedding.values,
            user_id=user_id,
            agent_name=agent_name,
            top_k=5,
            cfg=cfg,
        )
        hits = tuple(
            SearchHit(
                chunk_id=_result_value(row, "chunk_id"),
                source_path=_result_value(row, "source_path"),
                heading_path=tuple(_result_value(row, "heading_path") or ()),
                symbol=_result_value(row, "symbol"),
                start_line=_result_value(row, "start_line"),
                end_line=_result_value(row, "end_line"),
                citation=_result_value(row, "citation"),
                distance=_result_value(row, "distance"),
            )
            for row in rows
        )
        return SearchResponse(
            hits=hits,
            bytes_processed=None,
            truncated=embedding.truncated,
        )

    return execute


_QUERY_KEYS = frozenset(
    {"id", "query", "expected_source_path", "expected_heading", "expected_symbol"}
)
_REQUIRED_QUERY_KEYS = frozenset({"id", "query", "expected_source_path"})
_QUERY_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _nonempty_string(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"query fixture field {field!r} must be a non-empty string")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"query fixture field {field!r} contains invalid whitespace")
    return value


def _parse_query(row: Any, index: int) -> EvaluationQuery:
    if not isinstance(row, dict) or any(type(key) is not str for key in row):
        raise ValueError(f"query fixture entry {index} must be an object")
    keys = frozenset(row)
    if not _REQUIRED_QUERY_KEYS <= keys or not keys <= _QUERY_KEYS:
        raise ValueError(f"query fixture entry {index} has missing or unknown fields")
    query_id = _nonempty_string(row["id"], "id")
    if _QUERY_ID.fullmatch(query_id) is None:
        raise ValueError("query fixture field 'id' must use lowercase kebab-case")
    query_text = _nonempty_string(row["query"], "query")
    if len(query_text) > 500:
        raise ValueError("query fixture field 'query' exceeds 500 characters")
    source_path = _nonempty_string(row["expected_source_path"], "expected_source_path")
    if source_path not in APPROVED_PILOT_SOURCE_PATHS:
        raise ValueError(f"query fixture source path is not approved: {source_path}")
    labels: dict[str, str | None] = {}
    for field in ("expected_heading", "expected_symbol"):
        value = row.get(field)
        labels[field] = None if value is None else _nonempty_string(value, field)
    return EvaluationQuery(
        id=query_id,
        query=query_text,
        expected_source_path=source_path,
        **labels,
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"query fixture contains duplicate field {key!r}")
        result[key] = value
    return result


def load_queries(path: str | Path) -> tuple[EvaluationQuery, ...]:
    """Load and strictly validate a JSON-compatible YAML pilot fixture."""
    fixture_path = Path(path)
    try:
        document = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid query fixture {fixture_path}: {exc}") from exc
    if not isinstance(document, dict) or any(type(key) is not str for key in document):
        raise ValueError("query fixture must be an object with version 1")
    if (
        frozenset(document) != {"version", "queries"}
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("query fixture must contain only version 1 and queries")
    rows = document["queries"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("query fixture queries must be a non-empty array")
    queries = tuple(_parse_query(row, index) for index, row in enumerate(rows))
    ids = tuple(query.id for query in queries)
    if len(ids) != len(set(ids)):
        raise ValueError("query fixture query ids must be unique")
    return queries
