from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event, Lock
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from hermes_memory.embeddings import VertexEmbeddingClient


class RecordingModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = RecordingModels(response)


class SequencedModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def sequenced_client(outcomes):
    return SimpleNamespace(models=SequencedModels(outcomes))


class OutOfOrderModels:
    def __init__(self):
        self.first_started = Event()
        self.release_first = Event()
        self.completions = []

    def embed_content(self, **kwargs):
        text = kwargs["contents"]
        if text == "first":
            self.first_started.set()
            assert self.release_first.wait(timeout=1)
            value = 1.0
        else:
            assert self.first_started.wait(timeout=1)
            value = 2.0
            self.release_first.set()
        self.completions.append(text)
        return embedding_response([value])


class BlockingModels:
    def __init__(self):
        self.lock = Lock()
        self.release = Event()
        self.two_started = Event()
        self.four_started = Event()
        self.active = 0
        self.max_active = 0

    def embed_content(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= 2:
                self.two_started.set()
            if self.active >= 4:
                self.four_started.set()
        assert self.release.wait(timeout=1)
        with self.lock:
            self.active -= 1
        return embedding_response([float(kwargs["contents"])])


class DuplicateBlockingModels:
    def __init__(self):
        self.lock = Lock()
        self.release = Event()
        self.first_started = Event()
        self.second_started = Event()
        self.calls = 0

    def embed_content(self, **kwargs):
        with self.lock:
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
            else:
                self.second_started.set()
        assert self.release.wait(timeout=1)
        return embedding_response([0.1])


def embedding_response(values, *, token_count=None, truncated=None, billable_count=None):
    statistics = types.ContentEmbeddingStatistics(
        token_count=token_count,
        truncated=truncated,
    )
    return types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=values, statistics=statistics)],
        metadata=types.EmbedContentMetadata(billable_character_count=billable_count),
    )


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"model": ""},
        {"dimensions": 0},
        {"task_type": ""},
        {"concurrency": 0},
        {"max_attempts": 0},
        {"initial_retry_delay": -0.1},
    ],
)
def test_constructor_rejects_invalid_configuration_without_touching_client(invalid_config):
    class UntouchableClient:
        @property
        def models(self):
            raise AssertionError("constructor touched the SDK client")

    with pytest.raises((TypeError, ValueError)):
        VertexEmbeddingClient(client=UntouchableClient(), **invalid_config)


def test_valid_construction_does_not_touch_sdk_client_or_make_a_request():
    class UntouchableClient:
        @property
        def models(self):
            raise AssertionError("constructor touched the SDK client")

    gateway = VertexEmbeddingClient(client=UntouchableClient())

    assert gateway.model == "gemini-embedding-001"
    assert gateway.dimensions == 768
    assert gateway.task_type == "RETRIEVAL_DOCUMENT"


def test_embed_uses_exact_model_task_dimensions_and_one_input_per_request():
    client = FakeClient(embedding_response([0.1, 0.2, 0.3]))
    gateway = VertexEmbeddingClient(
        client=client,
        model="gemini-embedding-001",
        dimensions=3,
        task_type="RETRIEVAL_DOCUMENT",
    )

    result = gateway.embed("source document")

    assert result.values == (0.1, 0.2, 0.3)
    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == "source document"
    assert not isinstance(call["contents"], list)
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 3
    assert call["config"].auto_truncate is False


def test_embed_returns_token_billable_truncation_and_model_fields():
    client = FakeClient(
        embedding_response(
            [0.1, 0.2],
            token_count=17,
            truncated=False,
            billable_count=42,
        )
    )
    gateway = VertexEmbeddingClient(client=client, dimensions=2, task_type="SEMANTIC_SIMILARITY")

    result = gateway.embed("metadata")

    assert result.token_count == 17
    assert result.billable_character_count == 42
    assert result.truncated is False
    assert result.model == "gemini-embedding-001"


def test_embedding_result_is_deeply_immutable_and_detached_from_sdk_values():
    sdk_values = [0.1, 0.2]
    client = FakeClient(embedding_response(sdk_values))
    gateway = VertexEmbeddingClient(client=client, dimensions=2)

    result = gateway.embed("immutable")
    sdk_values[0] = 9.9

    assert result.values == (0.1, 0.2)
    with pytest.raises(FrozenInstanceError):
        result.model = "other-model"


def test_embed_rejects_any_reported_truncation():
    client = FakeClient(embedding_response([0.1], truncated=True))
    gateway = VertexEmbeddingClient(client=client, dimensions=1)

    with pytest.raises(ValueError, match="truncat"):
        gateway.embed("too long")


def test_embed_rejects_dimension_mismatch():
    client = FakeClient(embedding_response([0.1, 0.2]))
    gateway = VertexEmbeddingClient(client=client, dimensions=3)

    with pytest.raises(ValueError, match="dimension"):
        gateway.embed("wrong shape")


def test_embed_rejects_empty_embedding_response():
    client = FakeClient(types.EmbedContentResponse(embeddings=[]))
    gateway = VertexEmbeddingClient(client=client, dimensions=1)

    with pytest.raises(ValueError, match="exactly one embedding"):
        gateway.embed("empty")


def test_embed_rejects_malformed_embedding_response():
    response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=None, statistics=None)],
        metadata=None,
    )
    client = FakeClient(response)
    gateway = VertexEmbeddingClient(client=client, dimensions=1)

    with pytest.raises(ValueError, match="malformed"):
        gateway.embed("malformed")


@pytest.mark.parametrize("invalid_value", [True, "not-a-number", float("nan")])
def test_embed_rejects_non_numeric_or_non_finite_embedding_values(invalid_value):
    response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[invalid_value], statistics=None)],
        metadata=None,
    )
    gateway = VertexEmbeddingClient(client=FakeClient(response), dimensions=1)

    with pytest.raises(ValueError, match="finite numbers"):
        gateway.embed("invalid vector")


@pytest.mark.parametrize(
    ("truncated", "token_count", "billable_count"),
    [
        (1, None, None),
        (None, "ten", None),
        (None, None, 1.5),
    ],
)
def test_embed_rejects_malformed_usage_metadata(truncated, token_count, billable_count):
    response = SimpleNamespace(
        embeddings=[
            SimpleNamespace(
                values=[0.1],
                statistics=SimpleNamespace(truncated=truncated, token_count=token_count),
            )
        ],
        metadata=SimpleNamespace(billable_character_count=billable_count),
    )
    gateway = VertexEmbeddingClient(client=FakeClient(response), dimensions=1)

    with pytest.raises(ValueError, match="metadata"):
        gateway.embed("invalid metadata")


def test_embed_handles_absent_optional_usage_metadata():
    response = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.1])],
    )
    gateway = VertexEmbeddingClient(client=FakeClient(response), dimensions=1)

    result = gateway.embed("metadata unavailable")

    assert result.token_count is None
    assert result.billable_character_count is None
    assert result.truncated is False


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_embed_retries_documented_transient_statuses_with_exponential_backoff(status_code):
    error_type = errors.ClientError if status_code < 500 else errors.ServerError
    transient = error_type(status_code, {"error": {"status": "TRANSIENT"}})
    client = sequenced_client([transient, transient, embedding_response([0.1])])
    delays = []
    gateway = VertexEmbeddingClient(
        client=client,
        dimensions=1,
        max_attempts=3,
        initial_retry_delay=0.25,
        sleep=delays.append,
    )

    result = gateway.embed("retryable")

    assert result.values == (0.1,)
    assert len(client.models.calls) == 3
    assert all(call["config"].auto_truncate is False for call in client.models.calls)
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize(
    ("status_code", "status"),
    [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
        (404, "NOT_FOUND"),
        (501, "NOT_IMPLEMENTED"),
    ],
)
def test_embed_does_not_retry_invalid_auth_or_other_nontransient_api_errors(status_code, status):
    error_type = errors.ClientError if status_code < 500 else errors.ServerError
    failure = error_type(status_code, {"error": {"status": status}})
    client = sequenced_client([failure, embedding_response([0.1])])
    delays = []
    gateway = VertexEmbeddingClient(
        client=client,
        dimensions=1,
        max_attempts=2,
        sleep=delays.append,
    )

    with pytest.raises(error_type) as raised:
        gateway.embed("must not retry")

    assert raised.value is failure
    assert len(client.models.calls) == 1
    assert delays == []


def test_embed_does_not_retry_non_api_exceptions():
    failure = RuntimeError("transport contract failure")
    client = sequenced_client([failure, embedding_response([0.1])])
    gateway = VertexEmbeddingClient(client=client, dimensions=1, max_attempts=3)

    with pytest.raises(RuntimeError) as raised:
        gateway.embed("must not retry")

    assert raised.value is failure
    assert len(client.models.calls) == 1


def test_embed_caches_by_config_and_text_digest():
    sdk_values = [0.1]
    client = FakeClient(embedding_response(sdk_values))
    gateway = VertexEmbeddingClient(client=client, dimensions=1)

    first = gateway.embed("cached source")
    sdk_values[0] = 9.9
    second = gateway.embed("cached source")

    assert first == second
    assert second.values == (0.1,)
    assert len(client.models.calls) == 1


def test_cache_key_is_exactly_model_dimensions_task_and_text_sha256():
    client = sequenced_client(
        [
            embedding_response([1.0]),
            embedding_response([2.0]),
            embedding_response([3.0, 3.0]),
            embedding_response([4.0]),
            embedding_response([5.0]),
        ]
    )
    gateway = VertexEmbeddingClient(
        client=client,
        model="model-a",
        dimensions=1,
        task_type="TASK_A",
    )

    original = gateway.embed("same")
    gateway.model = "model-b"
    changed_model = gateway.embed("same")
    gateway.model = "model-a"
    gateway.dimensions = 2
    changed_dimensions = gateway.embed("same")
    gateway.dimensions = 1
    gateway.task_type = "TASK_B"
    changed_task = gateway.embed("same")
    gateway.task_type = "TASK_A"
    changed_text = gateway.embed("different")
    cached_original = gateway.embed("same")

    assert [
        original.values,
        changed_model.values,
        changed_dimensions.values,
        changed_task.values,
        changed_text.values,
        cached_original.values,
    ] == [(1.0,), (2.0,), (3.0, 3.0), (4.0,), (5.0,), (1.0,)]
    assert len(client.models.calls) == 5


def test_embed_many_preserves_input_order_when_requests_complete_out_of_order():
    models = OutOfOrderModels()
    gateway = VertexEmbeddingClient(
        client=SimpleNamespace(models=models),
        dimensions=1,
        concurrency=2,
    )

    results = gateway.embed_many(["first", "second"])

    assert models.completions == ["second", "first"]
    assert [result.values for result in results] == [(1.0,), (2.0,)]


def test_embed_many_never_exceeds_configured_concurrency():
    models = BlockingModels()
    gateway = VertexEmbeddingClient(
        client=SimpleNamespace(models=models),
        dimensions=1,
        concurrency=2,
    )

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(gateway.embed_many, ["1", "2", "3", "4"])
        assert models.two_started.wait(timeout=1)
        try:
            assert not models.four_started.wait(timeout=0.1)
        finally:
            models.release.set()
        results = future.result(timeout=1)

    assert models.max_active == 2
    assert [result.values for result in results] == [(1.0,), (2.0,), (3.0,), (4.0,)]


def test_concurrent_cache_misses_for_same_key_share_one_request():
    models = DuplicateBlockingModels()
    gateway = VertexEmbeddingClient(
        client=SimpleNamespace(models=models),
        dimensions=1,
        concurrency=2,
    )

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(gateway.embed_many, ["duplicate", "duplicate"])
        assert models.first_started.wait(timeout=1)
        try:
            assert not models.second_started.wait(timeout=0.1)
        finally:
            models.release.set()
        results = future.result(timeout=1)

    assert models.calls == 1
    assert results == [results[0], results[0]]
