"""Offline-testable gateway for Vertex document embeddings."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from threading import BoundedSemaphore, Lock
from time import sleep as default_sleep
from typing import Any, Callable

from google.genai import errors, types


_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_SAFE_REQUEST_ERROR = "embedding request failed"


class EmbeddingRequestError(RuntimeError):
    """A provider-independent embedding request failure."""


@dataclass(frozen=True)
class EmbeddingResult:
    """An immutable embedding and the usage metadata returned with it."""

    values: tuple[float, ...]
    token_count: float | None
    billable_character_count: int | None
    truncated: bool
    model: str


@dataclass(frozen=True)
class _EmbeddingConfig:
    model: str
    dimensions: int
    task_type: str


class VertexEmbeddingClient:
    """Embed text through an injected Google Gen AI SDK client."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
        concurrency: int = 4,
        max_attempts: int = 3,
        initial_retry_delay: float = 1.0,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_type must be a non-empty string")
        if type(concurrency) is not int or concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if (
            type(initial_retry_delay) not in (int, float)
            or not isfinite(initial_retry_delay)
            or initial_retry_delay < 0
        ):
            raise ValueError("initial_retry_delay must be a finite non-negative number")
        self._client = client
        self._config = _EmbeddingConfig(
            model=model,
            dimensions=dimensions,
            task_type=task_type,
        )
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._initial_retry_delay = initial_retry_delay
        self._sleep = sleep
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._admission = BoundedSemaphore(concurrency)
        self._cache: dict[tuple[str, int, str, str], EmbeddingResult] = {}
        self._inflight: dict[tuple[str, int, str, str], Future[EmbeddingResult]] = {}
        self._cache_lock = Lock()

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def task_type(self) -> str:
        return self._config.task_type

    def embed(self, text: str) -> EmbeddingResult:
        return self._submit(text).result()

    def _submit(self, text: str) -> Future[EmbeddingResult]:
        cache_key = (self.model, self.dimensions, self.task_type, sha256(text.encode()).hexdigest())
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                completed: Future[EmbeddingResult] = Future()
                completed.set_result(cached)
                return completed
            pending = self._inflight.get(cache_key)
            if pending is not None:
                return pending

        self._admission.acquire()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._admission.release()
                completed = Future()
                completed.set_result(cached)
                return completed
            pending = self._inflight.get(cache_key)
            if pending is not None:
                self._admission.release()
                return pending
            pending = Future()
            self._inflight[cache_key] = pending

        try:
            execution = self._executor.submit(self._execute, cache_key, text, pending)
        except BaseException as error:
            with self._cache_lock:
                self._inflight.pop(cache_key, None)
            pending.set_exception(error)
            self._admission.release()
        else:
            execution.add_done_callback(lambda _future: self._admission.release())
        return pending

    def _execute(
        self,
        cache_key: tuple[str, int, str, str],
        text: str,
        pending: Future[EmbeddingResult],
    ) -> None:
        try:
            result = self._embed_uncached(text)
        except BaseException as error:
            with self._cache_lock:
                self._inflight.pop(cache_key, None)
            pending.set_exception(error)
            return

        with self._cache_lock:
            self._cache[cache_key] = result
            self._inflight.pop(cache_key, None)
        pending.set_result(result)

    def _embed_uncached(self, text: str) -> EmbeddingResult:
        response = self._request(text)
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise ValueError("embedding response must contain exactly one embedding")
        embedding = embeddings[0]
        values = getattr(embedding, "values", None)
        if not isinstance(values, list):
            raise ValueError("embedding response is malformed: values must be a list")
        if len(values) != self.dimensions:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimensions}, received {len(values)}"
            )
        if not all(type(value) in (int, float) and isfinite(value) for value in values):
            raise ValueError("embedding response is malformed: values must be finite numbers")
        statistics = getattr(embedding, "statistics", None)
        metadata = getattr(response, "metadata", None)
        truncated = getattr(statistics, "truncated", None)
        token_count = getattr(statistics, "token_count", None)
        billable_character_count = getattr(metadata, "billable_character_count", None)
        if truncated is not None and type(truncated) is not bool:
            raise ValueError("embedding response metadata has invalid truncated field")
        if token_count is not None and (
            type(token_count) not in (int, float) or not isfinite(token_count) or token_count < 0
        ):
            raise ValueError("embedding response metadata has invalid token count")
        if billable_character_count is not None and (
            type(billable_character_count) is not int or billable_character_count < 0
        ):
            raise ValueError("embedding response metadata has invalid billable character count")
        if truncated is True:
            raise ValueError("embedding response reported truncated input")
        return EmbeddingResult(
            values=tuple(values),
            token_count=token_count,
            billable_character_count=billable_character_count,
            truncated=False,
            model=self.model,
        )

    def embed_many(self, texts: Iterable[str]) -> list[EmbeddingResult]:
        """Embed multiple inputs concurrently while preserving input order."""
        iterator = iter(texts)
        pending: deque[Future[EmbeddingResult]] = deque()
        for _ in range(self._concurrency):
            try:
                text = next(iterator)
            except StopIteration:
                break
            pending.append(self._submit(text))

        results: list[EmbeddingResult] = []
        while pending:
            results.append(pending.popleft().result())
            try:
                text = next(iterator)
            except StopIteration:
                continue
            pending.append(self._submit(text))
        return results

    def _request(self, text: str) -> Any:
        for attempt in range(self._max_attempts):
            retry_delay: float | None = None
            request_failed = False
            try:
                return self._client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=self.dimensions,
                        auto_truncate=False,
                    ),
                )
            except errors.APIError as error:
                can_retry = (
                    error.code in _TRANSIENT_STATUS_CODES and attempt + 1 < self._max_attempts
                )
                if can_retry:
                    retry_delay = self._initial_retry_delay * (2**attempt)
                else:
                    request_failed = True
            if request_failed:
                raise EmbeddingRequestError(_SAFE_REQUEST_ERROR)
            if retry_delay is not None:
                self._sleep(retry_delay)
        raise EmbeddingRequestError(_SAFE_REQUEST_ERROR)
