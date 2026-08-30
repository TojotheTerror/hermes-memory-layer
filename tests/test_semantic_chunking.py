from types import SimpleNamespace

import pytest

from hermes_memory.chunking import (
    AtomicUnit,
    _cosine_similarity,
    _select_semantic_boundary,
    pack_semantic_markdown_units,
    parse_markdown_units,
)


def _atomic(text: str, heading: str, line: int) -> AtomicUnit:
    return AtomicUnit(text, (heading,), None, line, line, 999_999)


class FailingGateway:
    task_type = "SEMANTIC_SIMILARITY"

    def __init__(self) -> None:
        self.calls = 0

    def embed_many(self, texts: list[str]):
        self.calls += 1
        raise RuntimeError("Vertex unavailable")


class FakeGateway:
    task_type = "SEMANTIC_SIMILARITY"

    def __init__(self, vectors: list[tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed_many(self, texts: list[str]):
        self.calls.append(list(texts))
        return [SimpleNamespace(values=vector) for vector in self._vectors]


def test_semantic_packing_cuts_at_weakest_adjacent_complete_unit_boundary():
    units = [
        _atomic("a" * 8, "Alpha", 2),
        _atomic("b" * 8, "Alpha", 3),
        _atomic("c" * 8, "Alpha", 4),
        _atomic("d" * 8, "Alpha", 5),
    ]
    gateway = FakeGateway([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

    packed = pack_semantic_markdown_units(
        units,
        gateway=gateway,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
    )

    assert [chunk.text for chunk in packed] == ["a" * 8 + "b" * 8 + "c" * 8, "d" * 8]
    assert gateway.calls == [[unit.text for unit in units]]


def test_boundary_tie_prefers_chunk_size_closest_to_target():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]
    vectors = [(1.0, 0.0)] * 4

    boundary = _select_semantic_boundary(
        units,
        vectors,
        start=0,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
    )

    assert boundary == 2


def test_boundary_tie_prefers_earliest_source_line():
    units = [
        _atomic("a" * 8, "Alpha", 1),
        _atomic("b" * 8, "Alpha", 2),
        _atomic("c" * 8, "Alpha", 20),
        _atomic("d" * 8, "Alpha", 10),
    ]
    vectors = [(1.0, 0.0)] * 4

    boundary = _select_semantic_boundary(
        units,
        vectors,
        start=0,
        min_tokens=3,
        target_tokens=5,
        max_tokens=6,
    )

    assert boundary == 3


def test_boundary_selection_returns_none_when_no_complete_boundary_is_in_range():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abc", 2)]

    boundary = _select_semantic_boundary(
        units,
        [(1.0, 0.0)] * 3,
        start=0,
        min_tokens=5,
        target_tokens=5,
        max_tokens=5,
    )

    assert boundary is None


def test_boundary_selection_rejects_vector_count_mismatch():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]

    with pytest.raises(ValueError, match="exactly one embedding"):
        _select_semantic_boundary(
            units,
            [(1.0, 0.0)],
            start=0,
            min_tokens=3,
            target_tokens=4,
            max_tokens=6,
        )


def test_dry_run_uses_structural_packing_without_calling_vertex():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]
    gateway = FailingGateway()

    packed = pack_semantic_markdown_units(
        units,
        gateway=gateway,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
        dry_run=True,
    )

    assert [chunk.text for chunk in packed] == ["a" * 8 + "b" * 8, "c" * 8 + "d" * 8]
    assert gateway.calls == 0


def test_dry_run_without_a_vertex_gateway_uses_structural_packing():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]

    packed = pack_semantic_markdown_units(
        units,
        gateway=None,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
        dry_run=True,
    )

    assert [chunk.text for chunk in packed] == ["a" * 8 + "b" * 8, "c" * 8 + "d" * 8]


def test_explicit_structural_strategy_never_calls_vertex():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]

    packed = pack_semantic_markdown_units(
        units,
        gateway=None,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
        strategy="structural",
    )

    assert [chunk.text for chunk in packed] == ["a" * 8 + "b" * 8, "c" * 8 + "d" * 8]


def test_vertex_failure_in_apply_mode_happens_before_caller_mutation():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]
    mutations: list[str] = []

    def apply() -> None:
        pack_semantic_markdown_units(
            units,
            gateway=FailingGateway(),
            min_tokens=3,
            target_tokens=4,
            max_tokens=6,
        )
        mutations.append("write")

    with pytest.raises(RuntimeError, match="Vertex unavailable"):
        apply()

    assert mutations == []


def test_semantic_overlap_uses_trailing_complete_units_within_maximum():
    units = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]
    gateway = FakeGateway([(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)])

    packed = pack_semantic_markdown_units(
        units,
        gateway=gateway,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
        overlap_tokens=2,
    )

    assert [chunk.text for chunk in packed] == [
        "a" * 8 + "b" * 8,
        "b" * 8 + "c" * 8 + "d" * 8,
    ]
    assert all(chunk.token_estimate <= 6 for chunk in packed)


def test_cosine_similarity_defines_zero_vector_similarity_as_zero():
    assert _cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_cosine_similarity_rejects_finite_values_that_overflow_calculation():
    with pytest.raises(ValueError, match="finite"):
        _cosine_similarity((1e308,), (1e308,))


@pytest.mark.parametrize(
    "huge_on_left",
    [True, False],
    ids=["left", "right"],
)
def test_cosine_similarity_rejects_huge_finite_integer_in_either_vector(huge_on_left):
    huge = 10**200
    left, right = ((huge,), (1,)) if huge_on_left else ((1,), (huge,))

    with pytest.raises(ValueError) as error:
        _cosine_similarity(left, right)

    assert type(error.value) is ValueError
    assert str(huge) not in str(error.value)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((), ()),
        ((1.0,), (1.0, 2.0)),
        ((True,), (1.0,)),
        ((float("nan"),), (1.0,)),
        ((float("inf"),), (1.0,)),
        ((10**1000,), (1.0,)),
    ],
)
def test_cosine_similarity_rejects_invalid_dimensions_or_values(left, right):
    with pytest.raises(ValueError):
        _cosine_similarity(left, right)


def test_heading_boundaries_remain_mandatory_for_semantic_chunks_and_overlap():
    alpha = [_atomic(character * 8, "Alpha", line) for line, character in enumerate("abcd", 2)]
    beta = [_atomic(character * 8, "Beta", line) for line, character in enumerate("efgh", 10)]
    gateway = FakeGateway([(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)])

    packed = pack_semantic_markdown_units(
        [*alpha, *beta],
        gateway=gateway,
        min_tokens=3,
        target_tokens=4,
        max_tokens=6,
        overlap_tokens=2,
    )

    assert [chunk.heading_path for chunk in packed] == [
        ("Alpha",),
        ("Alpha",),
        ("Beta",),
        ("Beta",),
    ]
    assert all(
        set(chunk.text) <= set(expected_characters)
        for chunk, expected_characters in zip(packed, ["ab", "bcd", "ef", "fgh"], strict=True)
    )
    assert gateway.calls == [[unit.text for unit in alpha], [unit.text for unit in beta]]


def test_one_unit_section_under_maximum_stays_whole_without_embedding():
    unit = _atomic("small", "Alpha", 7)

    packed = pack_semantic_markdown_units(
        [unit],
        gateway=None,
        min_tokens=2,
        target_tokens=3,
        max_tokens=4,
    )

    assert packed == [AtomicUnit("small", ("Alpha",), None, 7, 7, 2)]


def test_repeated_identical_markdown_headings_remain_distinct_physical_sections():
    source = "# Repeat\nfirst section\n# Repeat\nsecond section\n"
    units = parse_markdown_units(source)
    gateway = FakeGateway([(1.0, 0.0), (0.0, 1.0)])

    packed = pack_semantic_markdown_units(
        units,
        gateway=gateway,
        min_tokens=2,
        target_tokens=3,
        max_tokens=4,
    )

    assert [chunk.text for chunk in packed] == ["first section\n", "second section\n"]
    assert [(chunk.start_line, chunk.end_line) for chunk in packed] == [(2, 2), (4, 4)]
    assert gateway.calls == []


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_oversized_single_unit_preserves_source_ranges_and_maximum(line_ending):
    source = line_ending.join(("aaaa", "bbbb", "cccc"))
    unit = AtomicUnit(source, ("Alpha",), None, 10, 12, 999_999)
    gateway = FakeGateway([(1.0,)])

    packed = pack_semantic_markdown_units(
        [unit],
        gateway=gateway,
        min_tokens=1,
        target_tokens=2,
        max_tokens=2,
    )

    assert "".join(chunk.text for chunk in packed) == source
    assert [(chunk.start_line, chunk.end_line) for chunk in packed] == [(10, 11), (11, 12)]
    assert all(chunk.token_estimate <= 2 for chunk in packed)
    assert gateway.calls == [[source]]


def test_oversized_single_unit_rejects_empty_embedding_dimension():
    unit = _atomic("x" * 12, "Alpha", 2)

    with pytest.raises(ValueError, match="positive dimension"):
        pack_semantic_markdown_units(
            [unit],
            gateway=FakeGateway([()]),
            min_tokens=1,
            target_tokens=2,
            max_tokens=2,
        )


@pytest.mark.parametrize(
    ("min_tokens", "target_tokens", "max_tokens"),
    [(0, 4, 6), (5, 4, 6), (3, 7, 6), (True, 4, 6)],
)
def test_semantic_packing_rejects_invalid_min_target_max_relationships(
    min_tokens, target_tokens, max_tokens
):
    with pytest.raises(ValueError):
        pack_semantic_markdown_units(
            [],
            gateway=FakeGateway([]),
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )


def test_semantic_packing_requires_semantic_similarity_gateway_task():
    gateway = FakeGateway([])
    gateway.task_type = "RETRIEVAL_DOCUMENT"

    with pytest.raises(ValueError, match="SEMANTIC_SIMILARITY"):
        pack_semantic_markdown_units(
            [_atomic("x" * 20, "Alpha", 2)],
            gateway=gateway,
            min_tokens=2,
            target_tokens=3,
            max_tokens=4,
        )

    assert gateway.calls == []


@pytest.mark.parametrize("dry_run", [0, 1, None, "yes"])
def test_semantic_packing_requires_boolean_dry_run(dry_run):
    with pytest.raises(ValueError, match="dry_run"):
        pack_semantic_markdown_units(
            [],
            gateway=FakeGateway([]),
            min_tokens=2,
            target_tokens=3,
            max_tokens=4,
            dry_run=dry_run,
        )
