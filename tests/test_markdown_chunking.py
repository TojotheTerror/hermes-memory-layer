from dataclasses import FrozenInstanceError

import pytest

from hermes_memory.chunking import AtomicUnit, pack_markdown_units, parse_markdown_units


def _atomic(
    text: str,
    heading_path: tuple[str, ...],
    start_line: int,
    end_line: int,
) -> AtomicUnit:
    return AtomicUnit(text, heading_path, None, start_line, end_line, 999_999)


def test_pack_markdown_units_keeps_sections_under_maximum_whole():
    units = [
        _atomic("first block!", ("Alpha",), 2, 2),
        _atomic("second block", ("Alpha",), 4, 4),
        _atomic("other", ("Beta",), 7, 7),
    ]

    packed = pack_markdown_units(units, target_tokens=3, max_tokens=8, overlap_tokens=0)

    assert packed == [
        AtomicUnit("first block!second block", ("Alpha",), None, 2, 4, 6),
        AtomicUnit("other", ("Beta",), None, 7, 7, 2),
    ]
    with pytest.raises(FrozenInstanceError):
        packed[0].text = "changed"


def test_pack_markdown_units_greedily_reaches_target_without_exceeding_maximum():
    units = [
        _atomic("a" * 12, ("Large",), 2, 2),
        _atomic("b" * 12, ("Large",), 3, 3),
        _atomic("c" * 8, ("Large",), 4, 4),
    ]

    packed = pack_markdown_units(units, target_tokens=5, max_tokens=6, overlap_tokens=0)

    assert packed == [
        AtomicUnit("a" * 12 + "b" * 12, ("Large",), None, 2, 3, 6),
        AtomicUnit("c" * 8, ("Large",), None, 4, 4, 2),
    ]
    assert all(chunk.token_estimate <= 6 for chunk in packed)


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_pack_markdown_units_hard_splits_oversized_unit_without_changing_source(
    line_ending: str,
):
    source = line_ending.join(("aaaa", "bbbb", "cccc"))
    unit = _atomic(source, ("Large",), 10, 12)

    packed = pack_markdown_units([unit], target_tokens=2, max_tokens=2, overlap_tokens=0)

    assert "".join(chunk.text for chunk in packed) == source
    assert [(chunk.start_line, chunk.end_line) for chunk in packed] == [(10, 11), (11, 12)]
    assert all(chunk.token_estimate <= 2 for chunk in packed)


def test_pack_markdown_units_overlaps_only_trailing_complete_units_within_top_heading():
    units = [
        _atomic("a" * 8, ("Alpha",), 2, 2),
        _atomic("b" * 8, ("Alpha",), 3, 3),
        _atomic("c" * 8, ("Alpha",), 4, 4),
        _atomic("d" * 8, ("Alpha",), 5, 5),
        _atomic("e" * 8, ("Beta",), 8, 8),
    ]

    packed = pack_markdown_units(units, target_tokens=4, max_tokens=6, overlap_tokens=2)

    assert packed == [
        AtomicUnit("a" * 8 + "b" * 8, ("Alpha",), None, 2, 3, 4),
        AtomicUnit("b" * 8 + "c" * 8 + "d" * 8, ("Alpha",), None, 3, 5, 6),
        AtomicUnit("e" * 8, ("Beta",), None, 8, 8, 2),
    ]


@pytest.mark.parametrize(
    ("target_tokens", "max_tokens", "overlap_tokens"),
    [
        (0, 8, 0),
        (4, 0, 0),
        (9, 8, 0),
        (4, 8, -1),
        (4, 8, 4),
        (True, 8, 0),
    ],
)
def test_pack_markdown_units_rejects_invalid_numeric_relationships(
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
):
    with pytest.raises(ValueError, match="0 <= overlap_tokens < target_tokens <= max_tokens"):
        pack_markdown_units(
            [],
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )


def test_pack_markdown_units_is_byte_and_dataclass_deterministic_over_two_runs():
    units = [
        _atomic("α" * 8, ("Alpha",), 2, 2),
        _atomic("b" * 8, ("Alpha",), 3, 3),
        _atomic("c" * 8, ("Alpha",), 4, 4),
    ]

    first = pack_markdown_units(units, target_tokens=4, max_tokens=6, overlap_tokens=2)
    second = pack_markdown_units(units, target_tokens=4, max_tokens=6, overlap_tokens=2)

    assert first == second
    assert [chunk.text.encode("utf-8") for chunk in first] == [
        chunk.text.encode("utf-8") for chunk in second
    ]


def test_pack_markdown_units_does_not_overlap_hard_split_fragments():
    source = "a" * 20

    packed = pack_markdown_units(
        [_atomic(source, ("Alpha",), 2, 2)],
        target_tokens=2,
        max_tokens=2,
        overlap_tokens=1,
    )

    assert "".join(chunk.text for chunk in packed) == source


def test_parse_markdown_units_preserves_structural_boundaries_and_source_locations():
    text = """---
tags:
  - hermes
  - agent-memory
---
# Agent [[Memory]]
Keep [[wikilinks]] exactly.

## Storage
- Memory Bank facts.
  - Nested detail.
- BigQuery chunks.

```yaml
memory:
  backend: vertex
```

### Recovery
Replay first.

### Recovery
Replay second.
"""

    units = parse_markdown_units(text)

    assert [(unit.text, unit.heading_path, unit.start_line, unit.end_line) for unit in units] == [
        ("---\ntags:\n  - hermes\n  - agent-memory\n---\n", (), 1, 5),
        ("Keep [[wikilinks]] exactly.\n", ("Agent [[Memory]]",), 7, 7),
        (
            "- Memory Bank facts.\n  - Nested detail.\n- BigQuery chunks.\n\n",
            ("Agent [[Memory]]", "Storage"),
            10,
            13,
        ),
        (
            "```yaml\nmemory:\n  backend: vertex\n```\n",
            ("Agent [[Memory]]", "Storage"),
            14,
            17,
        ),
        (
            "Replay first.\n",
            ("Agent [[Memory]]", "Storage", "Recovery"),
            20,
            20,
        ),
        (
            "Replay second.\n",
            ("Agent [[Memory]]", "Storage", "Recovery"),
            23,
            23,
        ),
    ]


def test_tight_list_preserves_trailing_blank_separator_from_token_map():
    text = "# H\n- a\n- b\n\nparagraph\n"

    units = parse_markdown_units(text)

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        ("- a\n- b\n\n", 2, 4),
        ("paragraph\n", 5, 5),
    ]


def test_loose_list_preserves_trailing_blank_separator_from_token_map():
    text = "# H\n- a\n\n- b\n\nparagraph\n"

    units = parse_markdown_units(text)

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        ("- a\n\n- b\n\n", 2, 5),
        ("paragraph\n", 6, 6),
    ]


def test_unicode_line_separator_is_preserved_inside_markdown_text():
    text = "alpha\u2028beta\n\ngamma\n"

    units = parse_markdown_units(text)

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        ("alpha\u2028beta\n", 1, 1),
        ("gamma\n", 3, 3),
    ]


@pytest.mark.parametrize(
    "separator",
    ["\u000b", "\u000c", "\u0085", "\u001c", "\u001d", "\u001e", "\u2029"],
    ids=[
        "vertical-tab",
        "form-feed",
        "next-line",
        "file-separator",
        "group-separator",
        "record-separator",
        "paragraph-separator",
    ],
)
def test_python_only_line_separators_are_preserved_inside_markdown_text(separator: str):
    text = f"alpha{separator}beta\n\ngamma\n"

    units = parse_markdown_units(text)

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        (f"alpha{separator}beta\n", 1, 1),
        ("gamma\n", 3, 3),
    ]


def test_mixed_commonmark_line_endings_preserve_token_map_alignment():
    text = "alpha\r\n\r\nbeta\rgamma\n\ndelta"

    units = parse_markdown_units(text)

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        ("alpha\r\n", 1, 1),
        ("beta\rgamma\n", 3, 4),
        ("delta", 6, 6),
    ]
    assert all(unit.text.strip("\r\n") for unit in units)
