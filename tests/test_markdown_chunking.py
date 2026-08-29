import pytest

from hermes_memory.chunking import parse_markdown_units


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
