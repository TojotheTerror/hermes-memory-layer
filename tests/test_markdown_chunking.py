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

    assert [
        (unit.text, unit.heading_path, unit.start_line, unit.end_line)
        for unit in units
    ] == [
        ("---\ntags:\n  - hermes\n  - agent-memory\n---\n", (), 1, 5),
        ("Keep [[wikilinks]] exactly.\n", ("Agent [[Memory]]",), 7, 7),
        (
            "- Memory Bank facts.\n  - Nested detail.\n- BigQuery chunks.\n",
            ("Agent [[Memory]]", "Storage"),
            10,
            12,
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
