"""Parse Markdown into immutable, line-addressable structural units."""

from __future__ import annotations

from math import ceil

from markdown_it import MarkdownIt
from markdown_it.token import Token

from hermes_memory.documents import AtomicUnit

__all__ = ["AtomicUnit", "parse_markdown_units"]

_MARKDOWN = MarkdownIt("commonmark")
_UNIT_TOKEN_TYPES = {
    "blockquote_open",
    "bullet_list_open",
    "code_block",
    "fence",
    "html_block",
    "hr",
    "ordered_list_open",
    "paragraph_open",
}


def _frontmatter_end(lines: list[str]) -> int:
    """Return the exclusive end line for leading YAML frontmatter, or zero."""
    if not lines or lines[0].rstrip("\r\n") != "---":
        return 0

    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") in {"---", "..."}:
            return index + 1
    return 0


def _unit(lines: list[str], start: int, end: int, heading_path: tuple[str, ...]) -> AtomicUnit:
    source = "".join(lines[start:end])
    return AtomicUnit(
        text=source,
        heading_path=heading_path,
        symbol=None,
        start_line=start + 1,
        end_line=end,
        token_estimate=max(1, ceil(len(source) / 4)),
    )


def _heading_text(tokens: list[Token], index: int) -> str:
    if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
        return tokens[index + 1].content
    return ""


def parse_markdown_units(text: str) -> list[AtomicUnit]:
    """Return source-preserving Markdown blocks with one-based inclusive line ranges."""
    lines = text.splitlines(keepends=True)
    frontmatter_end = _frontmatter_end(lines)
    units: list[AtomicUnit] = []

    if frontmatter_end:
        units.append(_unit(lines, 0, frontmatter_end, ()))

    tokens = _MARKDOWN.parse("".join(lines[frontmatter_end:]))
    headings: dict[int, str] = {}
    covered_until = 0

    for index, token in enumerate(tokens):
        if token.map is None:
            continue

        start, end = token.map
        if start < covered_until:
            continue

        if token.type == "heading_open":
            level = int(token.tag[1:])
            headings = {
                heading_level: heading
                for heading_level, heading in headings.items()
                if heading_level < level
            }
            headings[level] = _heading_text(tokens, index)
            covered_until = end
            continue

        if token.type not in _UNIT_TOKEN_TYPES:
            continue

        heading_path = tuple(headings[level] for level in sorted(headings))
        absolute_start = frontmatter_end + start
        absolute_end = frontmatter_end + end
        units.append(_unit(lines, absolute_start, absolute_end, heading_path))
        covered_until = end

    return units
