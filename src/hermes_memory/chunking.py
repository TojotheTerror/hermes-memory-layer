"""Parse Markdown into immutable, line-addressable structural units."""

from __future__ import annotations

from math import ceil

from markdown_it import MarkdownIt
from markdown_it.token import Token

from hermes_memory.documents import AtomicUnit

__all__ = ["AtomicUnit", "pack_markdown_units", "parse_markdown_units"]

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


def _split_markdown_lines(text: str) -> list[str]:
    """Split on CommonMark line endings while retaining their exact spelling."""
    lines: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        if text[index] == "\n":
            index += 1
        elif text[index] == "\r":
            index += 1
            if index < len(text) and text[index] == "\n":
                index += 1
        else:
            index += 1
            continue

        lines.append(text[start:index])
        start = index

    if start < len(text):
        lines.append(text[start:])

    return lines


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
        token_estimate=_token_estimate(source),
    )


def _token_estimate_for_length(length: int) -> int:
    return max(1, ceil(length / 4))


def _token_estimate(text: str) -> int:
    return _token_estimate_for_length(len(text))


def _pack(units: list[AtomicUnit]) -> AtomicUnit:
    text = "".join(unit.text for unit in units)
    first = units[0]
    return AtomicUnit(
        text=text,
        heading_path=first.heading_path,
        symbol=first.symbol,
        start_line=first.start_line,
        end_line=units[-1].end_line,
        token_estimate=_token_estimate(text),
    )


def _line_break_ends(text: str) -> list[int]:
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index] == "\r":
            index += 1
            if index < len(text) and text[index] == "\n":
                index += 1
            ends.append(index)
        elif text[index] == "\n":
            index += 1
            ends.append(index)
        else:
            index += 1
    return ends


def _hard_split(unit: AtomicUnit, max_tokens: int) -> list[AtomicUnit]:
    if _token_estimate(unit.text) <= max_tokens:
        return [unit]

    max_characters = max_tokens * 4
    line_break_ends = _line_break_ends(unit.text)
    pieces: list[AtomicUnit] = []
    start_boundary_index = 0
    end_boundary_index = 0
    for start in range(0, len(unit.text), max_characters):
        end = min(start + max_characters, len(unit.text))
        text = unit.text[start:end]
        while (
            start_boundary_index < len(line_break_ends)
            and line_break_ends[start_boundary_index] <= start
        ):
            start_boundary_index += 1
        while (
            end_boundary_index < len(line_break_ends)
            and line_break_ends[end_boundary_index] <= end - 1
        ):
            end_boundary_index += 1
        start_line = unit.start_line + start_boundary_index
        end_line = unit.start_line + end_boundary_index
        pieces.append(
            AtomicUnit(
                text=text,
                heading_path=unit.heading_path,
                symbol=unit.symbol,
                start_line=start_line,
                end_line=end_line,
                token_estimate=_token_estimate(text),
            )
        )
    return pieces


def _pack_section(
    units: list[AtomicUnit],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[AtomicUnit]:
    expanded: list[tuple[AtomicUnit, bool]] = []
    for unit in units:
        pieces = _hard_split(unit, max_tokens)
        expanded.extend((piece, len(pieces) == 1) for piece in pieces)

    expanded_characters = sum(len(item.text) for item, _ in expanded)
    if _token_estimate_for_length(expanded_characters) <= max_tokens:
        return [_pack([item for item, _ in expanded])]

    groups: list[tuple[list[tuple[AtomicUnit, bool]], int]] = []
    current: list[tuple[AtomicUnit, bool]] = []
    current_characters = 0
    for item in expanded:
        item_characters = len(item[0].text)
        if current and (
            _token_estimate_for_length(current_characters + item_characters) > max_tokens
        ):
            groups.append((current, current_characters))
            current = []
            current_characters = 0
        current.append(item)
        current_characters += item_characters
        if _token_estimate_for_length(current_characters) >= target_tokens:
            groups.append((current, current_characters))
            current = []
            current_characters = 0
    if current:
        groups.append((current, current_characters))

    packed: list[AtomicUnit] = []
    for index, (group, group_characters) in enumerate(groups):
        overlap: list[tuple[AtomicUnit, bool]] = []
        if index:
            previous_group = groups[index - 1][0]
            overlap_start = len(previous_group)
            overlap_characters = 0
            for previous_index in range(len(previous_group) - 1, -1, -1):
                item = previous_group[previous_index]
                if not item[1]:
                    break
                candidate_overlap_characters = overlap_characters + len(item[0].text)
                if (
                    _token_estimate_for_length(candidate_overlap_characters) > overlap_tokens
                    or _token_estimate_for_length(candidate_overlap_characters + group_characters)
                    > max_tokens
                ):
                    break
                overlap_start = previous_index
                overlap_characters = candidate_overlap_characters
            overlap = previous_group[overlap_start:]
        packed.append(_pack([item for item, _ in [*overlap, *group]]))
    return packed


def _validate_packing_limits(target_tokens: int, max_tokens: int, overlap_tokens: int) -> None:
    values = (target_tokens, max_tokens, overlap_tokens)
    if not all(type(value) is int for value in values) or not (
        0 <= overlap_tokens < target_tokens <= max_tokens
    ):
        raise ValueError(
            "packing limits must satisfy 0 <= overlap_tokens < target_tokens <= max_tokens"
        )


def pack_markdown_units(
    units: list[AtomicUnit],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[AtomicUnit]:
    """Pack contiguous Markdown sections into immutable structural chunks."""
    _validate_packing_limits(target_tokens, max_tokens, overlap_tokens)
    if not units:
        return []

    sections: list[list[AtomicUnit]] = []
    for unit in units:
        if sections and (unit.heading_path, unit.symbol) == (
            sections[-1][0].heading_path,
            sections[-1][0].symbol,
        ):
            sections[-1].append(unit)
        else:
            sections.append([unit])
    return [
        chunk
        for section in sections
        for chunk in _pack_section(section, target_tokens, max_tokens, overlap_tokens)
    ]


def _heading_text(tokens: list[Token], index: int) -> str:
    if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
        return tokens[index + 1].content
    return ""


def parse_markdown_units(text: str) -> list[AtomicUnit]:
    """Return source-preserving Markdown blocks with one-based inclusive line ranges."""
    lines = _split_markdown_lines(text)
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
