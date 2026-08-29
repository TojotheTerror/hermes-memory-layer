"""Parse Markdown into immutable, line-addressable structural units."""

from __future__ import annotations

import ast
from math import ceil
import re

from markdown_it import MarkdownIt
from markdown_it.token import Token

from hermes_memory.documents import AtomicUnit

__all__ = ["AtomicUnit", "pack_markdown_units", "parse_code_units", "parse_markdown_units"]

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


def _token_estimate(text: str) -> int:
    return max(1, ceil(len(text) / 4))


def _python_symbols(
    tree: ast.AST,
) -> list[tuple[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, tuple[str, ...]]]:
    symbols: list[
        tuple[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, tuple[str, ...]]
    ] = []

    def visit(node: ast.AST, path: tuple[str, ...]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            path = (*path, node.name)
            symbols.append((node, path))
        for child in ast.iter_child_nodes(node):
            visit(child, path)

    visit(tree, ())
    return symbols


def _python_symbol_start(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
) -> int:
    """Return a zero-based start including decorators and adjacent comments."""
    first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    start = first_line - 1
    while start and lines[start - 1].rstrip("\r\n").lstrip(" \t").startswith("#"):
        start -= 1
    return start


def _generic_code_windows(text: str, *, max_tokens: int, max_lines: int) -> list[AtomicUnit]:
    """Split source deterministically while preserving every source character."""
    max_characters = max_tokens * 4
    windows: list[AtomicUnit] = []
    current: list[str] = []
    current_start = 0
    current_end = 0
    current_characters = 0

    def flush() -> None:
        nonlocal current, current_start, current_end, current_characters
        if not current:
            return
        source = "".join(current)
        windows.append(
            AtomicUnit(source, (), None, current_start, current_end, _token_estimate(source))
        )
        current = []
        current_start = 0
        current_end = 0
        current_characters = 0

    for line_number, line in enumerate(_split_markdown_lines(text), start=1):
        offset = 0
        while offset < len(line):
            if current and line_number - current_start + 1 > max_lines:
                flush()
            if not current:
                current_start = line_number
            available = max_characters - current_characters
            take = min(available, len(line) - offset)
            current.append(line[offset : offset + take])
            current_characters += take
            current_end = line_number
            offset += take
            if current_characters == max_characters:
                flush()
    flush()
    return windows


def parse_code_units(
    text: str,
    language: str,
    *,
    max_tokens: int,
    max_lines: int = 200,
) -> list[AtomicUnit]:
    """Return Python symbols or source-preserving generic language windows."""
    if type(language) is not str or re.fullmatch(r"[a-z][a-z0-9_.+#-]*", language) is None:
        raise ValueError("language must be a lowercase identifier")
    if (
        type(max_tokens) is not int
        or type(max_lines) is not int
        or max_tokens <= 0
        or max_lines <= 0
    ):
        raise ValueError("max_tokens and max_lines must be positive integers")
    if language != "python":
        return _generic_code_windows(text, max_tokens=max_tokens, max_lines=max_lines)

    lines = _split_markdown_lines(text)
    try:
        tree = ast.parse(text.replace("\r\n", "\n").replace("\r", "\n"))
    except SyntaxError:
        return _generic_code_windows(text, max_tokens=max_tokens, max_lines=max_lines)
    units: list[AtomicUnit] = []
    for node, path in _python_symbols(tree):
        start = _python_symbol_start(node, lines)
        end = node.end_lineno
        if end is None:  # pragma: no cover - populated by supported Python versions
            raise ValueError("Python AST node has no end line")
        source = "".join(lines[start:end])
        units.append(
            AtomicUnit(
                text=source,
                heading_path=(),
                symbol=".".join(path),
                start_line=start + 1,
                end_line=end,
                token_estimate=_token_estimate(source),
            )
        )
    return units


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
    for start in range(0, len(unit.text), max_characters):
        end = min(start + max_characters, len(unit.text))
        text = unit.text[start:end]
        start_line = unit.start_line + sum(boundary <= start for boundary in line_break_ends)
        end_line = unit.start_line + sum(boundary <= end - 1 for boundary in line_break_ends)
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

    if _token_estimate("".join(item.text for item, _ in expanded)) <= max_tokens:
        return [_pack([item for item, _ in expanded])]

    groups: list[list[tuple[AtomicUnit, bool]]] = []
    current: list[tuple[AtomicUnit, bool]] = []
    for item in expanded:
        candidate = "".join(unit.text for unit, _ in [*current, item])
        if current and _token_estimate(candidate) > max_tokens:
            groups.append(current)
            current = []
        current.append(item)
        if _token_estimate("".join(unit.text for unit, _ in current)) >= target_tokens:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    packed: list[AtomicUnit] = []
    for index, group in enumerate(groups):
        overlap: list[tuple[AtomicUnit, bool]] = []
        if index:
            for item in reversed(groups[index - 1]):
                if not item[1]:
                    break
                candidate_overlap = [item, *overlap]
                overlap_text = "".join(unit.text for unit, _ in candidate_overlap)
                chunk_text = overlap_text + "".join(unit.text for unit, _ in group)
                if (
                    _token_estimate(overlap_text) > overlap_tokens
                    or _token_estimate(chunk_text) > max_tokens
                ):
                    break
                overlap = candidate_overlap
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
