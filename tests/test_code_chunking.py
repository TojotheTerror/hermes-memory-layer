import ast
import sys
from dataclasses import FrozenInstanceError

import pytest

from hermes_memory.chunking import pack_markdown_units, parse_code_units


def test_code_parser_accepts_positional_language_with_a_bounded_line_default():
    units = parse_code_units("def ready():\n    pass\n", "python", max_tokens=20)

    assert [(unit.symbol, unit.start_line, unit.end_line) for unit in units] == [("ready", 1, 2)]


def test_valid_python_without_symbols_preserves_all_source_in_bounded_generic_units():
    source = '"""Module 雪."""\r\n\r\nimport os\r\nVALUE = "café"\r\n# trailing comment\r\n'

    first = parse_code_units(source, language="python", max_tokens=8, max_lines=2)
    second = parse_code_units(source, language="python", max_tokens=8, max_lines=2)

    assert first == second
    assert "".join(unit.text for unit in first) == source
    assert all(unit.symbol is None and unit.heading_path == () for unit in first)
    assert all(unit.token_estimate <= 8 for unit in first)
    assert all(unit.end_line - unit.start_line + 1 <= 2 for unit in first)


def test_python_symbols_and_generic_gaps_partition_mixed_source_in_order():
    source = (
        '"""Module documentation."""\n'
        "import os\n"
        "\n"
        "# outer comment\n"
        "@decorate\n"
        "class Outer:\n"
        "    # method comment\n"
        "    def café(self):\n"
        "        return '雪'\n"
        "    tail = True\n"
        "\n"
        "BETWEEN = 1\n"
        "\n"
        "# finish comment\n"
        "async def finish():\n"
        "    return os.name\n"
        "\n"
        "finish()\n"
    )
    lines = source.splitlines(keepends=True)

    first = parse_code_units(source, language="python", max_tokens=100, max_lines=100)
    second = parse_code_units(source, language="python", max_tokens=100, max_lines=100)

    assert first == second
    assert "".join(unit.text for unit in first) == source
    assert [(unit.symbol, unit.start_line, unit.end_line, unit.text) for unit in first] == [
        (None, 1, 3, "".join(lines[0:3])),
        ("Outer", 4, 6, "".join(lines[3:6])),
        ("Outer.café", 7, 9, "".join(lines[6:9])),
        ("Outer", 10, 10, lines[9]),
        (None, 11, 13, "".join(lines[10:13])),
        ("finish", 14, 16, "".join(lines[13:16])),
        (None, 17, 18, "".join(lines[16:18])),
    ]


def test_python_symbols_have_source_order_paths_and_inclusive_ranges():
    source = (
        "class Café:\n"
        "    def greet(self):\n"
        "        async def wave():\n"
        "            return '👋'\n"
        "        return wave\n"
        "\n"
        "async def finish():\n"
        "    return Café\n"
    )

    first = parse_code_units(source, language="python", max_tokens=100, max_lines=100)
    second = parse_code_units(source, language="python", max_tokens=100, max_lines=100)

    assert first == second
    lines = source.splitlines(keepends=True)
    assert "".join(unit.text for unit in first) == source
    assert [(unit.symbol, unit.start_line, unit.end_line, unit.text) for unit in first] == [
        ("Café", 1, 1, lines[0]),
        ("Café.greet", 2, 2, lines[1]),
        ("Café.greet.wave", 3, 4, "".join(lines[2:4])),
        ("Café.greet", 5, 5, lines[4]),
        (None, 6, 6, lines[5]),
        ("finish", 7, 8, "".join(lines[6:8])),
    ]
    assert all(unit.heading_path == () for unit in first)
    with pytest.raises(FrozenInstanceError):
        first[0].symbol = "changed"


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_python_symbol_keeps_comment_decorator_docstring_and_line_endings(line_ending: str):
    source = line_ending.join(
        (
            "# associé au symbole",
            "@registry.register",
            "def café():",
            '    """Documentation 雪."""',
            "    return 'exact'",
        )
    )

    units = parse_code_units(source, language="python", max_tokens=100, max_lines=100)

    assert [(unit.symbol, unit.start_line, unit.end_line, unit.text) for unit in units] == [
        ("café", 1, 5, source)
    ]


def test_packing_cuts_a_python_symbol_only_when_it_exceeds_the_maximum():
    source = "def large():\n" + "".join(f"    value_{index} = {index}\n" for index in range(8))
    units = parse_code_units(source, language="python", max_tokens=20, max_lines=100)

    roomy = pack_markdown_units(units, target_tokens=40, max_tokens=80)
    bounded = pack_markdown_units(units, target_tokens=10, max_tokens=20)

    assert roomy == units
    assert len(bounded) > 1
    assert "".join(unit.text for unit in bounded) == source
    assert all(unit.symbol == "large" for unit in bounded)
    assert all(unit.token_estimate <= 20 for unit in bounded)


def test_invalid_python_falls_back_to_bounded_windows_without_losing_source():
    source = "def broken(:\r\n" + "雪" * 25 + "\r" + "tail\n"

    first = parse_code_units(source, language="python", max_tokens=3, max_lines=2)
    second = parse_code_units(source, language="python", max_tokens=3, max_lines=2)

    assert first == second
    assert "".join(unit.text for unit in first) == source
    assert all(unit.symbol is None and unit.heading_path == () for unit in first)
    assert all(unit.token_estimate <= 3 for unit in first)
    assert all(unit.end_line - unit.start_line + 1 <= 2 for unit in first)


@pytest.mark.parametrize("statement", ["return", "break", "continue", "yield 1"])
def test_context_invalid_python_uses_complete_generic_fallback(statement: str):
    source = f"def valid():\r\n    return '雪'\r\n{statement}\r\n"

    first = parse_code_units(source, language="python", max_tokens=6, max_lines=2)
    second = parse_code_units(source, language="python", max_tokens=6, max_lines=2)

    assert first == second
    assert "".join(unit.text for unit in first) == source
    assert all(unit.symbol is None and unit.heading_path == () for unit in first)
    assert all(unit.token_estimate <= 6 for unit in first)
    assert all(unit.end_line - unit.start_line + 1 <= 2 for unit in first)


def test_pathological_python_compile_recursion_falls_back_without_losing_source():
    # Grammar-valid deeply chained attribute access: ast.parse() accepts it but
    # compile() overflows the interpreter stack with RecursionError. The parser
    # must fall back to bounded deterministic generic windows, not abort chunking.
    depth = 1500
    source = "x" + ".a" * depth + "\n"
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        tree = ast.parse(source)
        with pytest.raises(RecursionError):
            compile(tree, "<code>", "exec")

        first = parse_code_units(source, language="python", max_tokens=8, max_lines=2)
        second = parse_code_units(source, language="python", max_tokens=8, max_lines=2)
    finally:
        sys.setrecursionlimit(limit)

    assert first == second
    assert "".join(unit.text for unit in first) == source
    assert all(unit.symbol is None and unit.heading_path == () for unit in first)
    assert all(unit.token_estimate <= 8 for unit in first)
    assert all(unit.end_line - unit.start_line + 1 <= 2 for unit in first)


@pytest.mark.parametrize(
    ("language", "source", "ranges"),
    [
        (
            "go",
            'package main\r\n\r\nfunc main() {\r\n\tprintln("hi")\r\n}\r\n',
            [(1, 2), (3, 4), (5, 5)],
        ),
        (
            "rust",
            "def looks_python():\n    return 'but is unsupported'\n",
            [(1, 2), (2, 2)],
        ),
    ],
)
def test_go_and_unsupported_languages_use_only_bounded_generic_windows(
    language: str, source: str, ranges: list[tuple[int, int]]
):
    units = parse_code_units(source, language=language, max_tokens=8, max_lines=2)

    assert "".join(unit.text for unit in units) == source
    assert all(unit.symbol is None and unit.heading_path == () for unit in units)
    assert all(unit.token_estimate <= 8 for unit in units)
    assert all(unit.end_line - unit.start_line + 1 <= 2 for unit in units)
    assert [(unit.start_line, unit.end_line) for unit in units] == ranges


@pytest.mark.parametrize("language", [None, "", " python", "python ", "Python", "go lang", "go\n"])
def test_code_parser_rejects_noncanonical_language_identifiers(language: str):
    with pytest.raises(ValueError, match="language must be a lowercase identifier"):
        parse_code_units("", language=language, max_tokens=8, max_lines=2)


@pytest.mark.parametrize(
    ("max_tokens", "max_lines"),
    [(0, 1), (-1, 1), (True, 1), (1.5, 1), (1, 0), (1, -1), (1, False), (1, 2.5)],
)
def test_code_parser_rejects_nonpositive_or_noninteger_limits(max_tokens: int, max_lines: int):
    with pytest.raises(ValueError, match="max_tokens and max_lines must be positive integers"):
        parse_code_units("", language="python", max_tokens=max_tokens, max_lines=max_lines)
