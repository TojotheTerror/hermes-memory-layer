from dataclasses import FrozenInstanceError

import pytest

from hermes_memory.chunking import pack_markdown_units, parse_code_units


def test_code_parser_accepts_positional_language_with_a_bounded_line_default():
    units = parse_code_units("def ready():\n    pass\n", "python", max_tokens=20)

    assert [(unit.symbol, unit.start_line, unit.end_line) for unit in units] == [("ready", 1, 2)]


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
    assert [(unit.symbol, unit.start_line, unit.end_line, unit.text) for unit in first] == [
        (
            "Café",
            1,
            5,
            source.splitlines(keepends=True)[0] + "".join(source.splitlines(keepends=True)[1:5]),
        ),
        ("Café.greet", 2, 5, "".join(source.splitlines(keepends=True)[1:5])),
        ("Café.greet.wave", 3, 4, "".join(source.splitlines(keepends=True)[2:4])),
        ("finish", 7, 8, "".join(source.splitlines(keepends=True)[6:8])),
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
