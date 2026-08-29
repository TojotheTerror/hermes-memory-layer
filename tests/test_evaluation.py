from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"
EVALUATION_CASES = (
    {
        "query": "How are missing chunks replayed during storage recovery?",
        "relative_path": "Operations/agent-memory.md",
        "heading": "Recovery",
        "marker": "Replay only the missing chunk identifiers.",
    },
    {
        "query": "Which function constructs an empty fixture store?",
        "relative_path": "main.go",
        "symbol": "NewStore",
        "marker": "func NewStore() *InMemoryStore",
    },
    {
        "query": "How are snapshots ordered?",
        "relative_path": "main.go",
        "symbol": "(*InMemoryStore).Snapshot",
        "marker": "sort.Strings(ids)",
    },
)


def test_obsidian_fixture_covers_markdown_evaluation_boundaries():
    text = (FIXTURES / "obsidian" / "Operations" / "agent-memory.md").read_text()

    assert text.startswith("---\n")
    assert "tags:\n  - hermes\n  - agent-memory\n  - runbook" in text
    assert "## Storage\n\n### Layout" in text
    assert text.count("### Recovery") == 2
    assert "- Memory Bank facts for personalized recall." in text
    assert "```yaml\nmemory:" in text

    oversized = text.split("## Oversized Incident Narrative\n", 1)[1].split("\n## Runtime", 1)[0]
    assert len(oversized) > 4_000


def test_repo_fixture_covers_code_and_ignore_contracts():
    repo = FIXTURES / "repo"
    go_source = (repo / "main.go").read_text()
    readme = (repo / "README.md").read_text()
    ignored_environment = (repo / ".env.example").read_text().splitlines()

    assert "type Memory struct" in go_source
    assert "type Store interface" in go_source
    assert "func NewStore() *InMemoryStore" in go_source
    assert "func (s *InMemoryStore) Snapshot() []Memory" in go_source
    assert readme.count("## Recovery") == 2
    assert "```bash\ngo run ./main.go\n```" in readme
    assert ".env.example" in (repo / ".gitignore").read_text().splitlines()
    assert ignored_environment
    assert all(line.endswith("=FAKE_TEST_MARKER_ONLY") for line in ignored_environment)


def test_evaluation_cases_point_to_committed_fixture_answers():
    for case in EVALUATION_CASES:
        root = "obsidian" if case["relative_path"].startswith("Operations/") else "repo"
        fixture = FIXTURES / root / case["relative_path"]

        assert fixture.is_file(), case
        assert case["marker"] in fixture.read_text(), case
