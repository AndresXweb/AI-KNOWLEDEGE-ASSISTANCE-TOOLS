
import json

import pytest

from app.repo_analysis.toolset import RepoToolset


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "proyecto"

    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "node_modules").mkdir()
    (root / ".venv").mkdir()

    (root / "README.md").write_text("# Demo\nHola FastAPI\n")
    (root / "src" / "app.py").write_text(
        "import fastapi\n\napp = fastapi.FastAPI()\n"
    )
    (root / ".env").write_text("API_KEY=super-secreto\n")
    (root / ".env.example").write_text("API_KEY=\n")
    (root / "llave.pem").write_text("-----BEGIN-----")
    (root / ".git" / "config").write_text("secreto-git")
    (root / "node_modules" / "x.js").write_text("fastapi")
    (root / ".venv" / "y.py").write_text("fastapi")
    (root / "imagen.png").write_bytes(b"\x89PNG\x00\x00data")

    (tmp_path / "fuera.txt").write_text("fuera del repo")

    return RepoToolset(root)


def run(toolset, name, **args):
    return toolset.execute(name, json.dumps(args))


def test_exposes_three_read_only_tools(repo):
    names = {
        d["function"]["name"] for d in repo.definitions()
    }

    assert names == {"list_files", "read_file", "search_text"}


def test_list_files_skips_ignored_and_sensitive(repo):
    result = run(repo, "list_files")

    entries = result["data"]["entries"]

    assert "README.md" in entries
    assert "src/" in entries
    assert "src/app.py" in entries
    assert ".env.example" in entries
    assert ".env" not in entries
    assert "llave.pem" not in entries
    assert not any(e.startswith(".git") for e in entries)
    assert not any("node_modules" in e for e in entries)
    assert not any(".venv" in e for e in entries)


def test_list_files_respects_max_depth(repo):
    result = run(repo, "list_files", max_depth=1)

    entries = result["data"]["entries"]

    assert "src/" in entries
    assert "src/app.py" not in entries


def test_read_file_returns_numbered_lines(repo):
    result = run(repo, "read_file", path="src/app.py")

    data = result["data"]

    assert result["ok"] is True
    assert data["total_lines"] == 3
    assert data["content"].startswith("1: import fastapi")


def test_read_file_line_range(repo):
    result = run(
        repo,
        "read_file",
        path="src/app.py",
        start_line=3,
        end_line=3,
    )

    assert result["data"]["content"] == (
        "3: app = fastapi.FastAPI()"
    )


def test_read_file_blocks_path_traversal(repo):
    for path in ["../fuera.txt", "src/../../fuera.txt"]:
        result = run(repo, "read_file", path=path)

        assert result == {
            "ok": False,
            "error": "path_not_allowed",
        }


def test_read_file_blocks_absolute_path(repo, tmp_path):
    outside = str(tmp_path / "fuera.txt")

    result = run(repo, "read_file", path=outside)

    assert result["error"] == "path_not_allowed"


def test_read_file_blocks_sensitive_and_ignored(repo):
    for path in [".env", "llave.pem", ".git/config",
                 "node_modules/x.js"]:
        result = run(repo, "read_file", path=path)

        assert result["error"] == "path_not_allowed", path


def test_read_file_allows_env_example(repo):
    assert run(repo, "read_file", path=".env.example")["ok"]


def test_read_file_rejects_binary_and_missing(repo):
    assert (
        run(repo, "read_file", path="imagen.png")["error"]
        == "binary_file"
    )
    assert (
        run(repo, "read_file", path="no_existe.py")["error"]
        == "not_found"
    )


def test_read_file_caps_lines_per_call(repo, tmp_path):
    big = repo.root / "grande.txt"
    big.write_text("\n".join(f"l{i}" for i in range(1000)))

    data = run(repo, "read_file", path="grande.txt")["data"]

    assert data["end_line"] == 300
    assert data["truncated"] is True
    assert data["total_lines"] == 1000


def test_search_text_is_case_insensitive(repo):
    result = run(repo, "search_text", query="FASTAPI")

    found = {
        (r["path"], r["line"])
        for r in result["data"]["results"]
    }

    assert ("README.md", 2) in found
    assert ("src/app.py", 1) in found


def test_search_text_never_reads_secrets_or_ignored(repo):
    for query in ["super-secreto", "secreto-git", "BEGIN"]:
        result = run(repo, "search_text", query=query)

        assert result["data"]["results"] == []

    paths = {
        r["path"]
        for r in run(repo, "search_text", query="fastapi")
        ["data"]["results"]
    }

    assert not any("node_modules" in p or ".venv" in p
                   for p in paths)


def test_invalid_arguments_and_unknown_tool(repo):
    assert (
        repo.execute("read_file", "no-json")["error"]
        == "invalid_arguments"
    )
    assert (
        run(repo, "search_text", query="a")["error"]
        == "invalid_arguments"
    )
    assert (
        run(repo, "delete_file", path="x")["error"]
        == "unknown_tool"
    )


def test_symlink_pointing_outside_is_blocked(repo, tmp_path):
    link = repo.root / "enlace.txt"

    try:
        link.symlink_to(tmp_path / "fuera.txt")
    except OSError:
        pytest.skip("No se pueden crear symlinks aquí")

    assert (
        run(repo, "read_file", path="enlace.txt")["error"]
        == "path_not_allowed"
    )
