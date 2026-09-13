"""
Unit tests for tools.py — scoped to the success/fail *outcome* logic:
what each function returns when it works and when it doesn't.

Deliberately out of scope: the y/n confirmation prompts, the
decline-exits-the-program behavior in write_file/run_command, and
agent.py's LLM-call error handling. Those are control-flow/UX paths,
not "did the operation succeed or fail" logic.
"""

import subprocess
from types import SimpleNamespace

import tools


def make_tool_call(name, arguments):
    """Build a stand-in for the object Ollama passes to run_tool: it only
    needs to expose .function.name and .function.arguments."""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_success_lists_files_and_dirs(tmp_path):
    (tmp_path / "b.txt").write_text("content")
    (tmp_path / "a_dir").mkdir()

    result = tools.list_files(str(tmp_path))

    assert result.splitlines() == ["a_dir/", "b.txt"]  # sorted, dirs suffixed


def test_list_files_success_empty_directory(tmp_path):
    assert tools.list_files(str(tmp_path)) == "(empty directory)"


def test_list_files_fail_missing_path():
    result = tools.list_files("/no/such/path")
    assert result.startswith("Error listing files:")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_success(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world")

    assert tools.read_file(str(f)) == "hello world"


def test_read_file_fail_missing_file(tmp_path):
    result = tools.read_file(str(tmp_path / "missing.txt"))
    assert result.startswith("Error reading file:")


def test_read_file_fail_invalid_utf8(tmp_path):
    f = tmp_path / "binary.dat"
    f.write_bytes(b"\xff\xfe\x00\x01")  # not valid UTF-8

    result = tools.read_file(str(f))
    assert result.startswith("Error reading file:")


# ---------------------------------------------------------------------------
# write_file (confirmation mocked to "y" — the prompt itself isn't in scope)
# ---------------------------------------------------------------------------


def test_write_file_success(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    target = tmp_path / "out.txt"

    result = tools.write_file(str(target), "some content")

    assert target.read_text() == "some content"
    assert result == f"Saved {target} (12 characters)"


def test_write_file_fail_returns_error_string(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    # Writing to a path that is actually a directory raises an OSError
    # (IsADirectoryError on POSIX) — a reliable way to trigger the fail path.

    result = tools.write_file(str(tmp_path), "some content")

    assert result.startswith("Error writing file:")


# ---------------------------------------------------------------------------
# run_command (confirmation mocked to "y")
# ---------------------------------------------------------------------------


def test_run_command_success_with_output(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    result = tools.run_command("echo hello")

    assert result == "hello"


def test_run_command_success_no_output_reports_exit_code(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    result = tools.run_command("exit 1")

    assert result == "(no output, exit code 1)"


def test_run_command_fail_timeout(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="sleep 999", timeout=5)

    monkeypatch.setattr(tools.subprocess, "run", raise_timeout)

    result = tools.run_command("sleep 999")

    assert result == "Error: Command timed out after 5 seconds."


def test_run_command_fail_to_start(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    def raise_oserror(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(tools.subprocess, "run", raise_oserror)

    result = tools.run_command("whatever")

    assert result == "Error starting command: no such executable"


# ---------------------------------------------------------------------------
# run_tool dispatch
# ---------------------------------------------------------------------------


def test_run_tool_success_returns_underlying_result(monkeypatch):
    monkeypatch.setitem(tools.TOOLS, "list_files", lambda path=".": "ok")

    result = tools.run_tool(make_tool_call("list_files", {}))

    assert result == "ok"


def test_run_tool_fail_unknown_tool_name():
    result = tools.run_tool(make_tool_call("does_not_exist", {}))
    assert result == "Error: Tool 'does_not_exist' is not recognized."


def test_run_tool_fail_invalid_arguments():
    # list_files doesn't accept a `bogus` kwarg -> TypeError inside the call.
    result = tools.run_tool(make_tool_call("list_files", {"bogus": 1}))
    assert result.startswith("Error: Invalid arguments for tool 'list_files':")


def test_run_tool_fail_unexpected_exception(monkeypatch):
    def boom(**kwargs):
        raise ValueError("something went wrong")

    monkeypatch.setitem(tools.TOOLS, "list_files", boom)

    result = tools.run_tool(make_tool_call("list_files", {}))

    assert (
        result == "Error: tool 'list_files' failed unexpectedly. See logs for details."
    )
