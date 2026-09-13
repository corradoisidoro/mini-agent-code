"""
Unit tests for agent.py's run_agent loop — scoped to its own success/fail
outcomes: does it return the right thing when the model finishes, when it
calls tools, when it never converges, and when Ollama itself fails.

The real network call (agent.client.chat) and real tool execution
(agent.run_tool) are always mocked here — tools.py's own logic is already
covered in test_tools.py, and hitting a real Ollama server has no place
in a unit test.
"""

from types import SimpleNamespace

import httpx
import ollama
import pytest

import agent


def make_tool_call(name, arguments=None):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments or {})
    )


def chat_returning(*responses):
    """Build a fake agent.client.chat that returns each response in
    sequence, one per call, then keeps returning the last one."""
    responses = list(responses)

    def fake_chat(**kwargs):
        return responses.pop(0) if len(responses) > 1 else responses[0]

    return fake_chat


# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------


def test_run_agent_returns_plain_text_when_no_tool_calls(monkeypatch):
    monkeypatch.setattr(
        agent.client,
        "chat",
        chat_returning({"message": {"role": "assistant", "content": "all done"}}),
    )

    result = agent.run_agent([{"role": "user", "content": "hi"}])

    assert result == "all done"


def test_run_agent_runs_tool_calls_then_returns_final_text(monkeypatch):
    tool_call = make_tool_call("list_files", {"path": "."})
    first_response = {
        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    }
    second_response = {"message": {"role": "assistant", "content": "finished the task"}}

    monkeypatch.setattr(
        agent.client, "chat", chat_returning(first_response, second_response)
    )
    monkeypatch.setattr(agent, "run_tool", lambda tc: "a.txt\nb.txt")

    messages = [{"role": "user", "content": "list files"}]
    result = agent.run_agent(messages)

    assert result == "finished the task"
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages == [
        {"role": "tool", "name": "list_files", "content": "a.txt\nb.txt"}
    ]


def test_run_agent_stops_after_max_turns(monkeypatch):
    monkeypatch.setattr(agent.SETTINGS, "max_turns", 2)
    tool_call = make_tool_call("list_files")
    never_finishes = {
        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    }
    call_count = 0

    def fake_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        return never_finishes

    monkeypatch.setattr(agent.client, "chat", fake_chat)
    monkeypatch.setattr(agent, "run_tool", lambda tc: "ok")

    result = agent.run_agent([{"role": "user", "content": "loop forever"}])

    assert result == (
        "🟡 Stopped: reached the maximum number of tool-call turns for this request."
    )
    assert call_count == 2


# ---------------------------------------------------------------------------
# fail paths — Ollama itself is unreachable / rejects / times out
# ---------------------------------------------------------------------------


def test_run_agent_exits_on_connection_error(monkeypatch):
    def fake_chat(**kwargs):
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(agent.client, "chat", fake_chat)

    with pytest.raises(SystemExit) as exc_info:
        agent.run_agent([{"role": "user", "content": "hi"}])

    assert exc_info.value.code == 1


def test_run_agent_exits_on_timeout(monkeypatch):
    def fake_chat(**kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(agent.client, "chat", fake_chat)

    with pytest.raises(SystemExit) as exc_info:
        agent.run_agent([{"role": "user", "content": "hi"}])

    assert exc_info.value.code == 1


def test_run_agent_exits_on_ollama_response_error(monkeypatch):
    def fake_chat(**kwargs):
        raise ollama.ResponseError("model not found")

    monkeypatch.setattr(agent.client, "chat", fake_chat)

    with pytest.raises(SystemExit) as exc_info:
        agent.run_agent([{"role": "user", "content": "hi"}])

    assert exc_info.value.code == 1
