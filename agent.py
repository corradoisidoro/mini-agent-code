"""
Core agent loop.

Sends the running conversation to the LLM via Ollama, executes any tool
calls the model requests, feeds the results back, and repeats — until the
model responds with plain text (task done) or MAX_TURNS round-trips are
used up (safety valve against a looping model).
"""

import sys
import time

import httpx
import ollama

from config import SETTINGS, logger
from tools import TOOL_SCHEMAS, run_tool

# A dedicated client (rather than the module-level ollama.chat shortcut)
# so we can set a request timeout — without it, a hung Ollama server
# would hang the agent indefinitely with no way out.
client = ollama.Client(timeout=SETTINGS.command_timeout)

SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
You can list files, read files, write files, and run shell commands.
Never claim you created, modified, or ran something unless you actually called the corresponding tool and saw its result. 
If you haven't called a tool yet, call it — don't describe what you're about to do instead of doing it.
Use your tools to complete the user's task, then briefly summarize what you did.
The working directory is the folder the user launched you from."""


def run_agent(messages: list):
    """Run the LLM <-> tool-call loop for one user turn.

    `messages` is the full conversation history (including the new user
    message); it is mutated in place as the assistant/tool messages are
    appended, so the caller's history stays in sync across turns.

    Returns the model's final plain-text reply, or a "🟡 Stopped" message
    if MAX_TURNS is reached without one. Exits the process if Ollama isn't
    reachable at all, since there's nothing useful the agent can do without it.
    """

    for turn in range(1, SETTINGS.max_turns + 1):
        logger.info(
            "Calling LLM (turn %d/%d, model=%s)...",
            turn,
            SETTINGS.max_turns,
            SETTINGS.model,
        )
        start = time.perf_counter()
        try:
            response = client.chat(
                model=SETTINGS.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                think=SETTINGS.think,
                options={
                    "temperature": SETTINGS.temperature,
                    "num_ctx": SETTINGS.num_ctx,
                },
            )
        except ConnectionError as e:
            # Ollama isn't running / isn't reachable at all.
            logger.error("🔴 Error: Could not connect to the Ollama service '%s'", e)
            sys.exit(1)  # 1 indicates the program closed due to an error
        except httpx.TimeoutException as e:
            # Ollama accepted the connection but didn't respond within
            # COMMAND_TIMEOUT seconds — e.g. it's stuck loading the model or
            # overloaded. Not caught by ollama's own ConnectionError handling.
            logger.error(
                "🔴 Ollama call timed out after %ss: %s", SETTINGS.command_timeout, e
            )
            sys.exit(1)
        except ollama.ResponseError as e:
            # Ollama is reachable but rejected the request — e.g. the model
            # tag in .env isn't pulled, or the server hit an internal error.
            logger.error("🔴 Ollama returned an error: %s", e)
            sys.exit(1)  # 1 indicates the program closed due to an error

        elapsed = time.perf_counter() - start
        logger.info("LLM call completed in %.2fs", elapsed)

        message = response["message"]
        messages.append(message)

        # No tool calls means the model is done and answerd in plain text
        if not message.get("tool_calls"):
            return message.get("content", "")

        for tool_call in message["tool_calls"]:
            result = run_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": result,
                }
            )

    # Only reached if the loop finishes without the model ever giving a plain-text reply
    return "🟡 Stopped: reached the maximum number of tool-call turns for this request."
