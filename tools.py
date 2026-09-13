"""
Tool implementations available to the agent, their JSON schemas (as
required by Ollama's function-calling API), and the dispatch logic that
runs a tool the model asked for.

Every tool function returns a string rather than raising, so a failure —
a missing file, a bad path — becomes something the model can see and
react to, instead of crashing the whole session. The one deliberate
exception: declining the write_file/run_command confirmation exits the
program immediately, by design — see those functions' docstrings.
"""

import os
import subprocess
import sys

from config import SETTINGS, logger


def list_files(path="."):
    """Return a newline-separated listing of a directory's contents.

    Directories are suffixed with '/'. Defaults to the current directory.
    Returns an error message string (not an exception) if the path is
    invalid or inaccessible.
    """

    try:
        entries = []
        for entry in os.scandir(path):
            entries.append(entry.name + ("/" if entry.is_dir() else ""))
        return "\n".join(sorted(entries)) or "(empty directory)"
    except OSError as e:
        # Expected failure modes: missing path, not a directory, no permission.
        return f"Error listing files: {e}"


def read_file(path):
    """Read and return the full text contents of a file.

    Returns an error message string if the file doesn't exist, isn't
    readable, or isn't valid UTF-8 text.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        # OSError covers missing/unreadable files; UnicodeDecodeError covers
        # non-UTF-8 (e.g. binary) files. A real bug still propagates.
        return f"Error reading file: {e}"


def write_file(path, content):
    """Write text content to a file, after asking the user to confirm.

    Warns explicitly if this would overwrite an existing file. Returns a
    message describing what happened (saved / error) if the write goes
    ahead. If the user declines, this exits the whole program immediately
    instead of returning a "declined" message — kept simple on purpose,
    see the module docstring.
    """

    if os.path.exists(path):
        prompt = f"   ⚠️ Overwrite existing file '{path}'? [y/n] "
    else:
        prompt = f"   Create file '{path}'? [y/n] "

    answer = input(prompt)
    if answer.strip().lower() != "y":
        # Kept simple on purpose: declining exits the program rather than
        # returning a message the model could react to.
        logger.info("The user declined to write this file.")
        sys.exit(0)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {path} ({len(content)} characters)"
    except OSError as e:
        return f"Error writing file: {e}"


def run_command(command):
    """Run a shell command after asking the user to confirm.

    Captures stdout/stderr and returns them combined if the command runs.
    If the user declines, this exits the whole program immediately instead
    of returning a "declined" message — kept simple on purpose, see the
    module docstring.
    """

    # Safety confirmation check preserved.
    answer = input(f"   Run '{command}'? [y/n] ")
    if answer.strip().lower() != "y":
        # Kept simple on purpose: declining exits the program rather than
        # returning a message the model could react to.
        logger.info("The user declined to run this command.")
        sys.exit(0)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SETTINGS.command_timeout,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return output or f"(no output, exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {SETTINGS.command_timeout} seconds."
    except OSError as e:
        return f"Error starting command: {e}"


# Maps a tool name (as the model will call it) to the Python function that
# implements it. Keep this, TOOL_SCHEMAS, and run_tool in sync when adding
# a new tool: the schema tells the model the tool exists, this dict lets
# run_tool actually call it.
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

# JSON schemas describing each tool to the model, in the format Ollama's function-calling API expects.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in a directory. Folders end with /.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, e.g. '.' (defaults to the current directory if omitted)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes text content to a specified file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path where content should be written.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write into the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a shell command. The user will be prompted to confirm execution before the command runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal/shell command string to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def run_tool(tool_call):
    """Execute the tool an LLM tool_call requested, and return its result as a string.

    `tool_call` is the object Ollama returns for a single requested call:
    it exposes `.function.name` and `.function.arguments` (a dict of
    kwargs). Unknown tool names and argument mismatches are reported back
    as strings rather than raised, so a confused model gets feedback
    instead of crashing the agent.
    """

    name = tool_call.function.name
    args = tool_call.function.arguments

    # Track tool invocation with the logging system
    logger.info("Executing tool: %s with arguments: %s", name, args)

    if name not in TOOLS:
        logger.error("Tool '%s' is not recognized.", name)
        return f"Error: Tool '{name}' is not recognized."

    try:
        # Safe argument unpack safeguarding unexpected parameters from LLM model
        result = str(TOOLS[name](**args))
        # Truncate long results (e.g. read_file on a big file, or verbose shell
        # output) to keep the log readable — the full result still goes back to
        # the model via the return value, this is only for what gets logged.
        preview = result if len(result) < 80 else result[:77] + "..."
        logger.info("Tool '%s' finished: %s", name, preview)
        return result
    except TypeError as e:
        # The model sent arguments that don't match the tool's signature.
        logger.error("Invalid arguments for tool '%s': %s", name, e)
        return f"Error: Invalid arguments for tool '{name}': {e}"
    except Exception:
        # Last line of defense: an unexpected bug in a tool function, not
        # one of its normal failure modes. logger.exception logs the full
        # traceback so the bug is visible (not silently hidden), while
        # still keeping it from crashing the whole agent session.
        logger.exception("Unexpected error running tool '%s'", name)
        return f"Error: tool '{name}' failed unexpectedly. See logs for details."
