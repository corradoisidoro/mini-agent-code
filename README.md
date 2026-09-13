# Building AI Agents in Pure Python

A small coding agent built directly in Python — no agent framework — to
show exactly what happens between typing a task and a tool call actually
running on your machine.

> **The LLM never runs a tool. It requests one. This code is what
> actually runs it.**

That's the one idea this repo exists to make concrete. Everything else —
the confirmation prompts, the turn limit, the config — is just what it
takes to make that idea safe enough to actually try.

This is a demo, not a hardened tool: it runs shell commands with your
permission but no sandboxing, and it's meant to be read, not installed
and forgotten. Keep that in mind and it's a good one to poke at.

---

## What an agent actually is

Strip away the hype and an agent is three things:

1. **A model** — an API you send messages to. It sends one message back.
   That's it. It generates text; it cannot touch your filesystem, run a
   command, or do anything else on its own.
2. **Tools** — plain Python functions you write (read a file, run a
   command, ...). The model can only *ask* for one to run; your code is
   what actually runs it.
3. **A loop** — call the model, run whatever tools it asked for, feed the
   results back, repeat until it answers in plain text instead of asking
   for another tool. Without the loop you just have a chatbot that can
   describe what it would do.

```
                 ┌───────┐
   You type a    │  You  │
   task          └───┬───┘
                     ▼
              ┌─────────────┐
        ┌────►│    Model    │
        │     └──────┬──────┘
        │            ▼
        │     tool call? ──── no ──► final answer, printed to you
        │            │
        │           yes
        │            ▼
        │     ┌─────────────┐
        │     │  Your code  │  ← this repo
        │     │  runs it    │
        │     └──────┬──────┘
        │            ▼
        │       tool result
        │            │
        └────────────┘
        (result goes back into the conversation, model is called again)
```

This project implements exactly that loop, nothing more.

---

## The four tools

| Tool | What it does |
|---|---|
| `list_files` | See what's in a folder |
| `read_file` | Pull a file's contents into the conversation |
| `write_file` | Create or overwrite a file |
| `run_command` | Run a shell command |

`write_file` and `run_command` can change or break things, so both ask
for an explicit `y/n` confirmation first:

```
Create file 'game.py'? [y/n]
Run 'pytest'? [y/n]
```

**One thing worth knowing before you try this:** declining either prompt
exits the whole program immediately, rather than just skipping that one
step. That's a deliberate simplicity trade-off, not a bug — see the
docstrings in `tools.py` if you're curious why, or want to change it.

---

## How the loop works, in practice

```
You type a task           → appended to messages as role "user"
Call the model             → it replies with text, or with tool calls
No tool calls?              → print the reply, done
Tool calls?                 → your code runs them, results go back in
                               as role "tool", then call the model again
```

The conversation history (the `messages` list) *is* the model's memory —
it doesn't remember anything on its own. Everything it "knows" about the
task is whatever's currently in that list, which is why it's passed back
in full on every call.

A `MAX_TURNS` setting (`.env`) is the safety valve: if the model keeps
asking for tools without ever giving a plain-text answer, the loop stops
and reports it instead of running forever.

### Message roles

| Role | Set by | Meaning |
|---|---|---|
| `system` | you | how the agent should behave |
| `user` | you | the task |
| `assistant` | the model | a reply, or a tool call |
| `tool` | your code | the result of running a requested tool |

### Try it yourself

These are two real prompts used to test this agent — a good way to *see*
the loop rather than just read about it. Run `python main.py` and try:

```
You: Can you make a new directory in the current folder here for a
     snake game, then make a simple snake game in Python using Pygame
     that I can play.
```

```
You: Can you make a new directory in the current folder here for a
     hangman game, then make a simple hangman game in Python using
     Pygame that I can play, and make sure it runs.
```

Watch the terminal logs (`Calling LLM (turn N/…)`, `Executing tool: …`)
while you run these — each line is one iteration of the loop above. The
second prompt is a good one to watch closely: "make sure it runs" is an
open-ended instruction, so it's up to the model to decide *how* to check
that — usually another `run_command` call — which is exactly the kind of
model-driven behavior this project is meant to make visible.

---

## Requirements

- Python >= 3.14
- [Ollama](https://ollama.com) running locally, with a tool-calling-capable
  model pulled (e.g. `ollama pull qwen3:8b`)

## Setup

```bash
cp .env.example .env   # set MODEL to a tool-calling-capable model you've pulled
uv sync                 # or: pip install -e .
```

## Run

```bash
uv run main.py          # if you used uv sync
python main.py          # if you used pip install -e . (with the venv activated)
```

Type a task at the `You:` prompt; type `exit` or `quit` to leave.

## Configuration (`.env`)

| Setting | Meaning |
|---|---|
| `MODEL` | Ollama model tag to use |
| `THINK` | `false` skips chain-of-thought tokens — a big speedup on CPU |
| `TEMPERATURE` | Sampling temperature passed to the model |
| `NUM_CTX` | Context window size (tokens) requested from Ollama |
| `COMMAND_TIMEOUT` | Seconds before a shell command is killed — and before an unresponsive Ollama call is aborted (the two share one setting, on purpose, to keep config small) |
| `MAX_TURNS` | Safety valve: max model↔tool round-trips per task |

All fields are required — `config.py` validates them at startup and exits
immediately with a clear error if any are missing, rather than failing
partway through a conversation.

## Tests

```bash
uv run pytest                                    # if you used uv sync
pip install -e . --group dev && pytest           # otherwise
```

The tests cover the success/fail outcome logic of the tools and the
agent loop (`test_tools.py`, `test_agent.py`) with the real network call
and real tool execution mocked out — no live Ollama server or shell
commands run as part of the test suite.

## Project structure

```
main.py     # REPL entry point
agent.py    # the model <-> tool-call loop
tools.py    # tool implementations + their schemas
config.py   # settings, loaded and validated from .env
tests/      # unit tests for tools.py and agent.py
```

---

## Why no framework

Frameworks like LangChain or CrewAI are useful, but they also hide the
exact mechanics this repo is trying to show. Writing the loop, the tool
schemas, and the dispatch logic directly makes a few questions easy to
answer by just reading the code, instead of trusting an abstraction:

- Where does the model call actually happen?
- Where does a tool call actually get executed?
- Where does the result get added back to the context?
- Where's the loop, and what stops it?
- Where are permissions enforced?

Once those are concrete here, the same questions become a lot easier to
answer for any framework's agent implementation too.

## Limitations (on purpose)

- Declining `write_file`/`run_command` exits the whole program (see above).
- `run_command` executes with `shell=True` and no allow-list — the
  confirmation prompt is the only safety boundary.
- `COMMAND_TIMEOUT` covers both shell commands and the LLM call itself,
  rather than being two separate settings.
- No conversation persistence — history lives only for the life of the
  process.

Extend it if you want (add tools, swap in a different model API, split
the timeouts, add persistence) — the whole point is that there's nothing
hidden to work around.

## License

MIT.
