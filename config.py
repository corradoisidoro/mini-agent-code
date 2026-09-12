"""
Application configuration and logging setup.

Loads settings from environment variables (or a local .env file) into a
single validated `SETTINGS` object, and configures the shared `logger`
used across the agent. Every other module imports from here rather than
reading os.environ directly, so there is exactly one source of truth for
config values.
"""

import logging
import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Required configuration values, validated at startup.

    Values are read from environment variables (case-insensitive) or a
    `.env` file in the current directory. If any field is missing or has
    the wrong type, pydantic raises a ValidationError and the app exits
    immediately below, rather than failing later mid-conversation.
    """

    model: str  # Ollama model tag to use
    think: bool  # false: skip chain-of-thought reasoning tokens (big speedup on CPU)
    temperature: float  # Sampling temperature passed to the model
    num_ctx: int  # Context window size (tokens) requested from Ollama
    command_timeout: int  # Seconds before a shell command is killed, and before an unresponsive Ollama call is aborted
    max_turns: int  # Safety valve: max LLM round-trips per user message

    # Tell pydantic-settings how to locate and handle the file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Avoids validation errors if unrelated variables exist in your file
    )


# Load settings globally once, at import time, so every module shares the
# same validated instance instead of re-reading the environment.
try:
    SETTINGS = Settings()
except ValidationError as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")
