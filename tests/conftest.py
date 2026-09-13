"""
Shared pytest setup.

`config.py` validates all required settings at import time (fail-fast on
startup, by design). That means simply importing `tools` also imports
`config`, which needs a full set of settings present — so we set harmless
defaults here *before* anything imports `tools`, rather than requiring a
real `.env` file to exist just to run the test suite.
"""

import os
import sys
from pathlib import Path

# tests/ has no __init__.py, so pytest wouldn't otherwise put the project
# root (one level up) on sys.path — add it so `import tools` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# setdefault: don't clobber real values if the environment already has them.
os.environ.setdefault("MODEL", "test-model")
os.environ.setdefault("THINK", "false")
os.environ.setdefault("TEMPERATURE", "0.2")
os.environ.setdefault("NUM_CTX", "2048")
os.environ.setdefault("COMMAND_TIMEOUT", "5")
os.environ.setdefault("MAX_TURNS", "5")
