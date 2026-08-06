"""Pytest bootstrap: fake Isaac out before anything imports the simulation code."""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
SOURCE_DIR = os.path.join(REPO_DIR, "source")

for path in (SOURCE_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaac_stubs  # noqa: E402

# Must happen at import time: pytest imports conftest before collecting test
# modules, and those import the simulation code at their own module level.
isaac_stubs.install()
