"""Pytest bootstrap.

Loads ``.env`` before test collection so the API-key skip guards in
``test_graph_smoke.py`` see the configured key. Without this, pytest runs with a
bare environment and silently skips every live-LLM test.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is an optional convenience
    pass
