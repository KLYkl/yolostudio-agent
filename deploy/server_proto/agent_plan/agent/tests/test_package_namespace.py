from __future__ import annotations

import importlib
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def test_public_namespace_imports_agent_modules() -> None:
    module = importlib.import_module("yolostudio_agent.agent.client.session_state")
    assert hasattr(module, "SessionState")
