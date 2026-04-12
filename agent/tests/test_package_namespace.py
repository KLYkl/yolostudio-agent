from __future__ import annotations

import importlib
import sys
import warnings
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


def test_compatibility_namespace_still_imports() -> None:
    for key in list(sys.modules):
        if key == "agent_plan" or key.startswith("agent_plan."):
            sys.modules.pop(key, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("agent_plan.agent.client.session_state")
    assert hasattr(module, "SessionState")
    assert any("deprecated" in str(item.message).lower() for item in caught)
