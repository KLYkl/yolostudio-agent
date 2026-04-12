from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.tool_adapter import _stringify_tool_result


def main() -> None:
    value = [
        {'type': 'text', 'text': '{"ok": true, "summary": "hello"}'},
        {'type': 'meta', 'foo': 'bar'},
    ]
    text = _stringify_tool_result(value)
    assert 'hello' in text
    assert 'meta' in text
    print('tool adapter smoke ok')


if __name__ == '__main__':
    main()
