from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.tool_adapter import _stringify_tool_result


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
