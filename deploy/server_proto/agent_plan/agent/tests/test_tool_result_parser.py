from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')

    class _ToolMessage:
        def __init__(self, content=''):
            self.content = content

    messages_mod.ToolMessage = _ToolMessage
    core_mod.messages = messages_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod

from yolostudio_agent.agent.client.tool_result_parser import parse_tool_payload


def main() -> None:
    parsed = parse_tool_payload('{"ok": true, "summary": "done"}')
    assert parsed['ok'] is True, parsed
    assert parsed['summary'] == 'done', parsed

    invalid = parse_tool_payload('not-json-at-all')
    assert invalid['ok'] is False, invalid
    assert invalid['error'] == 'invalid_tool_result_payload', invalid
    assert invalid['raw'] == 'not-json-at-all', invalid

    empty = parse_tool_payload('')
    assert empty['ok'] is False, empty
    assert empty['error'] == 'empty_tool_result_payload', empty

    non_object = parse_tool_payload('[1, 2, 3]')
    assert non_object['ok'] is False, non_object
    assert non_object['error'] == 'non_object_tool_result_payload', non_object
    assert non_object['value'] == [1, 2, 3], non_object

    print('tool result parser ok')


if __name__ == '__main__':
    main()
