from __future__ import annotations

import sys
import types
import json
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_dependencies() -> None:
    core_mod = types.ModuleType('langchain_core')
    tools_mod = types.ModuleType('langchain_core.tools')
    pyd_mod = types.ModuleType('pydantic')

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool(_BaseTool):
        @classmethod
        def from_function(cls, func=None, coroutine=None, name='', description='', args_schema=None, return_direct=False):
            tool = cls()
            tool.func = func
            tool.coroutine = coroutine
            tool.name = name
            tool.description = description
            tool.args_schema = args_schema
            tool.return_direct = return_direct
            return tool

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.tools = tools_mod
    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.tools'] = tools_mod
    sys.modules['pydantic'] = pyd_mod


_install_fake_dependencies()

from yolostudio_agent.agent.client.tool_adapter import _serialize_tool_result_for_chat_model, _stringify_tool_result


def main() -> None:
    value = [
        {'type': 'text', 'text': '{"ok": true, "summary": "hello"}'},
        {'type': 'meta', 'foo': 'bar'},
    ]
    text = _stringify_tool_result(value)
    assert 'hello' in text
    assert 'meta' in text

    structured = _stringify_tool_result(
        {
            'ok': True,
            'summary': '训练前检查完成',
            'readiness_overview': {
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
            },
            'action_candidates': [
                {
                    'action': 'prepare_dataset',
                    'tool': 'prepare_dataset_for_training',
                    'description': '先准备数据集并生成 data.yaml',
                }
            ],
        }
    )
    assert '训练前检查完成' in structured
    assert '概览:' in structured
    assert '建议动作: 先准备数据集并生成 data.yaml' in structured

    preserved_dict = json.loads(
        _serialize_tool_result_for_chat_model({'ok': True, 'summary': 'hello', 'profiles': []})
    )
    assert preserved_dict['ok'] is True
    assert preserved_dict['profiles'] == []

    preserved_content_blocks = json.loads(
        _serialize_tool_result_for_chat_model([{'type': 'text', 'text': '{"ok": true, "profiles": []}'}])
    )
    assert preserved_content_blocks['ok'] is True
    assert preserved_content_blocks['profiles'] == []

    wrapped_raw = json.loads(_serialize_tool_result_for_chat_model('plain-text-result'))
    assert wrapped_raw['ok'] is True
    assert wrapped_raw['raw'] == 'plain-text-result'

    print('tool adapter smoke ok')


if __name__ == '__main__':
    main()
