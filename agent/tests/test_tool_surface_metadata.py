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


def _install_fake_dependencies() -> None:
    core_mod = types.ModuleType('langchain_core')
    tools_mod = types.ModuleType('langchain_core.tools')
    pyd_mod = types.ModuleType('pydantic')

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

        async def ainvoke(self, kwargs):
            return kwargs

        def invoke(self, kwargs):
            return kwargs

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

from yolostudio_agent.agent.client.remote_transfer_tools import build_local_transfer_tools
from yolostudio_agent.agent.client.tool_adapter import adapt_tool_for_chat_model


def main() -> None:
    remote_tools = {tool.name: tool for tool in build_local_transfer_tools()}
    upload_tool = remote_tools['upload_assets_to_remote']
    list_tool = remote_tools['list_remote_profiles']

    assert upload_tool.metadata['confirmation_required'] is True, upload_tool.metadata
    assert upload_tool.metadata['open_world'] is True, upload_tool.metadata
    assert 'remote-transfer' in upload_tool.tags, upload_tool.tags

    assert list_tool.metadata['read_only'] is True, list_tool.metadata
    assert list_tool.metadata['confirmation_required'] is False, list_tool.metadata

    class _AnnotatedTool(type(upload_tool)):
        pass

    base_tool = _AnnotatedTool()
    base_tool.name = 'inspect_training_run'
    base_tool.description = 'fake'
    base_tool.args_schema = None
    base_tool.metadata = {'confirmation_required': False, 'surface': 'mcp'}
    base_tool.tags = ['training', 'read-only']
    base_tool.annotations = types.SimpleNamespace(readOnlyHint=True, destructiveHint=False)

    adapted = adapt_tool_for_chat_model(base_tool)
    assert adapted.metadata['surface'] == 'mcp', adapted.metadata
    assert adapted.metadata['confirmation_required'] is False, adapted.metadata
    assert 'training' in adapted.tags, adapted.tags
    assert getattr(adapted, 'annotations', None) is not None
    assert adapted.annotations.readOnlyHint is True

    print('tool surface metadata ok')


if __name__ == '__main__':
    main()
