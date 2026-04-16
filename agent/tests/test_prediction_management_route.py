from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_test_dependencies() -> None:
    fake_openai = types.ModuleType('langchain_openai')
    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    fake_ollama.ChatOllama = _FakeChatOllama
    sys.modules['langchain_openai'] = fake_openai
    sys.modules['langchain_ollama'] = fake_ollama

    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseMessage:
        def __init__(self, content=''):
            self.content = content

    class _AIMessage(_BaseMessage):
        def __init__(self, content='', tool_calls=None):
            super().__init__(content)
            self.tool_calls = tool_calls or []

    class _HumanMessage(_BaseMessage):
        pass

    class _SystemMessage(_BaseMessage):
        pass

    class _ToolMessage(_BaseMessage):
        def __init__(self, content='', name='', tool_call_id=''):
            super().__init__(content)
            self.name = name
            self.tool_call_id = tool_call_id

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

    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = _BaseMessage
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    messages_mod.ToolMessage = _ToolMessage
    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.messages = messages_mod
    core_mod.tools = tools_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod
    sys.modules['langchain_core.tools'] = tools_mod

    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters.client'] = client_mod

    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in prediction management route tests')

    class _Command:
        def __init__(self, resume=None):
            self.resume = resume

    class _InMemorySaver:
        def __init__(self, *args, **kwargs):
            self.storage = {}
            self.writes = {}
            self.blobs = {}

    prebuilt_mod.create_react_agent = _fake_create_react_agent
    types_mod.Command = _Command
    checkpoint_mod.InMemorySaver = _InMemorySaver
    sys.modules['langgraph.prebuilt'] = prebuilt_mod
    sys.modules['langgraph.types'] = types_mod
    sys.modules['langgraph.checkpoint.memory'] = checkpoint_mod


_install_fake_test_dependencies()

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._coroutine_runner import run


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('prediction management follow-up should stay on routed flows, not fallback to graph')


WORK = Path(__file__).resolve().parent / '_tmp_prediction_management_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    client.session_state.active_prediction.output_dir = '/tmp/predict-out'
    client.session_state.active_prediction.report_path = '/tmp/predict-out/prediction_report.json'
    client.session_state.active_prediction.source_path = '/tmp/images'
    client.session_state.active_prediction.model = 'demo.pt'
    client.session_state.active_prediction.last_result = {'summary': '预测完成'}
    return client


async def _scenario_inspect_routes_to_prediction_output_inspection() -> None:
    client = _make_client('inspect')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '预测输出检查完成: 输出目录 /tmp/predict-out，已识别 3 个产物根路径',
            'mode': 'images',
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'artifact_roots': ['/tmp/predict-out', '/tmp/predict-out/annotated', '/tmp/predict-out/labels_yolo'],
            'path_list_files': {},
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('这次预测结果保存在哪？')
    assert turn['status'] == 'completed', turn
    assert calls == [('inspect_prediction_outputs', {'report_path': '/tmp/predict-out/prediction_report.json'})], calls
    assert '/tmp/predict-out' in turn['message'], turn
    assert client.session_state.active_prediction.last_inspection['output_dir'] == '/tmp/predict-out'


async def _scenario_export_routes_to_prediction_report_export() -> None:
    client = _make_client('export')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '预测报告导出完成: 已写出 markdown 报告',
            'mode': 'images',
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'export_path': '/tmp/predict-out/prediction_summary.md',
            'export_format': 'markdown',
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('把这次预测结果导成报告。')
    assert turn['status'] == 'completed', turn
    assert calls and calls[0][0] == 'export_prediction_report', calls
    assert calls[0][1]['report_path'] == '/tmp/predict-out/prediction_report.json', calls
    assert 'prediction_summary.md' in turn['message'], turn
    assert client.session_state.active_prediction.last_export['export_path'].endswith('prediction_summary.md')

    before_cached = len(calls)
    cached_turn = await client.chat('再把这次预测结果导成报告。')
    assert cached_turn['status'] == 'completed', cached_turn
    assert '预测报告导出完成' in cached_turn['message'], cached_turn
    assert len(calls) == before_cached, calls


async def _scenario_path_lists_route_exports_lists() -> None:
    client = _make_client('lists')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '预测路径清单导出完成: 命中 2 条 / 无命中 1 条 / 失败 0 条',
            'mode': 'images',
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'export_dir': '/tmp/predict-out/path_lists',
            'detected_items_path': '/tmp/predict-out/path_lists/detected_items.txt',
            'empty_items_path': '/tmp/predict-out/path_lists/empty_items.txt',
            'failed_items_path': '/tmp/predict-out/path_lists/failed_items.txt',
            'detected_count': 2,
            'empty_count': 1,
            'failed_count': 0,
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('给我这次预测的路径清单。')
    assert turn['status'] == 'completed', turn
    assert calls == [('export_prediction_path_lists', {'report_path': '/tmp/predict-out/prediction_report.json'})], calls
    assert 'path_lists' in turn['message'], turn
    assert client.session_state.active_prediction.last_path_lists['detected_count'] == 2

    before_cached = len(calls)
    cached_turn = await client.chat('再给我这次预测的路径清单。')
    assert cached_turn['status'] == 'completed', cached_turn
    assert '路径清单导出完成' in cached_turn['message'], cached_turn
    assert len(calls) == before_cached, calls


async def _scenario_organize_routes_to_prediction_result_organizer() -> None:
    client = _make_client('organize')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '预测结果整理完成: 已复制 2 个产物到 2 个目录桶',
            'mode': 'images',
            'source_output_dir': '/tmp/predict-out',
            'source_report_path': '/tmp/predict-out/prediction_report.json',
            'destination_dir': '/tmp/predict-out/organized_by_class',
            'organize_by': 'by_class',
            'artifact_preference': 'auto',
            'copied_items': 2,
            'bucket_stats': {'Excavator': 1, 'bulldozer': 1},
            'sample_outputs': ['/tmp/predict-out/organized_by_class/Excavator/a.jpg'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('按类别整理预测结果。')
    assert turn['status'] == 'completed', turn
    assert calls == [('organize_prediction_results', {'report_path': '/tmp/predict-out/prediction_report.json', 'organize_by': 'by_class', 'include_empty': False})], calls
    assert 'organized_by_class' in turn['message'], turn
    assert client.session_state.active_prediction.last_organized_result['copied_items'] == 2


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    try:
        run(_scenario_inspect_routes_to_prediction_output_inspection())
        run(_scenario_export_routes_to_prediction_report_export())
        run(_scenario_path_lists_route_exports_lists())
        run(_scenario_organize_routes_to_prediction_result_organizer())
        print('prediction management route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
