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
        raise AssertionError('create_react_agent should not be called in realtime route tests')

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
        raise AssertionError('realtime route should stay on routed flows, not fallback to graph')


WORK = Path(__file__).resolve().parent / '_tmp_realtime_prediction_route'


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    def __init__(self, reply) -> None:
        self.reply = reply

    async def ainvoke(self, messages):
        if callable(self.reply):
            return _FakePlannerResponse(self.reply(messages))
        return _FakePlannerResponse(self.reply)


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    client.session_state.active_prediction.model = 'demo.pt'
    return client


async def _scenario_scan_cameras_routes() -> None:
    client = _make_client('scan-cameras')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '摄像头扫描完成: 发现 1 个可用摄像头',
            'camera_count': 1,
            'cameras': [{'id': 0, 'name': '摄像头 0'}],
            'next_actions': ['如需开始实时预测，可继续调用 start_camera_prediction'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('先扫描可用摄像头。')
    assert turn['status'] == 'completed', turn
    assert calls == [('scan_cameras', {})], calls
    assert '摄像头 0' in turn['message'], turn


async def _scenario_rtsp_probe_routes() -> None:
    client = _make_client('rtsp-probe')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': 'RTSP 流测试通过：当前地址可连接并能读取视频帧',
            'rtsp_url': kwargs['rtsp_url'],
            'next_actions': ['如需开始实时预测，可继续调用 start_rtsp_prediction'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('测一下这个 RTSP 地址能不能用 rtsp://demo/live 超时 2 秒')
    assert turn['status'] == 'completed', turn
    assert calls == [('test_rtsp_stream', {'rtsp_url': 'rtsp://demo/live', 'timeout_ms': 2000})], calls
    assert 'rtsp://demo/live' in turn['message'], turn


async def _scenario_start_camera_routes() -> None:
    client = _make_client('start-camera')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '实时预测已启动: camera 源 camera:0',
            'session_id': 'realtime-camera-12345678',
            'source_type': 'camera',
            'source_label': 'camera:0',
            'output_dir': '/tmp/realtime-camera',
            'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('用 0 号摄像头开始预测，最多 10 帧')
    assert turn['status'] == 'completed', turn
    assert calls == [('start_camera_prediction', {'model': 'demo.pt', 'camera_id': 0, 'max_frames': 10})], calls
    assert 'realtime-camera-12345678' in turn['message'], turn
    assert client.session_state.active_prediction.realtime_session_id == 'realtime-camera-12345678'


async def _scenario_start_rtsp_routes() -> None:
    client = _make_client('start-rtsp')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '实时预测已启动: rtsp 源 rtsp://demo/live',
            'session_id': 'realtime-rtsp-12345678',
            'source_type': 'rtsp',
            'source_label': kwargs['rtsp_url'],
            'output_dir': '/tmp/realtime-rtsp',
            'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('用 rtsp://demo/live 开始预测，最多 5 帧')
    assert turn['status'] == 'completed', turn
    assert calls == [('start_rtsp_prediction', {'model': 'demo.pt', 'rtsp_url': 'rtsp://demo/live', 'max_frames': 5})], calls
    assert 'realtime-rtsp-12345678' in turn['message'], turn


async def _scenario_start_screen_routes() -> None:
    client = _make_client('start-screen')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '实时预测已启动: screen 源 screen:2',
            'session_id': 'realtime-screen-12345678',
            'source_type': 'screen',
            'source_label': 'screen:2',
            'output_dir': '/tmp/realtime-screen',
            'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('对 2 号屏幕开始预测，最多 20 帧')
    assert turn['status'] == 'completed', turn
    assert calls == [('start_screen_prediction', {'model': 'demo.pt', 'screen_id': 2, 'max_frames': 20})], calls
    assert 'realtime-screen-12345678' in turn['message'], turn


async def _scenario_status_and_stop_routes() -> None:
    client = _make_client('status-stop')
    client.session_state.active_prediction.realtime_session_id = 'realtime-camera-12345678'
    client.session_state.active_prediction.realtime_source_type = 'camera'
    client.session_state.active_prediction.realtime_source_label = 'camera:0'
    client.session_state.active_prediction.realtime_status = 'running'
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'check_realtime_prediction_status':
            result = {
                'ok': True,
                'summary': '实时预测运行中: 已处理 12 帧, 有检测 4 帧, 总检测 6',
                'session_id': kwargs['session_id'],
                'source_type': 'camera',
                'source_label': 'camera:0',
                'status': 'running',
                'processed_frames': 12,
                'detected_frames': 4,
                'total_detections': 6,
                'class_counts': {'excavator': 6},
                'output_dir': '/tmp/realtime-camera',
                'report_path': '',
                'next_actions': ['如需结束，可调用 stop_realtime_prediction'],
                'running': True,
            }
        else:
            result = {
                'ok': True,
                'summary': '实时预测已停止: 已处理 12 帧，检测到 6 个目标',
                'session_id': kwargs['session_id'],
                'source_type': 'camera',
                'source_label': 'camera:0',
                'status': 'stopped',
                'processed_frames': 12,
                'detected_frames': 4,
                'total_detections': 6,
                'class_counts': {'excavator': 6},
                'output_dir': '/tmp/realtime-camera',
                'report_path': '/tmp/realtime-camera/realtime_prediction_report.json',
                'next_actions': ['可查看实时预测报告: /tmp/realtime-camera/realtime_prediction_report.json'],
                'running': False,
            }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    status_turn = await client.chat('看下实时预测状态')
    assert status_turn['status'] == 'completed', status_turn
    assert calls[0] == ('check_realtime_prediction_status', {'session_id': 'realtime-camera-12345678'}), calls
    assert '已处理 12 帧' in status_turn['message'], status_turn

    stop_turn = await client.chat('停止实时预测')
    assert stop_turn['status'] == 'completed', stop_turn
    assert calls[1] == ('stop_realtime_prediction', {'session_id': 'realtime-camera-12345678'}), calls
    assert 'realtime_prediction_report.json' in stop_turn['message'], stop_turn


async def _scenario_followup_routes() -> None:
    client = _make_client('followup')
    client.session_state.active_prediction.realtime_session_id = 'realtime-camera-12345678'
    client.session_state.active_prediction.realtime_source_type = 'camera'
    client.session_state.active_prediction.realtime_source_label = 'camera:0'
    client.session_state.active_prediction.realtime_status = 'running'
    client.session_state.active_prediction.output_dir = '/tmp/realtime-camera'
    client.session_state.active_prediction.last_realtime_status = {
        'summary': '实时预测运行中: 已处理 12 帧, 有检测 4 帧, 总检测 6',
        'session_id': 'realtime-camera-12345678',
        'source_type': 'camera',
        'source_label': 'camera:0',
        'status': 'running',
        'processed_frames': 12,
        'detected_frames': 4,
        'total_detections': 6,
        'output_dir': '/tmp/realtime-camera',
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '实时预测跟进路由器' in text:
            return '{"action":"status","reason":"用户在追问当前实时预测的详细状态"}'
        return '实时预测运行中，当前已经处理 24 帧。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'check_realtime_prediction_status'
        result = {
            'ok': True,
            'summary': '实时预测运行中: 已处理 24 帧, 有检测 8 帧, 总检测 11',
            'session_id': kwargs['session_id'],
            'source_type': 'camera',
            'source_label': 'camera:0',
            'status': 'running',
            'processed_frames': 24,
            'detected_frames': 8,
            'total_detections': 11,
            'class_counts': {'excavator': 11},
            'output_dir': '/tmp/realtime-camera',
            'report_path': '',
            'action_candidates': [{'tool': 'stop_realtime_prediction', 'description': '如需结束，可停止实时预测'}],
            'running': True,
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    followup_turn = await client.chat('现在是什么情况了？我需要详细一点的实时预测信息')
    assert followup_turn['status'] == 'completed', followup_turn
    assert calls == [('check_realtime_prediction_status', {'session_id': 'realtime-camera-12345678'})], calls
    assert '24 帧' in followup_turn['message'], followup_turn


async def _scenario_cached_camera_scan_routes() -> None:
    client = _make_client('cached-scan-cameras')
    client.session_state.active_prediction.last_realtime_status = {
        'summary': '摄像头扫描完成: 发现 2 个可用摄像头',
        'camera_overview': {'camera_count': 2},
        'camera_count': 2,
        'cameras': [{'id': 0, 'name': '摄像头 0'}, {'id': 1, 'name': '摄像头 1'}],
        'action_candidates': [{'tool': 'start_camera_prediction', 'description': '选择一个摄像头开始实时预测'}],
    }

    async def _unexpected_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f'cached camera scan should not call direct tool: {tool_name} {kwargs}')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('先扫描可用摄像头。')
    assert turn['status'] == 'completed', turn
    assert '发现 2 个可用摄像头' in turn['message'], turn
    assert 'camera_count=2' in turn['message'], turn


async def _scenario_cached_realtime_status_routes() -> None:
    client = _make_client('cached-realtime-status')
    client.session_state.active_prediction.realtime_session_id = 'realtime-camera-12345678'
    client.session_state.active_prediction.realtime_source_type = 'camera'
    client.session_state.active_prediction.realtime_source_label = 'camera:0'
    client.session_state.active_prediction.realtime_status = 'stopped'
    client.session_state.active_prediction.last_realtime_status = {
        'summary': '实时预测已停止: 已处理 12 帧，检测到 6 个目标',
        'session_id': 'realtime-camera-12345678',
        'source_type': 'camera',
        'source_label': 'camera:0',
        'status': 'stopped',
        'processed_frames': 12,
        'detected_frames': 4,
        'total_detections': 6,
        'realtime_status_overview': {'processed_frames': 12},
        'report_path': '/tmp/realtime-camera/realtime_prediction_report.json',
        'action_candidates': [{'tool': 'inspect_prediction_outputs', 'description': '查看导出的实时预测报告'}],
    }

    async def _unexpected_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f'cached realtime status should not call direct tool: {tool_name} {kwargs}')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('看下实时预测状态')
    assert turn['status'] == 'completed', turn
    assert '已处理 12 帧' in turn['message'], turn
    assert 'realtime_prediction_report.json' in turn['message'], turn


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    try:
        run(_scenario_scan_cameras_routes())
        run(_scenario_rtsp_probe_routes())
        run(_scenario_start_camera_routes())
        run(_scenario_start_rtsp_routes())
        run(_scenario_start_screen_routes())
        run(_scenario_status_and_stop_routes())
        run(_scenario_followup_routes())
        run(_scenario_cached_camera_scan_routes())
        run(_scenario_cached_realtime_status_routes())
        print('realtime prediction route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
