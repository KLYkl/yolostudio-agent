from __future__ import annotations

import json
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
from langchain_core.messages import AIMessage, ToolMessage


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('realtime route should stay on routed flows, not fallback to graph')


class _ScriptedGraph:
    def __init__(self, script) -> None:
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        tool_name, args, result, final_text = self.script(messages)
        if tool_name:
            self.calls.append((tool_name, dict(args)))
            tool_call_id = f'call-{len(self.calls)}'
            messages.extend(
                [
                    AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': args}]),
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                ]
            )
        messages.append(AIMessage(content=final_text))
        return {'messages': messages}


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


def _make_client(session_id: str, graph=None) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph or _NoLLMGraph(), settings=settings, tool_registry={})
    client.session_state.active_prediction.model = 'demo.pt'
    return client


async def _scenario_scan_cameras_routes() -> None:
    def _scan_camera_script(messages):
        return (
            'scan_cameras',
            {},
            {
                'ok': True,
                'summary': '摄像头扫描完成: 发现 1 个可用摄像头',
                'camera_count': 1,
                'cameras': [{'id': 0, 'name': '摄像头 0'}],
                'next_actions': ['如需开始实时预测，可继续调用 start_camera_prediction'],
            },
            '摄像头扫描完成: 发现 1 个可用摄像头\n摄像头 0',
        )

    graph = _ScriptedGraph(_scan_camera_script)
    client = _make_client('scan-cameras', graph=graph)
    assert await client._try_handle_mainline_intent('先扫描可用摄像头。', 'thread-scan-cameras') is None
    turn = await client.chat('先扫描可用摄像头。')
    assert turn['status'] == 'completed', turn
    assert graph.calls == [('scan_cameras', {})], graph.calls
    assert '摄像头 0' in turn['message'], turn


async def _scenario_rtsp_probe_routes() -> None:
    def _rtsp_probe_script(messages):
        return (
            'test_rtsp_stream',
            {'rtsp_url': 'rtsp://demo/live', 'timeout_ms': 2000},
            {
                'ok': True,
                'summary': 'RTSP 流测试通过：当前地址可连接并能读取视频帧',
                'rtsp_url': 'rtsp://demo/live',
                'next_actions': ['如需开始实时预测，可继续调用 start_rtsp_prediction'],
            },
            'RTSP 流测试通过：当前地址可连接并能读取视频帧\nrtsp://demo/live',
        )

    graph = _ScriptedGraph(_rtsp_probe_script)
    client = _make_client('rtsp-probe', graph=graph)
    assert await client._try_handle_mainline_intent('测一下这个 RTSP 地址能不能用 rtsp://demo/live 超时 2 秒', 'thread-rtsp-probe') is None
    turn = await client.chat('测一下这个 RTSP 地址能不能用 rtsp://demo/live 超时 2 秒')
    assert turn['status'] == 'completed', turn
    assert graph.calls == [('test_rtsp_stream', {'rtsp_url': 'rtsp://demo/live', 'timeout_ms': 2000})], graph.calls
    assert 'rtsp://demo/live' in turn['message'], turn


async def _scenario_start_camera_routes() -> None:
    def _start_camera_script(messages):
        return (
            'start_camera_prediction',
            {'model': 'demo.pt', 'camera_id': 0, 'max_frames': 10},
            {
                'ok': True,
                'summary': '实时预测已启动: camera 源 camera:0',
                'session_id': 'realtime-camera-12345678',
                'source_type': 'camera',
                'source_label': 'camera:0',
                'output_dir': '/tmp/realtime-camera',
                'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
            },
            '实时预测已启动: camera 源 camera:0',
        )

    graph = _ScriptedGraph(_start_camera_script)
    client = _make_client('start-camera', graph=graph)
    assert await client._try_handle_mainline_intent('用 0 号摄像头开始预测，最多 10 帧', 'thread-start-camera') is None
    turn = await client.chat('用 0 号摄像头开始预测，最多 10 帧')
    assert turn['status'] == 'completed', turn
    assert graph.calls == [('start_camera_prediction', {'model': 'demo.pt', 'camera_id': 0, 'max_frames': 10})], graph.calls
    assert 'camera:0' in turn['message'], turn
    assert client.session_state.active_prediction.realtime_session_id == 'realtime-camera-12345678'


async def _scenario_start_rtsp_routes() -> None:
    def _start_rtsp_script(messages):
        return (
            'start_rtsp_prediction',
            {'model': 'demo.pt', 'rtsp_url': 'rtsp://demo/live', 'max_frames': 5},
            {
                'ok': True,
                'summary': '实时预测已启动: rtsp 源 rtsp://demo/live',
                'session_id': 'realtime-rtsp-12345678',
                'source_type': 'rtsp',
                'source_label': 'rtsp://demo/live',
                'output_dir': '/tmp/realtime-rtsp',
                'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
            },
            '实时预测已启动: rtsp 源 rtsp://demo/live',
        )

    graph = _ScriptedGraph(_start_rtsp_script)
    client = _make_client('start-rtsp', graph=graph)
    assert await client._try_handle_mainline_intent('用 rtsp://demo/live 开始预测，最多 5 帧', 'thread-start-rtsp') is None
    turn = await client.chat('用 rtsp://demo/live 开始预测，最多 5 帧')
    assert turn['status'] == 'completed', turn
    assert graph.calls == [('start_rtsp_prediction', {'model': 'demo.pt', 'rtsp_url': 'rtsp://demo/live', 'max_frames': 5})], graph.calls
    assert 'rtsp://demo/live' in turn['message'], turn


async def _scenario_start_screen_routes() -> None:
    def _start_screen_script(messages):
        return (
            'start_screen_prediction',
            {'model': 'demo.pt', 'screen_id': 2, 'max_frames': 20},
            {
                'ok': True,
                'summary': '实时预测已启动: screen 源 screen:2',
                'session_id': 'realtime-screen-12345678',
                'source_type': 'screen',
                'source_label': 'screen:2',
                'output_dir': '/tmp/realtime-screen',
                'next_actions': ['可继续调用 check_realtime_prediction_status 查看实时进度'],
            },
            '实时预测已启动: screen 源 screen:2',
        )

    graph = _ScriptedGraph(_start_screen_script)
    client = _make_client('start-screen', graph=graph)
    assert await client._try_handle_mainline_intent('对 2 号屏幕开始预测，最多 20 帧', 'thread-start-screen') is None
    turn = await client.chat('对 2 号屏幕开始预测，最多 20 帧')
    assert turn['status'] == 'completed', turn
    assert graph.calls == [('start_screen_prediction', {'model': 'demo.pt', 'screen_id': 2, 'max_frames': 20})], graph.calls
    assert 'screen:2' in turn['message'], turn


async def _scenario_status_and_stop_routes() -> None:
    def _status_script(messages):
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        if '停止实时预测' in user_text:
            return (
                'stop_realtime_prediction',
                {'session_id': 'realtime-camera-12345678'},
                {
                    'ok': True,
                    'summary': '实时预测已停止: 已处理 12 帧，检测到 6 个目标',
                    'session_id': 'realtime-camera-12345678',
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
                },
                '实时预测已停止: 已处理 12 帧，检测到 6 个目标。报告已写入 /tmp/realtime-camera/realtime_prediction_report.json',
            )
        return (
            'check_realtime_prediction_status',
            {'session_id': 'realtime-camera-12345678'},
            {
                'ok': True,
                'summary': '实时预测运行中: 已处理 12 帧, 有检测 4 帧, 总检测 6',
                'session_id': 'realtime-camera-12345678',
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
            },
            '实时预测运行中: 已处理 12 帧, 有检测 4 帧, 总检测 6',
        )

    graph = _ScriptedGraph(_status_script)
    client = _make_client('status-stop', graph=graph)
    client.session_state.active_prediction.realtime_session_id = 'realtime-camera-12345678'
    client.session_state.active_prediction.realtime_source_type = 'camera'
    client.session_state.active_prediction.realtime_source_label = 'camera:0'
    client.session_state.active_prediction.realtime_status = 'running'
    assert await client._try_handle_mainline_intent('看下实时预测状态', 'thread-status') is None
    status_turn = await client.chat('看下实时预测状态')
    assert status_turn['status'] == 'completed', status_turn
    assert graph.calls == [('check_realtime_prediction_status', {'session_id': 'realtime-camera-12345678'})], graph.calls
    assert '已处理 12 帧' in status_turn['message'], status_turn

    assert await client._try_handle_mainline_intent('停止实时预测', 'thread-stop') is None
    stop_turn = await client.chat('停止实时预测')
    assert stop_turn['status'] == 'completed', stop_turn
    assert graph.calls[-1] == ('stop_realtime_prediction', {'session_id': 'realtime-camera-12345678'}), graph.calls
    assert 'realtime_prediction_report.json' in stop_turn['message'], stop_turn


async def _scenario_followup_routes() -> None:
    def _followup_script(messages):
        del messages
        return (
            'check_realtime_prediction_status',
            {'session_id': 'realtime-camera-12345678'},
            {
                'ok': True,
                'summary': '实时预测运行中: 已处理 24 帧, 有检测 8 帧, 总检测 11',
                'session_id': 'realtime-camera-12345678',
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
            },
            '实时预测运行中，当前已经处理 24 帧。',
        )

    graph = _ScriptedGraph(_followup_script)
    client = _make_client('followup', graph=graph)
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
    followup_prompt = '现在是什么情况了？我需要详细一点的实时预测信息'
    assert await client._try_handle_mainline_intent(followup_prompt, 'thread-followup') is None
    followup_turn = await client.chat(followup_prompt)
    assert followup_turn['status'] == 'completed', followup_turn
    assert graph.calls == [('check_realtime_prediction_status', {'session_id': 'realtime-camera-12345678'})], graph.calls
    assert '24 帧' in followup_turn['message'], followup_turn


async def _scenario_cached_camera_scan_routes() -> None:
    def _cached_camera_scan_script(messages):
        del messages
        return (
            None,
            {},
            None,
            '摄像头扫描完成: 发现 2 个可用摄像头\ncamera_count=2',
        )

    graph = _ScriptedGraph(_cached_camera_scan_script)
    client = _make_client('cached-scan-cameras', graph=graph)
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
    assert await client._try_handle_mainline_intent('先扫描可用摄像头。', 'thread-cached-scan-cameras') is None
    turn = await client.chat('先扫描可用摄像头。')
    assert turn['status'] == 'completed', turn
    assert '发现 2 个可用摄像头' in turn['message'], turn
    assert 'camera_count=2' in turn['message'], turn
    assert graph.calls == [], graph.calls


async def _scenario_cached_realtime_status_routes() -> None:
    def _cached_status_script(messages):
        del messages
        return (
            None,
            {},
            None,
            '实时预测已停止: 已处理 12 帧，检测到 6 个目标。报告路径 /tmp/realtime-camera/realtime_prediction_report.json',
        )

    graph = _ScriptedGraph(_cached_status_script)
    client = _make_client('cached-realtime-status', graph=graph)
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
    assert await client._try_handle_mainline_intent('看下实时预测状态', 'thread-cached-status') is None
    turn = await client.chat('看下实时预测状态')
    assert turn['status'] == 'completed', turn
    assert '已处理 12 帧' in turn['message'], turn
    assert 'realtime_prediction_report.json' in turn['message'], turn
    assert graph.calls == [], graph.calls


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
