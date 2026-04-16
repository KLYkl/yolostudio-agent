from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import sys
import types

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
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

try:
    import langchain_mcp_adapters.client  # type: ignore  # noqa: F401
except Exception:
    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters.client'] = client_mod

try:
    import pydantic  # type: ignore  # noqa: F401
except Exception:
    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, description=''):
        del description
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

try:
    import langgraph.prebuilt  # type: ignore  # noqa: F401
    import langgraph.types  # type: ignore  # noqa: F401
    import langgraph.checkpoint.memory  # type: ignore  # noqa: F401
except Exception:
    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in extract route smoke')

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

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_extract_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='extract-route-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, object]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs):
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'preview_extract_images':
                assert kwargs['source_path'] == '/data/raw/images'
                assert kwargs['output_dir'] == '/tmp/extract_preview'
                assert kwargs['selection_mode'] == 'count'
                assert kwargs['count'] == 20
                result = {
                    'ok': True,
                    'summary': '预览完成: 可用图片 180 张，计划抽取 20 张（global / count）',
                    'available_images': 180,
                    'planned_extract_count': 20,
                    'sample_images': ['/data/raw/images/a.jpg'],
                    'output_dir': '/tmp/extract_preview',
                    'warnings': [],
                    'next_actions': ['可继续调用 extract_images 真正执行抽取'],
                }
                client._apply_to_state('preview_extract_images', result, kwargs)
                return result
            if tool_name == 'extract_images':
                assert kwargs['source_path'] == '/data/raw/images'
                assert kwargs['selection_mode'] == 'ratio'
                assert abs(float(kwargs['ratio']) - 0.1) < 1e-9
                result = {
                    'ok': True,
                    'summary': '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个',
                    'extracted': 18,
                    'labels_copied': 18,
                    'conflict_count': 0,
                    'output_dir': '/tmp/extract_run',
                    'workflow_ready_path': '/tmp/extract_run',
                    'warnings': [],
                    'next_actions': ['可直接对输出目录继续 scan_dataset / validate_dataset: /tmp/extract_run'],
                    'output_img_dir': '/tmp/extract_run/images',
                    'output_label_dir': '/tmp/extract_run/labels',
                }
                client._apply_to_state('extract_images', result, kwargs)
                return result
            if tool_name == 'scan_videos':
                assert kwargs['source_path'] == '/data/videos'
                result = {
                    'ok': True,
                    'summary': '视频扫描完成: 发现 3 个视频文件',
                    'total_videos': 3,
                    'sample_videos': ['/data/videos/a.mp4'],
                    'next_actions': ['如需抽帧，可继续调用 extract_video_frames'],
                }
                client._apply_to_state('scan_videos', result, kwargs)
                return result
            assert tool_name == 'extract_video_frames'
            assert kwargs['source_path'] == '/data/videos'
            assert kwargs['output_dir'] == '/tmp/frames_out'
            assert kwargs['mode'] == 'interval'
            assert kwargs['frame_interval'] == 10
            result = {
                'ok': True,
                'summary': '视频抽帧完成: 最终保留 12 帧（原始抽取 12 / 去重移除 0）',
                'total_frames': 120,
                'extracted': 12,
                'final_count': 12,
                'output_dir': '/tmp/frames_out',
                'warnings': [],
                'next_actions': ['可将抽帧输出目录继续作为图片输入使用: /tmp/frames_out'],
            }
            client._apply_to_state('extract_video_frames', result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        routed = await client._try_handle_mainline_intent('先预览一下从 /data/raw/images 抽 20 张图片到 /tmp/extract_preview，不要真的执行', 'thread-1')
        assert routed is not None, routed
        assert routed['status'] == 'completed', routed
        assert '计划抽取 20 张' in routed['message'], routed
        assert calls[-1][0] == 'preview_extract_images', calls
        preview_call_count = len(calls)

        routed_cached = await client._try_handle_mainline_intent('再预览一下从 /data/raw/images 抽 20 张图片到 /tmp/extract_preview，不要真的执行', 'thread-1b')
        assert routed_cached is not None, routed_cached
        assert routed_cached['status'] == 'completed', routed_cached
        assert '计划抽取 20 张' in routed_cached['message'], routed_cached
        assert len(calls) == preview_call_count, calls

        routed2 = await client._try_handle_mainline_intent('从 /data/raw/images 抽 10% 的图片到 /tmp/extract_run', 'thread-2')
        assert routed2 is not None, routed2
        assert routed2['status'] == 'completed', routed2
        assert '实际抽取 18 张' in routed2['message'], routed2
        assert client.session_state.active_dataset.dataset_root == '/tmp/extract_run'

        routed3 = await client._try_handle_mainline_intent('扫描一下 /data/videos 目录里有多少视频', 'thread-3')
        assert routed3 is not None, routed3
        assert '发现 3 个视频' in routed3['message'], routed3
        assert calls[-1][0] == 'scan_videos', calls
        video_scan_call_count = len(calls)

        routed3_cached = await client._try_handle_mainline_intent('再扫描一下 /data/videos 目录里有多少视频', 'thread-3b')
        assert routed3_cached is not None, routed3_cached
        assert '发现 3 个视频' in routed3_cached['message'], routed3_cached
        assert len(calls) == video_scan_call_count, calls

        routed4 = await client._try_handle_mainline_intent('从 /data/videos 抽帧，每 10 帧抽 1 帧，输出到 /tmp/frames_out', 'thread-4')
        assert routed4 is not None, routed4
        assert '最终保留 12 帧' in routed4['message'], routed4
        assert calls[-1][0] == 'extract_video_frames', calls

        print('extract route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
