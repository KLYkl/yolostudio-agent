from __future__ import annotations

import asyncio
import json
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
from langchain_core.messages import AIMessage, ToolMessage


class _FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        if '抽 20 张图片到 /tmp/extract_preview' in user_text:
            tool_name = 'preview_extract_images'
            args = {
                'source_path': '/data/raw/images',
                'output_dir': '/tmp/extract_preview',
                'selection_mode': 'count',
                'count': 20,
            }
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
            final_text = '预览完成: 可用图片 180 张，计划抽取 20 张（global / count）'
        elif '抽 10% 的图片到 /tmp/extract_run' in user_text:
            tool_name = 'extract_images'
            args = {
                'source_path': '/data/raw/images',
                'selection_mode': 'ratio',
                'ratio': 0.1,
                'output_dir': '/tmp/extract_run',
            }
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
            final_text = '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个'
        elif '扫描一下 /data/videos' in user_text:
            tool_name = 'scan_videos'
            args = {'source_path': '/data/videos'}
            result = {
                'ok': True,
                'summary': '视频扫描完成: 发现 3 个视频文件',
                'total_videos': 3,
                'sample_videos': ['/data/videos/a.mp4'],
                'next_actions': ['如需抽帧，可继续调用 extract_video_frames'],
            }
            final_text = '视频扫描完成: 发现 3 个视频文件'
        else:
            tool_name = 'extract_video_frames'
            args = {
                'source_path': '/data/videos',
                'output_dir': '/tmp/frames_out',
                'mode': 'interval',
                'frame_interval': 10,
            }
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
            final_text = '视频抽帧完成: 最终保留 12 帧（原始抽取 12 / 去重移除 0）'
        self.calls.append((tool_name, dict(args)))
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': args}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                AIMessage(content=final_text),
            ]
        }


WORK = Path(__file__).resolve().parent / '_tmp_extract_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='extract-route-smoke', memory_root=str(WORK))
        graph = _FakeGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

        preview_prompt = '先预览一下从 /data/raw/images 抽 20 张图片到 /tmp/extract_preview，不要真的执行'
        assert await client._try_handle_mainline_intent(preview_prompt, 'thread-1') is None
        preview_turn = await client.chat(preview_prompt)
        assert preview_turn['status'] == 'completed', preview_turn
        assert '计划抽取 20 张' in preview_turn['message'], preview_turn
        assert graph.calls[-1][0] == 'preview_extract_images', graph.calls

        extract_prompt = '从 /data/raw/images 抽 10% 的图片到 /tmp/extract_run'
        assert await client._try_handle_mainline_intent(extract_prompt, 'thread-2') is None
        extract_turn = await client.chat(extract_prompt)
        assert extract_turn['status'] == 'completed', extract_turn
        assert '实际抽取 18 张' in extract_turn['message'], extract_turn
        assert graph.calls[-1][0] == 'extract_images', graph.calls
        assert client.session_state.active_dataset.dataset_root == '/tmp/extract_run'

        scan_prompt = '扫描一下 /data/videos 目录里有多少视频'
        assert await client._try_handle_mainline_intent(scan_prompt, 'thread-3') is None
        scan_turn = await client.chat(scan_prompt)
        assert scan_turn['status'] == 'completed', scan_turn
        assert '发现 3 个视频' in scan_turn['message'], scan_turn
        assert graph.calls[-1][0] == 'scan_videos', graph.calls

        frame_prompt = '从 /data/videos 抽帧，每 10 帧抽 1 帧，输出到 /tmp/frames_out'
        assert await client._try_handle_mainline_intent(frame_prompt, 'thread-4') is None
        frame_turn = await client.chat(frame_prompt)
        assert frame_turn['status'] == 'completed', frame_turn
        assert '最终保留 12 帧' in frame_turn['message'], frame_turn
        assert graph.calls[-1][0] == 'extract_video_frames', graph.calls

        print('extract route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
