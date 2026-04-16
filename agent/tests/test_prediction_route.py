from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
import sys
import types

if __package__ in {None, ""}:
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
        raise AssertionError('create_react_agent should not be called in prediction route smoke')

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


WORK = Path(__file__).resolve().parent / '_tmp_prediction_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='prediction-route-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, str]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs):
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'predict_images':
                assert kwargs['source_path'] == '/data/images'
                assert kwargs['model'] == '/models/yolov8n.pt'
                result = {
                    'ok': True,
                    'summary': '预测完成: 已处理 2 张图片, 有检测 1, 无检测 1，主要类别 Excavator=1',
                    'model': kwargs['model'],
                    'source_path': kwargs['source_path'],
                    'processed_images': 2,
                    'detected_images': 1,
                    'empty_images': 1,
                    'class_counts': {'Excavator': 1},
                    'detected_samples': ['/data/images/a.jpg'],
                    'empty_samples': ['/data/images/b.jpg'],
                    'output_dir': '/tmp/predict',
                    'annotated_dir': '/tmp/predict/annotated',
                    'report_path': '/tmp/predict/prediction_report.json',
                    'warnings': [],
                    'next_actions': ['可查看标注结果目录: /tmp/predict/annotated'],
                }
                client._apply_to_state('predict_images', result, kwargs)
                return result
            if tool_name == 'inspect_prediction_outputs':
                assert kwargs['report_path'] == '/tmp/predict/prediction_report.json'
                result = {
                    'ok': True,
                    'summary': '预测输出检查完成',
                    'output_dir': '/tmp/predict',
                    'report_path': kwargs['report_path'],
                    'artifact_roots': ['/tmp/predict/annotated', '/tmp/predict/reports'],
                    'path_list_files': {'detected': '/tmp/predict/detected.txt'},
                    'action_candidates': [{'description': '可继续导出预测报告', 'tool': 'export_prediction_report'}],
                }
                client._apply_to_state('inspect_prediction_outputs', result, kwargs)
                return result
            assert tool_name == 'summarize_prediction_results'
            assert kwargs['report_path'] == '/tmp/predict/prediction_report.json'
            result = {
                'ok': True,
                'summary': '预测结果摘要: 已处理 2 张图片, 有检测 1, 无检测 1, 总检测框 1，主要类别 Excavator=1',
                'report_path': kwargs['report_path'],
                'output_dir': '/tmp/predict',
                'annotated_dir': '/tmp/predict/annotated',
                'processed_images': 2,
                'detected_images': 1,
                'empty_images': 1,
                'total_detections': 1,
                'class_counts': {'Excavator': 1},
                'detected_samples': ['/data/images/a.jpg'],
                'empty_samples': ['/data/images/b.jpg'],
                'warnings': [],
                'next_actions': ['可查看标注结果目录: /tmp/predict/annotated'],
                'model': '/models/yolov8n.pt',
                'source_path': '/data/images',
            }
            client._apply_to_state('summarize_prediction_results', result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        routed = await client._try_handle_mainline_intent('请用 /models/yolov8n.pt 预测 /data/images 这个目录里的图片', 'thread-1')
        assert routed is not None, routed
        assert routed['status'] == 'completed', routed
        assert '预测完成' in routed['message'], routed
        assert client.session_state.active_prediction.model == '/models/yolov8n.pt'
        assert client.session_state.active_prediction.source_path == '/data/images'

        routed2 = await client._try_handle_mainline_intent('总结一下刚才预测结果', 'thread-2')
        assert routed2 is not None, routed2
        assert routed2['status'] == 'completed', routed2
        assert '总检测框 1' in routed2['message'], routed2
        assert '标注结果目录' in routed2['message'], routed2
        assert calls[-1][0] == 'summarize_prediction_results', calls

        before = len(calls)
        routed3 = await client._try_handle_mainline_intent('总结预测结果', 'thread-3')
        assert routed3 is not None, routed3
        assert routed3['status'] == 'completed', routed3
        assert '总检测框 1' in routed3['message'], routed3
        assert len(calls) == before, calls

        def _planner_reply(messages):
            text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
            if '预测跟进路由器' in text:
                return '{"action":"inspect","reason":"用户要求更详细的预测信息"}'
            return '预测输出检查完成，可继续查看输出目录和报告。'

        client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
        routed4 = await client._try_handle_mainline_intent('现在是什么情况了？我需要详细一点的信息', 'thread-4')
        assert routed4 is not None, routed4
        assert routed4['status'] == 'completed', routed4
        assert '预测输出检查完成' in routed4['message'], routed4
        assert calls[-1][0] == 'inspect_prediction_outputs', calls

        before_cached_followup = len(calls)
        routed5 = await client._try_handle_mainline_intent('那个预测输出再详细一点', 'thread-5')
        assert routed5 is not None, routed5
        assert routed5['status'] == 'completed', routed5
        assert '预测输出检查完成' in routed5['message'], routed5
        assert len(calls) == before_cached_followup, calls

        routed6 = await client._try_handle_prediction_requests(
            user_text='看看 /tmp/predict/prediction_report.json 这个预测输出',
            normalized_text='看看 /tmp/predict/prediction_report.json 这个预测输出',
            prediction_path='/data/images',
            wants_train=False,
            training_command_like=False,
            wants_scan_videos=False,
            wants_extract_frames=False,
            wants_extract_preview=False,
            wants_extract_images=False,
            wants_remote_upload=False,
            wants_prediction_summary=False,
            wants_prediction_output_inspection=True,
            wants_prediction_report_export=False,
            wants_prediction_path_lists=False,
            wants_prediction_result_organize=False,
            has_prediction_followup_context=True,
            has_prediction_management_followup_context=True,
            has_explicit_prediction_target=True,
        )
        assert routed6 is not None, routed6
        assert routed6['status'] == 'completed', routed6
        assert '预测输出检查完成' in routed6['message'], routed6
        assert len(calls) == before_cached_followup, calls

        print('prediction route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
