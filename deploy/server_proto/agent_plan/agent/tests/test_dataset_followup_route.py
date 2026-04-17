from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
import sys
import types
from typing import Any

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
        raise AssertionError('create_react_agent should not be called in dataset follow-up smoke')

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
from yolostudio_agent.agent.client.dataset_fact_service import build_dataset_fact_followup_reply
from langchain_core.messages import AIMessage, ToolMessage


class _GraphState:
    def __init__(self, messages):
        self.values = {'messages': list(messages)}
        self.next = ()


class _DatasetGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self._last_state = _GraphState([])

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return self._last_state

    async def _cached_reply(self, payload, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        reply = await self.client._render_tool_result_message(tool_name, result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        messages = list(payload['messages']) + [AIMessage(content=reply)]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def _tool_reply(self, payload, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        assert self.client is not None
        observed = await self.client.direct_tool(tool_name, _state_mode='observe', **kwargs)
        reply = await self.client._render_tool_result_message(tool_name, observed)
        if not reply:
            reply = str(observed.get('summary') or observed.get('error') or '操作已完成')
        tool_call = {'id': f'tc-{len(payload["messages"])}', 'name': tool_name, 'args': kwargs}
        messages = list(payload['messages']) + [
            AIMessage(content='', tool_calls=[tool_call]),
            ToolMessage(content=json.dumps(observed, ensure_ascii=False), name=tool_name, tool_call_id=tool_call['id']),
            AIMessage(content=reply),
        ]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        summary = '\n'.join(str(getattr(message, 'content', message)) for message in payload['messages'][:2])
        user_text = str(getattr(payload['messages'][-1], 'content', ''))
        ds = self.client.session_state.active_dataset
        fact_reply = build_dataset_fact_followup_reply(
            self.client.session_state,
            user_text=user_text,
        )
        if fact_reply:
            messages = list(payload['messages']) + [AIMessage(content=fact_reply)]
            self._last_state = _GraphState(messages)
            return {'messages': messages}

        if '详细一点的数据集信息' in user_text:
            assert 'last_scan_summary:' in summary
            assert 'last_validate_summary:' in summary
            assert 'last_health_summary:' in summary
            reply = '数据集质量分析完成，可继续处理缺失标签和重复图片。'
            messages = list(payload['messages']) + [AIMessage(content=reply)]
            self._last_state = _GraphState(messages)
            return {'messages': messages}

        if '健康检查结果再详细一点' in user_text:
            assert 'last_health_summary:' in summary
            return await self._cached_reply(payload, 'run_dataset_health_check', dict(ds.last_health_check))

        if '/data/other' in user_text:
            return await self._tool_reply(payload, 'detect_duplicate_images', dataset_path='/data/other')

        if '/data/dataset' in user_text:
            assert 'img_dir: /data/dataset' in summary or 'dataset_root: /data/dataset' in summary
            if ds.last_duplicate_check:
                return await self._cached_reply(payload, 'detect_duplicate_images', dict(ds.last_duplicate_check))
            return await self._tool_reply(payload, 'detect_duplicate_images', dataset_path='/data/dataset')

        raise AssertionError(f'unexpected graph request: {user_text}')


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


WORK = Path(__file__).resolve().parent / '_tmp_dataset_followup_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='dataset-followup-smoke', memory_root=str(WORK))
        graph = _DatasetGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, object]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs):
            state_mode = str(kwargs.pop('_state_mode', 'persistent'))
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'scan_dataset':
                assert kwargs['img_dir'] == '/data/dataset'
                result = {
                    'ok': True,
                    'summary': '扫描完成: 共 200 张图片，类别 2 个',
                    'scan_overview': {'image_count': 200, 'class_count': 2},
                    'classes': ['epidural', 'subdural'],
                    'class_stats': {'epidural': 66, 'subdural': 24},
                    'top_classes': [{'class_name': 'epidural', 'count': 66}, {'class_name': 'subdural', 'count': 24}],
                    'missing_label_images': 3,
                    'missing_label_ratio': 0.015,
                    'warnings': ['发现少量缺失标签'],
                    'class_name_source': 'detected_classes_txt',
                    'detected_classes_txt': '/data/dataset/classes.txt',
                    'action_candidates': [{'description': '可继续做数据集健康检查', 'tool': 'run_dataset_health_check'}],
                }
                if state_mode != 'observe':
                    client._apply_to_state('scan_dataset', result, kwargs)
                return result
            if tool_name == 'validate_dataset':
                assert kwargs['img_dir'] == '/data/dataset'
                result = {
                    'ok': True,
                    'summary': '校验完成: 缺失标签 3 个，空标签 1 个',
                    'validation_overview': {'missing_labels': 3, 'empty_labels': 1},
                    'warnings': ['发现少量缺失标签'],
                    'action_candidates': [{'description': '建议修复缺失标签', 'tool': 'generate_missing_labels'}],
                }
                if state_mode != 'observe':
                    client._apply_to_state('validate_dataset', result, kwargs)
                return result
            if tool_name == 'run_dataset_health_check':
                assert kwargs['dataset_path'] == '/data/dataset'
                result = {
                    'ok': True,
                    'summary': '健康检查完成: 重复图片 2 组，损坏图片 0 个',
                    'health_overview': {'duplicate_groups': 2, 'corrupt_images': 0},
                    'duplicate_groups': 2,
                    'duplicate_extra_files': 2,
                    'warnings': [],
                    'action_candidates': [{'description': '可继续查看重复图片详情', 'tool': 'detect_duplicate_images'}],
                }
                if state_mode != 'observe':
                    client._apply_to_state('run_dataset_health_check', result, kwargs)
                return result
            assert tool_name == 'detect_duplicate_images'
            result = {
                'ok': True,
                'summary': '重复图片检查完成: 发现 2 组',
                'duplicate_overview': {'duplicate_groups': 2},
                'duplicate_groups': 2,
                'warnings': [],
                'action_candidates': [{'description': '建议整理重复图片', 'tool': 'organize_duplicates'}],
            }
            if state_mode != 'observe':
                client._apply_to_state('detect_duplicate_images', result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        initial = await client._complete_dataset_quality_reply('/data/dataset')
        assert initial['status'] == 'completed', initial
        assert '健康检查' in initial['message'] or '重复图片' in initial['message'] or '扫描完成' in initial['message'], initial

        routed = await client.chat('现在是什么情况了？我需要详细一点的数据集信息')
        assert routed['status'] == 'completed', routed
        assert '重复图片' in routed['message'] or '缺失标签' in routed['message'] or '数据集质量分析' in routed['message'], routed
        assert [name for name, _ in calls[-3:]] == ['scan_dataset', 'validate_dataset', 'run_dataset_health_check'], calls

        before_fact_followup = len(calls)
        least = await client.chat('哪个类别的标注最少？')
        assert least['status'] == 'completed', least
        assert 'subdural' in least['message'].lower(), least
        assert '24' in least['message'], least
        assert len(calls) == before_fact_followup, calls

        missing_labels = await client.chat('缺失标签有多少？')
        assert missing_labels['status'] == 'completed', missing_labels
        assert '3' in missing_labels['message'], missing_labels
        assert len(calls) == before_fact_followup, calls

        epidural = await client.chat('Epidural 有多少个？')
        assert epidural['status'] == 'completed', epidural
        assert '66' in epidural['message'], epidural
        assert len(calls) == before_fact_followup, calls

        classes = await client.chat('有哪些类别？')
        assert classes['status'] == 'completed', classes
        assert 'epidural' in classes['message'].lower(), classes
        assert 'subdural' in classes['message'].lower(), classes
        assert len(calls) == before_fact_followup, calls

        before_cached_followup = len(calls)
        routed2 = await client.chat('健康检查结果再详细一点')
        assert routed2['status'] == 'completed', routed2
        assert '健康检查完成' in routed2['message'], routed2
        assert len(calls) == before_cached_followup, calls

        first_dataset_duplicates = await client.chat('列出 /data/dataset 的重复图片')
        assert first_dataset_duplicates['status'] == 'completed', first_dataset_duplicates
        assert calls[-1] == ('detect_duplicate_images', {'dataset_path': '/data/dataset'}), calls

        before_duplicate_cached = len(calls)
        same_dataset_duplicates = await client.chat('列出 /data/dataset 的重复图片')
        assert same_dataset_duplicates['status'] == 'completed', same_dataset_duplicates
        assert len(calls) == before_duplicate_cached, calls

        other_dataset_duplicates = await client.chat('列出 /data/other 的重复图片')
        assert other_dataset_duplicates['status'] == 'completed', other_dataset_duplicates
        assert calls[-1] == ('detect_duplicate_images', {'dataset_path': '/data/other'}), calls

        print('dataset followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
