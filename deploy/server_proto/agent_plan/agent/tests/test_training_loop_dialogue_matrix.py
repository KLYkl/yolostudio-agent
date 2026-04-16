from __future__ import annotations

import asyncio
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Callable

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
        raise AssertionError('create_react_agent should not be called in training loop dialogue matrix')

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


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_dialogue_matrix'


class _DummyGraph:
    def get_state(self, config):
        return None


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    def __init__(self, reply: str | Callable[[list[Any]], str]) -> None:
        self.reply = reply
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> _FakePlannerResponse:
        self.calls.append(list(messages))
        if callable(self.reply):
            return _FakePlannerResponse(self.reply(list(messages)))
        return _FakePlannerResponse(self.reply)


async def _build_client(planner_llm: Any | None = None, *, session_id: str = 'training-loop-dialogue-matrix') -> tuple[YoloStudioAgentClient, list[tuple[str, dict[str, Any]]]]:
    settings = AgentSettings(session_id=session_id, memory_root=str(WORK / session_id))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={}, planner_llm=planner_llm)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            img_dir = str(kwargs.get('img_dir') or '')
            if img_dir.startswith('/missing'):
                result = {
                    'ok': False,
                    'summary': f'训练前检查失败：输入路径不存在：{img_dir}',
                    'error': f'输入路径不存在: {img_dir}',
                    'ready': False,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [f'输入路径不存在: {img_dir}'],
                }
            elif img_dir == '/home/kly/ct_loop/data_ct':
                result = {
                    'ok': True,
                    'summary': '当前还不能直接开启环训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': '/home/kly/ct_loop/data_ct',
                    'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                    'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
                    'resolved_data_yaml': '',
                    'ready': False,
                    'preparable': True,
                    'warnings': [],
                    'blockers': ['缺少可用的 data_yaml'],
                }
            else:
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：当前数据已具备训练条件。',
                    'dataset_root': '/data/loop',
                    'resolved_img_dir': '/data/loop/images',
                    'resolved_label_dir': '/data/loop/labels',
                    'resolved_data_yaml': '/data/loop/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'prepare_dataset_for_training':
            dataset_path = str(kwargs.get('dataset_path') or '')
            if dataset_path == '/home/kly/ct_loop/data_ct':
                result = {
                    'ok': True,
                    'ready': True,
                    'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                    'dataset_root': '/home/kly/ct_loop/data_ct',
                    'img_dir': '/home/kly/ct_loop/data_ct/images',
                    'label_dir': '/home/kly/ct_loop/data_ct/labels',
                    'data_yaml': '/home/kly/ct_loop/data_ct/data.yaml',
                    'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                    'next_actions': ['如需环训练，可继续 start_training_loop'],
                }
            else:
                raise AssertionError(f'unexpected prepare target: {dataset_path}')
        elif tool_name == 'start_training_loop':
            result = {
                'ok': True,
                'summary': '环训练已启动：helmet-loop（loop_id=loop-123）',
                'loop_id': 'loop-123',
                'loop_name': kwargs.get('loop_name') or 'helmet-loop',
                'status': 'queued',
                'managed_level': kwargs.get('managed_level', 'conservative_auto'),
                'boundaries': {
                    'max_rounds': kwargs.get('max_rounds', 5),
                    'target_metric': kwargs.get('target_metric', 'map50'),
                    'target_metric_value': kwargs.get('target_metric_value'),
                },
                'next_round_plan': {'round_index': 1, 'change_set': []},
            }
        elif tool_name == 'check_training_loop_status':
            result = {
                'ok': True,
                'summary': '第 2 轮训练已完成，准备下一轮',
                'loop_id': 'loop-123',
                'loop_name': 'helmet-loop',
                'status': 'awaiting_review',
                'current_round_index': 2,
                'max_rounds': 5,
                'best_round_index': 2,
                'best_target_metric': 0.68,
                'knowledge_gate_status': {
                    'outcome': 'awaiting_review',
                    'summary': '下一轮变更幅度偏大，当前停在审阅闸门。',
                },
                'latest_round_review': {
                    'recommended_action': 'continue_observing',
                    'why': 'mAP50 仍在提升，但幅度已经变小。',
                },
                'latest_round_memory': {
                    'next_focus': '继续观察',
                    'decision_type': 'await_review',
                },
                'latest_planner_output': {
                    'decision_type': 'await_review',
                    'decision_reason': '下一轮变更幅度偏大，等待确认。',
                },
                'latest_round_card': {
                    'round_index': 2,
                    'status': 'completed',
                    'vs_previous': {'highlights': ['mAP50提升 +0.0300']},
                    'next_plan': {'change_set': [{'field': 'epochs', 'old': 30, 'new': 40}]},
                    'round_review': {
                        'recommended_action': 'continue_observing',
                        'why': 'mAP50 仍在提升，但幅度已经变小。',
                    },
                    'round_memory': {
                        'next_focus': '继续观察',
                        'decision_type': 'await_review',
                    },
                    'planner_output': {
                        'decision_type': 'await_review',
                        'decision_reason': '下一轮变更幅度偏大，等待确认。',
                    },
                },
            }
        elif tool_name == 'inspect_training_loop':
            result = {
                'ok': True,
                'summary': '环训练详情已汇总',
                'loop_id': 'loop-123',
                'loop_name': 'helmet-loop',
                'status': 'awaiting_review',
                'round_cards': [
                    {'round_index': 1, 'status': 'completed'},
                    {'round_index': 2, 'status': 'completed'},
                ],
                'final_summary': None,
            }
        elif tool_name == 'list_training_loops':
            result = {
                'ok': True,
                'summary': '找到 2 条环训练记录',
                'loops': [
                    {'loop_id': 'loop-123', 'loop_name': 'helmet-loop', 'status': 'awaiting_review'},
                    {'loop_id': 'loop-099', 'loop_name': 'archive-loop', 'status': 'completed'},
                ],
            }
        elif tool_name == 'pause_training_loop':
            result = {
                'ok': True,
                'summary': '已记录暂停请求：当前第 2 轮结束后将停住',
                'loop_id': 'loop-123',
                'status': 'awaiting_review',
            }
        elif tool_name == 'resume_training_loop':
            result = {
                'ok': True,
                'summary': '环训练已恢复，准备启动第 3 轮',
                'loop_id': 'loop-123',
                'status': 'queued',
            }
        elif tool_name == 'stop_training_loop':
            result = {
                'ok': True,
                'summary': '环训练已停止',
                'loop_id': 'loop-123',
                'status': 'stopped',
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        client._record_secondary_event(tool_name, result)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return client, calls


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        client, calls = await _build_client(session_id='loop-dialogue-main')

        missing_model = await client.chat('给 /data/loop 开一个全托管环训练。')
        assert missing_model['status'] == 'completed', missing_model
        assert '缺少预训练权重' in missing_model['message'], missing_model

        missing_dataset = await client.chat('用 /missing/loop 数据集和 yolov8n.pt 开一个循环训练。')
        assert missing_dataset['status'] == 'completed', missing_dataset
        assert '输入路径不存在' in missing_dataset['message'], missing_dataset

        start_full = await client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 3 轮。')
        assert start_full['status'] == 'needs_confirmation', start_full
        assert start_full['tool_call']['args']['managed_level'] == 'full_auto', start_full

        confirm_full = await client.confirm(start_full['thread_id'], True)
        assert confirm_full['status'] == 'completed', confirm_full
        assert client.session_state.active_training.active_loop_id == 'loop-123'

        prepare_then_loop = await client.chat('用 /home/kly/ct_loop/data_ct 数据集和 /home/kly/yolov8n.pt 循环训一下，最多 2 轮。')
        assert prepare_then_loop['status'] == 'needs_confirmation', prepare_then_loop
        assert prepare_then_loop['tool_call']['name'] == 'prepare_dataset_for_training', prepare_then_loop
        assert prepare_then_loop['tool_call']['args']['dataset_path'] == '/home/kly/ct_loop/data_ct', prepare_then_loop
        assert client.session_state.active_training.training_plan_draft.get('next_step_tool') == 'prepare_dataset_for_training'

        loop_three_rounds_client, _ = await _build_client(session_id='loop-dialogue-three-rounds')
        prepare_loop_three_rounds = await loop_three_rounds_client.chat('用 /home/kly/ct_loop/data_ct 数据集和 /home/kly/yolov8n.pt 循环训练3轮。')
        assert prepare_loop_three_rounds['status'] == 'needs_confirmation', prepare_loop_three_rounds
        assert prepare_loop_three_rounds['tool_call']['name'] == 'prepare_dataset_for_training', prepare_loop_three_rounds
        assert loop_three_rounds_client.session_state.active_training.training_plan_draft.get('planned_loop_args', {}).get('max_rounds') == 3, prepare_loop_three_rounds
        assert 'epochs' not in loop_three_rounds_client.session_state.active_training.training_plan_draft.get('planned_loop_args', {}), prepare_loop_three_rounds

        def _loop_planner_reply(messages: list[Any]) -> str:
            combined = '\n'.join(str(getattr(item, 'content', '')) for item in messages)
            if '循环训练启动编排器' in combined and '/home/kly/ct_loop/data_ct' in combined and '"observed_tool_names": []' in combined:
                return '{"next_tool":"training_readiness","reason":"先确认数据是否已具备直接训练条件。"}'
            if '循环训练启动编排器' in combined and '/home/kly/ct_loop/data_ct' in combined and '缺少可用的 data_yaml' in combined:
                return '{"next_tool":"prepare_dataset_for_training","reason":"缺少 data_yaml，先 prepare。"}'
            if '循环训练启动编排器' in combined and '/home/kly/ct_loop/data_ct' in combined and '/home/kly/ct_loop/data_ct/data.yaml' in combined:
                return '{"next_tool":"start_training_loop","reason":"数据已经准备完成，进入 loop。"}'
            if '循环训练启动编排器' in combined and '/data/loop' in combined and '"observed_tool_names": []' in combined:
                return '{"next_tool":"training_readiness","reason":"先确认 data_yaml 是否已就绪。"}'
            if '循环训练启动编排器' in combined and '/data/loop' in combined and '/data/loop/data.yaml' in combined:
                return '{"next_tool":"start_training_loop","reason":"data_yaml 已就绪，直接启动 loop。"}'
            if '审批回复解释器' in combined:
                return '{"decision":"approve","reason":"用户批准执行。"}'
            return '这是模型整理后的说明。'

        planner_route_client, planner_route_calls = await _build_client(
            planner_llm=_FakePlannerLlm(_loop_planner_reply),
            session_id='loop-dialogue-planner-route',
        )
        planner_prepare = await planner_route_client.chat('用 /home/kly/ct_loop/data_ct 数据集和 /home/kly/yolov8n.pt 循环训练3轮。')
        assert planner_prepare['status'] == 'needs_confirmation', planner_prepare
        assert planner_prepare['tool_call']['name'] == 'prepare_dataset_for_training', planner_prepare
        assert planner_route_calls[-1][0] == 'training_readiness', planner_route_calls
        assert planner_route_client.session_state.active_training.training_plan_draft.get('planner_decision_source') == 'llm', planner_prepare
        assert planner_route_client.session_state.active_training.training_plan_draft.get('planner_decision') == 'prepare', planner_prepare

        planner_direct_start_client, _ = await _build_client(
            planner_llm=_FakePlannerLlm(_loop_planner_reply),
            session_id='loop-dialogue-planner-direct-start',
        )
        planner_direct_start = await planner_direct_start_client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 2 轮。')
        assert planner_direct_start['status'] == 'needs_confirmation', planner_direct_start
        assert planner_direct_start['tool_call']['name'] == 'start_training_loop', planner_direct_start
        assert planner_direct_start_client.session_state.active_training.training_plan_draft.get('planner_observed_tools') == ['training_readiness'], planner_direct_start
        assert planner_direct_start_client.session_state.active_training.training_plan_draft.get('planner_decision_source') == 'llm', planner_direct_start
        assert planner_direct_start_client.session_state.active_training.training_plan_draft.get('planner_decision') == 'start', planner_direct_start

        planner_prepare_followup = await planner_route_client.confirm(planner_prepare['thread_id'], True)
        assert planner_prepare_followup['status'] == 'needs_confirmation', planner_prepare_followup
        assert planner_prepare_followup['tool_call']['name'] == 'start_training_loop', planner_prepare_followup
        assert planner_route_client.session_state.active_training.training_plan_draft.get('planner_decision_source') == 'llm', planner_prepare_followup
        assert planner_route_client.session_state.active_training.training_plan_draft.get('planner_decision') == 'start', planner_prepare_followup

        confirm_prepare_then_loop = await client.confirm(prepare_then_loop['thread_id'], True)
        assert confirm_prepare_then_loop['status'] == 'needs_confirmation', confirm_prepare_then_loop
        assert confirm_prepare_then_loop['tool_call']['name'] == 'start_training_loop', confirm_prepare_then_loop
        assert confirm_prepare_then_loop['tool_call']['args']['model'] == '/home/kly/yolov8n.pt', confirm_prepare_then_loop
        assert confirm_prepare_then_loop['tool_call']['args']['data_yaml'] == '/home/kly/ct_loop/data_ct/data.yaml', confirm_prepare_then_loop
        assert confirm_prepare_then_loop['tool_call']['args']['max_rounds'] == 2, confirm_prepare_then_loop
        assert 'epochs' not in confirm_prepare_then_loop['tool_call']['args'], confirm_prepare_then_loop
        assert client.session_state.active_training.training_plan_draft.get('next_step_tool') == 'start_training_loop'
        loop_facts = client._training_plan_user_facts(client.session_state.active_training.training_plan_draft, pending=True)
        assert loop_facts['execution_mode'] == '先准备再进入循环训练', loop_facts
        assert loop_facts['loop_requested'] is True, loop_facts
        assert loop_facts['max_rounds'] == 2, loop_facts
        assert loop_facts['next_step'] == '启动循环训练', loop_facts

        confirm_loop_start = await client.confirm(confirm_prepare_then_loop['thread_id'], True)
        assert confirm_loop_start['status'] == 'completed', confirm_loop_start
        assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1, calls

        status_turn = await client.chat('环训练状态怎么样？')
        assert status_turn['status'] == 'completed', status_turn
        assert calls[-1][0] == 'check_training_loop_status', calls[-1]

        casual_status_turn = await client.chat('现在环训练怎么样了？')
        assert casual_status_turn['status'] == 'completed', casual_status_turn
        assert calls[-1][0] == 'check_training_loop_status', calls[-1]

        generic_status_turn = await client.chat('查看情况')
        assert generic_status_turn['status'] == 'completed', generic_status_turn
        assert calls[-1][0] == 'check_training_loop_status', calls[-1]

        inspect_turn = await client.chat('查看环训练详情')
        assert inspect_turn['status'] == 'completed', inspect_turn
        assert calls[-1][0] == 'inspect_training_loop', calls[-1]

        generic_inspect_turn = await client.chat('查看训练详情')
        assert generic_inspect_turn['status'] == 'completed', generic_inspect_turn
        assert calls[-1][0] == 'inspect_training_loop', calls[-1]

        list_turn = await client.chat('最近环训练有哪些？')
        assert list_turn['status'] == 'completed', list_turn
        assert calls[-1][0] == 'list_training_loops', calls[-1]

        pause_turn = await client.chat('暂停环训练')
        assert pause_turn['status'] == 'completed', pause_turn
        assert calls[-1][0] == 'pause_training_loop', calls[-1]

        resume_turn = await client.chat('恢复环训练')
        assert resume_turn['status'] == 'completed', resume_turn
        assert calls[-1][0] == 'resume_training_loop', calls[-1]

        stop_turn = await client.chat('立即终止当前环训练')
        assert stop_turn['status'] == 'completed', stop_turn
        assert calls[-1][0] == 'stop_training_loop', calls[-1]

        generic_stop_turn = await client.chat('停止训练')
        assert generic_stop_turn['status'] == 'completed', generic_stop_turn
        assert calls[-1][0] == 'stop_training_loop', calls[-1]

        inspect_by_id = await client.chat('查看 1713020000-loop-a 的环训练详情。')
        assert inspect_by_id['status'] == 'completed', inspect_by_id
        assert calls[-1][0] == 'inspect_training_loop', calls[-1]
        assert calls[-1][1]['loop_id'] == '1713020000-loop-a', calls[-1]

        review_client, _ = await _build_client(session_id='loop-dialogue-review')
        start_review = await review_client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个每轮都停的环训练。')
        assert start_review['status'] == 'needs_confirmation', start_review
        assert start_review['tool_call']['args']['managed_level'] == 'review', start_review

        short_client, _ = await _build_client(session_id='loop-dialogue-short')
        short_client.session_state.active_dataset.dataset_root = '/data/loop'
        short_client.session_state.active_dataset.data_yaml = '/data/loop/data.yaml'
        short_client.session_state.active_training.model = 'yolov8n.pt'
        short_turn = await short_client.chat('就这个，循环训一下，最多 2 轮。')
        assert short_turn['status'] == 'needs_confirmation', short_turn
        assert short_turn['tool_call']['name'] == 'start_training_loop', short_turn
        assert short_turn['tool_call']['args']['model'] == 'yolov8n.pt', short_turn
        assert short_turn['tool_call']['args']['data_yaml'] == '/data/loop/data.yaml', short_turn
        assert short_turn['tool_call']['args']['max_rounds'] == 2, short_turn
        assert 'epochs' not in short_turn['tool_call']['args'], short_turn

        def _loop_followup_planner(messages: list[Any]) -> str:
            combined = '\n'.join(str(getattr(item, 'content', '')) for item in messages)
            if '循环训练跟进路由器' in combined:
                if '现在是什么情况了？我需要详细一点的训练信息' in combined:
                    return '{"action":"inspect","reason":"用户要求更详细的当前环训练信息。"}'
                return '{"action":"status","reason":"这是当前环训练的泛状态追问。"}'
            return '这是模型整理后的 loop 状态说明。'

        planner = _FakePlannerLlm(_loop_followup_planner)
        llm_client, llm_calls = await _build_client(planner_llm=planner, session_id='loop-dialogue-llm-status')
        llm_client.session_state.active_training.active_loop_id = 'loop-123'
        llm_status = await llm_client.chat('环训练状态怎么样？')
        assert llm_status['status'] == 'completed', llm_status
        assert llm_calls[-1][0] == 'check_training_loop_status', llm_calls[-1]
        assert llm_status['message'] == '这是模型整理后的 loop 状态说明。', llm_status
        llm_status_again = await llm_client.chat('再次查看训练状态')
        assert llm_status_again['status'] == 'completed', llm_status_again
        assert llm_calls[-1][0] == 'check_training_loop_status', llm_calls[-1]
        assert planner.calls, planner.calls
        llm_prompt = str(getattr(planner.calls[-1][-1], 'content', ''))
        assert 'continue_observing' in llm_prompt, llm_prompt
        assert 'latest_round_memory' in llm_prompt, llm_prompt
        llm_detail_followup = await llm_client.chat('现在是什么情况了？我需要详细一点的训练信息')
        assert llm_detail_followup['status'] == 'completed', llm_detail_followup
        assert llm_calls[-1][0] == 'inspect_training_loop', llm_calls[-1]
        assert any('循环训练跟进路由器' in '\n'.join(str(getattr(item, 'content', '')) for item in call) for call in planner.calls), planner.calls

        def _planner_reply(messages: list[Any]) -> str:
            combined = '\n'.join(str(getattr(item, 'content', '')) for item in messages)
            if '循环训练启动编排器' in combined and '/data/loop' in combined and '"observed_tool_names": []' in combined:
                return '{"next_tool":"training_readiness","reason":"先确认 data_yaml 是否已就绪。"}'
            if '循环训练启动编排器' in combined and '/data/loop' in combined and '/data/loop/data.yaml' in combined:
                return '{"next_tool":"start_training_loop","reason":"data_yaml 已就绪，直接启动 loop。"}'
            if '审批回复解释器' in combined:
                return '{"decision":"approve","reason":"用户已经明确表示按当前安排继续。"}'
            if '确认说明器' in combined:
                return '这一步会按当前环训练计划继续；如果你同意，我就开始执行。'
            if '结果说明' in combined:
                return '环训练已经启动，后面会按计划继续推进。'
            return '好的。'

        approval_planner = _FakePlannerLlm(_planner_reply)
        approval_client, approval_calls = await _build_client(planner_llm=approval_planner, session_id='loop-dialogue-llm-approval')
        approval_start = await approval_client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 2 轮。')
        assert approval_start['status'] == 'needs_confirmation', approval_start
        approval_done = await approval_client.chat('那就按这个安排往下走。')
        assert approval_done['status'] == 'completed', approval_done
        assert approval_calls[-1][0] == 'start_training_loop', approval_calls[-1]
        assert any('审批回复解释器' in '\n'.join(str(getattr(item, 'content', '')) for item in call) for call in approval_planner.calls), approval_planner.calls

        def _multi_tool_renderer(messages: list[Any]) -> str:
            combined = '\n'.join(str(getattr(item, 'content', '')) for item in messages)
            if '组合结果说明器' in combined:
                return '这是模型整理后的组合执行结果。'
            return '好的。'

        render_planner = _FakePlannerLlm(_multi_tool_renderer)
        render_client, _ = await _build_client(planner_llm=render_planner, session_id='loop-dialogue-multi-render')
        combined_reply = await render_client._render_multi_tool_result_message(
            [
                ('summarize_training_run', {'ok': True, 'summary': '训练摘要：map50=0.58'}),
                ('analyze_training_outcome', {'ok': True, 'summary': '建议继续训练，并降低学习率。'}),
            ],
            objective='训练结果分析说明',
        )
        assert combined_reply == '这是模型整理后的组合执行结果。', combined_reply
        assert any('组合结果说明器' in '\n'.join(str(getattr(item, 'content', '')) for item in call) for call in render_planner.calls), render_planner.calls

        fallback_approve_client, fallback_approve_calls = await _build_client(session_id='loop-dialogue-fallback-approve')
        fallback_start = await fallback_approve_client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 2 轮。')
        assert fallback_start['status'] == 'needs_confirmation', fallback_start
        fallback_done = await fallback_approve_client.chat('开始训练')
        assert fallback_done['status'] == 'completed', fallback_done
        assert fallback_approve_calls[-1][0] == 'start_training_loop', fallback_approve_calls[-1]

        pending_guard_client, pending_guard_calls = await _build_client(session_id='loop-dialogue-pending-guard')
        pending_guard_start = await pending_guard_client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 2 轮。')
        assert pending_guard_start['status'] == 'needs_confirmation', pending_guard_start
        pending_guard_status = await pending_guard_client.chat('查看训练状态')
        assert pending_guard_status['status'] == 'needs_confirmation', pending_guard_status
        assert '训练计划草案' in pending_guard_status['message'], pending_guard_status
        assert pending_guard_status['tool_call']['name'] == 'start_training_loop', pending_guard_status
        assert [name for name, _ in pending_guard_calls].count('check_training_loop_status') == 0, pending_guard_calls

        print('training loop dialogue matrix ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
