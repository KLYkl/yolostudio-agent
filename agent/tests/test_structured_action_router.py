from __future__ import annotations

import asyncio
import shutil
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

        @classmethod
        def model_validate(cls, value):
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(**value)
            raise TypeError(f'cannot validate {value!r}')

        def model_dump(self):
            return dict(self.__dict__)

        def model_copy(self, *, update=None):
            payload = dict(self.__dict__)
            if update:
                payload.update(update)
            return type(self)(**payload)

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
        raise AssertionError('create_react_agent should not be called in structured action router tests')

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
from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings
from langchain_core.messages import AIMessage
from yolostudio_agent.agent.client.followup_router import (
    resolve_mainline_request_signals,
    resolve_training_run_query_signals,
)
from yolostudio_agent.agent.client.mainline_route_support import (
    resolve_mainline_context,
    resolve_mainline_dispatch_payload,
    resolve_mainline_guard_policy,
    resolve_mainline_guard_reply,
    resolve_mainline_guardrail_reply,
    resolve_mainline_route_state_payload,
)


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('graph should not be called in structured action router tests')


class _InterruptEnvelope:
    def __init__(self, value):
        self.value = value


class _InterruptTask:
    def __init__(self, payload):
        self.interrupts = [_InterruptEnvelope(payload)]


class _TrainingInterruptState:
    def __init__(self, payload: dict[str, object], *, messages=None) -> None:
        self.next = ('training_confirmation',)
        self.tasks = [_InterruptTask(payload)]
        self.values = {'messages': list(messages or [])}


class _TrainingInterruptGraph:
    def __init__(
        self,
        initial_payload: dict[str, object],
        next_payload: dict[str, object] | None = None,
        *,
        resume_handler=None,
    ) -> None:
        self._state = _TrainingInterruptState(initial_payload)
        self.next_payload = next_payload
        self.resume_handler = resume_handler
        self.resume_payloads: list[object] = []

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        del config
        resume = getattr(payload, 'resume', payload)
        self.resume_payloads.append(resume)
        if self.resume_handler is not None:
            result = self.resume_handler(self, resume)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if self.next_payload is not None:
            self._state = _TrainingInterruptState(self.next_payload)
            return {'messages': []}
        self._state = None
        return {'messages': [AIMessage(content='训练已启动')]}


class _FakeStructuredPlanner:
    def __init__(self, payload: dict[str, str]):
        self.payload = payload
        self.schemas: list[dict[str, object]] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    async def ainvoke(self, messages):
        del messages
        return dict(self.payload)


WORK = Path(__file__).resolve().parent / '_tmp_structured_action_router'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    client.helper_llm_settings = LlmProviderSettings(provider='ollama', model='gemma4:e4b', base_url='http://127.0.0.1:11434')
    client.primary_llm_settings = LlmProviderSettings(provider='ollama', model='gemma4:e4b', base_url='http://127.0.0.1:11434')
    return client


async def _scenario_uses_native_structured_output_for_ollama() -> None:
    client = _make_client('native-structured-output')
    planner = _FakeStructuredPlanner({'action': 'status'})
    client.planner_llm = planner  # type: ignore[assignment]
    action = await client._classify_structured_action(messages=[], allowed_actions={'status', 'inspect'})
    assert action == 'status', action
    assert planner.schemas, 'expected with_structured_output to be used'
    schema = planner.schemas[0]
    props = dict(schema.get('properties') or {})
    action_schema = dict(props.get('action') or {})
    assert action_schema.get('enum') == ['inspect', 'status'], schema


async def _scenario_generic_payload_uses_native_structured_output_for_ollama() -> None:
    client = _make_client('native-structured-payload')
    planner = _FakeStructuredPlanner({'decision': 'approve', 'reason': 'user approved'})
    client.planner_llm = planner  # type: ignore[assignment]
    payload = await client._invoke_structured_payload(
        messages=[],
        schema={
            'title': 'confirmation',
            'type': 'object',
            'properties': {
                'decision': {
                    'type': 'string',
                    'enum': ['approve', 'deny'],
                },
                'reason': {'type': 'string'},
            },
            'required': ['decision', 'reason'],
            'additionalProperties': False,
        },
    )
    assert payload == {'decision': 'approve', 'reason': 'user approved'}, payload
    assert planner.schemas, 'expected with_structured_output to be used'


async def _scenario_parse_user_decision_structures_loop_edits() -> None:
    client = _make_client('parse-user-decision-loop-edit')
    planner = _FakeStructuredPlanner({
        'action': 'edit',
        'reason': '用户把轮数改成 3，并保持循环训练',
        'edits': {'max_rounds': 3, 'epochs_per_round': 10, 'loop_name': 'ctxloop5'},
    })
    client.planner_llm = planner  # type: ignore[assignment]
    payload = await client._parse_user_decision(
        user_text='改成 3 轮，名字还是 ctxloop5，每轮 10 个 epoch',
        interrupt_payload={
            'phase': 'start',
            'plan': {
                'mode': 'loop',
                'dataset_path': '/data/demo',
                'model': 'yolov8n.pt',
                'max_rounds': 5,
                'epochs_per_round': 10,
                'loop_name': 'ctxloop5',
            },
        },
    )
    assert payload['action'] == 'edit', payload
    assert payload['edits'] == {'max_rounds': 3, 'epochs_per_round': 10, 'loop_name': 'ctxloop5'}, payload
    assert planner.schemas, 'expected structured decision schema'
    schema = planner.schemas[-1]
    props = dict(schema.get('properties') or {})
    edits = dict(props.get('edits') or {})
    edit_props = dict(edits.get('properties') or {})
    assert 'max_rounds' in edit_props, schema
    assert 'epochs_per_round' in edit_props, schema


async def _scenario_parse_user_decision_falls_back_to_unclear() -> None:
    client = _make_client('parse-user-decision-unclear')
    planner = _FakeStructuredPlanner({'action': 'something-else', 'reason': ''})
    client.planner_llm = planner  # type: ignore[assignment]
    payload = await client._parse_user_decision(
        user_text='嗯我再想想',
        interrupt_payload={
            'phase': 'prepare',
            'plan': {
                'mode': 'train',
                'dataset_path': '/data/demo',
                'model': 'yolov8n.pt',
                'epochs': 100,
            },
        },
    )
    assert payload == {'action': 'unclear', 'reason': '嗯我再想想'}, payload


async def _scenario_chat_resumes_training_confirmation_interrupt() -> None:
    interrupt_payload = {
        'type': 'training_confirmation',
        'phase': 'prepare',
        'plan': {
            'mode': 'train',
            'dataset_path': '/data/demo',
            'model': 'yolov8n.pt',
            'epochs': 20,
            'batch': 12,
        },
    }
    graph = _TrainingInterruptGraph(interrupt_payload)
    root = WORK / 'chat-resume-training-confirmation'
    settings = AgentSettings(session_id='chat-resume-training-confirmation', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    planner = _FakeStructuredPlanner({'action': 'approve', 'reason': '开始执行'})
    client.planner_llm = planner  # type: ignore[assignment]
    result = await client.chat('好，继续。')
    assert graph.resume_payloads == [{'action': 'approve', 'reason': '开始执行'}], graph.resume_payloads
    assert result['status'] == 'completed', result
    assert result['message'] == '训练已启动', result


async def _scenario_handoff_formats_training_confirmation_interrupt() -> None:
    interrupt_payload = {
        'type': 'training_confirmation',
        'phase': 'start',
        'plan': {
            'mode': 'loop',
            'dataset_path': '/data/demo',
            'model': 'yolov8n.pt',
            'max_rounds': 5,
            'epochs_per_round': 10,
            'loop_name': 'ctxloop5',
            'data_yaml': '/data/demo/split/data.yaml',
        },
    }
    graph = _TrainingInterruptGraph(interrupt_payload, next_payload=interrupt_payload)
    root = WORK / 'handoff-training-confirmation'
    settings = AgentSettings(session_id='handoff-training-confirmation', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    result = await client._handoff_current_runtime_to_graph(
        thread_id='handoff-training-confirmation-turn-1',
        user_text_hint='开一个 5 轮循环训练，每轮 10 epoch',
    )
    assert result['status'] == 'needs_confirmation', result
    assert '循环训练' in result['message'], result
    assert result['tool_call']['name'] == 'start_training_loop', result


async def _scenario_render_plan_handoffs_into_graph_interrupt() -> None:
    interrupt_payload = {
        'type': 'training_confirmation',
        'phase': 'prepare',
        'execution_mode': 'prepare_then_train',
        'plan': {
            'mode': 'train',
            'dataset_path': '/data/demo',
            'model': 'yolov8n.pt',
            'epochs': 20,
            'batch': 12,
        },
    }
    graph = _TrainingInterruptGraph(interrupt_payload, next_payload=interrupt_payload)
    root = WORK / 'render-plan-graph-interrupt'
    settings = AgentSettings(session_id='render-plan-graph-interrupt', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    result = await client._apply_training_plan_followup_action(
        followup_action={'action': 'render_plan'},
        thread_id='render-plan-graph-interrupt-turn-1',
        user_text='请继续当前训练确认',
        draft={
            'dataset_path': '/data/demo',
            'execution_mode': 'prepare_then_train',
            'next_step_tool': 'prepare_dataset_for_training',
            'planned_training_args': {
                'model': 'yolov8n.pt',
                'epochs': 20,
                'batch': 12,
            },
            'reasoning_summary': '当前数据先准备后再训练。',
        },
    )
    assert result['status'] == 'needs_confirmation', result
    assert result['interrupt_payload']['type'] == 'training_confirmation', result
    assert result['interrupt_payload']['phase'] == 'prepare', result


async def _scenario_graph_training_interrupt_edit_updates_plan() -> None:
    interrupt_payload = {
        'type': 'training_confirmation',
        'phase': 'start',
        'plan': {
            'mode': 'train',
            'dataset_path': '/data/demo',
            'model': 'yolov8n.pt',
            'epochs': 100,
            'batch': 16,
        },
    }

    def _resume_handler(graph, resume):
        payload = dict(interrupt_payload)
        edits = dict((resume or {}).get('edits') or {})
        plan = dict(payload.get('plan') or {})
        plan.update(edits)
        payload['plan'] = plan
        graph._state = _TrainingInterruptState(payload)
        return {'messages': []}

    graph = _TrainingInterruptGraph(interrupt_payload, resume_handler=_resume_handler)
    root = WORK / 'graph-training-interrupt-edit'
    settings = AgentSettings(session_id='graph-training-interrupt-edit', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    planner = _FakeStructuredPlanner({'action': 'edit', 'reason': '把 batch 改成 12，epochs 改成 20', 'edits': {'batch': 12, 'epochs': 20}})
    client.planner_llm = planner  # type: ignore[assignment]
    result = await client.chat('batch 改成 12，epochs 改成 20')
    payload = dict(result.get('interrupt_payload') or {})
    assert result['status'] == 'needs_confirmation', result
    assert payload.get('plan', {}).get('batch') == 12, payload
    assert payload.get('plan', {}).get('epochs') == 20, payload


async def _scenario_graph_training_interrupt_new_task_returns_result() -> None:
    interrupt_payload = {
        'type': 'training_confirmation',
        'phase': 'start',
        'plan': {
            'mode': 'loop',
            'dataset_path': '/data/demo',
            'model': 'yolov8n.pt',
            'max_rounds': 5,
            'epochs_per_round': 10,
            'loop_name': 'ctxloop5',
        },
    }

    def _resume_handler(graph, resume):
        graph._state = None
        return {'messages': [AIMessage(content='找到 2 条环训练')]}

    graph = _TrainingInterruptGraph(interrupt_payload, resume_handler=_resume_handler)
    root = WORK / 'graph-training-interrupt-new-task'
    settings = AgentSettings(session_id='graph-training-interrupt-new-task', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    planner = _FakeStructuredPlanner({'action': 'new_task', 'reason': '最近有哪些环训练'})
    client.planner_llm = planner  # type: ignore[assignment]
    result = await client.chat('最近有哪些环训练')
    assert result['message'] == '找到 2 条环训练', result


async def _scenario_training_loop_route_detects_control_followups_without_llm() -> None:
    client = _make_client('training-loop-route-followup')
    client.session_state.active_training.active_loop_id = 'loop-456'
    route = await client._resolve_training_loop_route(
        user_text='从下一轮开始继续。',
        normalized_text='从下一轮开始继续。'.lower(),
        wants_predict=False,
        wants_stop_training=False,
        explicit_run_ids=[],
    )
    assert route['action'] == 'followup', route
    assert route['has_context'] is True, route


async def _scenario_training_loop_route_falls_back_without_llm() -> None:
    client = _make_client('training-loop-route-fallback')
    client.session_state.active_training.active_loop_id = 'loop-789'
    client.session_state.active_training.last_loop_status = {'summary': '环训练还在运行'}
    route = await client._resolve_training_loop_route(
        user_text='现在环训练怎么样了？',
        normalized_text='现在环训练怎么样了？'.lower(),
        wants_predict=False,
        wants_stop_training=False,
        explicit_run_ids=[],
    )
    assert route == {'action': 'followup', 'loop_id': 'loop-789', 'has_context': True}, route


async def _scenario_training_run_query_signals_reuse_last_comparison() -> None:
    client = _make_client('training-run-query-signals-repeat-compare')
    client.session_state.active_training.last_run_comparison = {
        'left_run': {'run_id': 'train_log_200'},
        'right_run': {'run_id': 'train_log_100'},
    }
    signals = resolve_training_run_query_signals(
        session_state=client.session_state,
        user_text='把刚才那两次训练再比较一次',
        normalized_text='把刚才那两次训练再比较一次'.lower(),
        has_training_context=True,
        asks_metric_terms=False,
        metric_signals=[],
        explicit_run_ids=[],
    )
    assert signals['wants_training_run_compare'] is True, signals
    assert signals['comparison_run_ids'] == ['train_log_200', 'train_log_100'], signals
    assert signals['wants_best_training_run'] is False, signals


async def _scenario_training_run_query_signals_prior_statement_disables_compare_and_best() -> None:
    client = _make_client('training-run-query-signals-prior-statement')
    signals = resolve_training_run_query_signals(
        session_state=client.session_state,
        user_text='你上次不是说这次训练最好吗，基于哪次训练说的？',
        normalized_text='你上次不是说这次训练最好吗，基于哪次训练说的？'.lower(),
        has_training_context=True,
        asks_metric_terms=False,
        metric_signals=[],
        explicit_run_ids=['train_log_300'],
    )
    assert signals['wants_training_run_compare'] is False, signals
    assert signals['wants_best_training_run'] is False, signals
    assert signals['comparison_run_ids'] == [], signals
    assert signals['wants_training_run_inspect'] is False, signals
    assert signals['wants_training_knowledge'] is True, signals


async def _scenario_mainline_request_signals_remote_prediction_pipeline() -> None:
    client = _make_client('mainline-request-signals-remote-predict')
    signals = resolve_mainline_request_signals(
        session_state=client.session_state,
        user_text='把这批图片上传到服务器后直接预测',
        normalized_text='把这批图片上传到服务器后直接预测'.lower(),
    )
    assert signals['wants_predict'] is True, signals
    assert signals['wants_train'] is False, signals
    assert signals['wants_remote_upload'] is True, signals
    assert signals['wants_remote_prediction_pipeline'] is True, signals
    assert signals['wants_remote_training_pipeline'] is False, signals


async def _scenario_mainline_request_signals_training_entry() -> None:
    client = _make_client('mainline-request-signals-training-entry')
    signals = resolve_mainline_request_signals(
        session_state=client.session_state,
        user_text='开始训练，先默认划分，batch 改成 16',
        normalized_text='开始训练，先默认划分，batch 改成 16'.lower(),
    )
    assert signals['wants_train'] is True, signals
    assert signals['wants_predict'] is False, signals
    assert signals['wants_split'] is True, signals
    assert signals['training_command_like'] is True, signals
    assert signals['wants_training_revision'] is True, signals


async def _scenario_mainline_guard_reply_segmentation_training() -> None:
    reply = resolve_mainline_guard_reply(
        wants_segmentation_training=True,
        wants_predict=False,
        wants_continuous_parallel_predict=False,
        wants_prediction_and_training_mix=False,
        wants_prediction_result_as_training_data=False,
        wants_merge_extract_into_training=False,
    )
    assert '分割/SAM 训练暂不在这条主线上直接执行' in reply, reply


async def _scenario_mainline_guard_reply_predict_train_mix() -> None:
    reply = resolve_mainline_guard_reply(
        wants_segmentation_training=False,
        wants_predict=False,
        wants_continuous_parallel_predict=False,
        wants_prediction_and_training_mix=True,
        wants_prediction_result_as_training_data=False,
        wants_merge_extract_into_training=False,
    )
    assert '同时混了预测、训练或训练比较' in reply, reply


async def _scenario_mainline_guardrail_reply_push() -> None:
    reply = resolve_mainline_guardrail_reply(
        user_text='把这个分支 push 到 github',
        normalized_text='把这个分支 push 到 github'.lower(),
    )
    assert '不负责直接 push 代码仓库' in reply, reply


async def _scenario_mainline_guardrail_reply_ignores_safe_text() -> None:
    reply = resolve_mainline_guardrail_reply(
        user_text='先看看最近训练状态',
        normalized_text='先看看最近训练状态'.lower(),
    )
    assert reply == '', reply


async def _scenario_mainline_guard_policy_blocks_training_start_for_history_and_loop_status() -> None:
    policy = resolve_mainline_guard_policy(
        user_text='训练历史和当前环训练状态都看一下',
        normalized_text='训练历史和当前环训练状态都看一下'.lower(),
        wants_train=True,
        wants_predict=False,
        no_train=False,
        wants_readiness=False,
        training_command_like=False,
        wants_training_run_compare=False,
        wants_best_training_run=False,
        wants_stop_training=False,
        wants_training_run_list=True,
        wants_training_run_inspect=False,
        wants_failed_training_run_list=False,
        wants_completed_training_run_list=False,
        wants_stopped_training_run_list=False,
        wants_running_training_run_list=False,
        wants_analysis_ready_run_list=False,
        wants_training_loop_followup=True,
    )
    assert policy.blocks_training_start is True, policy
    assert policy.wants_train is False, policy


async def _scenario_collect_mainline_context_reuses_last_frame_extract() -> None:
    client = _make_client('collect-mainline-context-frame-followup')
    client.session_state.active_dataset.last_frame_extract = {'output_dir': '/data/frames'}
    context = client._collect_mainline_context('就用这些帧开始训练。')
    assert context['frame_followup_path'] == '/data/frames', context
    assert context['dataset_path'] == '/data/frames', context
    assert context['normalized_text'] == '就用这些帧开始训练。'.lower(), context


async def _scenario_resolve_mainline_context_helper_extracts_dataset_and_run_ids() -> None:
    client = _make_client('resolve-mainline-context-helper')
    helper_calls: dict[str, object] = {}

    def _metric_signal_extractor(text: str) -> list[str]:
        helper_calls['metric_text'] = text
        return ['precision']

    def _training_context_checker() -> bool:
        helper_calls['checked_training_context'] = True
        return True

    def _run_id_extractor(text: str) -> list[str]:
        helper_calls['run_id_text'] = text
        return ['train_log_123']

    context = resolve_mainline_context(
        session_state=client.session_state,
        user_text='用 /data/demo 这个数据集看看 train_log_123 的 precision',
        metric_signal_extractor=_metric_signal_extractor,
        training_context_checker=_training_context_checker,
        run_id_extractor=_run_id_extractor,
    )
    assert context['dataset_path'] == '/data/demo', context
    assert context['frame_followup_path'] == '', context
    assert context['metric_signals'] == ['precision'], context
    assert context['has_training_context'] is True, context
    assert context['explicit_run_ids'] == ['train_log_123'], context
    assert helper_calls == {
        'metric_text': '用 /data/demo 这个数据集看看 train_log_123 的 precision',
        'checked_training_context': True,
        'run_id_text': '用 /data/demo 这个数据集看看 train_log_123 的 precision',
    }, helper_calls


async def _scenario_resolve_mainline_route_state_payload_helper_aggregates_followups() -> None:
    client = _make_client('resolve-mainline-route-state-payload-helper')
    client.session_state.active_training.active_loop_id = 'loop-321'
    client.session_state.active_training.last_loop_status = {'summary': '环训练仍在运行'}
    user_text = '训练历史和当前环训练状态都看一下'
    mainline_context = client._collect_mainline_context(user_text)
    route_state = resolve_mainline_route_state_payload(
        session_state=client.session_state,
        user_text=user_text,
        normalized_text=str(mainline_context.get('normalized_text') or ''),
        has_training_context=bool(mainline_context.get('has_training_context')),
        mainline_signals=resolve_mainline_request_signals(
            session_state=client.session_state,
            user_text=user_text,
            normalized_text=str(mainline_context.get('normalized_text') or ''),
        ),
        metric_signals=list(mainline_context.get('metric_signals') or []),
        explicit_run_ids=list(mainline_context.get('explicit_run_ids') or []),
        loop_route={'action': 'followup'},
    )
    followup_flags = dict(route_state.get('followup_flags') or {})
    guard_policy = route_state.get('guard_policy')
    training_run_signals = dict(route_state.get('training_run_signals') or {})
    assert training_run_signals['wants_training_run_list'] is True, training_run_signals
    assert followup_flags['wants_training_run_list'] is True, followup_flags
    assert followup_flags['wants_training_loop_followup'] is True, followup_flags
    assert bool(guard_policy.blocks_training_start) is True, guard_policy


async def _scenario_resolve_mainline_dispatch_payload_helper_builds_request_args() -> None:
    client = _make_client('resolve-mainline-dispatch-payload-helper')
    client.session_state.active_training.active_loop_id = 'loop-654'
    user_text = '把这批图片上传到服务器后直接预测'
    mainline_context = client._collect_mainline_context(user_text)
    route_state = resolve_mainline_route_state_payload(
        session_state=client.session_state,
        user_text=user_text,
        normalized_text=str(mainline_context.get('normalized_text') or ''),
        has_training_context=bool(mainline_context.get('has_training_context')),
        mainline_signals=resolve_mainline_request_signals(
            session_state=client.session_state,
            user_text=user_text,
            normalized_text=str(mainline_context.get('normalized_text') or ''),
        ),
        metric_signals=list(mainline_context.get('metric_signals') or []),
        explicit_run_ids=list(mainline_context.get('explicit_run_ids') or []),
        loop_route={'action': ''},
    )
    dispatch_payload = resolve_mainline_dispatch_payload(
        mainline_context=mainline_context,
        route_state=route_state,
    )
    remote_request_args = dict(dispatch_payload.get('remote_request_args') or {})
    prediction_request_args = dict(dispatch_payload.get('prediction_request_args') or {})
    training_entrypoint_request_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
    assert 'training_context_request_args' not in dispatch_payload, dispatch_payload
    assert remote_request_args == {
        'wants_remote_profile_list': False,
        'wants_remote_upload': True,
        'wants_remote_prediction_pipeline': True,
        'wants_remote_training_pipeline': False,
    }, remote_request_args
    assert prediction_request_args['wants_predict'] is True, prediction_request_args
    assert prediction_request_args['training_command_like'] is False, prediction_request_args
    assert training_entrypoint_request_args['wants_train'] is False, training_entrypoint_request_args
    assert training_entrypoint_request_args['wants_predict'] is True, training_entrypoint_request_args
    assert training_entrypoint_request_args['blocks_training_start'] is False, training_entrypoint_request_args


async def _scenario_resolve_mainline_route_state_aggregates_followups() -> None:
    client = _make_client('resolve-mainline-route-state-aggregates-followups')
    client.session_state.active_training.active_loop_id = 'loop-321'
    client.session_state.active_training.last_loop_status = {'summary': '环训练仍在运行'}
    route_state = await client._resolve_mainline_route_state(
        '训练历史和当前环训练状态都看一下',
        client._collect_mainline_context('训练历史和当前环训练状态都看一下'),
    )
    followup_flags = dict(route_state.get('followup_flags') or {})
    guard_policy = route_state.get('guard_policy')
    assert followup_flags['wants_training_run_list'] is True, followup_flags
    assert followup_flags['wants_training_loop_followup'] is True, followup_flags
    assert bool(guard_policy.blocks_training_start) is True, guard_policy


async def _scenario_dispatch_mainline_requests_skips_standard_training_entrypoint() -> None:
    client = _make_client('dispatch-mainline-skips-standard-training')
    client._graph_has_training_entry = True
    calls: list[dict[str, object]] = []

    async def _unexpected_training_entrypoint(**kwargs):
        calls.append(dict(kwargs))
        return {'status': 'completed', 'message': 'should not happen', 'tool_call': None}

    client._try_handle_training_entrypoints = _unexpected_training_entrypoint  # type: ignore[method-assign]
    route_state = await client._resolve_mainline_route_state(
        '请用 /data/demo 基于 /models/yolov8n.pt 做一次训练，batch 改成 16',
        client._collect_mainline_context('请用 /data/demo 基于 /models/yolov8n.pt 做一次训练，batch 改成 16'),
    )
    result = await client._dispatch_mainline_requests(
        thread_id='dispatch-mainline-skips-standard-training-turn-1',
        user_text='请用 /data/demo 基于 /models/yolov8n.pt 做一次训练，batch 改成 16',
        mainline_context=client._collect_mainline_context('请用 /data/demo 基于 /models/yolov8n.pt 做一次训练，batch 改成 16'),
        route_state=route_state,
    )
    assert result is None, result
    assert calls == [], calls


async def _scenario_dispatch_mainline_requests_skips_loop_training_entrypoint() -> None:
    client = _make_client('dispatch-mainline-skips-loop-training')
    client._graph_has_training_entry = True
    calls: list[dict[str, object]] = []

    async def _unexpected_training_entrypoint(**kwargs):
        calls.append(dict(kwargs))
        return {'status': 'completed', 'message': 'should not happen', 'tool_call': None}

    client._try_handle_training_entrypoints = _unexpected_training_entrypoint  # type: ignore[method-assign]
    text = '基于 /data/demo 开一个 5 轮的循环训练，每轮 10 个 epoch'
    context = client._collect_mainline_context(text)
    route_state = await client._resolve_mainline_route_state(text, context)
    result = await client._dispatch_mainline_requests(
        thread_id='dispatch-mainline-skips-loop-training-turn-1',
        user_text=text,
        mainline_context=context,
        route_state=route_state,
    )
    assert result is None, result
    assert calls == [], calls


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_uses_native_structured_output_for_ollama()
        await _scenario_generic_payload_uses_native_structured_output_for_ollama()
        await _scenario_parse_user_decision_structures_loop_edits()
        await _scenario_parse_user_decision_falls_back_to_unclear()
        await _scenario_chat_resumes_training_confirmation_interrupt()
        await _scenario_handoff_formats_training_confirmation_interrupt()
        await _scenario_render_plan_handoffs_into_graph_interrupt()
        await _scenario_graph_training_interrupt_edit_updates_plan()
        await _scenario_graph_training_interrupt_new_task_returns_result()
        await _scenario_training_loop_route_detects_control_followups_without_llm()
        await _scenario_training_loop_route_falls_back_without_llm()
        await _scenario_training_run_query_signals_reuse_last_comparison()
        await _scenario_training_run_query_signals_prior_statement_disables_compare_and_best()
        await _scenario_mainline_request_signals_remote_prediction_pipeline()
        await _scenario_mainline_request_signals_training_entry()
        await _scenario_mainline_guard_reply_segmentation_training()
        await _scenario_mainline_guard_reply_predict_train_mix()
        await _scenario_mainline_guardrail_reply_push()
        await _scenario_mainline_guardrail_reply_ignores_safe_text()
        await _scenario_mainline_guard_policy_blocks_training_start_for_history_and_loop_status()
        await _scenario_collect_mainline_context_reuses_last_frame_extract()
        await _scenario_resolve_mainline_context_helper_extracts_dataset_and_run_ids()
        await _scenario_resolve_mainline_route_state_payload_helper_aggregates_followups()
        await _scenario_resolve_mainline_dispatch_payload_helper_builds_request_args()
        await _scenario_resolve_mainline_route_state_aggregates_followups()
        await _scenario_dispatch_mainline_requests_skips_standard_training_entrypoint()
        await _scenario_dispatch_mainline_requests_skips_loop_training_entrypoint()
        print('structured action router ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
