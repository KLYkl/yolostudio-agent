from __future__ import annotations

import asyncio
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
        raise AssertionError('create_react_agent should not be called in advanced plan dialogue smoke')

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

from langchain_core.messages import AIMessage
from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._pending_confirmation_test_support import seed_pending_confirmation
from yolostudio_agent.agent.client.mainline_route_support import resolve_mainline_dispatch_payload
from yolostudio_agent.agent.tests._training_plan_test_support import (
    clear_training_plan_draft,
    current_training_plan_context_payload,
    current_training_plan_draft,
    set_training_plan_draft,
)
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    build_training_plan_draft_from_context,
)
from yolostudio_agent.agent.client.training_dialogue_service import (
    run_training_plan_dialogue_flow,
    wants_training_advanced_details,
)
from yolostudio_agent.agent.client.training_request_service import (
    run_prepare_only_flow,
    run_training_request_entrypoint,
)


class _DummyGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        if not self.plan_context:
            return None
        return types.SimpleNamespace(values={'training_plan_context': dict(self.plan_context)})

    def update_state(self, config, update):
        del config
        if not isinstance(update, dict) or 'training_plan_context' not in update:
            return
        context = update.get('training_plan_context')
        self.plan_context = dict(context) if isinstance(context, dict) and context else None

    async def ainvoke(self, payload, config=None):
        assert self.client is not None
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        thread_id = str((((config or {}).get('configurable') or {}).get('thread_id') or '')).strip()
        if not plan_context:
            mainline_context = self.client._collect_mainline_context(user_text)
            route_state = await self.client._resolve_mainline_route_state(user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
            if training_entrypoint_args:
                prepare_only_followup = await run_prepare_only_flow(
                    user_text=user_text,
                    looks_like_prepare_only_request=self.client._looks_like_prepare_only_request,
                    extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                    local_path_exists=lambda path: Path(path).expanduser().exists(),
                    direct_tool=self.client.direct_tool,
                    collect_requested_training_args=self.client._collect_requested_training_args,
                    build_training_plan_draft_fn=self.client._build_training_plan_draft,
                    render_tool_result_message=self.client._render_tool_result_message,
                )
                if prepare_only_followup:
                    entrypoint_result = {
                        'reply': str(prepare_only_followup.get('reply') or '').strip(),
                        'draft': dict(prepare_only_followup.get('draft') or {}),
                        'defer_to_graph': str(prepare_only_followup.get('action') or '').strip() == 'save_draft_and_handoff',
                    }
                else:
                    entrypoint_result = await run_training_request_entrypoint(
                        session_state=self.client.session_state,
                        user_text=user_text,
                        normalized_text=str(training_entrypoint_args.get('normalized_text') or ''),
                        dataset_path=str(training_entrypoint_args.get('dataset_path') or ''),
                        frame_followup_path=str(training_entrypoint_args.get('frame_followup_path') or ''),
                        wants_train=bool(training_entrypoint_args.get('wants_train')),
                        wants_predict=bool(training_entrypoint_args.get('wants_predict')),
                        no_train=bool(training_entrypoint_args.get('no_train')),
                        readiness_only_query=bool(training_entrypoint_args.get('readiness_only_query')),
                        wants_training_outcome_analysis=bool(training_entrypoint_args.get('wants_training_outcome_analysis')),
                        wants_next_step_guidance=bool(training_entrypoint_args.get('wants_next_step_guidance')),
                        wants_training_knowledge=bool(training_entrypoint_args.get('wants_training_knowledge')),
                        wants_training_revision=bool(training_entrypoint_args.get('wants_training_revision')),
                        wants_stop_training=bool(training_entrypoint_args.get('wants_stop_training')),
                        blocks_training_start=bool(training_entrypoint_args.get('blocks_training_start')),
                        explicit_run_ids=list(training_entrypoint_args.get('explicit_run_ids') or []),
                        wants_split=bool(training_entrypoint_args.get('wants_split')),
                        current_training_plan_context=current_training_plan_context_payload(self.client),
                        direct_tool=self.client.direct_tool,
                        collect_requested_training_args=self.client._collect_requested_training_args,
                        is_training_discussion_only=self.client._is_training_discussion_only,
                        extract_training_execution_backend=self.client._extract_training_execution_backend_from_text,
                        build_training_plan_draft_fn=self.client._build_training_plan_draft,
                        render_training_plan_message=self.client._render_training_plan_message,
                    )
                draft = dict((entrypoint_result or {}).get('draft') or {})
                reply = str((entrypoint_result or {}).get('reply') or '').strip()
                discussion_only = self.client._is_training_discussion_only(user_text)
                if draft:
                    set_training_plan_draft(self.client, draft)
                    self.plan_context = build_training_plan_context_from_draft(draft)
                if draft and discussion_only:
                    rendered = reply or await self.client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}
                if bool((entrypoint_result or {}).get('defer_to_graph')) and draft:
                    plan_context = dict(self.plan_context or {})
                elif reply:
                    return {'messages': messages + [AIMessage(content=reply)]}
                elif draft:
                    rendered = await self.client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + [AIMessage(content=rendered)]}

        normalized = str(user_text or '').strip().lower()
        is_execute_turn = normalized in {'y', 'yes', '执行', '确认'} or any(
            token in user_text for token in ('开始吧', '就这样', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练', '按这个方案执行', '按这个最终方案执行')
        )
        if plan_context and not is_execute_turn:
            draft = build_training_plan_draft_from_context(plan_context) or {}
            dialogue_result = await run_training_plan_dialogue_flow(
                session_state=self.client.session_state,
                user_text=user_text,
                draft=draft,
                pending=self.client.get_pending_action() or None,
                explicit_run_ids=self.client._extract_training_run_ids_from_text(user_text),
                clear_fields=self.client._collect_training_clear_fields(user_text),
                readiness=self.client.session_state.active_dataset.last_readiness or {},
                data_yaml=str(self.client.session_state.active_dataset.data_yaml or '').strip(),
                is_training_discussion_only=self.client._is_training_discussion_only,
                custom_training_script_requested=bool(intent_parsing.extract_custom_training_script_from_text(user_text)),
                looks_like_prepare_only_request=self.client._looks_like_prepare_only_request,
                extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                local_path_exists=lambda path: Path(path).expanduser().exists(),
                collect_requested_training_args=self.client._collect_requested_training_args,
                extract_training_execution_backend=self.client._extract_training_execution_backend_from_text,
                wants_training_advanced_details=wants_training_advanced_details,
                direct_tool=self.client.direct_tool,
                build_training_plan_draft_fn=self.client._build_training_plan_draft,
                render_tool_result_message=self.client._render_tool_result_message,
                render_training_plan_message=self.client._render_training_plan_message,
            )
            draft_to_save = dict(dialogue_result.get('draft_to_save') or {})
            if draft_to_save:
                set_training_plan_draft(self.client, draft_to_save, thread_id=thread_id)
                self.plan_context = build_training_plan_context_from_draft(draft_to_save)
                draft = draft_to_save
            followup_action = dict(dialogue_result.get('followup_action') or {})
            action = str(followup_action.get('action') or '').strip()
            if action in {'reply', 'reply_with_pending', 'clear_draft_and_reply', 'save_draft_and_reply'}:
                reply = str(followup_action.get('reply') or '').strip()
                return {'messages': messages + ([AIMessage(content=reply)] if reply else [])}
            if action == 'cancel_draft':
                clear_training_plan_draft(self.client, thread_id=thread_id)
                self.plan_context = None
                return {'messages': messages + [AIMessage(content='已取消当前训练计划草案。')]}
            if action in {'render_draft', 'render_plan', 'render_completed'}:
                rendered = await self.client._render_training_plan_message(
                    draft,
                    pending=bool(draft.get('next_step_tool')),
                )
                return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}
            if action in {'save_draft_and_handoff', 'confirmation_message', 'refresh_confirmation', 'render_original_plan'} and draft:
                rendered = await self.client._render_training_plan_message(
                    draft,
                    pending=bool(draft.get('next_step_tool')),
                )
                return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}

        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        if next_tool:
            seed_pending_confirmation(self.client, 
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
        raise AssertionError(f'unexpected graph prompt: {user_text}')


WORK = Path(__file__).resolve().parent / '_tmp_training_plan_advanced_dialogue'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-plan-advanced-dialogue', memory_root=str(WORK))
        graph = _DummyGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：数据已具备训练条件。',
                    'dataset_root': '/data/project',
                    'resolved_img_dir': '/data/project/images',
                    'resolved_label_dir': '/data/project/labels',
                    'resolved_data_yaml': '/data/project/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 2 个可用训练环境，默认将使用 base',
                    'environments': [
                        {'name': 'base', 'display_name': 'base', 'selected_by_default': True},
                        {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': False},
                    ],
                    'default_environment': {'name': 'base', 'display_name': 'base'},
                }
            elif tool_name == 'training_preflight':
                selected_environment = kwargs.get('training_environment') or 'base'
                if selected_environment == 'missing-env':
                    result = {
                        'ok': True,
                        'ready_to_start': False,
                        'summary': '训练预检未通过：训练环境不存在: missing-env（可用: base, yolodo）',
                        'training_environment': None,
                        'resolved_args': {
                            'model': kwargs['model'],
                            'data_yaml': kwargs['data_yaml'],
                            'epochs': kwargs['epochs'],
                            'device': kwargs.get('device', 'auto') or 'auto',
                            'training_environment': 'missing-env',
                            'project': kwargs.get('project') or None,
                            'name': kwargs.get('name') or None,
                            'batch': kwargs.get('batch'),
                            'imgsz': kwargs.get('imgsz'),
                            'fraction': kwargs.get('fraction'),
                            'classes': kwargs.get('classes'),
                            'single_cls': kwargs.get('single_cls'),
                            'optimizer': kwargs.get('optimizer') or None,
                            'freeze': kwargs.get('freeze'),
                            'resume': kwargs.get('resume'),
                            'lr0': kwargs.get('lr0'),
                            'patience': kwargs.get('patience'),
                            'workers': kwargs.get('workers'),
                            'amp': kwargs.get('amp'),
                        },
                        'command_preview': [],
                        'blockers': ['训练环境不存在: missing-env（可用: base, yolodo）'],
                        'warnings': [],
                    }
                else:
                    result = {
                        'ok': True,
                        'ready_to_start': True,
                        'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                        'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                        'resolved_args': {
                            'model': kwargs['model'],
                            'data_yaml': kwargs['data_yaml'],
                            'epochs': kwargs['epochs'],
                            'device': kwargs.get('device', 'auto') or 'auto',
                            'training_environment': selected_environment,
                            'project': kwargs.get('project') or None,
                            'name': kwargs.get('name') or None,
                            'batch': kwargs.get('batch'),
                            'imgsz': kwargs.get('imgsz'),
                            'fraction': kwargs.get('fraction'),
                            'classes': kwargs.get('classes'),
                            'single_cls': kwargs.get('single_cls'),
                            'optimizer': kwargs.get('optimizer') or None,
                            'freeze': kwargs.get('freeze'),
                            'resume': kwargs.get('resume'),
                            'lr0': kwargs.get('lr0'),
                            'patience': kwargs.get('patience'),
                            'workers': kwargs.get('workers'),
                            'amp': kwargs.get('amp'),
                        },
                        'command_preview': ['yolo', 'train'],
                        'blockers': [],
                        'warnings': [],
                    }
            elif tool_name == 'start_training':
                result = {
                    'ok': True,
                    'summary': '训练已启动: model=yolov8s.pt, data=/data/project/data.yaml, device=auto',
                    'device': 'auto',
                    'pid': 8888,
                    'log_file': '/runs/train_advanced.txt',
                    'started_at': 123.4,
                    'resolved_args': dict(kwargs),
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            if tool_name == 'start_training' and result.get('ok'):
                clear_training_plan_draft(client)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat('数据在 /data/project，想用 /custom/train.py 配合 yolov8s.pt 训练，先给我计划，不要执行。')
        assert turn1['status'] == 'completed', turn1
        assert '执行后端: 自定义训练脚本' in turn1['message']
        assert '自定义脚本: /custom/train.py' in turn1['message']
        assert '当前自动执行链只支持标准 YOLO 训练' in turn1['message']
        assert (client.get_pending_action() or {}).get('tool_name', '') == ''

        turn2 = await client.chat('不用自定义脚本了，改成标准 yolo，用 yolodo 环境。输出放到 project /runs/ablation，name exp-blue，只训练类别 1,3，fraction 0.5。展开高级参数，把 lr0 改成 0.005，patience 20，workers 4，关闭 amp。')
        assert turn2['status'] == 'completed', turn2
        assert '执行后端: 标准 YOLO 训练' in turn2['message']
        assert '计划依据:' in turn2['message']
        assert '已从默认环境 base 切换到 yolodo' in turn2['message']
        assert '只计划使用约 50% 的训练数据' in turn2['message']
        assert '只训练指定类别 [1, 3]' in turn2['message']
        assert '训练环境: yolodo' in turn2['message']
        assert '输出组织: project=/runs/ablation, name=exp-blue' in turn2['message']
        assert '高级参数: fraction=0.5, classes=[1, 3], lr0=0.005, patience=20, workers=4, amp=False' in turn2['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] == 'yolodo'
        assert calls[-1][1]['project'] == '/runs/ablation'
        assert calls[-1][1]['name'] == 'exp-blue'
        assert calls[-1][1]['fraction'] == 0.5
        assert calls[-1][1]['classes'] == [1, 3]
        assert calls[-1][1]['lr0'] == 0.005
        assert calls[-1][1]['patience'] == 20
        assert calls[-1][1]['workers'] == 4
        assert calls[-1][1]['amp'] is False

        turn3 = await client.chat('那就按这个方案执行。')
        assert turn3['status'] == 'needs_confirmation', turn3
        assert turn3['tool_call']['name'] == 'start_training'
        assert turn3['tool_call']['args']['training_environment'] == 'yolodo'
        assert turn3['tool_call']['args']['project'] == '/runs/ablation'
        assert turn3['tool_call']['args']['name'] == 'exp-blue'
        assert turn3['tool_call']['args']['fraction'] == 0.5
        assert turn3['tool_call']['args']['classes'] == [1, 3]
        assert turn3['tool_call']['args']['lr0'] == 0.005
        assert turn3['tool_call']['args']['patience'] == 20
        assert turn3['tool_call']['args']['workers'] == 4
        assert turn3['tool_call']['args']['amp'] is False

        turn4 = await client.confirm(turn3['thread_id'], approved=True)
        assert turn4['status'] == 'completed', turn4
        assert '训练已启动' in turn4['message']
        assert client.session_state.active_training.training_environment == 'yolodo'
        assert client.session_state.active_training.project == '/runs/ablation'
        assert client.session_state.active_training.run_name == 'exp-blue'
        assert client.session_state.active_training.fraction == 0.5
        assert client.session_state.active_training.classes == [1, 3]
        assert client.session_state.active_training.lr0 == 0.005
        assert client.session_state.active_training.patience == 20
        assert client.session_state.active_training.workers == 4
        assert client.session_state.active_training.amp is False
        assert current_training_plan_draft(client) == {}

        turn5 = await client.chat('数据在 /data/project，用 yolov8s.pt，先给我计划，不执行，训练环境先用 missing-env。')
        assert turn5['status'] == 'completed', turn5
        assert '训练环境: missing-env' in turn5['message']
        assert '当前阻塞:' in turn5['message']
        assert '训练环境不存在: missing-env' in turn5['message']

        turn6 = await client.chat('为什么不行？那改成 yolodo，project /runs/review，name exp-fix，只训练类别 0,2，fraction 0.4，开启 single_cls。')
        assert turn6['status'] == 'completed', turn6
        assert '已从默认环境 base 切换到 yolodo' in turn6['message']
        assert '只训练指定类别 [0, 2]' in turn6['message']
        assert '启用了 single_cls' in turn6['message']
        assert '输出组织: project=/runs/review, name=exp-fix' in turn6['message']
        assert '高级参数: fraction=0.4, classes=[0, 2], single_cls=True' in turn6['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] == 'yolodo'
        assert calls[-1][1]['project'] == '/runs/review'
        assert calls[-1][1]['name'] == 'exp-fix'
        assert calls[-1][1]['fraction'] == 0.4
        assert calls[-1][1]['classes'] == [0, 2]
        assert calls[-1][1]['single_cls'] is True

        turn7 = await client.chat('那把类别限制去掉，恢复全量数据，环境恢复默认，project 不要了，name 不要了，不要单类别训练，重新开始训练。')
        assert turn7['status'] == 'completed', turn7
        assert '训练环境: base' in turn7['message']
        assert '已从默认环境 base 切换到' not in turn7['message']
        assert '只训练指定类别' not in turn7['message']
        assert '只计划使用约' not in turn7['message']
        assert '输出组织:' not in turn7['message']
        assert '高级参数: single_cls=False, resume=False' in turn7['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] in {'', 'base'}
        assert calls[-1][1]['project'] == ''
        assert calls[-1][1]['name'] == ''
        assert calls[-1][1]['fraction'] is None
        assert calls[-1][1]['classes'] is None
        assert calls[-1][1]['single_cls'] is False
        assert calls[-1][1]['resume'] is False

        turn8 = await client.chat('执行。')
        assert turn8['status'] == 'needs_confirmation', turn8
        assert turn8['tool_call']['args']['training_environment'] == 'base'
        assert turn8['tool_call']['args']['project'] == ''
        assert turn8['tool_call']['args']['name'] == ''
        assert turn8['tool_call']['args']['fraction'] is None
        assert turn8['tool_call']['args']['classes'] is None
        assert turn8['tool_call']['args']['single_cls'] is False
        assert turn8['tool_call']['args']['resume'] is False

        turn9 = await client.confirm(turn8['thread_id'], approved=False)
        assert turn9['status'] == 'cancelled', turn9
        assert '先不执行这一步' in turn9['message']
        assert '当前计划已保留' in turn9['message']
        assert current_training_plan_draft(client) != {}

        turn10 = await client.chat('为什么现在回到 base？那保留默认环境，但 project 改成 /runs/final-review，name exp-final，optimizer 改成 AdamW，freeze 6，再把类别限制改成 4,5。先给我计划。')
        assert turn10['status'] == 'completed', turn10
        assert '训练环境: base' in turn10['message'] or '训练环境:' not in turn10['message']
        assert '输出组织: project=/runs/final-review, name=exp-final' in turn10['message']
        assert '只训练指定类别 [4, 5]' in turn10['message']
        assert '高级参数: classes=[4, 5], single_cls=False, optimizer=AdamW, freeze=6, resume=False' in turn10['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] in {'', 'base'}
        assert calls[-1][1]['project'] == '/runs/final-review'
        assert calls[-1][1]['name'] == 'exp-final'
        assert calls[-1][1]['classes'] == [4, 5]
        assert calls[-1][1]['optimizer'] == 'AdamW'
        assert calls[-1][1]['freeze'] == 6

        turn11 = await client.chat('好，就按这个最终方案执行。')
        assert turn11['status'] == 'needs_confirmation', turn11
        assert turn11['tool_call']['args']['training_environment'] in {'', 'base'}
        assert turn11['tool_call']['args']['project'] == '/runs/final-review'
        assert turn11['tool_call']['args']['name'] == 'exp-final'
        assert turn11['tool_call']['args']['classes'] == [4, 5]
        assert turn11['tool_call']['args']['optimizer'] == 'AdamW'
        assert turn11['tool_call']['args']['freeze'] == 6

        turn12 = await client.confirm(turn11['thread_id'], approved=True)
        assert turn12['status'] == 'completed', turn12
        assert '训练已启动' in turn12['message']
        assert client.session_state.active_training.training_environment == 'base'
        assert client.session_state.active_training.project == '/runs/final-review'
        assert client.session_state.active_training.run_name == 'exp-final'
        assert client.session_state.active_training.classes == [4, 5]
        assert client.session_state.active_training.optimizer == 'AdamW'
        assert client.session_state.active_training.freeze == 6
        assert current_training_plan_draft(client) == {}
        print('training plan advanced dialogue ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


async def _run_backend_switch_followup() -> None:
    work = Path(__file__).resolve().parent / '_tmp_training_plan_backend_switch'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-plan-backend-switch', memory_root=str(work))
        graph = _DummyGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：数据已具备训练条件。',
                    'dataset_root': '/data/backend-switch',
                    'resolved_img_dir': '/data/backend-switch/images',
                    'resolved_label_dir': '/data/backend-switch/labels',
                    'resolved_data_yaml': '/data/backend-switch/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': ['当前先以讨论方案为主，确认后再真正启动'],
                    'blockers': [],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 2 个可用训练环境，默认将使用 base',
                    'environments': [
                        {'name': 'base', 'display_name': 'base', 'selected_by_default': True},
                        {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': False},
                    ],
                    'default_environment': {'name': 'base', 'display_name': 'base'},
                }
            elif tool_name == 'training_preflight':
                selected_environment = kwargs.get('training_environment') or 'base'
                result = {
                    'ok': True,
                    'ready_to_start': True,
                    'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                    'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                    'resolved_args': {
                        'model': kwargs['model'],
                        'data_yaml': kwargs['data_yaml'],
                        'epochs': kwargs['epochs'],
                        'device': kwargs.get('device', 'auto') or 'auto',
                        'training_environment': selected_environment,
                        'project': kwargs.get('project') or None,
                        'name': kwargs.get('name') or None,
                        'batch': kwargs.get('batch'),
                        'imgsz': kwargs.get('imgsz'),
                        'fraction': kwargs.get('fraction'),
                        'classes': kwargs.get('classes'),
                        'single_cls': kwargs.get('single_cls'),
                        'optimizer': kwargs.get('optimizer') or None,
                        'freeze': kwargs.get('freeze'),
                        'resume': kwargs.get('resume'),
                        'lr0': kwargs.get('lr0'),
                        'patience': kwargs.get('patience'),
                        'workers': kwargs.get('workers'),
                        'amp': kwargs.get('amp'),
                    },
                    'command_preview': ['yolo', 'train'],
                    'blockers': [],
                    'warnings': [],
                }
            elif tool_name == 'start_training':
                result = {
                    'ok': True,
                    'summary': '训练已启动: model=yolov8m.pt, data=/data/backend-switch/data.yaml, device=auto',
                    'device': 'auto',
                    'pid': 9999,
                    'log_file': '/runs/backend_switch.txt',
                    'started_at': 222.2,
                    'resolved_args': dict(kwargs),
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            if tool_name == 'start_training' and result.get('ok'):
                clear_training_plan_draft(client)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat('数据在 /data/backend-switch，用 yolov8m.pt 训练 40轮，先用 yolodo 环境，project /runs/backend-switch，name exp-a，batch 8，先给我计划。')
        assert turn1['status'] == 'completed', turn1
        assert '执行后端: 标准 YOLO 训练' in turn1['message']
        assert '训练环境: yolodo' in turn1['message']
        assert '输出组织: project=/runs/backend-switch, name=exp-a' in turn1['message']
        assert 'batch=8' in turn1['message']
        assert calls[-1][0] == 'training_preflight'

        turn2 = await client.chat('执行。')
        assert turn2['status'] == 'needs_confirmation', turn2
        assert turn2['tool_call']['name'] == 'start_training'
        assert turn2['tool_call']['args']['training_environment'] == 'yolodo'

        turn3 = await client.confirm(turn2['thread_id'], approved=False)
        assert turn3['status'] == 'cancelled', turn3
        assert '当前计划已保留' in turn3['message']
        assert current_training_plan_draft(client) != {}
        call_count_before_backend_switch = len(calls)

        turn4 = await client.chat('先别执行了，为什么还建议标准 yolo？改成用 /custom/research_train.py 跑，自定义 trainer 先不管，给我方案。')
        assert turn4['status'] == 'completed', turn4
        assert '执行方式: 先讨论方案，暂不执行' in turn4['message']
        assert '执行后端: 自定义训练脚本' in turn4['message']
        assert '自定义脚本: /custom/research_train.py' in turn4['message']
        assert '当前自动执行链只支持标准 YOLO 训练' in turn4['message']
        assert '输出组织: project=/runs/backend-switch, name=exp-a' in turn4['message']
        assert (client.get_pending_action() or {}).get('tool_name', '') == ''
        assert len(calls) == call_count_before_backend_switch

        turn5 = await client.chat('不用脚本了，切回标准 yolo，环境改成 base，project /runs/backend-final，name exp-b，optimizer 改成 SGD，freeze 4，batch 10，imgsz 736，执行前先给我计划。')
        assert turn5['status'] == 'completed', turn5
        assert '执行后端: 标准 YOLO 训练' in turn5['message']
        assert '训练环境: base' in turn5['message']
        assert '输出组织: project=/runs/backend-final, name=exp-b' in turn5['message']
        assert '高级参数: optimizer=SGD, freeze=4' in turn5['message']
        assert 'batch=10' in turn5['message']
        assert 'imgsz=736' in turn5['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] == 'base'
        assert calls[-1][1]['project'] == '/runs/backend-final'
        assert calls[-1][1]['name'] == 'exp-b'
        assert calls[-1][1]['optimizer'] == 'SGD'
        assert calls[-1][1]['freeze'] == 4
        assert calls[-1][1]['batch'] == 10
        assert calls[-1][1]['imgsz'] == 736

        turn6 = await client.chat('可以了，执行。')
        assert turn6['status'] == 'needs_confirmation', turn6
        assert turn6['tool_call']['args']['training_environment'] == 'base'
        assert turn6['tool_call']['args']['project'] == '/runs/backend-final'
        assert turn6['tool_call']['args']['name'] == 'exp-b'
        assert turn6['tool_call']['args']['optimizer'] == 'SGD'
        assert turn6['tool_call']['args']['freeze'] == 4
        assert turn6['tool_call']['args']['batch'] == 10
        assert turn6['tool_call']['args']['imgsz'] == 736

        turn7 = await client.confirm(turn6['thread_id'], approved=True)
        assert turn7['status'] == 'completed', turn7
        assert '训练已启动' in turn7['message']
        assert client.session_state.active_training.training_environment == 'base'
        assert client.session_state.active_training.project == '/runs/backend-final'
        assert client.session_state.active_training.run_name == 'exp-b'
        assert client.session_state.active_training.optimizer == 'SGD'
        assert client.session_state.active_training.freeze == 4
        assert client.session_state.active_training.batch == 10
        assert client.session_state.active_training.imgsz == 736
        assert current_training_plan_draft(client) == {}
        print('training plan backend switch ok')
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def _run_prepare_bridge_replanning() -> None:
    work = Path(__file__).resolve().parent / '_tmp_training_plan_prepare_bridge_advanced'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-plan-prepare-bridge-advanced', memory_root=str(work))
        graph = _DummyGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': '/data/prepare-advanced',
                    'resolved_img_dir': '/data/prepare-advanced/images',
                    'resolved_label_dir': '/data/prepare-advanced/labels',
                    'resolved_data_yaml': '',
                    'ready': False,
                    'preparable': True,
                    'primary_blocker_type': 'missing_yaml',
                    'warnings': ['先完成 prepare，再决定是否保留当前训练方案更稳'],
                    'blockers': ['缺少可用的 data_yaml'],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                    'environments': [
                        {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                        {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                    ],
                    'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                }
            elif tool_name == 'prepare_dataset_for_training':
                result = {
                    'ok': True,
                    'ready': True,
                    'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                    'dataset_root': '/data/prepare-advanced',
                    'img_dir': '/data/prepare-advanced/images',
                    'label_dir': '/data/prepare-advanced/labels',
                    'data_yaml': '/data/prepare-advanced/data.yaml',
                    'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                    'next_actions': ['如需训练，可继续 start_training'],
                }
            elif tool_name == 'training_preflight':
                selected_environment = kwargs.get('training_environment') or 'yolodo'
                result = {
                    'ok': True,
                    'ready_to_start': True,
                    'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                    'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                    'resolved_args': {
                        'model': kwargs['model'],
                        'data_yaml': kwargs['data_yaml'],
                        'epochs': kwargs['epochs'],
                        'device': kwargs.get('device', 'auto') or 'auto',
                        'training_environment': selected_environment,
                        'project': kwargs.get('project') or None,
                        'name': kwargs.get('name') or None,
                        'batch': kwargs.get('batch'),
                        'imgsz': kwargs.get('imgsz'),
                        'optimizer': kwargs.get('optimizer') or None,
                        'freeze': kwargs.get('freeze'),
                        'classes': kwargs.get('classes'),
                        'workers': kwargs.get('workers'),
                        'amp': kwargs.get('amp'),
                    },
                    'command_preview': ['yolo', 'train'],
                    'blockers': [],
                    'warnings': ['prepare 完成后的首轮训练，建议先确认参数再启动'],
                }
            elif tool_name == 'start_training':
                result = {
                    'ok': True,
                    'summary': '训练已启动: model=yolov8m.pt, data=/data/prepare-advanced/data.yaml, device=auto',
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'pid': 7777,
                    'log_file': '/runs/prepare_advanced.txt',
                    'started_at': 555.5,
                    'resolved_args': dict(kwargs),
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            if tool_name == 'start_training' and result.get('ok'):
                clear_training_plan_draft(client)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat(
            '数据在 /data/prepare-advanced，用 yolov8m.pt 训练 50轮，环境 yolodo，project /runs/prepare-advanced-a，name exp-a，batch 6，imgsz 800，freeze 2。先给我计划，不要执行。'
        )
        assert turn1['status'] == 'completed', turn1
        assert '执行方式: 先准备再训练' in turn1['message']
        assert '训练环境: yolodo' in turn1['message']
        assert '输出组织: project=/runs/prepare-advanced-a, name=exp-a' in turn1['message']
        assert '高级参数: freeze=2' in turn1['message']

        turn2 = await client.chat('执行。')
        assert turn2['status'] == 'needs_confirmation', turn2
        assert turn2['tool_call']['name'] == 'prepare_dataset_for_training'

        turn3 = await client.confirm(turn2['thread_id'], approved=True)
        assert turn3['status'] == 'needs_confirmation', turn3
        assert turn3['tool_call']['name'] == 'start_training'
        assert turn3['tool_call']['args']['data_yaml'] == '/data/prepare-advanced/data.yaml'
        assert turn3['tool_call']['args']['training_environment'] == 'yolodo'
        assert turn3['tool_call']['args']['project'] == '/runs/prepare-advanced-a'
        assert turn3['tool_call']['args']['name'] == 'exp-a'
        assert turn3['tool_call']['args']['batch'] == 6
        assert turn3['tool_call']['args']['imgsz'] == 800
        assert turn3['tool_call']['args']['freeze'] == 2

        turn4 = await client.confirm(turn3['thread_id'], approved=False)
        assert turn4['status'] == 'cancelled', turn4
        assert '当前计划已保留' in turn4['message']
        assert current_training_plan_draft(client) != {}

        preflight_count_before = [name for name, _ in calls].count('training_preflight')
        turn5 = await client.chat('为什么现在可以直接训练了？先别执行，改成 /custom/after_prepare.py 先讨论方案。')
        assert turn5['status'] == 'completed', turn5
        assert '执行方式: 先讨论方案，暂不执行' in turn5['message']
        assert '执行后端: 自定义训练脚本' in turn5['message']
        assert '自定义脚本: /custom/after_prepare.py' in turn5['message']
        assert '当前自动执行链只支持标准 YOLO 训练' in turn5['message']
        assert [name for name, _ in calls].count('training_preflight') == preflight_count_before

        turn6 = await client.chat('不用脚本了，切回标准 yolo，环境改成 base，project /runs/prepare-advanced-b，name exp-b，optimizer 改成 AdamW，workers 6，amp 关闭，类别改成 2,4，batch 12，imgsz 960，先给我计划。')
        assert turn6['status'] == 'completed', turn6
        assert '执行后端: 标准 YOLO 训练' in turn6['message']
        assert '训练环境: base' in turn6['message']
        assert '已从默认环境 yolodo 切换到 base' in turn6['message']
        assert '输出组织: project=/runs/prepare-advanced-b, name=exp-b' in turn6['message']
        assert '高级参数:' in turn6['message']
        assert 'classes=[2, 4]' in turn6['message']
        assert 'optimizer=AdamW' in turn6['message']
        assert 'freeze=2' in turn6['message']
        assert 'workers=6' in turn6['message']
        assert 'amp=False' in turn6['message']
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['data_yaml'] == '/data/prepare-advanced/data.yaml'
        assert calls[-1][1]['training_environment'] == 'base'
        assert calls[-1][1]['project'] == '/runs/prepare-advanced-b'
        assert calls[-1][1]['name'] == 'exp-b'
        assert calls[-1][1]['optimizer'] == 'AdamW'
        assert calls[-1][1]['workers'] == 6
        assert calls[-1][1]['amp'] is False
        assert calls[-1][1]['classes'] == [2, 4]
        assert calls[-1][1]['batch'] == 12
        assert calls[-1][1]['imgsz'] == 960

        turn7 = await client.chat('好，就按这个方案执行。')
        assert turn7['status'] == 'needs_confirmation', turn7
        assert turn7['tool_call']['name'] == 'start_training'
        assert turn7['tool_call']['args']['data_yaml'] == '/data/prepare-advanced/data.yaml'
        assert turn7['tool_call']['args']['training_environment'] == 'base'
        assert turn7['tool_call']['args']['project'] == '/runs/prepare-advanced-b'
        assert turn7['tool_call']['args']['name'] == 'exp-b'
        assert turn7['tool_call']['args']['optimizer'] == 'AdamW'
        assert turn7['tool_call']['args']['workers'] == 6
        assert turn7['tool_call']['args']['amp'] is False
        assert turn7['tool_call']['args']['classes'] == [2, 4]
        assert turn7['tool_call']['args']['batch'] == 12
        assert turn7['tool_call']['args']['imgsz'] == 960

        turn8 = await client.confirm(turn7['thread_id'], approved=True)
        assert turn8['status'] == 'completed', turn8
        assert '训练已启动' in turn8['message']
        assert client.session_state.active_training.training_environment == 'base'
        assert client.session_state.active_training.project == '/runs/prepare-advanced-b'
        assert client.session_state.active_training.run_name == 'exp-b'
        assert client.session_state.active_training.optimizer == 'AdamW'
        assert client.session_state.active_training.freeze == 2
        assert client.session_state.active_training.workers == 6
        assert client.session_state.active_training.amp is False
        assert client.session_state.active_training.classes == [2, 4]
        assert client.session_state.active_training.batch == 12
        assert client.session_state.active_training.imgsz == 960
        assert current_training_plan_draft(client) == {}
        assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
        assert [name for name, _ in calls].count('training_preflight') == preflight_count_before + 1
        assert [name for name, _ in calls].count('start_training') == 1
        print('training plan prepare bridge replanning ok')
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def _run_prepare_bridge_trainer_discussion() -> None:
    work = Path(__file__).resolve().parent / '_tmp_training_plan_prepare_bridge_trainer'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-plan-prepare-bridge-trainer', memory_root=str(work))
        graph = _DummyGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': '/data/prepare-trainer',
                    'resolved_img_dir': '/data/prepare-trainer/images',
                    'resolved_label_dir': '/data/prepare-trainer/labels',
                    'resolved_data_yaml': '',
                    'ready': False,
                    'preparable': True,
                    'primary_blocker_type': 'missing_yaml',
                    'warnings': ['建议先完成 prepare，再确认训练组织方式'],
                    'blockers': ['缺少可用的 data_yaml'],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                    'environments': [
                        {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                        {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                    ],
                    'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                }
            elif tool_name == 'prepare_dataset_for_training':
                result = {
                    'ok': True,
                    'ready': True,
                    'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                    'dataset_root': '/data/prepare-trainer',
                    'img_dir': '/data/prepare-trainer/images',
                    'label_dir': '/data/prepare-trainer/labels',
                    'data_yaml': '/data/prepare-trainer/data.yaml',
                    'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                    'next_actions': ['如需训练，可继续 start_training'],
                }
            elif tool_name == 'training_preflight':
                selected_environment = kwargs.get('training_environment') or 'yolodo'
                result = {
                    'ok': True,
                    'ready_to_start': True,
                    'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                    'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                    'resolved_args': {
                        'model': kwargs['model'],
                        'data_yaml': kwargs['data_yaml'],
                        'epochs': kwargs['epochs'],
                        'device': kwargs.get('device', 'auto') or 'auto',
                        'training_environment': selected_environment,
                        'project': kwargs.get('project') or None,
                        'name': kwargs.get('name') or None,
                        'batch': kwargs.get('batch'),
                        'imgsz': kwargs.get('imgsz'),
                        'patience': kwargs.get('patience'),
                        'resume': kwargs.get('resume'),
                    },
                    'command_preview': ['yolo', 'train'],
                    'blockers': [],
                    'warnings': ['prepare 完成后可以直接开训，但建议确认最终参数'],
                }
            elif tool_name == 'start_training':
                result = {
                    'ok': True,
                    'summary': '训练已启动: model=yolov8l.pt, data=/data/prepare-trainer/data.yaml, device=auto',
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'pid': 8881,
                    'log_file': '/runs/prepare_trainer.txt',
                    'started_at': 777.1,
                    'resolved_args': dict(kwargs),
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            if tool_name == 'start_training' and result.get('ok'):
                clear_training_plan_draft(client)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat(
            '数据在 /data/prepare-trainer，用 yolov8l.pt 训练 35轮，环境 yolodo，project /runs/prepare-trainer-a，name exp-a，batch 9，先给我计划，不要执行。'
        )
        assert turn1['status'] == 'completed', turn1
        assert '执行方式: 先准备再训练' in turn1['message']
        assert '训练环境: yolodo' in turn1['message']

        turn2 = await client.chat('执行。')
        assert turn2['status'] == 'needs_confirmation', turn2
        assert turn2['tool_call']['name'] == 'prepare_dataset_for_training'

        turn3 = await client.confirm(turn2['thread_id'], approved=True)
        assert turn3['status'] == 'needs_confirmation', turn3
        assert turn3['tool_call']['name'] == 'start_training'
        assert turn3['tool_call']['args']['data_yaml'] == '/data/prepare-trainer/data.yaml'
        preflight_count_before = [name for name, _ in calls].count('training_preflight')

        turn4 = await client.chat('先别执行，为什么现在就能训练了？改成自定义 trainer 先讨论。')
        assert turn4['status'] == 'completed', turn4
        assert '执行方式: 先讨论方案，暂不执行' in turn4['message']
        assert '执行后端: 自定义 Trainer' in turn4['message']
        assert '当前自动执行链只支持标准 YOLO 训练' in turn4['message']
        assert [name for name, _ in calls].count('training_preflight') == preflight_count_before

        turn5 = await client.chat('不用 trainer 了，切回标准 yolo，环境恢复默认，project 不要了，name 不要了，batch 14，imgsz 1024，patience 18，resume 不要，先给我计划。')
        assert turn5['status'] == 'completed', turn5
        assert '训练计划草案：' in turn5['message']
        assert '训练环境: yolodo' in turn5['message']
        assert '核心参数: model=yolov8l.pt, data=/data/prepare-trainer/data.yaml, epochs=35, batch=14, imgsz=1024, device=auto' in turn5['message']
        assert '输出组织:' not in turn5['message']
        assert turn5['tool_call'] is None
        assert calls[-1][0] == 'training_preflight'
        assert calls[-1][1]['training_environment'] in {'', 'yolodo'}
        assert calls[-1][1]['project'] == ''
        assert calls[-1][1]['name'] == ''
        assert calls[-1][1]['batch'] == 14
        assert calls[-1][1]['imgsz'] == 1024
        assert calls[-1][1]['patience'] == 18
        assert calls[-1][1]['resume'] is False

        turn6 = await client.chat('执行。')
        assert turn6['status'] == 'completed', turn6
        assert turn6['tool_call']['name'] == 'start_training'
        assert '训练已启动' in turn6['message']
        assert client.session_state.active_training.training_environment == 'yolodo'
        assert client.session_state.active_training.project == '/runs/prepare-trainer-a'
        assert client.session_state.active_training.run_name == 'exp-a'
        assert client.session_state.active_training.batch == 9
        assert client.session_state.active_training.imgsz is None
        assert client.session_state.active_training.patience is None
        assert client.session_state.active_training.resume is None
        assert current_training_plan_draft(client) == {}
        assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
        assert [name for name, _ in calls].count('start_training') == 1
        print('training plan prepare bridge trainer discussion ok')
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def _main() -> None:
    await _run()
    await _run_backend_switch_followup()
    await _run_prepare_bridge_replanning()
    await _run_prepare_bridge_trainer_discussion()


def main() -> None:
    asyncio.run(_main())


if __name__ == '__main__':
    main()
