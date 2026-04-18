from __future__ import annotations

import asyncio
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

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')

    class _BaseMessage:
        def __init__(self, content=''):
            self.content = content

    class _HumanMessage(_BaseMessage):
        pass

    class _SystemMessage(_BaseMessage):
        pass

    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    core_mod.messages = messages_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod

from yolostudio_agent.agent.client.training_plan_service import (
    build_training_recovery_base_args,
    build_training_revision_draft,
    build_training_preflight_tool_args,
    prepare_training_revision_context,
    resolve_training_recovery_bootstrap,
    resolve_training_plan_dialogue_context,
    resolve_training_plan_dialogue_existing_action,
    run_prepare_only_entrypoint,
    run_training_recovery_entrypoint,
    run_training_request_entrypoint,
    run_training_request_orchestration,
    run_training_recovery_orchestration,
    resolve_training_start_args,
)
from yolostudio_agent.agent.client.session_state import SessionState


def _build_training_plan_draft(**kwargs):
    return {
        'dataset_path': kwargs.get('dataset_path') or '',
        'planned_training_args': dict(kwargs.get('planned_training_args') or {}),
        'next_step_tool': str(kwargs.get('next_tool_name') or ''),
        'next_step_args': dict(kwargs.get('next_tool_args') or {}),
        'blockers': ['缺少可用 data_yaml'] if (kwargs.get('readiness') or {}).get('preparable') else [],
    }


async def _render_training_plan_message(draft: dict, pending: bool) -> str:
    label = 'pending' if pending else 'discussion'
    return f"{label}:{draft.get('next_step_tool') or 'none'}"


async def _render_tool_result_message(tool_name: str, parsed: dict) -> str:
    return str(parsed.get('summary') or parsed.get('error') or f'{tool_name}:done')


async def _run_async() -> None:
    request_calls: list[tuple[str, dict]] = []

    async def _fake_request_tool(tool_name: str, **kwargs):
        request_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_preflight':
            return {
                'ok': True,
                'ready_to_start': True,
                'resolved_args': {**kwargs, 'device': '1', 'training_environment': 'yolodo'},
            }
        raise AssertionError(tool_name)

    discussion_result = await run_training_request_orchestration(
        user_text='数据在 /data/train，用 yolov8n.pt 训练，先给我计划。',
        dataset_path='/data/train',
        readiness={'ready': True, 'resolved_data_yaml': '/data/train/data.yaml'},
        requested_args={'model': 'yolov8n.pt', 'epochs': 30},
        wants_split=False,
        discussion_only=True,
        execution_backend='standard_yolo',
        direct_tool=_fake_request_tool,
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert discussion_result['defer_to_graph'] is False, discussion_result
    assert discussion_result['reply'] == 'discussion:start_training', discussion_result
    assert discussion_result['draft']['next_step_tool'] == 'start_training', discussion_result

    pending_result = await run_training_request_orchestration(
        user_text='数据在 /data/preparable，用 yolov8n.pt 训练，执行。',
        dataset_path='/data/preparable',
        readiness={'ready': False, 'preparable': True},
        requested_args={'model': 'yolov8n.pt', 'classes_txt': '/data/preparable/classes.txt'},
        wants_split=True,
        discussion_only=False,
        execution_backend='standard_yolo',
        direct_tool=_fake_request_tool,
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert pending_result['defer_to_graph'] is True, pending_result
    assert pending_result['reply'] == 'pending:prepare_dataset_for_training', pending_result
    assert pending_result['draft']['next_step_args'] == {
        'dataset_path': '/data/preparable',
        'force_split': True,
        'classes_txt': '/data/preparable/classes.txt',
    }, pending_result

    assert request_calls == [
        ('training_preflight', {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/train/data.yaml',
            'epochs': 30,
            'device': 'auto',
            'training_environment': '',
            'project': '',
            'name': '',
            'optimizer': '',
            'batch': None,
            'imgsz': None,
            'fraction': None,
            'classes': None,
            'single_cls': None,
            'freeze': None,
            'resume': None,
            'lr0': None,
            'patience': None,
            'workers': None,
            'amp': None,
        }),
    ], request_calls

    hot_update_calls: list[tuple[str, dict]] = []

    async def _unexpected_hot_update_tool(tool_name: str, **kwargs):
        hot_update_calls.append((tool_name, dict(kwargs)))
        raise AssertionError(tool_name)

    running_state = SessionState(session_id='running')
    running_state.active_training.running = True
    hot_update_result = await run_training_request_entrypoint(
        session_state=running_state,
        user_text='batch 调到 16，继续训练。',
        normalized_text='batch 调到 16，继续训练。'.lower(),
        dataset_path='/data/train',
        frame_followup_path='',
        wants_train=True,
        wants_predict=False,
        no_train=False,
        readiness_only_query=False,
        wants_training_outcome_analysis=False,
        wants_next_step_guidance=False,
        wants_training_knowledge=False,
        wants_training_revision=True,
        wants_stop_training=False,
        blocks_training_start=False,
        explicit_run_ids=None,
        wants_split=False,
        direct_tool=_unexpected_hot_update_tool,
        collect_requested_training_args=lambda *_args, **_kwargs: {},
        is_training_discussion_only=lambda _text: False,
        extract_training_execution_backend=lambda _text: 'standard_yolo',
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert '不能直接热更新' in hot_update_result['reply'], hot_update_result
    assert hot_update_result['defer_to_graph'] is False, hot_update_result
    assert hot_update_calls == [], hot_update_calls

    missing_input_calls: list[tuple[str, dict]] = []

    async def _unexpected_missing_input_tool(tool_name: str, **kwargs):
        missing_input_calls.append((tool_name, dict(kwargs)))
        raise AssertionError(tool_name)

    missing_input_result = await run_training_request_entrypoint(
        session_state=SessionState(session_id='missing-input'),
        user_text='开始训练',
        normalized_text='开始训练'.lower(),
        dataset_path='',
        frame_followup_path='',
        wants_train=True,
        wants_predict=False,
        no_train=False,
        readiness_only_query=False,
        wants_training_outcome_analysis=False,
        wants_next_step_guidance=False,
        wants_training_knowledge=False,
        wants_training_revision=False,
        wants_stop_training=False,
        blocks_training_start=False,
        explicit_run_ids=None,
        wants_split=False,
        direct_tool=_unexpected_missing_input_tool,
        collect_requested_training_args=lambda *_args, **_kwargs: {},
        is_training_discussion_only=lambda _text: False,
        extract_training_execution_backend=lambda _text: 'standard_yolo',
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert '- 缺少数据集路径' in missing_input_result['reply'], missing_input_result
    assert '- 缺少预训练权重/模型' in missing_input_result['reply'], missing_input_result
    assert missing_input_calls == [], missing_input_calls

    entrypoint_calls: list[tuple[str, dict]] = []

    async def _fake_entrypoint_tool(tool_name: str, **kwargs):
        entrypoint_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            return {
                'ready': True,
                'resolved_data_yaml': '/data/frames/data.yaml',
            }
        if tool_name == 'list_training_environments':
            return {'items': ['yolodo']}
        if tool_name == 'training_preflight':
            return {
                'ok': True,
                'ready_to_start': True,
                'resolved_args': {**kwargs, 'device': '0', 'training_environment': 'yolodo'},
            }
        raise AssertionError(tool_name)

    preserved_state = SessionState(session_id='preserve-model')
    preserved_state.active_training.training_plan_draft = {
        'planned_training_args': {'model': '/weights/last.pt'}
    }
    preserved_state.active_training.model = '/weights/fallback.pt'
    preserved_model_result = await run_training_request_entrypoint(
        session_state=preserved_state,
        user_text='这些帧训练一下，直接开始。',
        normalized_text='这些帧训练一下，直接开始。'.lower(),
        dataset_path='/data/frames',
        frame_followup_path='/data/frames',
        wants_train=True,
        wants_predict=False,
        no_train=False,
        readiness_only_query=False,
        wants_training_outcome_analysis=False,
        wants_next_step_guidance=False,
        wants_training_knowledge=False,
        wants_training_revision=False,
        wants_stop_training=False,
        blocks_training_start=False,
        explicit_run_ids=None,
        wants_split=False,
        direct_tool=_fake_entrypoint_tool,
        collect_requested_training_args=lambda _text, *, data_yaml=None: {
            'epochs': 12,
            'data_yaml': data_yaml,
        },
        is_training_discussion_only=lambda _text: False,
        extract_training_execution_backend=lambda _text: 'standard_yolo',
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert preserved_model_result['defer_to_graph'] is True, preserved_model_result
    assert preserved_model_result['reply'] == 'pending:start_training', preserved_model_result
    assert preserved_model_result['draft']['planned_training_args']['model'] == '/weights/last.pt', preserved_model_result
    assert entrypoint_calls == [
        ('training_readiness', {'img_dir': '/data/frames'}),
        ('list_training_environments', {}),
        ('training_preflight', {
            'model': '/weights/last.pt',
            'data_yaml': '/data/frames/data.yaml',
            'epochs': 12,
            'device': 'auto',
            'training_environment': '',
            'project': '',
            'name': '',
            'optimizer': '',
            'batch': None,
            'imgsz': None,
            'fraction': None,
            'classes': None,
            'single_cls': None,
            'freeze': None,
            'resume': None,
            'lr0': None,
            'patience': None,
            'workers': None,
            'amp': None,
        }),
    ], entrypoint_calls

    recovery_calls: list[tuple[str, dict]] = []

    async def _fake_recovery_tool(tool_name: str, **kwargs):
        recovery_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_preflight':
            return {
                'ok': True,
                'ready_to_start': True,
                'resolved_args': {**kwargs, 'device': '0', 'training_environment': 'yolodo'},
            }
        raise AssertionError(tool_name)

    recovery_result = await run_training_recovery_orchestration(
        user_text='从最近状态继续训练。',
        dataset_path='/data/recovery',
        readiness={'ready': True, 'resolved_data_yaml': '/data/recovery/data.yaml'},
        base_args={
            'model': 'yolov8n.pt',
            'data_yaml': '/data/recovery/data.yaml',
            'epochs': 60,
            'device': 'auto',
            'resume': True,
            'project': '/runs/recovery',
        },
        direct_tool=_fake_recovery_tool,
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert recovery_result['defer_to_graph'] is True, recovery_result
    assert recovery_result['reply'] == 'pending:start_training', recovery_result
    assert recovery_result['draft']['next_step_tool'] == 'start_training', recovery_result
    assert recovery_result['draft']['next_step_args']['resume'] is True, recovery_result
    assert recovery_result['draft']['next_step_args']['device'] == '0', recovery_result
    assert recovery_calls == [
        ('training_preflight', {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/recovery/data.yaml',
            'epochs': 60,
            'device': 'auto',
            'training_environment': '',
            'project': '/runs/recovery',
            'name': '',
            'optimizer': '',
            'batch': None,
            'imgsz': None,
            'fraction': None,
            'classes': None,
            'single_cls': None,
            'freeze': None,
            'resume': True,
            'lr0': None,
            'patience': None,
            'workers': None,
            'amp': None,
        }),
    ], recovery_calls

    recovery_entrypoint_calls: list[tuple[str, dict]] = []

    async def _fake_recovery_entrypoint_tool(tool_name: str, **kwargs):
        recovery_entrypoint_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            return {
                'ready': True,
                'resolved_data_yaml': '/data/recovery/data.yaml',
            }
        if tool_name == 'list_training_environments':
            return {'items': ['yolodo']}
        if tool_name == 'training_preflight':
            return {
                'ok': True,
                'ready_to_start': True,
                'resolved_args': {**kwargs, 'device': '0', 'training_environment': 'yolodo'},
            }
        raise AssertionError(tool_name)

    recovery_entrypoint_state = SessionState(session_id='recovery-entrypoint')
    recovery_entrypoint_result = await run_training_recovery_entrypoint(
        session_state=recovery_entrypoint_state,
        user_text='从最近状态继续训练。',
        dataset_path='/data/recovery',
        base_args={
            'model': 'yolov8n.pt',
            'data_yaml': '/data/recovery/data.yaml',
            'epochs': 60,
            'device': 'auto',
            'resume': True,
        },
        direct_tool=_fake_recovery_entrypoint_tool,
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_training_plan_message=_render_training_plan_message,
    )
    assert recovery_entrypoint_result['defer_to_graph'] is True, recovery_entrypoint_result
    assert recovery_entrypoint_result['reply'] == 'pending:start_training', recovery_entrypoint_result
    assert recovery_entrypoint_calls == [
        ('training_readiness', {'img_dir': '/data/recovery'}),
        ('list_training_environments', {}),
        ('training_preflight', {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/recovery/data.yaml',
            'epochs': 60,
            'device': 'auto',
            'training_environment': '',
            'project': '',
            'name': '',
            'optimizer': '',
            'batch': None,
            'imgsz': None,
            'fraction': None,
            'classes': None,
            'single_cls': None,
            'freeze': None,
            'resume': True,
            'lr0': None,
            'patience': None,
            'workers': None,
            'amp': None,
        }),
    ], recovery_entrypoint_calls

    revision_calls: list[tuple[str, dict]] = []

    async def _fake_revision_tool(tool_name: str, **kwargs):
        revision_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_preflight':
            return {
                'ok': True,
                'ready_to_start': True,
                'resolved_args': {**kwargs, 'device': '2', 'training_environment': 'base'},
            }
        raise AssertionError(tool_name)

    revision_draft = await build_training_revision_draft(
        user_text='batch 改成 16，继续按新计划执行。',
        dataset_path='/data/revised',
        readiness={'ready': True, 'resolved_data_yaml': '/data/revised/data.yaml'},
        planned_args={
            'model': 'yolov8n.pt',
            'data_yaml': '/data/revised/data.yaml',
            'epochs': 40,
            'device': 'auto',
            'batch': 16,
        },
        next_tool_name='start_training',
        next_tool_args={},
        execution_mode='direct_train',
        execution_backend='standard_yolo',
        advanced_requested=True,
        direct_tool=_fake_revision_tool,
        build_training_plan_draft_fn=_build_training_plan_draft,
    )
    assert revision_draft['next_step_tool'] == 'start_training', revision_draft
    assert revision_draft['next_step_args']['device'] == '2', revision_draft
    assert revision_draft['next_step_args']['batch'] == 16, revision_draft
    assert revision_draft['advanced_details_requested'] is True, revision_draft
    assert revision_calls == [
        ('training_preflight', {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/revised/data.yaml',
            'epochs': 40,
            'device': 'auto',
            'training_environment': '',
            'project': '',
            'name': '',
            'optimizer': '',
            'batch': 16,
            'imgsz': None,
            'fraction': None,
            'classes': None,
            'single_cls': None,
            'freeze': None,
            'resume': None,
            'lr0': None,
            'patience': None,
            'workers': None,
            'amp': None,
        }),
    ], revision_calls

    revision_context_calls: list[tuple[str, dict]] = []

    async def _fake_revision_context_tool(tool_name: str, **kwargs):
        revision_context_calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            return {
                'ready': True,
                'resolved_data_yaml': '/data/newset/data.yaml',
            }
        if tool_name == 'list_training_environments':
            return {'items': ['yolodo']}
        raise AssertionError(tool_name)

    revision_context_state = SessionState(session_id='revision-context')
    revision_context_state.active_dataset.data_yaml = '/data/current/data.yaml'
    revision_context_result = await prepare_training_revision_context(
        session_state=revision_context_state,
        user_text='数据换成 /data/newset，只做准备，不要划分。',
        draft={
            'dataset_path': '/data/current',
            'execution_mode': 'prepare_then_train',
            'next_step_tool': 'prepare_dataset_for_training',
            'next_step_args': {'dataset_path': '/data/current', 'force_split': True},
            'planned_training_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/current/data.yaml', 'batch': 8},
        },
        pending={'name': 'prepare_dataset_for_training', 'args': {'dataset_path': '/data/current', 'force_split': True}},
        latest_dataset_path='/data/newset',
        clear_fields=[],
        switching_prepare_only_to_train=False,
        wants_prepare_only=True,
        wants_disable_split=True,
        collect_requested_training_args=lambda _text, *, data_yaml=None: {
            'model': 'yolov8n.pt',
            'data_yaml': data_yaml,
        },
        extract_training_execution_backend=lambda _text: 'standard_yolo',
        wants_training_advanced_details=lambda _text: False,
        direct_tool=_fake_revision_context_tool,
    )
    assert revision_context_result == {
        'revised_draft': {
            'dataset_path': '/data/current',
            'execution_mode': 'prepare_only',
            'next_step_tool': 'prepare_dataset_for_training',
            'next_step_args': {'dataset_path': '/data/current'},
            'planned_training_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/current/data.yaml', 'batch': 8},
        },
        'planned_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/newset/data.yaml',
            'batch': 8,
        },
        'dataset_path': '/data/newset',
        'readiness': {
            'ready': True,
            'resolved_data_yaml': '/data/newset/data.yaml',
        },
        'next_tool_name': 'prepare_dataset_for_training',
        'next_tool_args': {'dataset_path': '/data/current'},
        'execution_mode': 'prepare_only',
        'execution_backend': 'standard_yolo',
        'advanced_requested': False,
    }, revision_context_result
    assert revision_context_calls == [
        ('training_readiness', {'img_dir': '/data/newset'}),
        ('list_training_environments', {}),
    ], revision_context_calls

    prepare_only_calls: list[tuple[str, dict]] = []

    async def _fake_prepare_only_tool(tool_name: str, **kwargs):
        prepare_only_calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'dataset_training_readiness', tool_name
        return {
            'ok': True,
            'ready': False,
            'resolved_img_dir': '/data/prep/images',
            'resolved_label_dir': '/data/prep/labels',
        }

    prepare_only_result = await run_prepare_only_entrypoint(
        user_text='数据在 /data/prep，用 yolov8n.pt 训练，只做准备，按默认比例划分。',
        dataset_path='/data/prep',
        direct_tool=_fake_prepare_only_tool,
        collect_requested_training_args=lambda _text, *, data_yaml=None: {
            'model': 'yolov8n.pt',
            'data_yaml': data_yaml,
        },
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_tool_result_message=_render_tool_result_message,
    )
    assert prepare_only_result['defer_to_graph'] is True, prepare_only_result
    assert prepare_only_result['reply'] == '', prepare_only_result
    assert prepare_only_result['draft']['next_step_tool'] == 'prepare_dataset_for_training', prepare_only_result
    assert prepare_only_result['draft']['next_step_args'] == {
        'dataset_path': '/data/prep',
        'force_split': True,
    }, prepare_only_result
    assert prepare_only_result['draft']['execution_mode'] == 'prepare_only', prepare_only_result
    assert prepare_only_calls == [
        ('dataset_training_readiness', {'img_dir': '/data/prep'}),
    ], prepare_only_calls

    prepare_ready_calls: list[tuple[str, dict]] = []

    async def _fake_prepare_ready_tool(tool_name: str, **kwargs):
        prepare_ready_calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'dataset_training_readiness', tool_name
        return {
            'ok': True,
            'ready': True,
            'resolved_data_yaml': '/data/prep/data.yaml',
            'resolved_img_dir': '/data/prep/images',
            'resolved_label_dir': '/data/prep/labels',
        }

    prepare_ready_result = await run_prepare_only_entrypoint(
        user_text='数据在 /data/prep，只做准备。',
        dataset_path='/data/prep',
        direct_tool=_fake_prepare_ready_tool,
        collect_requested_training_args=lambda _text, *, data_yaml=None: {},
        build_training_plan_draft_fn=_build_training_plan_draft,
        render_tool_result_message=_render_tool_result_message,
    )
    assert prepare_ready_result == {
        'status': 'completed',
        'reply': '当前数据已经可训练，现成 data.yaml: /data/prep/data.yaml。如果你只是想准备数据，这一步已经完成。',
        'draft': None,
        'clear_draft': True,
        'defer_to_graph': False,
    }, prepare_ready_result
    assert prepare_ready_calls == [
        ('dataset_training_readiness', {'img_dir': '/data/prep'}),
    ], prepare_ready_calls


def _run() -> None:
    preflight_payload = build_training_preflight_tool_args(
        {
            'epochs': 20,
            'device': '',
            'training_environment': 'yolodo',
            'batch': 8,
            'optimizer': 'AdamW',
            'resume': False,
        },
        fallback_model='yolov8n.pt',
        fallback_data_yaml='/data/train.yaml',
    )
    assert preflight_payload == {
        'model': 'yolov8n.pt',
        'data_yaml': '/data/train.yaml',
        'epochs': 20,
        'device': 'auto',
        'training_environment': 'yolodo',
        'project': '',
        'name': '',
        'optimizer': 'AdamW',
        'batch': 8,
        'imgsz': None,
        'fraction': None,
        'classes': None,
        'single_cls': None,
        'freeze': None,
        'resume': False,
        'lr0': None,
        'patience': None,
        'workers': None,
        'amp': None,
    }, preflight_payload

    resolved_args = resolve_training_start_args(
        {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/original.yaml',
            'epochs': 50,
            'device': 'auto',
            'training_environment': 'base',
            'project': '/runs/original',
            'name': 'exp-old',
            'batch': 8,
            'optimizer': 'SGD',
            'resume': True,
        },
        {
            'resolved_args': {
                'model': '/weights/yolo11n.pt',
                'data_yaml': '/data/final.yaml',
                'device': '1',
                'project': None,
                'batch': None,
                'optimizer': 'AdamW',
            }
        },
    )
    assert resolved_args == {
        'model': '/weights/yolo11n.pt',
        'data_yaml': '/data/final.yaml',
        'epochs': 50,
        'device': '1',
        'training_environment': 'base',
        'project': '/runs/original',
        'name': 'exp-old',
        'optimizer': 'AdamW',
        'batch': None,
        'imgsz': None,
        'fraction': None,
        'classes': None,
        'single_cls': None,
        'freeze': None,
        'resume': True,
        'lr0': None,
        'patience': None,
        'workers': None,
        'amp': None,
    }, resolved_args

    recovery_state = SessionState(session_id='recovery-base')
    recovery_state.active_training.last_start_result = {
        'resolved_args': {
            'epochs': 40,
            'device': '0',
        }
    }
    recovery_state.active_training.model = '/weights/fallback.pt'
    recovery_state.active_training.data_yaml = '/data/fallback.yaml'
    recovery_state.active_training.training_environment = 'yolodo'
    recovery_state.active_dataset.data_yaml = '/data/dataset.yaml'
    assert build_training_recovery_base_args(recovery_state) == {
        'epochs': 40,
        'device': '0',
        'model': '/weights/fallback.pt',
        'data_yaml': '/data/fallback.yaml',
        'training_environment': 'yolodo',
    }

    bootstrap_running_state = SessionState(session_id='bootstrap-running')
    bootstrap_running_state.active_training.running = True
    bootstrap_running_state.active_training.training_run_summary = {'run_state': 'running'}
    running_bootstrap = resolve_training_recovery_bootstrap(
        session_state=bootstrap_running_state,
        user_text='resume train_log_old',
        normalized_text='resume train_log_old',
        latest_dataset_path='',
        explicit_run_ids=['train_log_old'],
        requested_execute=False,
        wants_repeat_prepare=False,
        wants_retry_last_plan=False,
        wants_resume_recent_training=True,
        wants_analysis_only=False,
    )
    assert running_bootstrap == {
        'reply': '当前已有活动训练；如果你想恢复或切换到另一个历史 run，请先停止当前训练，再明确要恢复的 run。',
        'defer_to_graph': False,
        'proceed': False,
    }, running_bootstrap

    bootstrap_analysis_state = SessionState(session_id='bootstrap-analysis')
    bootstrap_analysis_state.active_training.last_summary = {'summary': '最近训练已完成', 'run_state': 'completed'}
    analysis_bootstrap = resolve_training_recovery_bootstrap(
        session_state=bootstrap_analysis_state,
        user_text='resume 上次训练，但不要接着训，只分析就行。',
        normalized_text='resume 上次训练，但不要接着训，只分析就行。'.lower(),
        latest_dataset_path='',
        explicit_run_ids=[],
        requested_execute=False,
        wants_repeat_prepare=False,
        wants_retry_last_plan=False,
        wants_resume_recent_training=True,
        wants_analysis_only=True,
    )
    assert analysis_bootstrap == {
        'reply': '',
        'defer_to_graph': True,
        'proceed': False,
    }, analysis_bootstrap

    bootstrap_repeat_prepare_state = SessionState(session_id='bootstrap-repeat-prepare')
    bootstrap_repeat_prepare_state.active_dataset.last_readiness = {'ready': True}
    bootstrap_repeat_prepare_state.active_dataset.data_yaml = '/data/ready/data.yaml'
    repeat_prepare_bootstrap = resolve_training_recovery_bootstrap(
        session_state=bootstrap_repeat_prepare_state,
        user_text='再 prepare 一次',
        normalized_text='再 prepare 一次'.lower(),
        latest_dataset_path='',
        explicit_run_ids=[],
        requested_execute=False,
        wants_repeat_prepare=True,
        wants_retry_last_plan=False,
        wants_resume_recent_training=False,
        wants_analysis_only=False,
    )
    assert repeat_prepare_bootstrap == {
        'reply': '当前数据集已经准备完成：/data/ready/data.yaml；不需要重复 prepare。你可以直接继续训练或重新规划。',
        'defer_to_graph': False,
        'proceed': False,
    }, repeat_prepare_bootstrap

    bootstrap_resume_state = SessionState(session_id='bootstrap-resume')
    bootstrap_resume_state.active_training.last_status = {'run_state': 'stopped'}
    bootstrap_resume_state.active_training.last_start_result = {
        'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/resume/data.yaml', 'epochs': 20}
    }
    bootstrap_resume_state.active_dataset.dataset_root = '/data/resume'
    proceed_bootstrap = resolve_training_recovery_bootstrap(
        session_state=bootstrap_resume_state,
        user_text='恢复上次训练',
        normalized_text='恢复上次训练'.lower(),
        latest_dataset_path='',
        explicit_run_ids=[],
        requested_execute=False,
        wants_repeat_prepare=False,
        wants_retry_last_plan=False,
        wants_resume_recent_training=True,
        wants_analysis_only=False,
    )
    assert proceed_bootstrap == {
        'reply': '',
        'defer_to_graph': False,
        'proceed': True,
        'base_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/resume/data.yaml',
            'epochs': 20,
            'resume': True,
        },
        'dataset_path': '/data/resume',
    }, proceed_bootstrap

    dialogue_context = resolve_training_plan_dialogue_context(
        user_text='resume 上次训练，但不要接着训，只分析就行。',
        explicit_run_ids=[],
        is_training_discussion_only=lambda _text: False,
    )
    assert dialogue_context['requested_execute'] is False, dialogue_context
    assert dialogue_context['wants_resume_recent_training'] is True, dialogue_context
    assert dialogue_context['wants_analysis_only'] is True, dialogue_context
    assert dialogue_context['is_loop_dialogue'] is False, dialogue_context

    dataset_revision_context = resolve_training_plan_dialogue_context(
        user_text='数据换成 /data/newset，现在用这个继续训练。',
        explicit_run_ids=[],
        is_training_discussion_only=lambda _text: False,
    )
    assert dataset_revision_context['latest_dataset_path'] == '/data/newset', dataset_revision_context
    assert dataset_revision_context['dataset_path_revision_requested'] is True, dataset_revision_context

    existing_action = resolve_training_plan_dialogue_existing_action(
        draft={'next_step_tool': 'start_training'},
        pending=None,
        flag_context={'has_revision': False},
        requested_execute=True,
        wants_repeat_prepare=False,
        readiness={},
        data_yaml='',
    )
    assert existing_action == {'action': 'defer_to_graph'}, existing_action

    repeat_prepare_action = resolve_training_plan_dialogue_existing_action(
        draft={'dataset_path': '/data/ready'},
        pending={'name': 'prepare_dataset_for_training', 'args': {'dataset_path': '/data/ready'}},
        flag_context={
            'has_revision': False,
            'wants_prepare_output_explanation': True,
        },
        requested_execute=False,
        wants_repeat_prepare=False,
        readiness={},
        data_yaml='',
    )
    assert repeat_prepare_action == {
        'action': 'reply_with_pending',
        'reply': (
            '如果继续 prepare，我会基于数据集 /data/ready 生成可训练产物；'
            '预期会产出可用的 data_yaml（通常是 /data/ready/data.yaml），完成后我会把真实路径写回状态。'
        ),
    }, repeat_prepare_action

    ready_repeat_action = resolve_training_plan_dialogue_existing_action(
        draft={'dataset_path': '/data/ready'},
        pending=None,
        flag_context={'has_revision': False},
        requested_execute=False,
        wants_repeat_prepare=True,
        readiness={'ready': True},
        data_yaml='/data/ready/data.yaml',
    )
    assert ready_repeat_action == {
        'action': 'reply',
        'reply': '当前数据集已经准备完成：/data/ready/data.yaml；不需要重复 prepare。你可以直接继续训练或重新规划。',
    }, ready_repeat_action

    asyncio.run(_run_async())

    print('training plan service helpers ok')


if __name__ == '__main__':
    _run()
