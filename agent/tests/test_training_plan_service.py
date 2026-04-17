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
    build_training_revision_draft,
    build_training_preflight_tool_args,
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

    asyncio.run(_run_async())

    print('training plan service helpers ok')


if __name__ == '__main__':
    _run()
