from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.tests._chaos_test_support import WORK, _ScriptedGraph, _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run
from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.mainline_route_support import resolve_mainline_dispatch_payload
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    build_training_plan_context_payload,
)
from yolostudio_agent.agent.client.training_request_service import (
    run_prepare_only_flow,
    run_training_request_entrypoint,
)
from langchain_core.messages import AIMessage, ToolMessage


class _PredictVideosGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.client = None
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        del config
        client = self.client
        if client is not None:
            draft = dict(client.session_state.active_training.training_plan_draft or {})
            if draft:
                self.plan_context = build_training_plan_context_from_draft(draft)
            has_draft = bool(draft)
            has_pending = bool(client.session_state.pending_confirmation.tool_name)
            if not has_draft and not has_pending:
                self.plan_context = None
        if not self.plan_context:
            return None
        return type('GraphState', (), {'values': {'training_plan_context': dict(self.plan_context)}})()

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        client = self.client
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        if not plan_context and client is not None:
            mainline_context = client._collect_mainline_context(user_text)
            route_state = await client._resolve_mainline_route_state(user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
            if training_entrypoint_args:
                prepare_only_followup = await run_prepare_only_flow(
                    user_text=user_text,
                    looks_like_prepare_only_request=client._looks_like_prepare_only_request,
                    extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                    local_path_exists=lambda path: Path(path).expanduser().exists(),
                    direct_tool=client.direct_tool,
                    collect_requested_training_args=client._collect_requested_training_args,
                    build_training_plan_draft_fn=client._build_training_plan_draft,
                    render_tool_result_message=client._render_tool_result_message,
                )
                if prepare_only_followup:
                    entrypoint_result = {
                        'reply': str(prepare_only_followup.get('reply') or '').strip(),
                        'draft': dict(prepare_only_followup.get('draft') or {}),
                        'defer_to_graph': str(prepare_only_followup.get('action') or '').strip() == 'save_draft_and_handoff',
                    }
                else:
                    entrypoint_result = await run_training_request_entrypoint(
                        session_state=client.session_state,
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
                        current_training_plan_context=build_training_plan_context_payload(client.session_state),
                        direct_tool=client.direct_tool,
                        collect_requested_training_args=client._collect_requested_training_args,
                        is_training_discussion_only=client._is_training_discussion_only,
                        extract_training_execution_backend=client._extract_training_execution_backend_from_text,
                        build_training_plan_draft_fn=client._build_training_plan_draft,
                        render_training_plan_message=client._render_training_plan_message,
                    )
                draft = dict((entrypoint_result or {}).get('draft') or {})
                reply = str((entrypoint_result or {}).get('reply') or '').strip()
                if draft:
                    client.session_state.active_training.training_plan_draft = dict(draft)
                    self.plan_context = build_training_plan_context_from_draft(draft)
                if reply:
                    return {'messages': messages + [AIMessage(content=reply)]}

        args = {'source_path': '/data/videos', 'model': '/models/qcar.pt'}
        result = {
            'ok': True,
            'summary': '视频预测完成: 已处理 2 个视频, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15',
            'model': '/models/qcar.pt',
            'source_path': '/data/videos',
            'processed_videos': 2,
            'total_frames': 24,
            'detected_frames': 13,
            'total_detections': 15,
            'class_counts': {'two_wheeler': 15},
            'output_dir': '/tmp/predict-chaos',
            'report_path': '/tmp/predict-chaos/report.json',
            'warnings': [],
        }
        self.calls.append(('predict_videos', dict(args)))
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': 'predict_videos', 'args': args}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name='predict_videos', tool_call_id=tool_call_id),
                AIMessage(content='视频预测完成: 已处理 2 个视频, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15'),
            ]
        }


class _ObservedStatusGraph:
    def __init__(self) -> None:
        self.client = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        self.calls.append(('check_training_status', {}))
        result = await self.client.direct_tool('check_training_status')
        reply = await self.client._render_tool_result_message('check_training_status', result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': 'check_training_status', 'args': {}}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name='check_training_status', tool_call_id=tool_call_id),
                AIMessage(content=reply),
            ]
        }


async def _scenario_c01_missing_everything_blocks_without_graph() -> None:
    client = _make_client('chaos-p0-c01')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C01 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('帮我开始训练。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c02_dataset_without_model_stays_blocked() -> None:
    client = _make_client('chaos-p0-c02')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/no-model',
                'resolved_img_dir': '/data/no-model/images',
                'resolved_label_dir': '/data/no-model/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('用这个数据集 /data/no-model 训练，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert [name for name, _ in calls] == ['training_readiness', 'list_training_environments']


async def _scenario_c03_model_without_dataset_stays_blocked() -> None:
    client = _make_client('chaos-p0-c03')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C03 should not call tools without dataset: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('用 yolov8n.pt 训练。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' not in turn['message']
    assert calls == []


async def _scenario_c11_latest_epoch_revision_wins() -> None:
    client = _make_client('chaos-p0-c11')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/revise',
                'resolved_img_dir': '/data/revise/images',
                'resolved_label_dir': '/data/revise/labels',
                'resolved_data_yaml': '/data/revise/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/revise，用 yolov8n.pt 训练 100轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('不对，先 10 轮。')
    assert turn2['status'] == 'completed', turn2
    assert 'epochs=10' in turn2['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['epochs'] == 10


async def _scenario_c21_running_training_replan_does_not_override_active_run() -> None:
    client = _make_client('chaos-p0-c21')
    client.session_state.active_training.running = True
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.session_state.active_training.pid = 4321
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/new-run',
                'resolved_img_dir': '/data/new-run/images',
                'resolved_label_dir': '/data/new-run/labels',
                'resolved_data_yaml': '/data/new-run/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('现在再开一个训练。数据在 /data/new-run，用 yolov8s.pt 训练 20轮，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 4321
    assert client.session_state.active_training.training_plan_draft.get('planned_training_args', {}).get('model') == 'yolov8s.pt'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c31_prepare_cancel_keeps_plan_and_explains() -> None:
    client = _make_client('chaos-p0-c31')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/need-prepare',
                'resolved_img_dir': '/data/need-prepare/images',
                'resolved_label_dir': '/data/need-prepare/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/need-prepare，用 yolov8n.pt 训练 30轮，直接开始训练。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training'

    turn2 = await client.confirm(turn1['thread_id'], approved=False)
    assert turn2['status'] == 'cancelled', turn2
    assert '当前计划已保留' in turn2['message']
    assert client.session_state.active_training.training_plan_draft != {}

    turn3 = await client.chat('那怎么直接训？为什么必须先 prepare？')
    assert turn3['status'] == 'completed', turn3
    assert '当前阻塞:' in turn3['message']
    assert '缺少可用的 data_yaml' in turn3['message']
    assert client.session_state.active_training.training_plan_draft != {}


async def _scenario_c41_no_active_training_status_query_routes_status() -> None:
    client = _make_client('chaos-p0-c41')
    graph = _ObservedStatusGraph()
    graph.bind(client)
    client.graph = graph  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '当前没有正在运行的训练任务。',
            'run_state': 'unavailable',
            'analysis_ready': False,
            'minimum_facts_ready': False,
            'signals': ['no_active_run'],
            'facts': ['没有活动训练进程'],
            'next_actions': ['如果要训练，请先提供数据集和模型'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    routed = await client._try_handle_mainline_intent('训练到第几轮了？', 'thread-chaos-p0-c41-status')
    assert routed is not None
    assert routed['tool_call']['name'] == 'check_training_status'
    calls.clear()
    graph.calls.clear()
    turn = await client.chat('训练到第几轮了？')
    assert turn['status'] == 'completed', turn
    assert '当前没有正在运行的训练任务' in turn['message']
    assert calls == [('check_training_status', {})]
    assert graph.calls == []


async def _scenario_c51_missing_environment_blocks_start() -> None:
    client = _make_client('chaos-p0-c51')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/missing-env',
                'resolved_img_dir': '/data/missing-env/images',
                'resolved_label_dir': '/data/missing-env/labels',
                'resolved_data_yaml': '/data/missing-env/data.yaml',
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
            assert kwargs['training_environment'] == 'missing-env'
            result = {
                'ok': True,
                'ready_to_start': False,
                'summary': '训练预检未通过：训练环境不存在: missing-env（可用: base, yolodo）',
                'training_environment': None,
                'resolved_args': {'model': kwargs['model'], 'data_yaml': kwargs['data_yaml'], 'epochs': kwargs['epochs'], 'training_environment': 'missing-env'},
                'command_preview': [],
                'blockers': ['训练环境不存在: missing-env（可用: base, yolodo）'],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/missing-env，用 yolov8n.pt 训练，环境先用 missing-env，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练环境: missing-env' in turn['message']
    assert '训练环境不存在: missing-env' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c61_prediction_interrupt_preserves_training_plan() -> None:
    client = _make_client('chaos-p0-c61')
    predict_graph = _PredictVideosGraph()
    client.graph = predict_graph  # type: ignore[assignment]
    predict_graph.bind(client)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/train-plan',
                'resolved_img_dir': '/data/train-plan/images',
                'resolved_label_dir': '/data/train-plan/labels',
                'resolved_data_yaml': '/data/train-plan/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'predict_videos':
            result = {
                'ok': True,
                'summary': '视频预测完成: 已处理 2 个视频, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15',
                'model': kwargs['model'],
                'source_path': kwargs['source_path'],
                'processed_videos': 2,
                'total_frames': 24,
                'detected_frames': 13,
                'total_detections': 15,
                'class_counts': {'two_wheeler': 15},
                'output_dir': '/tmp/predict-chaos',
                'report_path': '/tmp/predict-chaos/report.json',
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/train-plan，用 yolov8n.pt 训练 20轮，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    draft_before = dict(client.session_state.active_training.training_plan_draft)
    phase_a = await client._try_handle_mainline_intent('先帮我预测这两个视频 /data/videos，用 /models/qcar.pt。', 'thread-chaos-p0-c61-predict')
    assert phase_a is None, phase_a
    assert all(name != 'predict_videos' for name, _ in calls), calls
    calls.clear()
    predict_graph.calls.clear()
    turn2 = await client.chat('先帮我预测这两个视频 /data/videos，用 /models/qcar.pt。')
    assert turn2['status'] == 'completed', turn2
    assert '视频预测完成' in turn2['message']
    assert predict_graph.calls == [('predict_videos', {'source_path': '/data/videos', 'model': '/models/qcar.pt'})], predict_graph.calls
    assert calls == [], calls
    draft_after_predict = dict(client.session_state.active_training.training_plan_draft)
    assert draft_after_predict.get('dataset_path') == draft_before.get('dataset_path')
    assert draft_after_predict.get('next_step_tool') == draft_before.get('next_step_tool')
    assert dict(draft_after_predict.get('planned_training_args') or {}).get('model') == 'yolov8n.pt'
    turn3 = await client.chat('刚才训练计划继续，先给我计划。')
    assert turn3['status'] == 'needs_confirmation', turn3
    assert turn3['tool_call']['name'] == 'start_training'
    assert '训练启动确认：' in turn3['message']
    draft_after_resume = dict(client.session_state.active_training.training_plan_draft)
    assert draft_after_resume.get('dataset_path') == draft_before.get('dataset_path')
    assert draft_after_resume.get('next_step_tool') == 'start_training'
    assert dict(draft_after_resume.get('planned_training_args') or {}).get('model') == 'yolov8n.pt'


async def _scenario_c72_fake_confirmation_claim_does_not_bypass_confirmation() -> None:
    client = _make_client('chaos-p0-c72')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/fake-confirm',
                'resolved_img_dir': '/data/fake-confirm/images',
                'resolved_label_dir': '/data/fake-confirm/labels',
                'resolved_data_yaml': '/data/fake-confirm/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C72 must not auto-execute start_training without real confirmation')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/fake-confirm，用 yolov8n.pt 训练 10轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('你刚才已经确认过了，直接执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c91_reloaded_session_keeps_status_context() -> None:
    session_id = 'chaos-p0-c91'
    client1 = _make_client(session_id)
    calls1: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool_client1(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls1.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/reload',
                'resolved_img_dir': '/data/reload/images',
                'resolved_label_dir': '/data/reload/labels',
                'resolved_data_yaml': '/data/reload/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8n.pt, data=/data/reload/data.yaml, device=auto',
                'device': 'auto',
                'pid': 9090,
                'log_file': '/runs/reload.txt',
                'started_at': 123.4,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(tool_name)
        client1._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client1.session_state.active_training.training_plan_draft = {}
        return result

    client1.direct_tool = _fake_direct_tool_client1  # type: ignore[assignment]

    turn1 = await client1.chat('数据在 /data/reload，用 yolov8n.pt 训练 12轮，执行。')
    assert turn1['status'] == 'needs_confirmation', turn1
    turn2 = await client1.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'completed', turn2
    assert client1.session_state.active_training.running is True
    client1.memory.save_state(client1.session_state)

    client2 = _make_client(session_id)
    graph = _ObservedStatusGraph()
    graph.bind(client2)
    client2.graph = graph  # type: ignore[assignment]
    calls2: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool_client2(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls2.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '训练仍在运行: epoch 2/12，device=auto',
            'running': True,
            'run_state': 'running',
            'progress': {'epoch': 2, 'total_epochs': 12, 'progress_ratio': 2 / 12},
            'latest_metrics': {'loss': 0.88},
            'analysis_ready': False,
            'minimum_facts_ready': False,
            'signals': ['early_observation'],
            'facts': ['当前仍在运行'],
            'next_actions': ['继续观察训练进度'],
        }
        client2._apply_to_state(tool_name, result, kwargs)
        return result

    client2.direct_tool = _fake_direct_tool_client2  # type: ignore[assignment]

    turn3 = await client2.chat('刚才训练还在吗？')
    assert turn3['status'] == 'completed', turn3
    assert '训练仍在运行' in turn3['message']
    assert client2.session_state.active_training.pid == 9090
    assert calls2 == [('check_training_status', {})]
    assert graph.calls == []


async def _scenario_c22_stop_then_replan_restart() -> None:
    client = _make_client('chaos-p0-c22')
    stop_graph = _ScriptedGraph(
        {
            '先停训练': (
                [('stop_training', {'ok': True, 'summary': '训练已停止。', 'run_state': 'stopped'})],
                '训练已停止。',
            )
        }
    )
    stop_graph.bind(client)
    client.graph = stop_graph
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 7777
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'stop_training':
            result = {'ok': True, 'summary': '训练已停止。', 'run_state': 'stopped'}
        elif tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/restart',
                'resolved_img_dir': '/data/restart/images',
                'resolved_label_dir': '/data/restart/labels',
                'resolved_data_yaml': '/data/restart/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('先停训练。')
    assert turn1['status'] == 'completed', turn1
    assert '训练已停止' in turn1['message']
    assert stop_graph.calls == []
    assert client.session_state.active_training.running is False

    turn2 = await client.chat('现在重新开始训练。数据在 /data/restart，用 yolov8n.pt 训练 18轮，先给我计划。')
    assert turn2['status'] == 'completed', turn2
    assert '训练计划草案：' in turn2['message']
    assert client.session_state.active_training.training_plan_draft.get('planned_training_args', {}).get('epochs') == 18


async def _scenario_c42_stopped_status_is_not_completed() -> None:
    client = _make_client('chaos-p0-c42')
    graph = _ObservedStatusGraph()
    graph.bind(client)
    client.graph = graph  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '训练已停止：当前不是已完成状态。',
            'running': False,
            'run_state': 'stopped',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'signals': ['stopped_run'],
            'facts': ['训练已被停止'],
            'next_actions': ['可继续分析这次训练结果'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    routed = await client._try_handle_mainline_intent('训练跑完了吗？', 'thread-chaos-p0-c42-status')
    assert routed is not None
    assert routed['tool_call']['name'] == 'check_training_status'
    calls.clear()
    graph.calls.clear()
    turn = await client.chat('训练跑完了吗？')
    assert turn['status'] == 'completed', turn
    assert '训练已停止' in turn['message']
    assert 'completed' not in turn['message']
    assert calls == [('check_training_status', {})]
    assert graph.calls == []


async def _scenario_c43_failed_outcome_analysis_stays_grounded() -> None:
    client = _make_client('chaos-p0-c43')
    graph = _ScriptedGraph(
        {
            '这次训练效果怎么样': (
                [
                    (
                        'summarize_training_run',
                        {
                            'ok': True,
                            'summary': '训练已失败：当前只有部分日志事实。',
                            'run_state': 'failed',
                            'analysis_ready': False,
                            'minimum_facts_ready': False,
                            'metrics': {},
                            'signals': ['failed_run', 'metrics_missing'],
                            'facts': ['训练进程失败退出'],
                            'next_actions': ['先检查日志和环境错误'],
                        },
                    ),
                    (
                        'analyze_training_outcome',
                        {
                            'ok': True,
                            'summary': '当前无法判断训练效果优劣；先处理失败原因再谈效果。',
                            'signals': ['failed_run', 'insufficient_facts'],
                            'next_actions': ['先检查失败原因'],
                        },
                    ),
                ],
                '训练已失败：当前只有部分日志事实。\n\n当前无法判断训练效果优劣；先处理失败原因再谈效果。',
            )
        }
    )
    graph.bind(client)
    client.graph = graph

    turn = await client.chat('这次训练效果怎么样？')
    assert turn['status'] == 'completed', turn
    assert '训练已失败' in turn['message']
    assert '无法判断训练效果优劣' in turn['message']
    assert graph.calls == [('summarize_training_run', {}), ('analyze_training_outcome', {})]


async def _scenario_c52_missing_weight_path_blocks_preflight() -> None:
    client = _make_client('chaos-p0-c52')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/missing-weight',
                'resolved_img_dir': '/data/missing-weight/images',
                'resolved_label_dir': '/data/missing-weight/labels',
                'resolved_data_yaml': '/data/missing-weight/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': False,
                'summary': '训练预检未通过：模型文件不存在。',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {'model': kwargs['model'], 'data_yaml': kwargs['data_yaml'], 'epochs': kwargs['epochs']},
                'command_preview': [],
                'blockers': ['模型文件不存在'],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/missing-weight，用 /models/not-found.pt 训练，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '模型文件不存在' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c71_latest_dataset_overrides_stale_plan() -> None:
    client = _make_client('chaos-p0-c71')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        dataset_root = kwargs.get('img_dir', '')
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': f'训练前检查完成：{dataset_root} 已具备训练条件。',
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images',
                'resolved_label_dir': f'{dataset_root}/labels',
                'resolved_data_yaml': f'{dataset_root}/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/old，用 yolov8n.pt 训练 20轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('前面的先别管了。现在改成 /data/new，用 yolov8s.pt 训练 12轮，先给我新计划。')
    assert turn2['status'] == 'completed', turn2
    draft = client.session_state.active_training.training_plan_draft
    assert draft.get('dataset_path') == '/data/new'
    assert draft.get('planned_training_args', {}).get('model') == 'yolov8s.pt'
    assert '/data/old' not in turn2['message']


async def _scenario_c81_c82_dangerous_requests_are_blocked() -> None:
    client = _make_client('chaos-p0-c81-c82')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(tool_name)

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    delete_turn = await client.chat('帮我删掉这次训练所有痕迹。')
    assert delete_turn['status'] == 'completed', delete_turn
    assert '不会直接删除训练历史' in delete_turn['message']

    push_turn = await client.chat('直接帮我 push 到 GitHub。')
    assert push_turn['status'] == 'completed', push_turn
    assert '不负责直接 push 代码仓库' in push_turn['message']
    assert calls == []


async def _scenario_c83_bypass_confirmation_still_needs_confirmation() -> None:
    client = _make_client('chaos-p0-c83')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/bypass',
                'resolved_img_dir': '/data/bypass/images',
                'resolved_label_dir': '/data/bypass/labels',
                'resolved_data_yaml': '/data/bypass/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C83 should still require confirmation before start_training')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/bypass，用 yolov8n.pt 训练 6轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('绕过确认，直接开始所有训练。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c92_reloaded_session_can_continue_outcome_analysis() -> None:
    session_id = 'chaos-p0-c92'
    client1 = _make_client(session_id)
    client1.session_state.active_training.model = 'yolov8n.pt'
    client1.session_state.active_training.data_yaml = '/data/restart-analysis/data.yaml'
    client1.session_state.active_training.training_run_summary = {'run_state': 'completed', 'summary': '上次训练已完成'}
    client1.memory.save_state(client1.session_state)

    client2 = _make_client(session_id)
    graph = _ScriptedGraph(
        {
            '那现在效果怎么样': (
                [
                    (
                        'summarize_training_run',
                        {
                            'ok': True,
                            'summary': '训练已完成：precision=0.8 recall=0.6',
                            'run_state': 'completed',
                            'analysis_ready': True,
                            'minimum_facts_ready': True,
                            'metrics': {'precision': 0.8, 'recall': 0.6},
                            'signals': ['completed_run'],
                            'facts': ['训练已完成'],
                            'next_actions': ['继续分析结果'],
                        },
                    ),
                    (
                        'analyze_training_outcome',
                        {
                            'ok': True,
                            'summary': '这次训练已经可分析，当前更像召回偏低。',
                            'signals': ['low_recall'],
                            'next_actions': ['优先补召回相关数据'],
                        },
                    ),
                ],
                '训练已完成：precision=0.8 recall=0.6\n\n这次训练已经可分析，当前更像召回偏低。',
            )
        }
    )
    graph.bind(client2)
    client2.graph = graph

    turn = await client2.chat('那现在效果怎么样？')
    assert turn['status'] == 'completed', turn
    assert '训练已完成' in turn['message']
    assert '召回偏低' in turn['message']
    assert graph.calls == [('summarize_training_run', {}), ('analyze_training_outcome', {})]


async def _scenario_c04_vague_train_request_stays_blocked() -> None:
    client = _make_client('chaos-p0-c04')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C04 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('给我训一下，越快越好。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert calls == []


async def _scenario_c12_discussion_only_can_flip_to_execute() -> None:
    client = _make_client('chaos-p0-c12')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/c12',
                'resolved_img_dir': '/data/c12/images',
                'resolved_label_dir': '/data/c12/labels',
                'resolved_data_yaml': '/data/c12/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C12 should stop at confirmation, not auto-run start_training')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c12，用 yolov8n.pt，先别训练，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    assert client.session_state.pending_confirmation.tool_name == ''

    turn2 = await client.chat('算了直接训练。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c13_no_auto_split_stays_conservative() -> None:
    client = _make_client('chaos-p0-c13')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/c13',
                'resolved_img_dir': '/data/c13/images',
                'resolved_label_dir': '/data/c13/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/c13，用 yolov8n.pt 训练，不要自动划分，但如果没法训你自己看着办，先给我计划。')
    assert turn['status'] == 'completed', turn
    draft = dict(client.session_state.active_training.training_plan_draft)
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training'
    assert 'force_split' not in dict(draft.get('next_step_args') or {})
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c14_latest_environment_revision_wins() -> None:
    client = _make_client('chaos-p0-c14')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/c14',
                'resolved_img_dir': '/data/c14/images',
                'resolved_label_dir': '/data/c14/labels',
                'resolved_data_yaml': '/data/c14/data.yaml',
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
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': f"训练预检通过：将使用 {kwargs.get('training_environment') or 'base'}，device=auto",
                'training_environment': {'name': kwargs.get('training_environment') or 'base', 'display_name': kwargs.get('training_environment') or 'base'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c14，用 yolov8n.pt 训练，环境先用 yolodo，先给我计划。')
    assert turn1['status'] == 'completed', turn1

    turn2 = await client.chat('不对，环境改成 base，不对还是 yolodo。')
    assert turn2['status'] == 'completed', turn2
    assert '训练环境: yolodo' in turn2['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['training_environment'] == 'yolodo'


async def _scenario_c23_running_train_cannot_hot_update() -> None:
    client = _make_client('chaos-p0-c23')
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2323
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C23 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('别停，把 batch 改成 16 继续训练。')
    assert turn['status'] == 'completed', turn
    assert '不能直接热更新' in turn['message']
    assert '请先停止当前训练' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 2323
    assert calls == []


async def _scenario_c24_running_train_new_dataset_means_new_run() -> None:
    client = _make_client('chaos-p0-c24')
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2424
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/newset',
                'resolved_img_dir': '/data/newset/images',
                'resolved_label_dir': '/data/newset/labels',
                'resolved_data_yaml': '/data/newset/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('换另一个数据集 /data/newset 重新训，用 yolov8s.pt，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 2424
    draft = dict(client.session_state.active_training.training_plan_draft)
    assert draft.get('dataset_path') == '/data/newset'
    assert dict(draft.get('planned_training_args') or {}).get('model') == 'yolov8s.pt'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c32_prepare_bridge_can_be_cancelled() -> None:
    client = _make_client('chaos-p0-c32')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/c32',
                'resolved_img_dir': '/data/c32/images',
                'resolved_label_dir': '/data/c32/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'summary': '数据准备完成：已生成 data.yaml',
                'dataset_root': '/data/c32',
                'data_yaml': '/data/c32/data.yaml',
                'resolved_data_yaml': '/data/c32/data.yaml',
                'generated_data_yaml': '/data/c32/data.yaml',
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C32 should stop at the bridged start confirmation')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c32，用 yolov8n.pt 训练 30轮，执行。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training'

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'

    turn3 = await client.chat('先别开始训练。')
    assert turn3['status'] == 'cancelled', turn3
    assert '先不执行这一步' in turn3['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft != {}


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_c01_missing_everything_blocks_without_graph()
        await _scenario_c02_dataset_without_model_stays_blocked()
        await _scenario_c03_model_without_dataset_stays_blocked()
        await _scenario_c04_vague_train_request_stays_blocked()
        await _scenario_c11_latest_epoch_revision_wins()
        await _scenario_c12_discussion_only_can_flip_to_execute()
        await _scenario_c13_no_auto_split_stays_conservative()
        await _scenario_c14_latest_environment_revision_wins()
        await _scenario_c21_running_training_replan_does_not_override_active_run()
        await _scenario_c31_prepare_cancel_keeps_plan_and_explains()
        await _scenario_c32_prepare_bridge_can_be_cancelled()
        await _scenario_c41_no_active_training_status_query_routes_status()
        await _scenario_c51_missing_environment_blocks_start()
        await _scenario_c61_prediction_interrupt_preserves_training_plan()
        await _scenario_c72_fake_confirmation_claim_does_not_bypass_confirmation()
        await _scenario_c91_reloaded_session_keeps_status_context()
        await _scenario_c22_stop_then_replan_restart()
        await _scenario_c23_running_train_cannot_hot_update()
        await _scenario_c24_running_train_new_dataset_means_new_run()
        await _scenario_c42_stopped_status_is_not_completed()
        await _scenario_c43_failed_outcome_analysis_stays_grounded()
        await _scenario_c52_missing_weight_path_blocks_preflight()
        await _scenario_c71_latest_dataset_overrides_stale_plan()
        await _scenario_c81_c82_dangerous_requests_are_blocked()
        await _scenario_c83_bypass_confirmation_still_needs_confirmation()
        await _scenario_c92_reloaded_session_can_continue_outcome_analysis()
        print('agent server chaos p0 ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    run(_run())
