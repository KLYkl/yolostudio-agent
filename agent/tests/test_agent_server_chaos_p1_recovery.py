from __future__ import annotations

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

from yolostudio_agent.agent.tests._chaos_test_support import WORK as P0_WORK, _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run


def _fresh_client(session_id: str):
    shutil.rmtree(P0_WORK / session_id, ignore_errors=True)
    return _make_client(session_id)


def _install_recovery_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_preflight':
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
        elif tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            result = {
                'ok': True,
                'summary': '数据已具备训练条件。',
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images' if dataset_root else '',
                'resolved_label_dir': f'{dataset_root}/labels' if dataset_root else '',
                'resolved_data_yaml': '/data/recovery/data.yaml',
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
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


def _seed_recent_run(client, *, run_state: str) -> None:
    client.session_state.active_dataset.dataset_root = '/data/recovery'
    client.session_state.active_dataset.img_dir = '/data/recovery/images'
    client.session_state.active_dataset.label_dir = '/data/recovery/labels'
    client.session_state.active_dataset.data_yaml = '/data/recovery/data.yaml'
    client.session_state.active_dataset.last_readiness = {
        'ready': True,
        'preparable': False,
        'resolved_data_yaml': '/data/recovery/data.yaml',
        'summary': '数据已具备训练条件。',
    }
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/recovery/data.yaml'
    client.session_state.active_training.training_environment = 'yolodo'
    client.session_state.active_training.last_environment_probe = {
        'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
        'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
    }
    client.session_state.active_training.last_start_result = {
        'resolved_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/recovery/data.yaml',
            'epochs': 60,
            'device': 'auto',
            'training_environment': 'yolodo',
            'project': 'recovery_project',
            'name': 'run_recovery',
            'batch': 16,
        }
    }
    client.session_state.active_training.last_status = {
        'run_state': run_state,
        'summary': f'最近一次训练状态：{run_state}',
    }
    client.session_state.active_training.training_run_summary = {
        'run_state': run_state,
        'summary': f'最近一次训练状态：{run_state}',
    }
    client.memory.save_state(client.session_state)



async def _scenario_c97_resume_cancel_resume_stays_stable() -> None:
    client = _fresh_client('chaos-p1-c97')
    calls = _install_recovery_tools(client)
    _seed_recent_run(client, run_state='stopped')

    first = await client.chat('从最近状态继续训练。')
    assert first['status'] == 'needs_confirmation', first
    assert first['tool_call']['name'] == 'start_training'
    assert first['tool_call']['args']['resume'] is True

    second = await client.chat('取消。')
    assert second['status'] == 'cancelled', second
    assert (client.get_pending_action() or {}).get('tool_name', '') == ''

    call_count = len(calls)
    third = await client.chat('从最近状态继续训练。')
    assert third['status'] == 'needs_confirmation', third
    assert third['tool_call']['name'] == 'start_training'
    assert third['tool_call']['args']['resume'] is True
    assert len(calls) == call_count + 1
    assert calls[-1][0] == 'training_preflight'


async def _scenario_c98_retry_failed_run_rebuilds_plan() -> None:
    client = _fresh_client('chaos-p1-c98')
    calls = _install_recovery_tools(client)
    _seed_recent_run(client, run_state='failed')

    turn = await client.chat('按原计划重试一次。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'start_training'
    assert turn['tool_call']['args']['model'] == 'yolov8n.pt'
    assert turn['tool_call']['args']['data_yaml'] == '/data/recovery/data.yaml'
    assert turn['tool_call']['args']['resume'] is False
    assert 'resume=False' in turn['message']
    assert calls[0][0] == 'training_preflight'


async def _scenario_c99_resume_stopped_run_uses_resume_true() -> None:
    client = _fresh_client('chaos-p1-c99')
    calls = _install_recovery_tools(client)
    _seed_recent_run(client, run_state='stopped')

    turn = await client.chat('从最近状态继续训练。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'start_training'
    assert turn['tool_call']['args']['resume'] is True
    assert 'resume=True' in turn['message']
    assert calls[0][0] == 'training_preflight'


async def _scenario_completed_run_cannot_resume_recent_state() -> None:
    client = _fresh_client('chaos-p1-completed-resume')
    calls = _install_recovery_tools(client)
    _seed_recent_run(client, run_state='completed')

    turn = await client.chat('从最近状态继续训练。')
    assert turn['status'] == 'completed', turn
    assert '只有已停止的训练才适合按最近状态继续' in turn['message']
    assert calls == []


async def _run() -> None:
    await _scenario_c97_resume_cancel_resume_stays_stable()
    await _scenario_c98_retry_failed_run_rebuilds_plan()
    await _scenario_c99_resume_stopped_run_uses_resume_true()
    await _scenario_completed_run_cannot_resume_recent_state()
    print('agent server chaos p1 recovery ok')


if __name__ == '__main__':
    run(_run())
