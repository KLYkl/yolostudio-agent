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


def _install_repeat_tools(client, *, ready: bool):
    calls: list[tuple[str, dict[str, Any]]] = []
    start_count = {'value': 0}
    prepare_count = {'value': 0}

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            if ready:
                result = {
                    'ok': True,
                    'summary': '数据已具备训练条件。',
                    'dataset_root': '/data/repeat',
                    'resolved_img_dir': '/data/repeat/images',
                    'resolved_label_dir': '/data/repeat/labels',
                    'resolved_data_yaml': '/data/repeat/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
            else:
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': '/data/repeat',
                    'resolved_img_dir': '/data/repeat/images',
                    'resolved_label_dir': '/data/repeat/labels',
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
            start_count['value'] += 1
            result = {
                'ok': True,
                'summary': '训练已启动',
                'resolved_args': dict(kwargs),
                'device': 'auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'pid': 4455,
                'log_file': '/tmp/repeat-train.log',
                'started_at': 'now',
            }
            client._apply_to_state(tool_name, result, kwargs)
            client._clear_training_plan_draft()
            return result
        elif tool_name == 'prepare_dataset_for_training':
            prepare_count['value'] += 1
            result = {
                'ok': True,
                'summary': '数据准备完成：当前数据集已具备训练条件。',
                'dataset_root': '/data/repeat-prepared',
                'img_dir': '/data/repeat-prepared/images',
                'label_dir': '/data/repeat-prepared/labels',
                'data_yaml': '/data/repeat-prepared/data.yaml',
                'ready': True,
                'steps_completed': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls, start_count, prepare_count


async def _scenario_c94_duplicate_execute_does_not_start_twice() -> None:
    client = _fresh_client('chaos-p1-c94')
    calls, start_count, _ = _install_repeat_tools(client, ready=True)

    first = await client.chat('用 /data/repeat 和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first

    second = await client.chat('执行。')
    assert second['status'] == 'needs_confirmation', second
    assert second['tool_call']['name'] == 'start_training'

    third = await client.confirm(second['thread_id'], approved=True)
    assert third['status'] == 'completed', third
    assert '训练已启动' in third['message']
    assert start_count['value'] == 1

    fourth = await client.chat('执行。')
    assert fourth['status'] == 'completed', fourth
    assert '当前训练已经在运行' in fourth['message']
    assert start_count['value'] == 1
    assert [name for name, _ in calls].count('start_training') == 1


async def _scenario_c95_repeat_prepare_after_prepare_finished_is_blocked() -> None:
    client = _fresh_client('chaos-p1-c95')
    calls, _, prepare_count = _install_repeat_tools(client, ready=False)

    first = await client.chat('用 /data/repeat 和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first

    second = await client.chat('执行。')
    assert second['status'] == 'needs_confirmation', second
    assert second['tool_call']['name'] == 'prepare_dataset_for_training'

    third = await client.confirm(second['thread_id'], approved=True)
    assert third['status'] == 'needs_confirmation', third
    assert third['tool_call']['name'] == 'start_training'
    assert prepare_count['value'] == 1
    call_count = len(calls)

    fourth = await client.chat('再 prepare 一次。')
    assert fourth['status'] == 'needs_confirmation', fourth
    assert fourth['tool_call']['name'] == 'start_training'
    assert prepare_count['value'] == 1
    assert len(calls) == call_count
    assert client.session_state.pending_confirmation.tool_name == 'start_training'


async def _run() -> None:
    await _scenario_c94_duplicate_execute_does_not_start_twice()
    await _scenario_c95_repeat_prepare_after_prepare_finished_is_blocked()
    print('agent server chaos p1 repeat tolerance ok')


if __name__ == '__main__':
    run(_run())
