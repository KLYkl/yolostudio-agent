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


def _install_context_tools(client, readiness_map: dict[str, dict[str, Any]]):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            config = dict(readiness_map[dataset_root])
            ready = bool(config.get('ready'))
            data_yaml = str(config.get('data_yaml') or '')
            summary = str(config.get('summary') or ('训练前检查完成：数据已具备训练条件。' if ready else '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training'))
            result = {
                'ok': True,
                'summary': summary,
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images' if dataset_root else '',
                'resolved_label_dir': f'{dataset_root}/labels' if dataset_root else '',
                'resolved_data_yaml': data_yaml,
                'ready': ready,
                'preparable': bool(config.get('preparable')),
                'primary_blocker_type': str(config.get('primary_blocker_type') or ''),
                'warnings': list(config.get('warnings') or []),
                'blockers': list(config.get('blockers') or []),
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
    return calls


async def _scenario_c71_latest_dataset_wins_after_long_replanning() -> None:
    client = _fresh_client('chaos-p2-c71')
    calls = _install_context_tools(
        client,
        {
            '/data/a': {'ready': True, 'data_yaml': '/data/a/data.yaml'},
            '/data/b': {'ready': True, 'data_yaml': '/data/b/data.yaml'},
        },
    )

    first = await client.chat('用 /data/a 和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first
    await client.chat('batch 改 8。')
    await client.chat('project 改成 /outputs/alpha。')
    switched = await client.chat('把数据换成 /data/b。')
    assert switched['status'] == 'completed', switched
    assert '- 数据集: /data/b' in switched['message']
    assert '/data/a/data.yaml' not in switched['message']

    await client.chat('name 改成 beta。')
    final = await client.chat('就按最新版本执行。')
    assert final['status'] == 'needs_confirmation', final
    assert final['tool_call']['name'] == 'start_training'
    args = final['tool_call']['args']
    assert args['model'] == 'yolov8n.pt'
    assert args['data_yaml'] == '/data/b/data.yaml'
    assert args['batch'] == 8
    assert args['project'] == '/outputs/alpha'
    assert args['name'] == 'beta'
    assert '- 数据集: /data/b' in final['message']
    assert '/data/a/data.yaml' not in final['message']
    assert ('training_readiness', {'img_dir': '/data/b'}) in calls


async def _scenario_c79_old_yaml_cannot_skip_new_dataset_prepare() -> None:
    client = _fresh_client('chaos-p2-c79')
    _install_context_tools(
        client,
        {
            '/data/a': {'ready': True, 'data_yaml': '/data/a/data.yaml'},
            '/data/b': {
                'ready': False,
                'data_yaml': '',
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'blockers': ['缺少可用的 data_yaml'],
            },
        },
    )

    first = await client.chat('用 /data/a 和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first

    second = await client.chat('把数据换成 /data/b。')
    assert second['status'] == 'completed', second
    assert '- 数据集: /data/b' in second['message']
    assert '缺少可用的 data_yaml' in second['message']

    third = await client.chat('前面那个 data.yaml 肯定还是对的，直接执行。')
    assert third['status'] == 'needs_confirmation', third
    assert third['tool_call']['name'] == 'prepare_dataset_for_training'
    assert third['tool_call']['args']['dataset_path'] == '/data/b'
    assert '/data/a/data.yaml' not in third['message']
    assert '缺少可用的 data_yaml' in third['message']


async def _scenario_c80_first_plan_query_does_not_restore_stale_draft() -> None:
    client = _fresh_client('chaos-p2-c80')
    _install_context_tools(
        client,
        {
            '/data/c80': {'ready': True, 'data_yaml': '/data/c80/data.yaml'},
        },
    )

    await client.chat('用 /data/c80 和 yolov8n.pt 训练，先给我计划。')
    await client.chat('batch 改 16。')
    await client.chat('模型改成 yolov8s.pt。')

    turn = await client.chat('最开始那套呢？')
    assert turn['status'] == 'completed', turn
    assert '当前只保留最新训练计划草案' in turn['message']
    assert 'model=yolov8s.pt' in turn['message']
    assert 'batch=16' in turn['message']


async def _scenario_c100_execute_latest_version_after_many_revisions() -> None:
    client = _fresh_client('chaos-p2-c100')
    _install_context_tools(
        client,
        {
            '/data/c100': {'ready': True, 'data_yaml': '/data/c100/data.yaml'},
        },
    )

    await client.chat('用 /data/c100 和 yolov8n.pt 训练，先给我计划。')
    await client.chat('batch 改 8。')
    await client.chat('imgsz 改 960。')
    await client.chat('project 改成 /outputs/finalproj。')
    await client.chat('name 改成 runv1。')
    await client.chat('fraction 改 0.5。')
    await client.chat('类别改成 1,2。')
    await client.chat('取消类别限制。')
    await client.chat('amp 关闭。')
    await client.chat('name 改成 run-final。')
    await client.chat('project 改成 /outputs/finalproj2。')

    final = await client.chat('就按最新版本执行。')
    assert final['status'] == 'needs_confirmation', final
    assert final['tool_call']['name'] == 'start_training'
    args = final['tool_call']['args']
    assert args['model'] == 'yolov8n.pt'
    assert args['data_yaml'] == '/data/c100/data.yaml'
    assert args['batch'] == 8
    assert args['imgsz'] == 960
    assert args['project'] == '/outputs/finalproj2'
    assert args['name'] == 'run-final'
    assert args['fraction'] == 0.5
    assert args.get('classes') in ([], None)
    assert args['amp'] is False
    assert 'project=/outputs/finalproj2' in final['message']
    assert 'name=run-final' in final['message']
    assert 'fraction=0.5' in final['message']
    assert 'amp=False' in final['message']


async def _run() -> None:
    await _scenario_c71_latest_dataset_wins_after_long_replanning()
    await _scenario_c79_old_yaml_cannot_skip_new_dataset_prepare()
    await _scenario_c80_first_plan_query_does_not_restore_stale_draft()
    await _scenario_c100_execute_latest_version_after_many_revisions()
    print('agent server chaos p2 context ok')


if __name__ == '__main__':
    run(_run())
