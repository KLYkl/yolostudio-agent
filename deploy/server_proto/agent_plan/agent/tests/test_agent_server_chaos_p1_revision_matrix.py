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

from yolostudio_agent.agent.tests.test_agent_server_chaos_p0 import WORK as P0_WORK
from yolostudio_agent.agent.tests.test_agent_server_chaos_p0 import _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run


def _fresh_client(session_id: str):
    shutil.rmtree(P0_WORK / session_id, ignore_errors=True)
    return _make_client(session_id)


def _install_ready_training_tools(client, *, dataset_root: str = '/data/revision', yaml_path: str = '/data/revision/data.yaml'):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images',
                'resolved_label_dir': f'{dataset_root}/labels',
                'resolved_data_yaml': yaml_path,
                'ready': True,
                'preparable': False,
                'warnings': [],
                'blockers': [],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                'environments': [
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                ],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': str(kwargs.get('training_environment') or 'yolodo'), 'display_name': str(kwargs.get('training_environment') or 'yolodo')},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'warnings': [],
                'blockers': [],
            }
        elif tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '最近训练已完成：precision=0.76 recall=0.54 mAP50=0.58',
                'run_state': 'completed',
                'analysis_ready': True,
                'minimum_facts_ready': True,
                'signals': ['completed_run'],
                'facts': ['最近训练已完成'],
                'next_actions': ['可以继续分析结果'],
                'metrics': {'precision': 0.76, 'recall': 0.54, 'mAP50': 0.58},
            }
        elif tool_name == 'analyze_training_outcome':
            result = {
                'ok': True,
                'summary': '当前更适合先分析结果，不直接恢复训练。',
                'assessment': 'review_first',
                'signals': ['completed_run'],
                'matched_rule_ids': ['workflow_review_first'],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c15_batch_override_can_restore_default() -> None:
    client = _fresh_client('chaos-p1-c15')
    _install_ready_training_tools(client)
    await client.chat('用 /data/revision 和 yolov8n.pt 训练，batch 8，先给我计划。')
    await client.chat('batch 改 16。')
    final = await client.chat('batch 恢复默认，执行。')
    assert final['status'] == 'needs_confirmation', final
    assert final['tool_call']['name'] == 'start_training'
    assert final['tool_call']['args'].get('batch') is None


async def _scenario_c16_classes_and_single_cls_do_not_cross_contaminate() -> None:
    client = _fresh_client('chaos-p1-c16')
    _install_ready_training_tools(client)
    await client.chat('用 /data/revision 和 yolov8n.pt 训练，只训练类别 1,2，先给我计划。')
    await client.chat('取消类别限制。')
    final = await client.chat('还是单类别训练，执行。')
    assert final['status'] == 'needs_confirmation', final
    args = final['tool_call']['args']
    assert args.get('classes') in ([], None)
    assert args.get('single_cls') is True


async def _scenario_c17_explain_then_follow_latest_intent() -> None:
    client = _fresh_client('chaos-p1-c17')

    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/preparable',
                'resolved_img_dir': '/data/preparable/images',
                'resolved_label_dir': '/data/preparable/labels',
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
                'summary': '数据准备完成：当前数据集已具备训练条件。',
                'dataset_root': '/data/preparable',
                'img_dir': '/data/preparable/images',
                'label_dir': '/data/preparable/labels',
                'data_yaml': '/data/preparable/data.yaml',
                'ready': True,
                'steps_completed': [],
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'warnings': [],
                'blockers': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    first = await client.chat('数据在 /data/preparable，用 yolov8n.pt 训练，只做准备。')
    assert first['status'] == 'needs_confirmation', first
    second = await client.chat('为什么不能直接训练？')
    assert second['status'] == 'needs_confirmation', second
    assert '缺少可用的 data_yaml' in second['message']
    third = await client.chat('那你直接训练。')
    assert third['status'] == 'needs_confirmation', third
    assert third['tool_call']['name'] == 'start_training'
    assert third['tool_call']['args'].get('data_yaml') == '/data/preparable/data.yaml'
    assert [name for name, _ in calls] == ['training_readiness', 'list_training_environments', 'prepare_dataset_for_training', 'training_preflight']


async def _scenario_c18_same_turn_conflict_stays_conservative() -> None:
    client = _fresh_client('chaos-p1-c18')
    turn = await client.chat('不要训练，开始训练。')
    assert turn['status'] == 'completed', turn
    assert '同时出现了“不要训练”和“开始训练”' in turn['message']


async def _scenario_c19_resume_but_analysis_only_does_not_restart() -> None:
    client = _fresh_client('chaos-p1-c19')
    _install_ready_training_tools(client)
    client.session_state.active_training.training_run_summary = {
        'summary': '最近训练已完成：precision=0.76 recall=0.54',
        'run_state': 'completed',
    }
    client.memory.save_state(client.session_state)
    turn = await client.chat('resume 上次训练，但不要接着训，只分析就行。')
    assert turn['status'] == 'completed', turn
    assert '当前更适合先分析结果' in turn['message'] or '最近训练已完成' in turn['message']


async def _scenario_c20_project_and_name_can_be_cleared() -> None:
    client = _fresh_client('chaos-p1-c20')
    _install_ready_training_tools(client)
    await client.chat('用 /data/revision 和 yolov8n.pt 训练，project /runs/x，name exp-x，先给我计划。')
    final = await client.chat('project 不要了，name 不要了，执行。')
    assert final['status'] == 'needs_confirmation', final
    args = final['tool_call']['args']
    assert args.get('project') in ('', None)
    assert args.get('name') in ('', None)


async def _run() -> None:
    await _scenario_c15_batch_override_can_restore_default()
    await _scenario_c16_classes_and_single_cls_do_not_cross_contaminate()
    await _scenario_c17_explain_then_follow_latest_intent()
    await _scenario_c18_same_turn_conflict_stays_conservative()
    await _scenario_c19_resume_but_analysis_only_does_not_restart()
    await _scenario_c20_project_and_name_can_be_cleared()
    print('agent server chaos p1 revision matrix ok')


if __name__ == '__main__':
    run(_run())
