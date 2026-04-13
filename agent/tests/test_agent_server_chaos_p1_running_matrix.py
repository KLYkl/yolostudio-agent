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


def _install_running_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'check_training_status':
            result = {
                'ok': True,
                'summary': '训练仍在运行：epoch=3/20',
                'run_state': 'running',
                'progress': {'epoch': 3, 'total_epochs': 20},
                'analysis_ready': False,
                'minimum_facts_ready': False,
                'signals': ['running'],
                'facts': ['训练正在运行'],
                'next_actions': ['继续观察'],
            }
        elif tool_name == 'predict_videos':
            result = {
                'ok': True,
                'summary': '视频预测完成。',
                'output_dir': '/tmp/predict-out',
                'report_path': '/tmp/predict-out/report.json',
            }
        elif tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
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
    return calls


async def _scenario_c25_running_train_cannot_resume_other_run() -> None:
    client = _fresh_client('chaos-p1-c25')
    _install_running_tools(client)
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 1111
    client.memory.save_state(client.session_state)
    turn = await client.chat('这次效果不行，直接 resume 另一个 run train_log_999。')
    assert turn['status'] == 'completed', turn
    assert '当前已有活动训练' in turn['message']


async def _scenario_c26_repeated_status_queries_stay_on_status() -> None:
    client = _fresh_client('chaos-p1-c26')
    calls = _install_running_tools(client)
    client.session_state.active_training.running = True
    client.memory.save_state(client.session_state)
    for query in ('现在第几轮了？', '现在第几轮了？', '现在第几轮了？'):
        turn = await client.chat(query)
        assert turn['status'] == 'completed', turn
        assert '训练仍在运行' in turn['message']
    assert [name for name, _ in calls].count('check_training_status') == 3


async def _scenario_c27_cannot_erase_running_history() -> None:
    client = _fresh_client('chaos-p1-c27')
    turn = await client.chat('我后悔了，把这次当没发生。')
    assert turn['status'] == 'completed', turn
    assert '不能把已经发生的训练当成没发生' in turn['message']


async def _scenario_c28_running_training_can_temporarily_switch_to_prediction() -> None:
    client = _fresh_client('chaos-p1-c28')
    calls = _install_running_tools(client)
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2222
    client.session_state.active_training.model = 'yolov8n.pt'
    client.memory.save_state(client.session_state)
    turn = await client.chat('训练先放着，先帮我预测这两个视频 /data/videos。')
    assert turn['status'] == 'completed', turn
    assert '视频预测完成' in turn['message']
    assert client.session_state.active_training.running is True
    assert [name for name, _ in calls][-1] == 'predict_videos'


async def _scenario_c29_running_training_cannot_switch_gpu_live() -> None:
    client = _fresh_client('chaos-p1-c29')
    _install_running_tools(client)
    client.session_state.active_training.running = True
    client.memory.save_state(client.session_state)
    turn = await client.chat('训练别停，把 GPU 换成 1。')
    assert turn['status'] == 'completed', turn
    assert '不能直接热更新' in turn['message']


async def _scenario_c30_second_experiment_becomes_new_plan_without_overriding_run() -> None:
    client = _fresh_client('chaos-p1-c30')
    calls = _install_running_tools(client)
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 3333
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/running/data.yaml'
    client.memory.save_state(client.session_state)
    turn = await client.chat('开始第二个不同模型实验，用 /data/exp2 和 yolov8s.pt 训练，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案' in turn['message']
    assert 'yolov8s.pt' in turn['message']
    assert client.session_state.active_training.running is True
    assert any(name == 'training_readiness' for name, _ in calls)


async def _run() -> None:
    await _scenario_c25_running_train_cannot_resume_other_run()
    await _scenario_c26_repeated_status_queries_stay_on_status()
    await _scenario_c27_cannot_erase_running_history()
    await _scenario_c28_running_training_can_temporarily_switch_to_prediction()
    await _scenario_c29_running_training_cannot_switch_gpu_live()
    await _scenario_c30_second_experiment_becomes_new_plan_without_overriding_run()
    print('agent server chaos p1 running matrix ok')


if __name__ == '__main__':
    run(_run())