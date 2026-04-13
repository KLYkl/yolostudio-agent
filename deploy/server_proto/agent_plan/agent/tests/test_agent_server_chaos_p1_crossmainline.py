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


def _install_crossmainline_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []
    stop_count = {'value': 0}

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images' if dataset_root else '',
                'resolved_label_dir': f'{dataset_root}/labels' if dataset_root else '',
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
        elif tool_name == 'predict_videos':
            result = {
                'ok': True,
                'summary': '视频预测完成',
                'source_path': kwargs.get('source_path'),
                'output_dir': '/tmp/p1-preds',
                'report_path': '/tmp/p1-preds/report.json',
                'processed_videos': 1,
                'total_frames': 12,
                'detected_frames': 6,
                'total_detections': 8,
                'class_counts': {'two_wheeler': 8},
            }
        elif tool_name == 'stop_training':
            stop_count['value'] += 1
            if stop_count['value'] == 1:
                result = {
                    'ok': True,
                    'summary': '训练已停止：已终止当前训练进程。',
                    'run_state': 'stopped',
                    'signals': ['stopped_run'],
                    'facts': ['训练已停止'],
                    'next_actions': ['可以继续分析当前结果或重新规划下一次训练。'],
                }
            else:
                result = {
                    'ok': False,
                    'summary': '当前没有活动训练进程可停止。',
                    'error': '当前没有活动训练进程可停止。',
                }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c61_training_plan_survives_prediction() -> None:
    client = _fresh_client('chaos-p1-c61')
    calls = _install_crossmainline_tools(client)

    first = await client.chat('用 /data/c61 这个数据集和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first
    draft = client.session_state.active_training.training_plan_draft
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training'
    assert ((draft.get('planned_training_args') or {}).get('model')) == 'yolov8n.pt'

    second = await client.chat('先帮我预测 /videos/c61.mp4。')
    assert second['status'] == 'completed', second
    assert '视频预测完成' in second['message']
    assert calls[-1][0] == 'predict_videos'

    draft = client.session_state.active_training.training_plan_draft
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training'
    assert ((draft.get('planned_training_args') or {}).get('model')) == 'yolov8n.pt'


async def _scenario_c62_prediction_done_then_resume_training_plan() -> None:
    client = _fresh_client('chaos-p1-c62')
    calls = _install_crossmainline_tools(client)

    first = await client.chat('用 /data/c62 这个数据集和 yolov8n.pt 训练，先给我计划。')
    assert first['status'] == 'completed', first
    second = await client.chat('先帮我预测 /videos/c62.mp4。')
    assert second['status'] == 'completed', second
    call_count = len(calls)

    third = await client.chat('刚才训练计划继续。')
    assert third['status'] == 'completed', third
    assert '训练计划草案：' in third['message']
    assert '- 数据集: /data/c62' in third['message']
    assert 'model=yolov8n.pt' in third['message']
    assert len(calls) == call_count, calls


async def _scenario_c63_train_from_recent_frames_followup() -> None:
    client = _fresh_client('chaos-p1-c63')
    calls = _install_crossmainline_tools(client)
    client.session_state.active_dataset.last_frame_extract = {
        'source_path': '/videos/raw.mp4',
        'output_dir': '/tmp/p1-frames',
        'final_count': 20,
        'summary': '抽帧完成：已输出 20 帧。',
    }
    client.session_state.active_training.model = 'yolov8n.pt'
    client.memory.save_state(client.session_state)

    turn = await client.chat('就用这些帧开始训练。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training'
    assert turn['tool_call']['args']['dataset_path'] == '/tmp/p1-frames'
    assert '- 数据集: /tmp/p1-frames' in turn['message']
    assert 'model=yolov8n.pt' in turn['message']
    assert calls[0] == ('training_readiness', {'img_dir': '/tmp/p1-frames'})


async def _scenario_c93_duplicate_stop_keeps_state_clean() -> None:
    client = _fresh_client('chaos-p1-c93')
    _install_crossmainline_tools(client)
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2222
    client.memory.save_state(client.session_state)

    first = await client.chat('停止训练。')
    assert first['status'] == 'completed', first
    assert '训练已停止' in first['message']
    assert client.session_state.active_training.running is False

    second = await client.chat('再停一次。')
    assert second['status'] == 'error', second
    assert '当前没有活动训练进程可停止' in second['message']
    assert client.session_state.active_training.running is False
    assert client.session_state.active_training.pid is None


async def _run() -> None:
    await _scenario_c61_training_plan_survives_prediction()
    await _scenario_c62_prediction_done_then_resume_training_plan()
    await _scenario_c63_train_from_recent_frames_followup()
    await _scenario_c93_duplicate_stop_keeps_state_clean()
    print('agent server chaos p1 crossmainline ok')


if __name__ == '__main__':
    run(_run())
