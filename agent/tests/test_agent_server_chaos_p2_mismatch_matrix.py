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


def _install_mismatch_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            if dataset_root == '/data/no-yaml':
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': dataset_root,
                    'resolved_img_dir': f'{dataset_root}/images',
                    'resolved_label_dir': f'{dataset_root}/labels',
                    'resolved_data_yaml': '',
                    'ready': False,
                    'preparable': True,
                    'primary_blocker_type': 'missing_yaml',
                    'warnings': [],
                    'blockers': ['缺少可用的 data_yaml'],
                }
            else:
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
                'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                'environments': [
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                ],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'training_preflight':
            model = str(kwargs.get('model') or '')
            dataset_yaml = str(kwargs.get('data_yaml') or '')
            blockers: list[str] = []
            warnings: list[str] = []
            if 'video' in model:
                warnings.append('当前模型更像视频/时序模型，与图片检测训练任务不完全匹配。')
            if str(kwargs.get('device') or '') == 'cpu':
                warnings.append('当前配置在 CPU 上可能非常慢，请谨慎执行。')
            batch = kwargs.get('batch')
            if batch == 512:
                blockers.append('batch=512 过大，当前环境无法安全启动。')
            classes = kwargs.get('classes') or []
            if classes == [99]:
                blockers.append('classes 超出当前数据集类别范围。')
            fraction = kwargs.get('fraction')
            if fraction is not None and not (0 < float(fraction) <= 1):
                blockers.append('fraction 必须在 (0, 1] 范围内')
            if dataset_yaml == '/data/old/data.yaml' and 'new' in str(kwargs.get('project') or ''):
                blockers.append('data_yaml 与当前数据集上下文不一致。')
            result = {
                'ok': not blockers,
                'ready_to_start': not blockers,
                'summary': '训练预检通过：将使用 yolodo，device=auto' if not blockers else '训练预检未通过。',
                'training_environment': {'name': str(kwargs.get('training_environment') or 'yolodo'), 'display_name': str(kwargs.get('training_environment') or 'yolodo')},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'warnings': warnings,
                'blockers': blockers,
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c53_segmentation_training_stays_outside_mainline() -> None:
    client = _fresh_client('chaos-p2-c53')
    turn = await client.chat('用 /data/detect 和 yolov8n.pt 做分割训练。')
    assert turn['status'] == 'completed', turn
    assert 'YOLO detection' in turn['message']


async def _scenario_c54_force_start_without_yaml_still_goes_prepare() -> None:
    client = _fresh_client('chaos-p2-c54')
    _install_mismatch_tools(client)
    turn = await client.chat('数据在 /data/no-yaml，用 yolov8n.pt 强制 start_training。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training'


async def _scenario_c55_video_model_mismatch_surfaces_warning() -> None:
    client = _fresh_client('chaos-p2-c55')
    calls = _install_mismatch_tools(client)
    turn = await client.chat('用 /data/detect 和 /models/video_model.pt 训练，执行。')
    assert turn['status'] == 'needs_confirmation', turn
    assert any(name == 'training_preflight' for name, _ in calls)
    assert '不完全匹配' in turn['message']


async def _scenario_c56_cpu_large_scale_surfaces_warning() -> None:
    client = _fresh_client('chaos-p2-c56')
    _install_mismatch_tools(client)
    turn = await client.chat('用 /data/detect 和 yolov8n.pt 训练，device cpu，执行。')
    assert turn['status'] == 'needs_confirmation', turn
    assert 'CPU 上可能非常慢' in turn['message']


async def _scenario_c57_huge_batch_gets_blocked() -> None:
    client = _fresh_client('chaos-p2-c57')
    _install_mismatch_tools(client)
    turn = await client.chat('用 /data/detect 和 yolov8n.pt 训练，batch 512，执行。')
    assert turn['status'] == 'completed', turn
    assert 'batch=512 过大' in turn['message']


async def _scenario_c58_classes_out_of_range_gets_blocked() -> None:
    client = _fresh_client('chaos-p2-c58')
    _install_mismatch_tools(client)
    turn = await client.chat('用 /data/detect 和 yolov8n.pt 训练，只训练类别 99，执行。')
    assert turn['status'] == 'completed', turn
    assert 'classes 超出当前数据集类别范围' in turn['message']


async def _scenario_c59_invalid_fraction_is_parsed_then_blocked() -> None:
    client = _fresh_client('chaos-p2-c59')
    calls = _install_mismatch_tools(client)
    turn = await client.chat('用 /data/detect 和 yolov8n.pt 训练，fraction 1.5，执行。')
    assert turn['status'] == 'completed', turn
    assert 'fraction 必须在 (0, 1] 范围内' in turn['message']
    preflight_calls = [kwargs for name, kwargs in calls if name == 'training_preflight']
    assert preflight_calls and preflight_calls[-1]['fraction'] == 1.5


async def _scenario_c60_old_yaml_cannot_follow_new_dataset() -> None:
    client = _fresh_client('chaos-p2-c60')
    _install_mismatch_tools(client)
    client.session_state.active_training.data_yaml = '/data/old/data.yaml'
    client.session_state.active_dataset.data_yaml = '/data/old/data.yaml'
    client.memory.save_state(client.session_state)
    turn = await client.chat('新数据在 /data/new，用 yolov8n.pt 和旧 run 的 data_yaml 继续训练，project /runs/newctx，执行。')
    assert turn['status'] == 'needs_confirmation', turn
    assert '/data/new/data.yaml' in turn['message'] or turn['tool_call']['args'].get('data_yaml') == '/data/new/data.yaml'


async def _run() -> None:
    await _scenario_c53_segmentation_training_stays_outside_mainline()
    await _scenario_c54_force_start_without_yaml_still_goes_prepare()
    await _scenario_c55_video_model_mismatch_surfaces_warning()
    await _scenario_c56_cpu_large_scale_surfaces_warning()
    await _scenario_c57_huge_batch_gets_blocked()
    await _scenario_c58_classes_out_of_range_gets_blocked()
    await _scenario_c59_invalid_fraction_is_parsed_then_blocked()
    await _scenario_c60_old_yaml_cannot_follow_new_dataset()
    print('agent server chaos p2 mismatch matrix ok')


if __name__ == '__main__':
    run(_run())
