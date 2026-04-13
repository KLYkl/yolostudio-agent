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


def _install_readiness_tools(client, readiness_map: dict[str, dict[str, Any]]):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            dataset_root = str(kwargs.get('img_dir') or '')
            config = dict(readiness_map[dataset_root])
            result = {
                'ok': True,
                'summary': str(config.get('summary') or '训练前检查完成。'),
                'dataset_root': dataset_root,
                'resolved_img_dir': str(config.get('resolved_img_dir') or ''),
                'resolved_label_dir': str(config.get('resolved_label_dir') or ''),
                'resolved_data_yaml': str(config.get('data_yaml') or ''),
                'ready': bool(config.get('ready')),
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
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c05_empty_directory_blocks_cleanly() -> None:
    client = _fresh_client('chaos-p1-c05')
    calls = _install_readiness_tools(
        client,
        {
            '/data/empty': {
                'ready': False,
                'preparable': False,
                'summary': '当前还不能直接训练：未发现可用图片或标签。',
                'blockers': ['未发现可用图片或标签'],
                'primary_blocker_type': 'no_valid_labels',
            }
        },
    )
    turn = await client.chat('空目录在 /data/empty，训练它。')
    assert turn['status'] == 'completed', turn
    assert '未发现可用图片或标签' in turn['message']
    assert calls and calls[0] == ('training_readiness', {'img_dir': '/data/empty'})
    assert all(name not in {'prepare_dataset_for_training', 'training_preflight', 'start_training'} for name, _ in calls)
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.running is False
    assert client.session_state.active_dataset.dataset_root == '/data/empty'


async def _scenario_c06_file_path_is_not_dataset_root() -> None:
    client = _fresh_client('chaos-p1-c06')
    calls = _install_readiness_tools(
        client,
        {
            '/data/not_dataset.txt': {
                'ready': False,
                'preparable': False,
                'summary': '当前路径更像单个文件，不是可训练数据集根目录。',
                'blockers': ['路径类型异常：不是数据集根目录'],
                'primary_blocker_type': 'path_type_error',
            }
        },
    )
    turn = await client.chat('这是数据集 /data/not_dataset.txt，开始训练。')
    assert turn['status'] == 'completed', turn
    assert '不是可训练数据集根目录' in turn['message']
    assert calls and calls[0] == ('training_readiness', {'img_dir': '/data/not_dataset.txt'})
    assert all(name not in {'prepare_dataset_for_training', 'training_preflight', 'start_training'} for name, _ in calls)
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.running is False
    assert client.session_state.active_dataset.last_readiness.get('primary_blocker_type') == 'path_type_error'


async def _scenario_c07_video_directory_should_not_train_directly() -> None:
    client = _fresh_client('chaos-p1-c07')
    calls = _install_readiness_tools(
        client,
        {
            '/data/videos': {
                'ready': False,
                'preparable': False,
                'summary': '当前路径更像视频目录，不能直接训练；请先抽帧或准备数据。',
                'blockers': ['检测到视频目录，需先抽帧/准备'],
                'primary_blocker_type': 'video_source',
            }
        },
    )
    turn = await client.chat('给视频目录 /data/videos 直接训练。')
    assert turn['status'] == 'completed', turn
    assert '不能直接训练' in turn['message']
    assert '抽帧' in turn['message']
    assert calls and calls[0] == ('training_readiness', {'img_dir': '/data/videos'})
    assert all(name not in {'prepare_dataset_for_training', 'training_preflight', 'start_training'} for name, _ in calls)
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.running is False
    assert client.session_state.active_dataset.last_readiness.get('primary_blocker_type') == 'video_source'


async def _scenario_c08_labels_directory_only_stays_blocked() -> None:
    client = _fresh_client('chaos-p1-c08')
    calls = _install_readiness_tools(
        client,
        {
            '/data/labels_only': {
                'ready': False,
                'preparable': False,
                'summary': '当前只有 labels 目录，缺少 images 和 data_yaml。',
                'blockers': ['缺少 images 目录', '缺少 data_yaml'],
                'primary_blocker_type': 'missing_images',
            }
        },
    )
    turn = await client.chat('目录 /data/labels_only 现在开始训练。')
    assert turn['status'] == 'completed', turn
    assert '缺少 images' in turn['message'] or '只有 labels' in turn['message']
    assert calls and calls[0] == ('training_readiness', {'img_dir': '/data/labels_only'})
    assert all(name not in {'prepare_dataset_for_training', 'training_preflight', 'start_training'} for name, _ in calls)
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.running is False
    assert client.session_state.active_dataset.last_readiness.get('primary_blocker_type') == 'missing_images'


async def _scenario_c09_vague_old_dataset_requires_explicit_path() -> None:
    client = _fresh_client('chaos-p1-c09')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C09 should not call tools without an explicit dataset path: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('就是前几天那个数据集，直接训。')
    assert turn['status'] == 'completed', turn
    assert '缺少数据集路径' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_c10_second_push_without_info_stays_blocked() -> None:
    client = _fresh_client('chaos-p1-c10')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C10 should stay blocked without enough information: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    first = await client.chat('现在能不能训练？')
    assert first['status'] == 'completed', first
    second = await client.chat('开始训练吧。')
    assert second['status'] == 'completed', second
    assert '缺少数据集路径' in second['message']
    assert '缺少预训练权重/模型' in second['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}


async def _run() -> None:
    await _scenario_c05_empty_directory_blocks_cleanly()
    await _scenario_c06_file_path_is_not_dataset_root()
    await _scenario_c07_video_directory_should_not_train_directly()
    await _scenario_c08_labels_directory_only_stays_blocked()
    await _scenario_c09_vague_old_dataset_requires_explicit_path()
    await _scenario_c10_second_push_without_info_stays_blocked()
    print('agent server chaos p1 input matrix ok')


if __name__ == '__main__':
    run(_run())
