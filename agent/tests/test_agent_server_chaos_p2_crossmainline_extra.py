from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

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


def _install_no_tools(client):
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: object) -> dict[str, object]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'crossmainline extra guardrail should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c65_failed_then_predict_with_best_weight_requires_real_path() -> None:
    client = _fresh_client('chaos-p2-c65')
    calls = _install_no_tools(client)
    client.session_state.active_training.best_run_selection = {
        'best_run': {'run_id': 'train_log_best'},
        'summary': '最近最佳训练为 train_log_best。',
    }
    client.memory.save_state(client.session_state)
    before = copy.deepcopy(client.session_state.active_training.best_run_selection)
    turn = await client.chat('训练失败后，用最好权重去预测视频 /data/videos。')
    assert turn['status'] == 'completed', turn
    assert '不能直接假定“最佳训练”的权重文件路径' in turn['message']
    assert calls == []
    assert client.session_state.active_training.best_run_selection == before
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_prediction.source_path == ''


async def _scenario_c66_mixed_predict_train_compare_request_is_split() -> None:
    client = _fresh_client('chaos-p2-c66')
    calls = _install_no_tools(client)
    turn = await client.chat('先预测视频 /data/videos，再继续上次训练，然后比较两次训练。')
    assert turn['status'] == 'completed', turn
    assert '请拆成连续步骤' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}
    assert client.session_state.active_prediction.source_path == ''


async def _scenario_c67_merge_frames_into_old_dataset_requires_prepare_chain() -> None:
    client = _fresh_client('chaos-p2-c67')
    calls = _install_no_tools(client)
    turn = await client.chat('把刚才抽帧结果合并到旧数据集再训练。')
    assert turn['status'] == 'completed', turn
    assert '先走数据准备/校验' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_c68_predict_using_best_run_after_compare_still_needs_weight_path() -> None:
    client = _fresh_client('chaos-p2-c68')
    calls = _install_no_tools(client)
    client.session_state.active_training.last_run_comparison = {
        'left_run': {'run_id': 'train_log_new'},
        'right_run': {'run_id': 'train_log_old'},
        'summary': '最近一次训练更好。',
    }
    client.memory.save_state(client.session_state)
    before = copy.deepcopy(client.session_state.active_training.last_run_comparison)
    turn = await client.chat('那就用最佳训练去预测视频 /data/videos。')
    assert turn['status'] == 'completed', turn
    assert '不能直接假定“最佳训练”的权重文件路径' in turn['message']
    assert calls == []
    assert client.session_state.active_training.last_run_comparison == before
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_prediction.source_path == ''


async def _scenario_c69_prediction_result_cannot_become_training_data_directly() -> None:
    client = _fresh_client('chaos-p2-c69')
    calls = _install_no_tools(client)
    turn = await client.chat('就拿刚才的预测结果目录直接开始训练。')
    assert turn['status'] == 'completed', turn
    assert '不能直接当训练数据开训' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_c70_no_continuous_parallel_prediction_during_training() -> None:
    client = _fresh_client('chaos-p2-c70')
    calls = _install_no_tools(client)
    turn = await client.chat('边训练边不断做视频预测。')
    assert turn['status'] == 'completed', turn
    assert '不支持“边训练边持续做视频预测”' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}
    assert client.session_state.active_prediction.source_path == ''


async def _run() -> None:
    await _scenario_c65_failed_then_predict_with_best_weight_requires_real_path()
    await _scenario_c66_mixed_predict_train_compare_request_is_split()
    await _scenario_c67_merge_frames_into_old_dataset_requires_prepare_chain()
    await _scenario_c68_predict_using_best_run_after_compare_still_needs_weight_path()
    await _scenario_c69_prediction_result_cannot_become_training_data_directly()
    await _scenario_c70_no_continuous_parallel_prediction_during_training()
    print('agent server chaos p2 crossmainline extra ok')


if __name__ == '__main__':
    run(_run())
