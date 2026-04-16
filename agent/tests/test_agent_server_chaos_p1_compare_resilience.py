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

from yolostudio_agent.agent.tests._chaos_test_support import WORK as P0_WORK, _ScriptedGraph, _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run


def _fresh_client(session_id: str):
    shutil.rmtree(P0_WORK / session_id, ignore_errors=True)
    return _make_client(session_id)


def _install_compare_tools(client):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'compare_training_runs':
            result = {
                'ok': False,
                'summary': '训练对比失败：缺少训练记录 train_log_100。',
                'error': '缺少训练记录 train_log_100。',
                'missing_run_id': 'train_log_100',
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return calls


async def _scenario_c96_rerun_last_comparison_surfaces_missing_run() -> None:
    client = _fresh_client('chaos-p1-c96')
    calls = _install_compare_tools(client)
    client.graph = _ScriptedGraph(
        {
            '刚才那个对比再比较一次': (
                [
                    (
                        'compare_training_runs',
                        {
                            'ok': False,
                            'summary': '训练对比失败：缺少训练记录 train_log_100。',
                            'error': '缺少训练记录 train_log_100。',
                            'missing_run_id': 'train_log_100',
                        },
                    )
                ],
                '训练对比失败：缺少训练记录 train_log_100。',
            )
        }
    )  # type: ignore[assignment]
    client.session_state.active_training.last_run_comparison = {
        'left_run': {'run_id': 'train_log_200'},
        'right_run': {'run_id': 'train_log_100'},
        'summary': '上一次对比完成。',
    }
    client.memory.save_state(client.session_state)

    assert await client._try_handle_mainline_intent('刚才那个对比再比较一次。', 'thread-chaos-p1-c96') is None
    turn = await client.chat('刚才那个对比再比较一次。')
    assert turn['status'] == 'completed', turn
    assert client.graph.calls == [('compare_training_runs', {})], client.graph.calls
    assert calls == []
    assert '缺少训练记录 train_log_100' in turn['message']


async def _run() -> None:
    await _scenario_c96_rerun_last_comparison_surfaces_missing_run()
    print('agent server chaos p1 compare resilience ok')


if __name__ == '__main__':
    run(_run())
