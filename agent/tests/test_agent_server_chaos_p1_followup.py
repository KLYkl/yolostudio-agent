from __future__ import annotations

import json
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

from yolostudio_agent.agent.tests._chaos_test_support import _ScriptedGraph, _make_client
from yolostudio_agent.agent.tests._coroutine_runner import run
from langchain_core.messages import AIMessage, ToolMessage


class _ObservedStatusGraph:
    def __init__(self) -> None:
        self.client = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        self.calls.append(('check_training_status', {}))
        result = await self.client.direct_tool('check_training_status')
        reply = await self.client._render_tool_result_message('check_training_status', result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': 'check_training_status', 'args': {}}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name='check_training_status', tool_call_id=tool_call_id),
                AIMessage(content=reply),
            ]
        }


async def _scenario_c44_completed_next_step_guidance_routes_cleanly() -> None:
    client = _make_client('chaos-p1-c44')
    client.session_state.active_dataset.last_readiness = {
        'ready': True,
        'summary': '数据已具备训练条件。',
        'resolved_data_yaml': '/data/c44/data.yaml',
    }
    client.graph = _ScriptedGraph(
        {
            '下一步先补数据还是调参数': (
                [
                    (
                        'summarize_training_run',
                        {
                            'ok': True,
                            'summary': '训练已完成：precision=0.82 recall=0.58 mAP50=0.61',
                            'run_state': 'completed',
                            'analysis_ready': True,
                            'minimum_facts_ready': True,
                            'signals': ['completed_run', 'low_recall'],
                            'facts': ['训练已完成'],
                            'next_actions': ['优先补召回相关数据'],
                            'metrics': {'precision': 0.82, 'recall': 0.58, 'mAP50': 0.61},
                        },
                    ),
                    (
                        'recommend_next_training_step',
                        {
                            'ok': True,
                            'summary': '当前更适合先补召回相关数据，再考虑参数微调。',
                            'recommended_action': 'fix_data_quality',
                            'signals': ['low_recall', 'data_quality_risk'],
                            'matched_rule_ids': ['workflow_low_recall'],
                        },
                    ),
                ],
                '当前更适合先补召回相关数据，再考虑参数微调。',
            )
        }
    )  # type: ignore[assignment]
    assert await client._try_handle_mainline_intent('下一步先补数据还是调参数？', 'thread-chaos-p1-c44') is None
    turn = await client.chat('下一步先补数据还是调参数？')
    assert turn['status'] == 'completed', turn
    assert any(token in turn['message'] for token in ('优先补召回相关数据', '先补召回相关数据', '更适合先补召回相关数据')), turn
    assert client.graph.calls == [('summarize_training_run', {}), ('recommend_next_training_step', {})]


async def _scenario_c45_stopped_convergence_question_stays_conservative() -> None:
    client = _make_client('chaos-p1-c45')
    client.graph = _ScriptedGraph(
        {
            '是不是已经收敛了': (
                [
                    (
                        'summarize_training_run',
                        {
                            'ok': True,
                            'summary': '训练已停止：epoch=9/30，当前只有阶段性结果。',
                            'run_state': 'stopped',
                            'analysis_ready': True,
                            'minimum_facts_ready': True,
                            'signals': ['stopped_run', 'early_stop'],
                            'facts': ['训练已停止'],
                            'next_actions': ['谨慎解释当前结果'],
                            'metrics': {'precision': 0.71, 'recall': 0.41},
                        },
                    ),
                    (
                        'analyze_training_outcome',
                        {
                            'ok': True,
                            'summary': '这次训练已停止，目前只能做阶段性判断，不能当成最终收敛结论。',
                            'assessment': 'incomplete_observation',
                            'signals': ['stopped_run', 'not_final'],
                            'matched_rule_ids': ['workflow_stopped_not_final'],
                            'next_actions': ['如果要下最终结论，先补完整训练或重新验证。'],
                        },
                    ),
                ],
                '这次训练已停止，目前只能做阶段性判断，不能当成最终收敛结论。',
            )
        }
    )  # type: ignore[assignment]
    assert await client._try_handle_mainline_intent('是不是已经收敛了？', 'thread-chaos-p1-c45') is None
    turn = await client.chat('是不是已经收敛了？')
    assert turn['status'] == 'completed', turn
    assert '不能当成最终收敛结论' in turn['message']
    assert client.graph.calls == [('summarize_training_run', {}), ('analyze_training_outcome', {})]


async def _scenario_c46_status_phrase_now_routes_status() -> None:
    client = _make_client('chaos-p1-c46')
    client.session_state.active_training.last_status = {}
    client.memory.save_state(client.session_state)
    graph = _ObservedStatusGraph()
    graph.bind(client)
    client.graph = graph  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '训练已完成：当前没有活动训练进程。',
            'run_state': 'completed',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'signals': ['completed_run'],
            'facts': ['最近一次训练已完成'],
            'next_actions': ['可以继续分析训练结果'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    routed = await client._try_handle_mainline_intent('现在状态呢？', 'thread-chaos-p1-c46-status')
    assert routed is not None
    assert routed['tool_call']['name'] == 'check_training_status'
    calls.clear()
    graph.calls.clear()
    turn = await client.chat('现在状态呢？')
    assert turn['status'] == 'completed', turn
    assert '训练已完成' in turn['message']
    assert calls == [('check_training_status', {})]
    assert graph.calls == []


async def _scenario_c47_provenance_question_uses_last_comparison() -> None:
    client = _make_client('chaos-p1-c47')
    client.session_state.active_training.last_run_comparison = {
        'summary': '最近一次训练相对上一次召回下降。',
        'left_run': {'run_id': 'train_log_new'},
        'right_run': {'run_id': 'train_log_old'},
        'signals': ['latest_run_regressed'],
    }
    client.memory.save_state(client.session_state)
    client.graph = _ScriptedGraph(
        {
            '你基于哪次训练说的': (
                [],
                '我当前主要基于训练对比结果：train_log_new 对比 train_log_old。\n- 对比摘要: 最近一次训练相对上一次召回下降。',
            )
        }
    )  # type: ignore[assignment]

    assert await client._try_handle_mainline_intent('你基于哪次训练说的？', 'thread-chaos-p1-c47') is None
    turn = await client.chat('你基于哪次训练说的？')
    assert turn['status'] == 'completed', turn
    assert 'train_log_new' in turn['message']
    assert 'train_log_old' in turn['message']
    assert client.graph.calls == []


async def _scenario_c48_compare_followup_routes_compare_analysis() -> None:
    client = _make_client('chaos-p1-c48')
    client.graph = _ScriptedGraph(
        {
            '刚刚那次和上次比哪个好': (
                [
                    (
                        'compare_training_runs',
                        {
                            'ok': True,
                            'summary': '最近一次训练相对上一次 precision 更高，但 recall 更低。',
                            'left_run': {'run_id': 'train_log_new', 'summary': 'new'},
                            'right_run': {'run_id': 'train_log_old', 'summary': 'old'},
                            'signals': ['latest_run_more_conservative'],
                            'highlights': ['precision 上升', 'recall 下降'],
                            'next_actions': ['结合任务目标决定是否继续补召回'],
                        },
                    ),
                ],
                '最近一次训练相对上一次 precision 更高，但 recall 更低。',
            )
        }
    )  # type: ignore[assignment]
    assert await client._try_handle_mainline_intent('刚刚那次和上次比哪个好？', 'thread-chaos-p1-c48') is None
    turn = await client.chat('刚刚那次和上次比哪个好？')
    assert turn['status'] == 'completed', turn
    assert 'precision 更高' in turn['message'] or 'recall 更低' in turn['message']
    assert client.graph.calls == [('compare_training_runs', {})]


async def _scenario_c49_best_run_followup_routes_selection() -> None:
    client = _make_client('chaos-p1-c49')
    client.graph = _ScriptedGraph(
        {
            '最近哪次最值得参考，怎么看': (
                [
                    (
                        'select_best_training_run',
                        {
                            'ok': True,
                            'summary': '最近最值得参考的训练是 train_log_best。',
                            'best_run': {'run_id': 'train_log_best', 'summary': 'best run'},
                            'candidates': [{'run_id': 'train_log_best'}, {'run_id': 'train_log_prev'}],
                            'signals': ['analysis_ready_run'],
                            'next_actions': ['基于最佳训练继续分析或做后续决策'],
                        },
                    ),
                    (
                        'analyze_training_outcome',
                        {
                            'ok': True,
                            'summary': 'train_log_best 当前是最值得参考的一次训练。',
                            'assessment': 'best_recent_run',
                            'signals': ['best_recent_run'],
                            'matched_rule_ids': ['workflow_best_run'],
                        },
                    ),
                ],
                'train_log_best 当前是最值得参考的一次训练。',
            )
        }
    )  # type: ignore[assignment]
    assert await client._try_handle_mainline_intent('最近哪次最值得参考，怎么看？', 'thread-chaos-p1-c49') is None
    turn = await client.chat('最近哪次最值得参考，怎么看？')
    assert turn['status'] == 'completed', turn
    assert 'train_log_best' in turn['message']
    assert client.graph.calls == [('select_best_training_run', {}), ('analyze_training_outcome', {})]


async def _scenario_c50_evidence_question_uses_state_facts() -> None:
    client = _make_client('chaos-p1-c50')
    client.session_state.active_training.training_run_summary = {
        'summary': '训练已完成：precision=0.81 recall=0.43，召回明显偏低。',
        'signals': ['low_recall', 'completed_run'],
    }
    client.session_state.active_knowledge.last_analysis = {
        'summary': '当前更像数据质量问题，尤其是召回相关数据不足。',
        'signals': ['data_quality_risk', 'low_recall'],
        'matched_rule_ids': ['workflow_low_recall'],
    }
    client.session_state.active_knowledge.last_recommendation = {
        'summary': '建议先补数据，再考虑微调参数。',
        'recommended_action': 'fix_data_quality',
        'signals': ['data_quality_risk'],
        'matched_rule_ids': ['workflow_fix_data'],
    }
    client.memory.save_state(client.session_state)
    client.graph = _ScriptedGraph(
        {
            '刚才你说数据有问题，依据是什么': (
                [],
                '当前判断主要基于这些事实：\n- 训练事实: 训练已完成：precision=0.81 recall=0.43，召回明显偏低。\n- 训练信号: low_recall, completed_run\n- 分析结论: 当前更像数据质量问题，尤其是召回相关数据不足。\n- 分析信号: data_quality_risk, low_recall\n- 建议依据: 建议先补数据，再考虑微调参数。\n- 当前建议动作: fix_data_quality',
            )
        }
    )  # type: ignore[assignment]

    assert await client._try_handle_mainline_intent('刚才你说数据有问题，依据是什么？', 'thread-chaos-p1-c50') is None
    turn = await client.chat('刚才你说数据有问题，依据是什么？')
    assert turn['status'] == 'completed', turn
    assert '训练事实' in turn['message']
    assert '召回明显偏低' in turn['message']
    assert 'data_quality_risk' in turn['message']
    assert client.graph.calls == []
    assert 'fix_data_quality' in turn['message']


async def _run() -> None:
    await _scenario_c44_completed_next_step_guidance_routes_cleanly()
    await _scenario_c45_stopped_convergence_question_stays_conservative()
    await _scenario_c46_status_phrase_now_routes_status()
    await _scenario_c47_provenance_question_uses_last_comparison()
    await _scenario_c48_compare_followup_routes_compare_analysis()
    await _scenario_c49_best_run_followup_routes_selection()
    await _scenario_c50_evidence_question_uses_state_facts()
    print('agent server chaos p1 followup ok')


if __name__ == '__main__':
    run(_run())
