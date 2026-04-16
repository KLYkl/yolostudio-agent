from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_real_llm_dialogue'


class _DummyGraph:
    def get_state(self, config: dict[str, Any]) -> None:
        del config
        return None


def _contains_any(text: str, tokens: list[str]) -> bool:
    lowered = str(text or '').strip().lower()
    return any(str(token or '').strip().lower() in lowered for token in tokens if str(token or '').strip())


def _call_names(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in calls]


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(output_path)


async def _build_client(
    *,
    planner_llm: Any,
    memory_root: Path,
) -> tuple[YoloStudioAgentClient, list[tuple[str, dict[str, Any]]]]:
    settings = AgentSettings(
        provider=os.getenv('YOLOSTUDIO_TEST_PROVIDER', os.getenv('YOLOSTUDIO_LLM_PROVIDER', 'ollama')),
        model=os.getenv('YOLOSTUDIO_TEST_MODEL', os.getenv('YOLOSTUDIO_AGENT_MODEL', 'gemma4:e4b')),
        base_url=os.getenv('YOLOSTUDIO_LLM_BASE_URL', os.getenv('YOLOSTUDIO_OLLAMA_URL', 'http://127.0.0.1:11434')),
        api_key=os.getenv('YOLOSTUDIO_LLM_API_KEY', ''),
        temperature=float(os.getenv('YOLOSTUDIO_LLM_TEMPERATURE', '0')),
        session_id=f'training-loop-real-llm-{uuid.uuid4().hex[:8]}',
        memory_root=str(memory_root),
    )
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={}, planner_llm=planner_llm)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            img_dir = str(kwargs.get('img_dir') or '')
            if img_dir.startswith('/missing'):
                result = {
                    'ok': False,
                    'summary': f'训练前检查失败：输入路径不存在：{img_dir}',
                    'error': f'输入路径不存在: {img_dir}',
                    'ready': False,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [f'输入路径不存在: {img_dir}'],
                }
            else:
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：当前数据已具备训练条件。',
                    'dataset_root': '/data/loop',
                    'resolved_img_dir': '/data/loop/images',
                    'resolved_label_dir': '/data/loop/labels',
                    'resolved_data_yaml': '/data/loop/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
        elif tool_name == 'list_training_environments':
            env_name = os.getenv('YOLOSTUDIO_TEST_TRAIN_ENV', 'yolostudio-agent-server')
            result = {
                'ok': True,
                'summary': f'发现 1 个可用训练环境，默认将使用 {env_name}',
                'environments': [{'name': env_name, 'display_name': env_name, 'selected_by_default': True}],
                'default_environment': {'name': env_name, 'display_name': env_name},
            }
        elif tool_name == 'start_training_loop':
            result = {
                'ok': True,
                'summary': '环训练已启动：helmet-loop（loop_id=loop-llm-123）',
                'loop_id': 'loop-llm-123',
                'loop_name': kwargs.get('loop_name') or 'helmet-loop',
                'status': 'queued',
                'managed_level': kwargs.get('managed_level', 'full_auto'),
                'boundaries': {
                    'max_rounds': kwargs.get('max_rounds', 2),
                    'target_metric': kwargs.get('target_metric', 'map50'),
                    'target_metric_value': kwargs.get('target_metric_value'),
                },
                'next_round_plan': {'round_index': 1, 'change_set': []},
            }
        elif tool_name == 'check_training_loop_status':
            result = {
                'ok': True,
                'summary': '第 2 轮训练已完成，准备下一轮。',
                'loop_id': 'loop-llm-123',
                'loop_name': 'helmet-loop',
                'status': 'awaiting_review',
                'managed_level': 'full_auto',
                'current_round_index': 2,
                'completed_rounds': 2,
                'max_rounds': 2,
                'best_round_index': 2,
                'best_target_metric': 0.68,
                'knowledge_gate_status': {
                    'outcome': 'awaiting_review',
                    'summary': '下一轮变更幅度偏大，当前停在审阅闸门。',
                },
                'latest_round_review': {
                    'recommended_action': 'continue_observing',
                    'why': 'mAP50 仍在提升，但幅度已经变小。',
                    'next_focus': '重点看误检样本是否下降。',
                },
                'latest_round_memory': {
                    'next_focus': '继续观察误检样本变化',
                    'decision_type': 'await_review',
                    'carry_forward': ['当前 batch/epochs 组合已验证可稳定完成。'],
                },
                'latest_planner_output': {
                    'decision_type': 'await_review',
                    'decision_reason': '下一轮变更幅度偏大，等待确认。',
                    'planner_source': 'llm',
                    'planner_summary': '建议在下一轮只小步调整 epochs。',
                },
                'latest_round_card': {
                    'round_index': 2,
                    'status': 'completed',
                    'target_metric': 'map50',
                    'target_metric_value': 0.68,
                    'vs_previous': {'highlights': ['mAP50 提升 +0.0300']},
                    'round_review': {
                        'recommended_action': 'continue_observing',
                        'why': 'mAP50 仍在提升，但幅度已经变小。',
                    },
                    'round_memory': {
                        'next_focus': '继续观察误检样本变化',
                        'decision_type': 'await_review',
                    },
                    'planner_output': {
                        'decision_type': 'await_review',
                        'decision_reason': '下一轮变更幅度偏大，等待确认。',
                        'planner_source': 'llm',
                    },
                    'next_plan': {'change_set': [{'field': 'epochs', 'old': 5, 'new': 8}]},
                },
                'next_actions': ['可以确认是否继续下一轮。'],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        client._record_secondary_event(tool_name, result)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    return client, calls


async def _run(output_path: str | Path | None = None) -> dict[str, Any]:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'provider': os.getenv('YOLOSTUDIO_TEST_PROVIDER', os.getenv('YOLOSTUDIO_LLM_PROVIDER', 'ollama')),
        'model': os.getenv('YOLOSTUDIO_TEST_MODEL', os.getenv('YOLOSTUDIO_AGENT_MODEL', 'gemma4:e4b')),
    }
    llm_settings = LlmProviderSettings(
        provider=payload['provider'],
        model=payload['model'],
        base_url=os.getenv('YOLOSTUDIO_LLM_BASE_URL', os.getenv('YOLOSTUDIO_OLLAMA_URL', 'http://127.0.0.1:11434')),
        api_key=os.getenv('YOLOSTUDIO_LLM_API_KEY', ''),
        temperature=float(os.getenv('YOLOSTUDIO_LLM_TEMPERATURE', '0')),
        ollama_keep_alive=os.getenv('YOLOSTUDIO_OLLAMA_KEEP_ALIVE', ''),
    )
    payload['llm'] = provider_summary(llm_settings)
    planner_llm = build_llm(llm_settings)

    try:
        client, calls = await _build_client(planner_llm=planner_llm, memory_root=WORK / 'main')
        missing_path = await client.chat('用 /missing/loop 数据集和 yolov8n.pt 循环训一下。')
        assert missing_path['status'] == 'completed', missing_path
        main_call_names = _call_names(calls)
        assert 'training_readiness' in main_call_names, main_call_names
        missing_message = str(missing_path.get('message') or '')
        assert '/missing/loop' in missing_message, missing_message
        assert _contains_any(missing_message, ['不存在', '路径']), missing_message

        start_turn = await client.chat('用 /data/loop 数据集和 yolov8n.pt 循环训一下，最多 2 轮。')
        assert start_turn['status'] == 'needs_confirmation', start_turn
        assert (start_turn.get('tool_call') or {}).get('name') == 'start_training_loop', start_turn
        assert (start_turn['tool_call']['args']).get('max_rounds') == 2, start_turn

        confirm_turn = await client.confirm(start_turn['thread_id'], approved=True)
        assert confirm_turn['status'] == 'completed', confirm_turn
        main_call_names = _call_names(calls)
        assert 'start_training_loop' in main_call_names, main_call_names
        assert client.session_state.active_training.active_loop_id == 'loop-llm-123'

        status_turn = await client.chat('现在环训练怎么样了？')
        assert status_turn['status'] == 'completed', status_turn
        main_call_names = _call_names(calls)
        assert 'check_training_loop_status' in main_call_names, main_call_names
        status_message = str(status_turn.get('message') or '')
        assert status_message.strip(), status_turn
        assert _contains_any(
            status_message,
            ['helmet-loop', '第 2 轮', '第2轮', '等待审阅', '继续观察', '0.68', '误检'],
        ), status_message

        short_client, short_calls = await _build_client(planner_llm=planner_llm, memory_root=WORK / 'short')
        short_client.session_state.active_dataset.dataset_root = '/data/loop'
        short_client.session_state.active_dataset.data_yaml = '/data/loop/data.yaml'
        short_client.session_state.active_training.model = 'yolov8n.pt'
        short_turn = await short_client.chat('就这个，循环训一下，最多 2 轮。')
        assert short_turn['status'] == 'needs_confirmation', short_turn
        short_call_names = _call_names(short_calls)
        assert 'training_readiness' not in short_call_names or short_turn['tool_call']['args']['data_yaml'] == '/data/loop/data.yaml', short_call_names
        assert (short_turn.get('tool_call') or {}).get('name') == 'start_training_loop', short_turn
        assert short_turn['tool_call']['args']['model'] == 'yolov8n.pt', short_turn
        assert short_turn['tool_call']['args']['data_yaml'] == '/data/loop/data.yaml', short_turn
        assert short_turn['tool_call']['args']['max_rounds'] == 2, short_turn

        payload['ok'] = True
        payload['transcript'] = {
            'missing_path': missing_path,
            'start_turn': start_turn,
            'confirm_turn': confirm_turn,
            'status_turn': status_turn,
            'short_turn': short_turn,
        }
        payload['tool_calls'] = {
            'main': [{'tool': name, 'args': kwargs} for name, kwargs in calls],
            'short': [{'tool': name, 'args': kwargs} for name, kwargs in short_calls],
        }
        payload['status_message'] = status_message
    finally:
        if output_path:
            payload['output_path'] = _write_json(output_path, payload)
        shutil.rmtree(WORK, ignore_errors=True)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a real-LLM multi-turn smoke for training-loop dialogue handling.')
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    payload = asyncio.run(_run(args.output or None))
    print(payload.get('output_path') or json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
