from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient, build_agent_client

OUT = Path(__file__).resolve().parent / 'test_agent_capability_range_output.json'


def _now_id() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


async def ensure_no_training(agent: YoloStudioAgentClient) -> dict[str, Any]:
    status = await agent.direct_tool('check_training_status')
    if status.get('running'):
        stop = await agent.direct_tool('stop_training')
        return {'status': status, 'stop': stop}
    return {'status': status}


async def run_prompt(agent: YoloStudioAgentClient, prompt: str, decisions: list[bool] | None = None) -> dict[str, Any]:
    decisions = list(decisions or [])
    transcript: list[dict[str, Any]] = []
    result = await agent.chat(prompt)
    transcript.append({'kind': 'chat', 'prompt': prompt, 'result': result})
    idx = 0
    while result.get('status') == 'needs_confirmation' and idx < len(decisions):
        approved = decisions[idx]
        idx += 1
        result = await agent.confirm(result['thread_id'], approved=approved)
        transcript.append({'kind': 'confirm', 'approved': approved, 'result': result})
    return {'result': result, 'transcript': transcript}


def summarize_events(agent: YoloStudioAgentClient, limit: int = 20) -> dict[str, Any]:
    events = agent.memory.read_events(agent.session_state.session_id, limit=limit)
    tools = []
    confirmations = []
    for event in events:
        if event.get('type') == 'tool_result':
            tools.append({
                'tool': event.get('tool'),
                'ok': event.get('result', {}).get('ok'),
                'summary': event.get('result', {}).get('summary') or event.get('result', {}).get('message'),
            })
        elif event.get('type', '').startswith('confirmation_'):
            confirmations.append({'type': event.get('type'), 'tool': event.get('tool')})
    latest_prepare = agent.memory.latest_event(agent.session_state.session_id, 'tool_result', tool_name='prepare_dataset_for_training')
    latest_start = agent.memory.latest_event(agent.session_state.session_id, 'tool_result', tool_name='start_training')
    return {
        'tools': tools,
        'confirmations': confirmations,
        'latest_prepare': latest_prepare,
        'latest_start': latest_start,
    }


def state_snapshot(agent: YoloStudioAgentClient) -> dict[str, Any]:
    ds = agent.session_state.active_dataset
    tr = agent.session_state.active_training
    pending = agent.get_pending_action() or {}
    return {
        'dataset_root': ds.dataset_root,
        'img_dir': ds.img_dir,
        'label_dir': ds.label_dir,
        'data_yaml': ds.data_yaml,
        'training_running': tr.running,
        'training_model': tr.model,
        'training_data_yaml': tr.data_yaml,
        'training_device': tr.device,
        'pending_tool': pending.get('tool_name', ''),
        'pending_args': pending.get('tool_args', {}),
    }


async def cleanup_training(agent: YoloStudioAgentClient) -> dict[str, Any] | None:
    status = await agent.direct_tool('check_training_status')
    if status.get('running'):
        stop = await agent.direct_tool('stop_training')
        return {'status': status, 'stop': stop}
    return {'status': status}


async def make_agent(provider: str, model: str, session_id: str) -> YoloStudioAgentClient:
    return await build_agent_client(AgentSettings(provider=provider, model=model, session_id=session_id))


async def case_root_train_chain(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-root-train')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '数据在 /data/test_dataset/，按默认划分比例，然后用 yolov8n 模型训练 2 轮。', [True, True])
    cleanup = await cleanup_training(agent)
    summary = summarize_events(agent)
    return {
        'case': 'root_train_chain',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'cleanup': cleanup,
        'assessment': {
            'used_prepare': any(t['tool'] == 'prepare_dataset_for_training' for t in summary['tools']),
            'used_start_training': any(t['tool'] == 'start_training' for t in summary['tools']),
            'resolved_root_ok': agent.session_state.active_dataset.img_dir == '/data/test_dataset/images',
            'resolved_yaml_ok': agent.session_state.active_dataset.data_yaml == '/data/test_dataset/data.yaml',
        },
    }


async def case_conditional_direct_train(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-conditional-direct')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '先检查 /data/test_dataset/ 是否已经可以直接训练；如果可以，不要重新划分，直接用 yolov8n 模型训练 2 轮。', [True, True])
    cleanup = await cleanup_training(agent)
    summary = summarize_events(agent)
    prepare = (summary.get('latest_prepare') or {}).get('result', {})
    steps = [step.get('step') for step in prepare.get('steps_completed', [])]
    return {
        'case': 'conditional_direct_train',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'cleanup': cleanup,
        'assessment': {
            'used_prepare': any(t['tool'] == 'prepare_dataset_for_training' for t in summary['tools']),
            'skipped_split': 'split' not in steps,
            'used_start_training': any(t['tool'] == 'start_training' for t in summary['tools']),
        },
    }


async def case_followup_context_train(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-followup-context')
    await ensure_no_training(agent)
    first = await run_prompt(agent, '扫描 /data/test_dataset/')
    second = await run_prompt(agent, '那就直接训练 2 轮', [True, True])
    cleanup = await cleanup_training(agent)
    summary = summarize_events(agent)
    return {
        'case': 'followup_context_train',
        'provider': provider,
        'model': model,
        'transcript': first['transcript'] + second['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'cleanup': cleanup,
        'assessment': {
            'used_scan': any(t['tool'] == 'scan_dataset' for t in summary['tools']),
            'used_start_training': any(t['tool'] == 'start_training' for t in summary['tools']),
            'final_status': second['result'].get('status'),
        },
    }


async def case_status_branch(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-status-branch')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '如果现在有训练在跑就停止；如果没有，就只告诉我当前没有训练，不要启动新的训练。')
    summary = summarize_events(agent)
    return {
        'case': 'status_branch',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'assessment': {
            'used_status': any(t['tool'] == 'check_training_status' for t in summary['tools']),
            'did_not_start': not any(t['tool'] == 'start_training' for t in summary['tools']),
        },
    }


async def case_nonstandard_dataset(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-nonstandard')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '数据在 /data/agent_cap_tests/nonstandard_dataset/，帮我准备到可训练状态。', [True])
    summary = summarize_events(agent)
    return {
        'case': 'nonstandard_dataset',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'assessment': {
            'started_training': any(t['tool'] == 'start_training' for t in summary['tools']),
            'prepare_ok': (summary.get('latest_prepare') or {}).get('result', {}).get('ok'),
            'prepare_summary': (summary.get('latest_prepare') or {}).get('result', {}).get('summary'),
        },
    }


async def case_no_start_info(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-no-start-info')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '检查 /data/test_dataset/ 是否能直接训练，如果可以，只告诉我会使用哪个 data.yaml 和 auto 会选哪张卡，不要启动训练。')
    summary = summarize_events(agent)
    return {
        'case': 'no_start_info',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'assessment': {
            'did_not_start': not any(t['tool'] == 'start_training' for t in summary['tools']),
            'used_readiness_or_gpu': any(t['tool'] in {'training_readiness', 'check_gpu_status'} for t in summary['tools']),
        },
    }


async def case_cancel_and_recall(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-cancel-recall')
    await ensure_no_training(agent)
    first = await run_prompt(agent, '数据在 /data/test_dataset/，用 yolov8n 模型训练 2 轮。', [True, False])
    second = await run_prompt(agent, '刚才待确认的训练参数是什么？')
    summary = summarize_events(agent)
    return {
        'case': 'cancel_and_recall',
        'provider': provider,
        'model': model,
        'transcript': first['transcript'] + second['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'assessment': {
            'cancelled': any(c['type'] == 'confirmation_cancelled' for c in summary['confirmations']),
            'recall_message': second['result'].get('message'),
        },
    }


async def case_prepare_only_no_train(provider: str, model: str, prefix: str) -> dict[str, Any]:
    agent = await make_agent(provider, model, f'{prefix}-prepare-only')
    await ensure_no_training(agent)
    run = await run_prompt(agent, '数据在 /data/test_dataset/。如果已经可训练就不要重新划分；如果还不能，请按默认比例划分并生成 yaml。但无论如何都不要启动训练，只告诉我结果。', [True])
    summary = summarize_events(agent)
    prepare = (summary.get('latest_prepare') or {}).get('result', {})
    return {
        'case': 'prepare_only_no_train',
        'provider': provider,
        'model': model,
        'transcript': run['transcript'],
        'state': state_snapshot(agent),
        'events': summary,
        'assessment': {
            'used_prepare': any(t['tool'] == 'prepare_dataset_for_training' for t in summary['tools']),
            'did_not_start': not any(t['tool'] == 'start_training' for t in summary['tools']),
            'ready': prepare.get('ready'),
        },
    }


async def main() -> None:
    run_id = _now_id()
    cases: list[dict[str, Any]] = []

    gemma_provider = ('ollama', 'gemma4:e4b', f'stress-gemma-{run_id}')
    deepseek_provider = ('deepseek', 'deepseek-chat', f'stress-deepseek-{run_id}') if os.getenv('DEEPSEEK_API_KEY') else None

    gemma_cases = [
        case_root_train_chain,
        case_conditional_direct_train,
        case_followup_context_train,
        case_status_branch,
        case_nonstandard_dataset,
        case_no_start_info,
        case_cancel_and_recall,
        case_prepare_only_no_train,
    ]

    for fn in gemma_cases:
        try:
            cases.append(await fn(*gemma_provider))
        except Exception as exc:
            cases.append({'case': fn.__name__, 'provider': gemma_provider[0], 'model': gemma_provider[1], 'error': repr(exc)})

    if deepseek_provider:
        deepseek_cases = [
            case_root_train_chain,
            case_nonstandard_dataset,
            case_no_start_info,
            case_cancel_and_recall,
        ]
        for fn in deepseek_cases:
            try:
                cases.append(await fn(*deepseek_provider))
            except Exception as exc:
                cases.append({'case': fn.__name__, 'provider': deepseek_provider[0], 'model': deepseek_provider[1], 'error': repr(exc)})

    payload = {
        'run_id': run_id,
        'cases': cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'run_id': run_id, 'case_count': len(cases), 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
