from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
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

OUT = Path(__file__).resolve().parent / 'test_zyb_10_method_output.json'
DATASET_ROOT = '/data/agent_cap_tests/zyb'
MODEL_PATH = '/models/yolov8n.pt'


def _now_id() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


async def make_agent(provider: str, model: str, session_id: str) -> YoloStudioAgentClient:
    settings = AgentSettings(provider=provider, model=model, session_id=session_id)
    if provider != 'ollama':
        settings.base_url = os.getenv('YOLOSTUDIO_LLM_BASE_URL', settings.base_url)
        settings.api_key = os.getenv('YOLOSTUDIO_LLM_API_KEY', settings.api_key)
    return await build_agent_client(settings)


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


async def cleanup_training(agent: YoloStudioAgentClient) -> dict[str, Any]:
    status = await agent.direct_tool('check_training_status')
    payload: dict[str, Any] = {'status': status}
    if status.get('running'):
        payload['stop'] = await agent.direct_tool('stop_training')
    return payload


async def case_01_tool_scan(agent: YoloStudioAgentClient) -> dict[str, Any]:
    result = await agent.direct_tool('scan_dataset', img_dir=DATASET_ROOT)
    return {
        'method': 'tool_scan_root',
        'kind': 'tool',
        'input': {'img_dir': DATASET_ROOT},
        'result': result,
        'assessment': {
            'ok': result.get('ok') is True,
            'detected_classes_txt': bool(result.get('detected_classes_txt')),
            'class_name_source': result.get('class_name_source'),
            'missing_label_images': result.get('missing_label_images'),
            'risk_level': result.get('risk_level'),
        },
    }


async def case_02_tool_validate(agent: YoloStudioAgentClient) -> dict[str, Any]:
    result = await agent.direct_tool('validate_dataset', img_dir=DATASET_ROOT)
    return {
        'method': 'tool_validate_root',
        'kind': 'tool',
        'input': {'img_dir': DATASET_ROOT},
        'result': result,
        'assessment': {
            'ok': result.get('ok') is True,
            'has_risks': result.get('has_risks'),
            'missing_label_images': result.get('missing_label_images'),
            'risk_level': result.get('risk_level'),
            'summary_mentions_missing': '缺少标签' in str(result.get('summary', '')),
        },
    }


async def case_03_tool_readiness(agent: YoloStudioAgentClient) -> dict[str, Any]:
    result = await agent.direct_tool('training_readiness', img_dir=DATASET_ROOT)
    return {
        'method': 'tool_training_readiness',
        'kind': 'tool',
        'input': {'img_dir': DATASET_ROOT},
        'result': result,
        'assessment': {
            'ok': result.get('ok') is True,
            'ready': result.get('ready'),
            'has_warnings': bool(result.get('warnings')),
            'blockers': result.get('blockers'),
            'risk_level': result.get('risk_level'),
        },
    }


async def case_04_tool_prepare(agent: YoloStudioAgentClient) -> dict[str, Any]:
    result = await agent.direct_tool('prepare_dataset_for_training', dataset_path=DATASET_ROOT)
    return {
        'method': 'tool_prepare_dataset',
        'kind': 'tool',
        'input': {'dataset_path': DATASET_ROOT},
        'result': result,
        'assessment': {
            'ok': result.get('ok') is True,
            'ready': result.get('ready'),
            'data_yaml': result.get('data_yaml'),
            'class_name_source': result.get('class_name_source'),
            'risk_level': result.get('risk_level'),
            'summary_mentions_risk': '数据质量风险' in str(result.get('summary', '')),
        },
    }


async def case_05_tool_train_lifecycle(agent: YoloStudioAgentClient) -> dict[str, Any]:
    prep = await agent.direct_tool('prepare_dataset_for_training', dataset_path=DATASET_ROOT)
    start = await agent.direct_tool(
        'start_training',
        model=MODEL_PATH,
        data_yaml=prep.get('data_yaml', ''),
        epochs=3,
        device='auto',
    )
    await asyncio.sleep(8)
    status = await agent.direct_tool('check_training_status')
    stop = await agent.direct_tool('stop_training')
    final_status = await agent.direct_tool('check_training_status')
    return {
        'method': 'tool_training_lifecycle',
        'kind': 'tool',
        'result': {
            'prepare': prep,
            'start': start,
            'status': status,
            'stop': stop,
            'final_status': final_status,
        },
        'assessment': {
            'started': start.get('ok') is True,
            'running_after_start': status.get('running') is True,
            'stop_ok': stop.get('ok') is True,
            'stopped_after_stop': final_status.get('running') is False,
        },
    }


async def case_06_gemma_dirty_summary() -> dict[str, Any]:
    agent = await make_agent('ollama', 'gemma4:e4b', f'zyb-gemma-summary-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    prompt = '请扫描 /data/agent_cap_tests/zyb/ ，然后用中文总结这个数据集目前最值得注意的 3 个点。'
    run = await run_prompt(agent, prompt)
    msg = str(run['result'].get('message', ''))
    return {
        'method': 'gemma_dirty_summary',
        'kind': 'agent',
        'provider': 'ollama',
        'model': 'gemma4:e4b',
        'transcript': run['transcript'],
        'assessment': {
            'completed': run['result'].get('status') == 'completed',
            'mentions_missing_labels': '缺失标签' in msg or '缺少标签' in msg,
            'mentions_classes_txt_or_classes': ('classes.txt' in msg) or ('Excavator' in msg),
            'mentions_image_count': '7027' in msg,
        },
    }


async def case_07_gemma_no_train_constraint() -> dict[str, Any]:
    agent = await make_agent('ollama', 'gemma4:e4b', f'zyb-gemma-no-train-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    prompt = '请检查 /data/agent_cap_tests/zyb/ 是否能直接训练。如果还不能，请只告诉我原因和建议，不要启动训练，也不要先做划分。'
    run = await run_prompt(agent, prompt)
    events = agent.memory.read_events(agent.session_state.session_id, limit=20)
    started = any(e.get('type') == 'tool_result' and e.get('tool') == 'start_training' for e in events)
    prepared = any(e.get('type') == 'tool_result' and e.get('tool') == 'prepare_dataset_for_training' for e in events)
    return {
        'method': 'gemma_no_train_constraint',
        'kind': 'agent',
        'provider': 'ollama',
        'model': 'gemma4:e4b',
        'transcript': run['transcript'],
        'assessment': {
            'completed': run['result'].get('status') == 'completed',
            'did_not_start_training': not started,
            'did_not_prepare': not prepared,
        },
    }


async def case_08_gemma_full_chain_train() -> dict[str, Any]:
    agent = await make_agent('ollama', 'gemma4:e4b', f'zyb-gemma-full-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    prompt = '数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。'
    run = await run_prompt(agent, prompt, [True, True])
    await asyncio.sleep(8)
    status = await agent.direct_tool('check_training_status')
    stop = await agent.direct_tool('stop_training')
    events = agent.memory.read_events(agent.session_state.session_id, limit=30)
    used_prepare = any(e.get('type') == 'tool_result' and e.get('tool') == 'prepare_dataset_for_training' for e in events)
    used_start = any(e.get('type') == 'tool_result' and e.get('tool') == 'start_training' for e in events)
    return {
        'method': 'gemma_full_chain_train',
        'kind': 'agent',
        'provider': 'ollama',
        'model': 'gemma4:e4b',
        'transcript': run['transcript'],
        'status': status,
        'stop': stop,
        'assessment': {
            'used_prepare': used_prepare,
            'used_start_training': used_start,
            'running_after_confirm': status.get('running') is True,
            'stop_ok': stop.get('ok') is True,
        },
    }


async def case_09_gemma_cancel_and_recall() -> dict[str, Any]:
    agent = await make_agent('ollama', 'gemma4:e4b', f'zyb-gemma-recall-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    first = await run_prompt(agent, '数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。', [True, False])
    second = await run_prompt(agent, '刚才待确认的训练参数是什么？这个数据集最大的风险又是什么？')
    msg = str(second['result'].get('message', ''))
    return {
        'method': 'gemma_cancel_and_recall',
        'kind': 'agent',
        'provider': 'ollama',
        'model': 'gemma4:e4b',
        'transcript': first['transcript'] + second['transcript'],
        'assessment': {
            'cancelled': first['transcript'][-1]['result'].get('status') == 'cancelled',
            'mentions_model': 'yolov8n' in msg,
            'mentions_risk': ('缺失标签' in msg) or ('缺少标签' in msg),
            'mentions_auto_or_device': ('auto' in msg) or ('device' in msg) or ('显卡' in msg),
        },
    }


async def case_10_deepseek_full_chain() -> dict[str, Any]:
    agent = await make_agent('deepseek', 'deepseek-chat', f'zyb-deepseek-full-{uuid.uuid4().hex[:8]}')
    await ensure_no_training(agent)
    prompt = '数据在 /data/agent_cap_tests/zyb/，按默认划分比例，然后用 yolov8n 模型进行训练。'
    run = await run_prompt(agent, prompt, [True, True])
    await asyncio.sleep(8)
    status = await agent.direct_tool('check_training_status')
    stop = await agent.direct_tool('stop_training')
    events = agent.memory.read_events(agent.session_state.session_id, limit=30)
    used_prepare = any(e.get('type') == 'tool_result' and e.get('tool') == 'prepare_dataset_for_training' for e in events)
    used_start = any(e.get('type') == 'tool_result' and e.get('tool') == 'start_training' for e in events)
    return {
        'method': 'deepseek_full_chain_train',
        'kind': 'agent',
        'provider': 'deepseek',
        'model': 'deepseek-chat',
        'transcript': run['transcript'],
        'status': status,
        'stop': stop,
        'assessment': {
            'used_prepare': used_prepare,
            'used_start_training': used_start,
            'running_after_confirm': status.get('running') is True,
            'stop_ok': stop.get('ok') is True,
        },
    }


async def main() -> None:
    run_id = _now_id()
    tool_agent = await make_agent('ollama', 'gemma4:e4b', f'zyb-tool-{run_id}')
    await ensure_no_training(tool_agent)

    cases: list[dict[str, Any]] = []
    cases.append(await case_01_tool_scan(tool_agent))
    cases.append(await case_02_tool_validate(tool_agent))
    cases.append(await case_03_tool_readiness(tool_agent))
    cases.append(await case_04_tool_prepare(tool_agent))
    cases.append(await case_05_tool_train_lifecycle(tool_agent))
    cases.append(await case_06_gemma_dirty_summary())
    cases.append(await case_07_gemma_no_train_constraint())
    cases.append(await case_08_gemma_full_chain_train())
    cases.append(await case_09_gemma_cancel_and_recall())
    if os.getenv('DEEPSEEK_API_KEY'):
        cases.append(await case_10_deepseek_full_chain())
    else:
        cases.append({
            'method': 'deepseek_full_chain_train',
            'kind': 'agent',
            'provider': 'deepseek',
            'model': 'deepseek-chat',
            'skipped': True,
            'reason': 'DEEPSEEK_API_KEY 未配置',
        })

    payload = {
        'run_id': run_id,
        'dataset_root': DATASET_ROOT,
        'case_count': len(cases),
        'cases': cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'run_id': run_id, 'case_count': len(cases), 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
