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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent_plan.agent.client.agent_client import AgentSettings, build_agent_client

DATASET_ROOT = os.getenv('YOLOSTUDIO_TEST_DATASET_ROOT', '/home/kly/agent_cap_tests/zyb')
MCP_URL = os.getenv('YOLOSTUDIO_TEST_MCP_URL', 'http://127.0.0.1:18080/mcp')
OLLAMA_URL = os.getenv('YOLOSTUDIO_TEST_OLLAMA_URL', 'http://127.0.0.1:11435')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
OUTPUT = Path(r'D:\yolodo2.0\agent_plan\agent\tests\test_zyb_large_dataset_output.json')
MEM_ROOT = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_zyb_memory')


def _norm(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        raw = '\n'.join((item.get('text', '') if isinstance(item, dict) else str(item)) for item in payload).strip()
        if not raw:
            return {'ok': True, 'raw': raw}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {'ok': True, 'value': obj}
        except Exception:
            return {'ok': True, 'raw': raw}
    return {'ok': True, 'raw': str(payload)}


async def _tool_map() -> dict[str, Any]:
    client = MultiServerMCPClient({'y': {'transport': 'streamable-http', 'url': MCP_URL}})
    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}


async def _call_tool(tool_map: dict[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    payload = await tool_map[name].ainvoke(args)
    result = _norm(payload)
    return {
        'tool': name,
        'args': args,
        'duration_sec': round(time.time() - start, 3),
        'result': result,
    }


async def _build_provider(provider: str, model: str, *, session_id: str) -> Any:
    settings = AgentSettings(
        provider=provider,
        model=model,
        session_id=session_id,
        memory_root=str(MEM_ROOT),
        mcp_url=MCP_URL,
        ollama_url=OLLAMA_URL,
    )
    if provider == 'deepseek':
        settings.api_key = DEEPSEEK_API_KEY
        settings.base_url = 'https://api.deepseek.com'
    return await build_agent_client(settings)


async def _chat_case(agent: Any, prompt: str, *, approve_first: bool | None = None, approve_second: bool | None = None, label: str) -> dict[str, Any]:
    start = time.time()
    first = await agent.chat(prompt)
    record: dict[str, Any] = {
        'label': label,
        'prompt': prompt,
        'first': first,
        'duration_sec': None,
    }
    if first.get('status') == 'needs_confirmation' and approve_first is not None:
        second = await agent.confirm(first['thread_id'], approved=approve_first)
        record['second'] = second
        if second.get('status') == 'needs_confirmation' and approve_second is not None:
            third = await agent.confirm(second['thread_id'], approved=approve_second)
            record['third'] = third
    record['duration_sec'] = round(time.time() - start, 3)
    record['state'] = {
        'dataset_root': agent.session_state.active_dataset.dataset_root,
        'img_dir': agent.session_state.active_dataset.img_dir,
        'label_dir': agent.session_state.active_dataset.label_dir,
        'data_yaml': agent.session_state.active_dataset.data_yaml,
        'training_running': agent.session_state.active_training.running,
        'training_model': agent.session_state.active_training.model,
        'training_data_yaml': agent.session_state.active_training.data_yaml,
        'training_device': agent.session_state.active_training.device,
    }
    return record


async def main() -> None:
    MEM_ROOT.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        'dataset_root': DATASET_ROOT,
        'mcp_url': MCP_URL,
        'ollama_url': OLLAMA_URL,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tool_tests': [],
        'agent_tests': [],
    }

    tool_map = await _tool_map()

    # Tool-level tests
    out['tool_tests'].append(await _call_tool(tool_map, 'scan_dataset', {'img_dir': DATASET_ROOT}))
    out['tool_tests'].append(await _call_tool(tool_map, 'validate_dataset', {'img_dir': DATASET_ROOT}))
    out['tool_tests'].append(await _call_tool(tool_map, 'training_readiness', {'img_dir': DATASET_ROOT}))
    prepare_default = await _call_tool(tool_map, 'prepare_dataset_for_training', {'dataset_path': DATASET_ROOT})
    out['tool_tests'].append(prepare_default)

    prepared_yaml = ''
    prepare_result = prepare_default.get('result', {})
    if prepare_result.get('ok') and prepare_result.get('data_yaml'):
        prepared_yaml = str(prepare_result['data_yaml'])

    train_case: dict[str, Any] = {'tool': 'start_training_smoke', 'args': {}, 'duration_sec': None, 'result': {'ok': False, 'error': 'prepare did not yield trainable yaml'}}
    if prepared_yaml:
        start_ts = time.time()
        started = _norm(await tool_map['start_training'].ainvoke({'model': '/home/kly/yolov8n.pt', 'data_yaml': prepared_yaml, 'epochs': 2, 'device': 'auto'}))
        await asyncio.sleep(6)
        status = _norm(await tool_map['check_training_status'].ainvoke({}))
        stop = _norm(await tool_map['stop_training'].ainvoke({})) if status.get('running') else {'ok': True, 'summary': 'training already finished before stop'}
        train_case = {
            'tool': 'start_training_smoke',
            'args': {'model': '/home/kly/yolov8n.pt', 'data_yaml': prepared_yaml, 'epochs': 2, 'device': 'auto'},
            'duration_sec': round(time.time() - start_ts, 3),
            'result': {
                'start': started,
                'status_after_6s': status,
                'stop': stop,
            },
        }
    out['tool_tests'].append(train_case)

    # Agent tests - Gemma
    gemma = await _build_provider('ollama', 'gemma4:e4b', session_id=f'zyb-gemma-{uuid.uuid4().hex[:8]}')
    out['agent_tests'].append(await _chat_case(gemma, f'请扫描 {DATASET_ROOT}，并用中文总结图片数、标签数、缺失标签、类别情况。', label='gemma_scan_summary'))
    out['agent_tests'].append(await _chat_case(gemma, f'请判断 {DATASET_ROOT} 能不能直接训练，不要启动训练，只告诉我原因和下一步建议。', label='gemma_readiness_no_train'))
    out['agent_tests'].append(await _chat_case(gemma, f'{DATASET_ROOT} 的 labels 目录里可能混有 classes.txt。请检查这个数据集，判断这会不会影响训练准备，并给出处理建议，但不要启动训练。', label='gemma_classes_diagnostic'))
    out['agent_tests'].append(await _chat_case(gemma, f'数据在 {DATASET_ROOT}，按默认划分比例准备成可训练状态，但不要开始训练。', approve_first=True, label='gemma_prepare_only'))
    gemma_train = await _chat_case(gemma, f'数据在 {DATASET_ROOT}，按默认划分比例，然后用 yolov8n 模型训练 2 轮。', approve_first=True, approve_second=True, label='gemma_prepare_and_train')
    gemma_status = await gemma.direct_tool('check_training_status')
    gemma_stop = await gemma.direct_tool('stop_training') if gemma_status.get('running') else {'ok': True, 'summary': 'training already stopped'}
    gemma_train['post_tools'] = {'status': gemma_status, 'stop': gemma_stop}
    out['agent_tests'].append(gemma_train)

    # Agent test - DeepSeek compare on the hardest chain
    deepseek = await _build_provider('deepseek', 'deepseek-chat', session_id=f'zyb-deepseek-{uuid.uuid4().hex[:8]}')
    ds_train = await _chat_case(deepseek, f'数据在 {DATASET_ROOT}，按默认划分比例，然后用 yolov8n 模型训练 2 轮。', approve_first=True, approve_second=False, label='deepseek_prepare_then_cancel_train')
    out['agent_tests'].append(ds_train)

    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUTPUT))


if __name__ == '__main__':
    asyncio.run(main())
