from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langchain_mcp_adapters.client import MultiServerMCPClient
from agent_plan.agent.client.agent_client import AgentSettings, build_agent_client

OUTPUT_PATH = Path(__file__).with_name('test_comprehensive_validation_output.json')


def normalize_tool_result(payload):
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        texts = []
        for item in payload:
            if isinstance(item, dict):
                texts.append(item.get('text', ''))
            else:
                texts.append(str(item))
        raw = '\n'.join(part for part in texts if part).strip()
        if not raw:
            return {'ok': True, 'raw': raw}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {'ok': True, 'value': parsed}
        except Exception:
            return {'ok': True, 'raw': raw}
    return {'ok': True, 'raw': str(payload)}


async def call_tool(tool, args: dict) -> dict:
    return normalize_tool_result(await tool.ainvoke(args))


async def wait_for_train_finish(tool, timeout_sec: int = 240, interval_sec: int = 8) -> dict:
    history: list[dict] = []
    started = time.time()
    while time.time() - started < timeout_sec:
        status = await call_tool(tool, {})
        record = {
            'ts': round(time.time() - started, 1),
            'running': status.get('running'),
            'device': status.get('device'),
            'summary': status.get('summary'),
            'return_code': status.get('return_code'),
            'latest_metrics': status.get('latest_metrics'),
        }
        history.append(record)
        if not status.get('running'):
            return {'final': status, 'history': history, 'timed_out': False}
        await asyncio.sleep(interval_sec)
    status = await call_tool(tool, {})
    history.append({
        'ts': round(time.time() - started, 1),
        'running': status.get('running'),
        'device': status.get('device'),
        'summary': status.get('summary'),
        'return_code': status.get('return_code'),
        'latest_metrics': status.get('latest_metrics'),
    })
    return {'final': status, 'history': history, 'timed_out': True}


async def ensure_idle(status_tool, stop_tool, result: dict) -> None:
    status = await call_tool(status_tool, {})
    result['precheck_training_status'] = status
    if status.get('running'):
        result['precheck_stop_existing'] = await call_tool(stop_tool, {})
        await asyncio.sleep(3)
        result['precheck_training_status_after_stop'] = await call_tool(status_tool, {})


async def main() -> None:
    result: dict = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tool_checks': {},
        'train_runs': {},
        'agent_runs': {},
        'findings': [],
    }

    client = MultiServerMCPClient({
        'yolostudio': {
            'transport': 'streamable-http',
            'url': 'http://127.0.0.1:8080/mcp',
        }
    })
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}
    result['tool_checks']['tool_names'] = sorted(tool_map.keys())

    start_tool = tool_map['start_training']
    status_tool = tool_map['check_training_status']
    stop_tool = tool_map['stop_training']
    await ensure_idle(status_tool, stop_tool, result['tool_checks'])

    # 1) 基础数据集验证
    result['tool_checks']['scan_test_dataset_root'] = await call_tool(tool_map['scan_dataset'], {
        'img_dir': '/home/kly/test_dataset',
    })
    result['tool_checks']['scan_test_dataset_explicit'] = await call_tool(tool_map['scan_dataset'], {
        'img_dir': '/home/kly/test_dataset/images',
        'label_dir': '/home/kly/test_dataset/labels',
    })
    result['tool_checks']['validate_test_dataset_explicit'] = await call_tool(tool_map['validate_dataset'], {
        'img_dir': '/home/kly/test_dataset/images',
        'label_dir': '/home/kly/test_dataset/labels',
    })
    result['tool_checks']['readiness_test_dataset'] = await call_tool(tool_map['training_readiness'], {
        'img_dir': '/home/kly/test_dataset/images',
        'label_dir': '/home/kly/test_dataset/labels',
    })
    result['tool_checks']['gpu_status'] = await call_tool(tool_map['check_gpu_status'], {})
    result['tool_checks']['scan_dataset_1ch'] = await call_tool(tool_map['scan_dataset'], {
        'img_dir': '/home/kly/dataset_1ch/images',
        'label_dir': '/home/kly/dataset_1ch/labels',
    })
    result['tool_checks']['readiness_dataset_1ch'] = await call_tool(tool_map['training_readiness'], {
        'img_dir': '/home/kly/dataset_1ch/images',
        'label_dir': '/home/kly/dataset_1ch/labels',
        'data_yaml': '/home/kly/dataset_1ch.yaml',
    })

    # 2) 多轮训练：直接 tool 路径
    run1 = await call_tool(start_tool, {
        'model': '/home/kly/yolov8n.pt',
        'data_yaml': '/home/kly/test_dataset/data.yaml',
        'epochs': 4,
        'device': 'auto',
    })
    result['train_runs']['tool_run_small_4epochs_start'] = run1
    if run1.get('ok'):
        wait1 = await wait_for_train_finish(status_tool, timeout_sec=240, interval_sec=8)
        result['train_runs']['tool_run_small_4epochs_status'] = wait1
        if wait1['timed_out'] and wait1['final'].get('running'):
            result['train_runs']['tool_run_small_4epochs_stop'] = await call_tool(stop_tool, {})
    else:
        result['findings'].append({'type': 'tool_run_small_4epochs_start_failed', 'detail': run1})

    run2 = await call_tool(start_tool, {
        'model': '/home/kly/yolov8n.pt',
        'data_yaml': '/home/kly/test_dataset/data.yaml',
        'epochs': 6,
        'device': 'auto',
    })
    result['train_runs']['tool_run_small_6epochs_start'] = run2
    if run2.get('ok'):
        await asyncio.sleep(12)
        mid_status = await call_tool(status_tool, {})
        result['train_runs']['tool_run_small_6epochs_mid_status'] = mid_status
        result['train_runs']['tool_run_small_6epochs_stop'] = await call_tool(stop_tool, {})
        result['train_runs']['tool_run_small_6epochs_after_stop'] = await call_tool(status_tool, {})
    else:
        result['findings'].append({'type': 'tool_run_small_6epochs_start_failed', 'detail': run2})

    # 3) Agent 路径：DeepSeek 多轮实际使用
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise RuntimeError('DEEPSEEK_API_KEY 未设置')

    await ensure_idle(status_tool, stop_tool, result['agent_runs'])
    settings = AgentSettings(
        provider='deepseek',
        model='deepseek-chat',
        session_id='realworld-validation',
    )
    agent = await build_agent_client(settings)

    turns = [
        '请扫描 /home/kly/test_dataset/images 和 /home/kly/test_dataset/labels。',
        '这个数据集现在能不能直接训练？如果不能请说明缺什么。',
        '请先调用 check_gpu_status 再告诉我现在是否适合训练。',
        '请用 /home/kly/yolov8n.pt 基于刚才的数据训练 3 轮。',
        '现在训练状态怎么样？',
    ]
    agent_records: list[dict] = []
    for text in turns:
        reply = await agent.chat(text)
        rec = {
            'user': text,
            'status': reply['status'],
            'message': reply['message'],
            'tool_call': reply.get('tool_call'),
            'data_yaml': agent.session_state.active_dataset.data_yaml,
            'img_dir': agent.session_state.active_dataset.img_dir,
            'label_dir': agent.session_state.active_dataset.label_dir,
            'pending_tool': agent.session_state.pending_confirmation.tool_name,
        }
        agent_records.append(rec)
        if reply['status'] == 'needs_confirmation':
            confirmed = await agent.confirm(reply['thread_id'], approved=True)
            agent_records.append({
                'user': 'y',
                'status': confirmed['status'],
                'message': confirmed['message'],
                'tool_call': confirmed.get('tool_call'),
                'data_yaml': agent.session_state.active_dataset.data_yaml,
                'img_dir': agent.session_state.active_dataset.img_dir,
                'label_dir': agent.session_state.active_dataset.label_dir,
                'pending_tool': agent.session_state.pending_confirmation.tool_name,
            })
    result['agent_runs']['deepseek_training_flow'] = agent_records

    agent_wait = await wait_for_train_finish(status_tool, timeout_sec=180, interval_sec=8)
    result['agent_runs']['deepseek_training_flow_status'] = agent_wait
    if agent_wait['timed_out'] and agent_wait['final'].get('running'):
        result['agent_runs']['deepseek_training_flow_stop'] = await call_tool(stop_tool, {})

    # 4) 自动 finding
    root_total = result['tool_checks']['scan_test_dataset_root'].get('total_images')
    explicit_total = result['tool_checks']['scan_test_dataset_explicit'].get('total_images')
    if isinstance(root_total, int) and isinstance(explicit_total, int) and root_total != explicit_total:
        result['findings'].append({
            'type': 'scan_root_ambiguity',
            'root_total_images': root_total,
            'explicit_total_images': explicit_total,
        })

    run1_final = result['train_runs'].get('tool_run_small_4epochs_status', {}).get('final', {})
    if run1_final.get('running') is False and run1_final.get('latest_metrics') in (None, {}):
        result['findings'].append({'type': 'latest_metrics_missing_after_training', 'run': 'tool_run_small_4epochs'})

    run2_mid = result['train_runs'].get('tool_run_small_6epochs_mid_status', {})
    if run2.get('ok') and run2_mid.get('latest_metrics') in (None, {}):
        result['findings'].append({'type': 'latest_metrics_missing_mid_training', 'run': 'tool_run_small_6epochs'})

    last_agent_record = result['agent_runs']['deepseek_training_flow'][-1] if result['agent_runs']['deepseek_training_flow'] else {}
    if last_agent_record.get('message') and '0,1' in last_agent_record.get('message', ''):
        result['findings'].append({'type': 'agent_free_style_gpu_rule', 'message': last_agent_record['message']})

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT_PATH), 'finding_count': len(result['findings'])}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
