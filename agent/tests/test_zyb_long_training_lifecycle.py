from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

OUT = Path(r'D:\yolodo2.0\agent_plan\agent\tests\test_zyb_long_training_lifecycle_output.json')
DATASET_ROOT = '/home/kly/agent_cap_tests/zyb'
MODEL_PATH = '/home/kly/yolov8n.pt'
EPOCHS = 30
STATUS_DELAYS = [15, 20, 20]


def _norm(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        raw = '\n'.join((item.get('text', '') if isinstance(item, dict) else str(item)) for item in payload).strip()
        return json.loads(raw) if raw else {'ok': True}
    return {'ok': True, 'raw': str(payload)}


async def main() -> None:
    client = MultiServerMCPClient({'y': {'transport': 'streamable-http', 'url': 'http://127.0.0.1:8080/mcp'}})
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}

    pre_status = _norm(await tool_map['check_training_status'].ainvoke({}))
    pre_stop = None
    if pre_status.get('running'):
        pre_stop = _norm(await tool_map['stop_training'].ainvoke({}))
        await asyncio.sleep(3)

    prepare = _norm(await tool_map['prepare_dataset_for_training'].ainvoke({'dataset_path': DATASET_ROOT}))
    if not prepare.get('ok') or not prepare.get('ready'):
        payload = {
            'ok': False,
            'stage': 'prepare',
            'pre_status': pre_status,
            'pre_stop': pre_stop,
            'prepare': prepare,
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'ok': False, 'stage': 'prepare', 'output': str(OUT)}, ensure_ascii=False))
        return

    start = _norm(await tool_map['start_training'].ainvoke({
        'model': MODEL_PATH,
        'data_yaml': prepare['data_yaml'],
        'epochs': EPOCHS,
        'device': 'auto',
    }))

    statuses: list[dict[str, Any]] = []
    for delay in STATUS_DELAYS:
        await asyncio.sleep(delay)
        statuses.append(_norm(await tool_map['check_training_status'].ainvoke({})))

    stop = None
    final_status = None
    if statuses and statuses[-1].get('running'):
        stop = _norm(await tool_map['stop_training'].ainvoke({}))
        await asyncio.sleep(3)
        final_status = _norm(await tool_map['check_training_status'].ainvoke({}))
    else:
        final_status = statuses[-1] if statuses else _norm(await tool_map['check_training_status'].ainvoke({}))

    running_count = sum(1 for item in statuses if item.get('running'))
    payload = {
        'ok': True,
        'dataset_root': DATASET_ROOT,
        'epochs': EPOCHS,
        'pre_status': pre_status,
        'pre_stop': pre_stop,
        'prepare': prepare,
        'start': start,
        'statuses': statuses,
        'stop': stop,
        'final_status': final_status,
        'assessment': {
            'prepare_ready': prepare.get('ready') is True,
            'start_ok': start.get('ok') is True,
            'running_observed_count': running_count,
            'status_window_seconds': sum(STATUS_DELAYS),
            'stopped_after_manual_stop': (stop or {}).get('ok') is True and (final_status or {}).get('running') is False if stop else None,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'epochs': EPOCHS,
        'running_observed_count': running_count,
        'final_running': (final_status or {}).get('running'),
        'output': str(OUT),
    }, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
