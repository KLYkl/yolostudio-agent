from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None  # type: ignore[assignment]

DEFAULT_OUT = str(Path(__file__).with_name('test_zyb_long_training_lifecycle_output.json'))
DEFAULT_DATASET_ROOT = '/data/example_dataset'
DEFAULT_MODEL_PATH = '/models/yolov8n.pt'
DEFAULT_EPOCHS = 30
DEFAULT_STATUS_DELAYS = [15, 35, 60]
DEFAULT_MCP_URL = 'http://127.0.0.1:8080/mcp'
DEFAULT_TEST_MODE = 'mcp'
DEFAULT_TARGET_EPOCH = 2
DEFAULT_EXTRA_POLL_INTERVAL = 30
DEFAULT_EXTRA_POLL_LIMIT = 8


def _norm(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        raw = '\n'.join((item.get('text', '') if isinstance(item, dict) else str(item)) for item in payload).strip()
        return json.loads(raw) if raw else {'ok': True}
    return {'ok': True, 'raw': str(payload)}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    return int(value)


def _env_csv_ints(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def _build_direct_tool_map() -> dict[str, Any]:
    import yolostudio_agent.agent.server.tools.combo_tools as combo_tools
    import yolostudio_agent.agent.server.tools.data_tools as data_tools
    import yolostudio_agent.agent.server.tools.knowledge_tools as knowledge_tools
    import yolostudio_agent.agent.server.tools.train_tools as train_tools

    mapping = {
        'training_readiness': data_tools.training_readiness,
        'prepare_dataset_for_training': combo_tools.prepare_dataset_for_training,
        'start_training': train_tools.start_training,
        'check_training_status': train_tools.check_training_status,
        'stop_training': train_tools.stop_training,
        'summarize_training_run': train_tools.summarize_training_run,
        'analyze_training_outcome': knowledge_tools.analyze_training_outcome,
        'recommend_next_training_step': knowledge_tools.recommend_next_training_step,
    }
    return mapping


async def _invoke_tool(tool_map: dict[str, Any], tool_name: str, kwargs: dict[str, Any], test_mode: str) -> dict[str, Any]:
    if test_mode == 'direct_tools':
        return _norm(tool_map[tool_name](**kwargs))
    return _norm(await tool_map[tool_name].ainvoke(kwargs))


async def main() -> None:
    out_path = Path(os.environ.get('YOLO_TRAIN_OUT', DEFAULT_OUT))
    dataset_root = os.environ.get('YOLO_TRAIN_DATASET_ROOT', DEFAULT_DATASET_ROOT).strip() or DEFAULT_DATASET_ROOT
    model_path = os.environ.get('YOLO_TRAIN_MODEL_PATH', DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH
    epochs = _env_int('YOLO_TRAIN_EPOCHS', DEFAULT_EPOCHS)
    status_delays = _env_csv_ints('YOLO_TRAIN_STATUS_DELAYS', DEFAULT_STATUS_DELAYS)
    mcp_url = os.environ.get('YOLOSTUDIO_MCP_URL', DEFAULT_MCP_URL).strip() or DEFAULT_MCP_URL
    test_mode = os.environ.get('YOLO_TRAIN_TEST_MODE', DEFAULT_TEST_MODE).strip() or DEFAULT_TEST_MODE
    target_epoch = _env_int('YOLO_TRAIN_TARGET_EPOCH', DEFAULT_TARGET_EPOCH)
    extra_poll_interval = _env_int('YOLO_TRAIN_EXTRA_POLL_INTERVAL', DEFAULT_EXTRA_POLL_INTERVAL)
    extra_poll_limit = _env_int('YOLO_TRAIN_EXTRA_POLL_LIMIT', DEFAULT_EXTRA_POLL_LIMIT)

    if test_mode == 'direct_tools':
        tool_map = _build_direct_tool_map()
    else:
        if MultiServerMCPClient is None:
            raise RuntimeError('当前环境缺少 langchain_mcp_adapters，无法以 mcp 模式运行')
        client = MultiServerMCPClient({'y': {'transport': 'streamable-http', 'url': mcp_url}})
        tools = await client.get_tools()
        tool_map = {tool.name: tool for tool in tools}

    pre_status = await _invoke_tool(tool_map, 'check_training_status', {}, test_mode)
    pre_stop = None
    if pre_status.get('running'):
        pre_stop = await _invoke_tool(tool_map, 'stop_training', {}, test_mode)
        await asyncio.sleep(3)

    readiness = await _invoke_tool(tool_map, 'training_readiness', {'img_dir': dataset_root}, test_mode)
    prepare = await _invoke_tool(tool_map, 'prepare_dataset_for_training', {'dataset_path': dataset_root}, test_mode)
    if not prepare.get('ok') or not prepare.get('ready'):
        payload = {
            'ok': False,
            'stage': 'prepare',
            'test_mode': test_mode,
            'mcp_url': mcp_url,
            'dataset_root': dataset_root,
            'model_path': model_path,
            'epochs': epochs,
            'status_delays': status_delays,
            'pre_status': pre_status,
            'pre_stop': pre_stop,
            'readiness': readiness,
            'prepare': prepare,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'ok': False, 'stage': 'prepare', 'output': str(out_path)}, ensure_ascii=False))
        return

    start = await _invoke_tool(tool_map, 'start_training', {
        'model': model_path,
        'data_yaml': prepare['data_yaml'],
        'epochs': epochs,
        'device': 'auto',
    }, test_mode)

    statuses: list[dict[str, Any]] = []
    for delay in status_delays:
        await asyncio.sleep(delay)
        statuses.append(await _invoke_tool(tool_map, 'check_training_status', {}, test_mode))

    extra_polls = 0
    while extra_polls < extra_poll_limit:
        observed_epochs = [int((item.get('progress') or {}).get('epoch') or 0) for item in statuses]
        max_observed_epoch = max(observed_epochs or [0])
        latest_status = statuses[-1] if statuses else {}
        if max_observed_epoch >= target_epoch or not latest_status.get('running'):
            break
        await asyncio.sleep(extra_poll_interval)
        statuses.append(await _invoke_tool(tool_map, 'check_training_status', {}, test_mode))
        extra_polls += 1

    stop = None
    final_status = None
    if statuses and statuses[-1].get('running'):
        stop = await _invoke_tool(tool_map, 'stop_training', {}, test_mode)
        await asyncio.sleep(3)
        final_status = await _invoke_tool(tool_map, 'check_training_status', {}, test_mode)
    else:
        final_status = statuses[-1] if statuses else await _invoke_tool(tool_map, 'check_training_status', {}, test_mode)

    summary = await _invoke_tool(tool_map, 'summarize_training_run', {}, test_mode)
    analysis = await _invoke_tool(tool_map, 'analyze_training_outcome', {'metrics': summary}, test_mode)
    next_step = await _invoke_tool(tool_map, 'recommend_next_training_step', {
        'readiness': readiness,
        'status': summary,
    }, test_mode)

    observed_epochs = [int((item.get('progress') or {}).get('epoch') or 0) for item in statuses]
    running_count = sum(1 for item in statuses if item.get('running'))
    payload = {
        'ok': True,
        'test_mode': test_mode,
        'mcp_url': mcp_url,
        'dataset_root': dataset_root,
        'model_path': model_path,
        'epochs': epochs,
        'status_delays': status_delays,
        'target_epoch': target_epoch,
        'extra_poll_interval': extra_poll_interval,
        'extra_poll_limit': extra_poll_limit,
        'pre_status': pre_status,
        'pre_stop': pre_stop,
        'readiness': readiness,
        'prepare': prepare,
        'start': start,
        'statuses': statuses,
        'stop': stop,
        'final_status': final_status,
        'summary_run': summary,
        'analysis': analysis,
        'next_step': next_step,
        'assessment': {
            'readiness_ok': readiness.get('ok') is True,
            'readiness_ready': readiness.get('ready') is True,
            'prepare_ready': prepare.get('ready') is True,
            'start_ok': start.get('ok') is True,
            'running_observed_count': running_count,
            'status_window_seconds': sum(status_delays),
            'extra_polls': extra_polls,
            'max_observed_epoch': max(observed_epochs or [0]),
            'summary_ok': summary.get('ok') is True,
            'summary_run_state': summary.get('run_state'),
            'analysis_ok': analysis.get('ok') is True,
            'next_step_ok': next_step.get('ok') is True,
            'stopped_after_manual_stop': (stop or {}).get('ok') is True and (final_status or {}).get('running') is False if stop else None,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'epochs': epochs,
        'running_observed_count': running_count,
        'final_running': (final_status or {}).get('running'),
        'max_observed_epoch': max(observed_epochs or [0]),
        'summary_run_state': summary.get('run_state'),
        'recommended_action': next_step.get('recommended_action'),
        'output': str(out_path),
    }, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
