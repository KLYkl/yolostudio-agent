from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from langchain_mcp_adapters.client import MultiServerMCPClient
from agent_plan.agent.client.agent_client import AgentSettings, build_agent_client


def _norm(payload):
    if isinstance(payload, dict):
        return payload
    raw = '\n'.join((item.get('text', '') if isinstance(item, dict) else str(item)) for item in payload).strip()
    return json.loads(raw)


async def main() -> None:
    provider = os.getenv('YOLOSTUDIO_TEST_PROVIDER', 'ollama')
    model = os.getenv('YOLOSTUDIO_TEST_MODEL', 'gemma4:e4b')
    session_id = f'test-complex-flow-{provider}-{uuid.uuid4().hex[:8]}'
    settings = AgentSettings(provider=provider, model=model, session_id=session_id)
    if provider != 'ollama':
        settings.base_url = os.getenv('YOLOSTUDIO_LLM_BASE_URL', settings.base_url)
        settings.api_key = os.getenv('YOLOSTUDIO_LLM_API_KEY', settings.api_key)
    agent = await build_agent_client(settings)
    first = await agent.chat('数据在 /home/kly/test_dataset/，按默认划分比例，然后用yolov8n模型进行训练')
    print('first', json.dumps(first, ensure_ascii=False))
    if first.get('status') != 'needs_confirmation':
        raise SystemExit('expected first confirmation')

    second = await agent.confirm(first['thread_id'], approved=True)
    print('second', json.dumps(second, ensure_ascii=False))
    if second.get('status') != 'needs_confirmation' or (second.get('tool_call') or {}).get('name') != 'start_training':
        raise SystemExit('expected second confirmation for start_training')

    third = await agent.confirm(second['thread_id'], approved=False)
    print('third', json.dumps(third, ensure_ascii=False))
    if third.get('status') != 'cancelled':
        raise SystemExit('expected cancellation path for start_training')

    state = {
        'dataset_root': agent.session_state.active_dataset.dataset_root,
        'img_dir': agent.session_state.active_dataset.img_dir,
        'label_dir': agent.session_state.active_dataset.label_dir,
        'data_yaml': agent.session_state.active_dataset.data_yaml,
    }
    print('state', json.dumps(state, ensure_ascii=False))

    client = MultiServerMCPClient({'y': {'transport': 'streamable-http', 'url': 'http://127.0.0.1:8080/mcp'}})
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}
    status = _norm(await tool_map['check_training_status'].ainvoke({}))
    print('status', json.dumps(status, ensure_ascii=False))
    if status.get('running'):
        stop = _norm(await tool_map['stop_training'].ainvoke({}))
        print('stop', json.dumps(stop, ensure_ascii=False))

    print(f'complex prompt flow smoke ok ({provider})')


if __name__ == '__main__':
    asyncio.run(main())
