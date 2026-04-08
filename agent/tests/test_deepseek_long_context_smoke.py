from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, build_agent_client

SESSION_ID = 'deepseek-long-context-smoke'

USER_TURNS = [
    '请扫描 /home/kly/dataset_1ch/images 和 /home/kly/dataset_1ch/labels。',
    '这个数据集质量怎么样？请先校验再回答。',
    '现在这个数据集一共有多少张图？',
    '类别分布里最多的是哪两类？',
    '再告诉我一次当前数据集的图片目录和标签目录。',
    '这个数据集现在能不能直接训练？请给我结论。',
    '请调用 check_gpu_status 工具并告诉我现在哪张卡空闲。',
    '当前是否已经有训练在跑？',
    '请总结一下到目前为止你已经知道的上下文。',
    '请用 /home/kly/yolov8n.pt 基于刚才的数据训练1轮。',
    '如果我取消，会发生什么？',
    '重新说一下刚才待确认的训练参数。',
    '请再回答一次：现在这个数据集是哪一个？',
    '这个数据集之前校验有没有发现问题？',
    '如果现在开始训练，你预计会用哪张 GPU？',
    '请不要调用工具，直接用你记住的上下文回答：当前数据集图片目录是什么？',
    '同样不要调用工具，直接说最近一次扫描的总图片数。',
    '再说一下最近一次校验的问题数。',
    '我们刚才有没有进入过人工确认？',
    '请最后把当前会话的关键状态整理成 5 条短句。',
]


async def main() -> None:
    settings = AgentSettings(provider='deepseek', model='deepseek-chat', session_id=SESSION_ID)
    agent = await build_agent_client(settings)
    results: list[dict] = []

    for idx, user_text in enumerate(USER_TURNS, start=1):
        result = await agent.chat(user_text)
        record = {
            'turn': idx,
            'user': user_text,
            'status': result['status'],
            'message': result['message'],
            'tool_call': result.get('tool_call'),
            'history_len': len(agent.messages),
            'dataset_img_dir': agent.session_state.active_dataset.img_dir,
            'dataset_label_dir': agent.session_state.active_dataset.label_dir,
            'data_yaml': agent.session_state.active_dataset.data_yaml,
            'last_scan': agent.session_state.active_dataset.last_scan,
            'last_validate': agent.session_state.active_dataset.last_validate,
            'pending_tool': agent.session_state.pending_confirmation.tool_name,
        }
        results.append(record)
        if result['status'] == 'needs_confirmation':
            cancelled = await agent.confirm(result['thread_id'], approved=False)
            results.append({
                'turn': f'{idx}-confirm-cancel',
                'user': 'n',
                'status': cancelled['status'],
                'message': cancelled['message'],
                'tool_call': cancelled.get('tool_call'),
                'history_len': len(agent.messages),
                'dataset_img_dir': agent.session_state.active_dataset.img_dir,
                'dataset_label_dir': agent.session_state.active_dataset.label_dir,
                'data_yaml': agent.session_state.active_dataset.data_yaml,
                'last_scan': agent.session_state.active_dataset.last_scan,
                'last_validate': agent.session_state.active_dataset.last_validate,
                'pending_tool': agent.session_state.pending_confirmation.tool_name,
            })
        if idx == 10:
            agent = await build_agent_client(settings)
            results.append({
                'turn': 'reload-after-10',
                'user': '[reload]',
                'status': 'reloaded',
                'message': 'rebuilt agent client from persisted session',
                'tool_call': None,
                'history_len': len(agent.messages),
                'dataset_img_dir': agent.session_state.active_dataset.img_dir,
                'dataset_label_dir': agent.session_state.active_dataset.label_dir,
                'data_yaml': agent.session_state.active_dataset.data_yaml,
                'last_scan': agent.session_state.active_dataset.last_scan,
                'last_validate': agent.session_state.active_dataset.last_validate,
                'pending_tool': agent.session_state.pending_confirmation.tool_name,
            })

    memory_root = Path(settings.memory_root)
    session_path = memory_root / 'sessions' / f'{SESSION_ID}.json'
    event_path = memory_root / 'events' / f'{SESSION_ID}.jsonl'
    summary = {
        'session_path': str(session_path),
        'event_path': str(event_path),
        'session_exists': session_path.exists(),
        'event_exists': event_path.exists(),
        'session_size': session_path.stat().st_size if session_path.exists() else 0,
        'event_lines': sum(1 for _ in event_path.open('r', encoding='utf-8')) if event_path.exists() else 0,
        'final_history_len': len(agent.messages),
    }
    print(json.dumps({'results': results, 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
