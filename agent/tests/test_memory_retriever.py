from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.event_retriever import EventRetriever
from agent_plan.agent.client.memory_store import MemoryStore
from agent_plan.agent.client.session_state import SessionState


def main() -> None:
    root = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_memory_retriever_test')
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        store = MemoryStore(root)
        state = SessionState(session_id='memory-retriever-smoke')
        state.active_dataset.img_dir = '/data/images'
        state.active_dataset.label_dir = '/data/labels'
        state.active_dataset.last_scan = {
            'total_images': 128,
            'missing_labels': 0,
            'empty_labels': 1,
            'summary': '128 张图，1 个空标签',
        }
        state.active_dataset.last_validate = {
            'issue_count': 1,
            'has_issues': True,
        }
        state.active_training.model = '/models/yolov8n.pt'
        state.active_training.data_yaml = '/data/data.yaml'
        state.active_training.device = '1'
        state.active_training.last_status = {
            'running': False,
            'device': '1',
        }
        store.save_state(state)
        store.append_event(state.session_id, 'tool_result', {
            'tool': 'scan_dataset',
            'args': {'img_dir': '/data/images', 'label_dir': '/data/labels'},
            'result': {'ok': True, 'summary': '128 张图，1 个空标签'},
        })
        store.append_event(state.session_id, 'tool_result', {
            'tool': 'validate_dataset',
            'args': {'img_dir': '/data/images', 'label_dir': '/data/labels'},
            'result': {'ok': True, 'issue_count': 1, 'has_issues': True},
        })
        store.append_event(state.session_id, 'confirmation_requested', {
            'tool': 'start_training',
            'args': {'model': '/models/yolov8n.pt', 'epochs': 1},
            'thread_id': 't-1',
        })
        store.append_event(state.session_id, 'confirmation_cancelled', {
            'tool': 'start_training',
            'args': {'model': '/models/yolov8n.pt', 'epochs': 1},
        })

        retriever = EventRetriever(store)
        digest = retriever.build_digest(state.session_id, state)
        text = digest.to_text()

        assert '最近扫描' in text
        assert '最近校验' in text
        assert '人工确认记录' in text
        assert 'scan_dataset' in text
        assert 'validate_dataset' in text
        assert len(digest.recent_events) == 4

        recent = store.read_events(state.session_id, limit=2)
        assert len(recent) == 2
        assert recent[-1]['type'] == 'confirmation_cancelled'
        assert store.latest_event(state.session_id, 'tool_result', tool_name='validate_dataset') is not None

        print('memory retriever smoke ok')
    finally:
        if root.exists():
            shutil.rmtree(root)


if __name__ == '__main__':
    main()
