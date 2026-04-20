from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.session_state import SESSION_STATE_SCHEMA_VERSION


WORK = Path(__file__).resolve().parent / '_tmp_memory_store_schema_migration'


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    try:
        store = MemoryStore(WORK)
        session_path = store.sessions_dir / 'legacy-session.json'
        session_path.write_text(
            json.dumps(
                {
                    'created_at': '2026-04-17T00:00:00+00:00',
                    'updated_at': '2026-04-17T00:00:00+00:00',
                    'active_training': {
                        'workflow_state': 'pending_confirmation',
                        'training_plan_draft': ['legacy-stale-draft'],
                    },
                    'pending_confirmation': {
                        'tool_name': 'start_training',
                        'tool_args': {'model': 'yolov8n.pt'},
                        'allowed_decisions': 'approve',
                        'review_config': ['legacy-review'],
                        'decision_context': ['legacy-context'],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

        state = store.load_state('legacy-session')
        assert state.session_id == 'legacy-session'
        assert state.schema_version == SESSION_STATE_SCHEMA_VERSION
        assert not hasattr(state, 'pending_confirmation')
        assert not hasattr(state.active_training, 'training_plan_draft')

        rewritten = json.loads(session_path.read_text(encoding='utf-8'))
        assert rewritten['session_id'] == 'legacy-session'
        assert rewritten['schema_version'] == SESSION_STATE_SCHEMA_VERSION
        assert 'pending_confirmation' not in rewritten
        assert 'training_plan_draft' not in rewritten['active_training']
        events = store.read_events('legacy-session')
        assert any(
            event.get('type') == 'state_schema_migrated'
            and event.get('from_version') == 1
            and event.get('to_version') == SESSION_STATE_SCHEMA_VERSION
            for event in events
        ), events
        print('memory store schema migration ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
