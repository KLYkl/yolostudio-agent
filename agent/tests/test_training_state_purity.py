from __future__ import annotations

import shutil
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(r'D:\yolodo2.0\agent_plan\agent\tests\_tmp_training_state_purity')


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='purity-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})

        assert client.session_state.active_training.model == ''
        assert client.session_state.active_training.data_yaml == ''
        assert client.session_state.active_training.device == ''

        client._apply_to_state('check_training_status', {
            'ok': True,
            'running': False,
            'device': '1',
            'pid': 4321,
            'log_file': '/tmp/train.log',
            'started_at': 123.4,
            'command': ['yolo', 'train', 'model=/tmp/yolov8n.pt', 'data=/tmp/data.yaml', 'device=1'],
            'summary': '当前没有在训练',
        })

        tr = client.session_state.active_training
        assert tr.running is False
        assert tr.model == ''
        assert tr.data_yaml == ''
        assert tr.device == ''
        assert tr.pid is None
        assert tr.log_file == ''
        assert tr.started_at is None
        assert tr.last_status.get('running') is False

        print('training state purity smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
