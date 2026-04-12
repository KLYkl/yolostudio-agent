from __future__ import annotations

import shutil
from pathlib import Path
import sys
import types

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(r'C:\workspace\yolodo2.0\agent_plan\agent\tests\_tmp_training_state_purity')


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
            'resolved_args': {'model': '/tmp/yolov8n.pt', 'data_yaml': '/tmp/data.yaml'},
            'command': ['yolo', 'train', 'model=/tmp/yolov8n.pt', 'data=/tmp/data.yaml', 'device=1'],
            'summary': '当前没有在训练',
            'run_state': 'completed',
        })

        tr = client.session_state.active_training
        assert tr.running is False
        assert tr.model == '/tmp/yolov8n.pt'
        assert tr.data_yaml == '/tmp/data.yaml'
        assert tr.device == '1'
        assert tr.pid is None
        assert tr.log_file == '/tmp/train.log'
        assert tr.started_at == 123.4
        assert tr.last_status.get('running') is False

        print('training state purity smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
