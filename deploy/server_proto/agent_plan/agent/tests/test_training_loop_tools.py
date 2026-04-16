from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.tools.training_loop_tools as training_loop_tools


class _DummyLoopService:
    def start_loop(self, **kwargs):
        return {
            'ok': True,
            'summary': '环训练已启动',
            'loop_id': 'loop-123',
            'status': 'queued',
            'received': kwargs,
        }

    def check_loop_status(self, loop_id: str = ''):
        return {
            'ok': True,
            'summary': '环训练状态已就绪',
            'loop_id': loop_id or 'loop-123',
            'status': 'running_round',
        }


def main() -> None:
    original_service = training_loop_tools.service
    assert getattr(original_service, 'loop_llm', None) is None, getattr(original_service, 'loop_llm', None)
    training_loop_tools.service = _DummyLoopService()
    try:
        started = training_loop_tools.start_training_loop(model='yolov8n.pt', data_yaml='data.yaml')
        assert started['ok'] is True
        assert started['loop_id'] == 'loop-123'
        assert started['received']['epochs'] == training_loop_tools._DEFAULT_LOOP_EPOCHS
        checked = training_loop_tools.check_training_loop_status('loop-123')
        assert checked['ok'] is True
        assert checked['status'] == 'running_round'
        print('training loop tools ok')
    finally:
        training_loop_tools.service = original_service


if __name__ == '__main__':
    main()
