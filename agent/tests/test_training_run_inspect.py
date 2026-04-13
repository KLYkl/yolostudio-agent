from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.train_service import TrainService


WORK = Path(__file__).resolve().parent / '_tmp_training_run_inspect'

COMPLETED_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.19G      3.267      4.407      2.853          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.420        0.55     0.365    0.210
3 epochs completed in 0.001 hours.
Results saved to /tmp/runs/train100
'''

RUNNING_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        2/10      2.13G      3.076      3.867      2.729         35        640: 100% ━━━━━━━━━━━━ 2/2 2.2s
'''


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _spawn_sleep(seconds: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'])


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        completed_log = WORK / 'train_log_100.txt'
        completed_log.write_text(COMPLETED_LOG, encoding='utf-8')
        running_log = WORK / 'train_log_200.txt'
        running_log.write_text(RUNNING_LOG, encoding='utf-8')

        now = time.time()
        proc = _spawn_sleep(60)
        _write_json(
            WORK / 'active_train_job.json',
            {
                'pid': proc.pid,
                'log_file': str(running_log),
                'started_at': now - 20,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=10', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 10, 'device': '1'},
                'running': True,
                'updated_at': now - 1,
            },
        )
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 90000,
                'log_file': str(completed_log),
                'started_at': now - 120,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=3', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 3, 'device': '1'},
                'running': False,
                'return_code': 0,
                'forced': False,
                'stop_reason': 'process_exit',
                'updated_at': now - 10,
                'stopped_at': now - 10,
            },
        )

        service = TrainService(state_dir=WORK)

        latest = service.inspect_training_run()
        assert latest['ok'] is True
        assert latest['selected_run_id'] == 'train_log_200'
        assert latest['run_state'] == 'running'
        assert latest['observation_stage'] == 'early'

        explicit = service.inspect_training_run('train_log_100')
        assert explicit['ok'] is True
        assert explicit['selected_run_id'] == 'train_log_100'
        assert explicit['run_state'] == 'completed'
        assert explicit['status_source'] == 'last_run'
        assert explicit['analysis_ready'] is True
        assert explicit['progress']['epoch'] == 3
        assert explicit['save_dir'] == '/tmp/runs/train100'

        missing = service.inspect_training_run('train_log_missing')
        assert missing['ok'] is False
        assert '未找到对应训练记录' in missing['summary']
        print('training run inspect ok')
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
