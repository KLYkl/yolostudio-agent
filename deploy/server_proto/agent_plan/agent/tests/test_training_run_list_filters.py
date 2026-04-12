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


WORK = Path(__file__).resolve().parent / '_tmp_training_run_list_filters'

COMPLETED_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.19G      3.267      4.407      2.853          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.420        0.55     0.365    0.210
3 epochs completed in 0.001 hours.
'''

FAILED_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/30      2.10G      4.021      6.322      3.102          8        640: 100% ━━━━━━━━━━━━ 2/2 0.4s
RuntimeError: CUDA out of memory
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
        failed_log = WORK / 'train_log_200.txt'
        failed_log.write_text(FAILED_LOG, encoding='utf-8')
        running_log = WORK / 'train_log_300.txt'
        running_log.write_text(COMPLETED_LOG, encoding='utf-8')

        now = time.time()
        proc = _spawn_sleep(60)
        _write_json(
            WORK / 'active_train_job.json',
            {
                'pid': proc.pid,
                'log_file': str(running_log),
                'started_at': now - 15,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/a.yaml', 'epochs': 10, 'device': '1'},
                'running': True,
                'updated_at': now - 1,
            },
        )
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 95270,
                'log_file': str(completed_log),
                'started_at': now - 100,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train'],
                'resolved_args': {'model': 'yolov8s.pt', 'data_yaml': '/tmp/b.yaml', 'epochs': 3, 'device': '1'},
                'running': False,
                'return_code': 0,
                'forced': False,
                'stop_reason': 'process_exit',
                'updated_at': now - 10,
                'stopped_at': now - 10,
            },
        )
        older = now - 220
        _write_json(
            WORK / 'train_log_200.json',
            {
                'dummy': True,
            },
        )
        import os
        os.utime(failed_log, (older, older))

        service = TrainService(state_dir=WORK)

        failed = service.list_training_runs(run_state='failed', limit=5)
        assert failed['ok'] is True
        assert failed['count'] == 1
        assert failed['runs'][0]['run_id'] == 'train_log_200'
        assert failed['applied_filters']['run_state'] == 'failed'

        completed = service.list_training_runs(run_state='completed', analysis_ready=True, limit=5)
        assert completed['ok'] is True
        assert completed['count'] == 1
        assert completed['runs'][0]['run_id'] == 'train_log_100'
        assert completed['applied_filters']['analysis_ready'] is True

        running = service.list_training_runs(run_state='running', limit=5)
        assert running['count'] == 1
        assert running['runs'][0]['run_id'] == 'train_log_300'
        print('training run list filters ok')
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
