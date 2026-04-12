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


WORK = Path('/mnt/d/yolodo2.0/agent_plan/agent/tests/_tmp_training_run_summary')

COMPLETED_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.19G      3.267      4.407      2.853          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.420        0.55     0.365    0.210
3 epochs completed in 0.001 hours.
Results saved to /tmp/runs/train5
'''

LOSS_ONLY_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/10      2.13G      3.076      3.867      2.729         35        640: 100% ━━━━━━━━━━━━ 2/2 2.2s
'''

RUNNING_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/5      2.13G      3.076      3.867      2.729         35        640: 100% ━━━━━━━━━━━━ 2/2 2.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.0s
                   all          5          5    0.810        0.33     0.220    0.110
'''


def _spawn_sleep(seconds: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        completed_log = WORK / 'completed.log'
        completed_log.write_text(COMPLETED_LOG, encoding='utf-8')
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 12345,
                'log_file': str(completed_log),
                'started_at': time.time() - 60,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=3', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 3, 'device': '1'},
                'running': False,
                'return_code': 0,
                'forced': False,
                'stop_reason': 'process_exit',
                'updated_at': time.time() - 1,
                'stopped_at': time.time() - 1,
            },
        )
        service = TrainService(state_dir=WORK)
        completed = service.summarize_run()
        assert completed['ok'] is True
        assert completed['run_state'] == 'completed'
        assert completed['analysis_ready'] is True
        assert completed['minimum_facts_ready'] is True
        assert completed['metrics']['precision'] == 0.42
        assert completed['latest_metrics']['metrics']['precision'] == 0.42
        assert completed['progress']['progress_ratio'] == 1.0
        assert completed['status_source'] == 'last_run'
        assert completed['observation_stage'] == 'final'

        loss_log = WORK / 'loss_only.log'
        loss_log.write_text(LOSS_ONLY_LOG, encoding='utf-8')
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 12346,
                'log_file': str(loss_log),
                'started_at': time.time() - 20,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=10', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 10, 'device': '1'},
                'running': False,
                'return_code': -15,
                'forced': False,
                'stop_reason': 'manual_stop',
                'updated_at': time.time() - 1,
                'stopped_at': time.time() - 1,
            },
        )
        stopped = service.summarize_run()
        assert stopped['run_state'] == 'stopped'
        assert stopped['analysis_ready'] is False
        assert 'loss_only_metrics' in stopped['signals']
        assert 'insufficient_eval_metrics' in stopped['signals']
        assert stopped['minimum_facts_ready'] is True
        assert stopped['progress']['progress_ratio'] == 0.1
        assert stopped['observation_stage'] == 'final'

        incomplete_log = WORK / 'completed_without_metrics.log'
        incomplete_log.write_text('', encoding='utf-8')
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 12347,
                'log_file': str(incomplete_log),
                'started_at': time.time() - 20,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=3', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 3, 'device': '1'},
                'running': False,
                'return_code': 0,
                'forced': False,
                'stop_reason': 'process_exit',
                'updated_at': time.time() - 1,
                'stopped_at': time.time() - 1,
            },
        )
        incomplete = service.summarize_run()
        assert incomplete['run_state'] == 'completed'
        assert incomplete['analysis_ready'] is False
        assert incomplete['minimum_facts_ready'] is False
        assert '已完成' in incomplete['summary']
        assert '缺少可分析日志或指标' in incomplete['summary']
        assert incomplete['observation_stage'] == 'final'

        running_log = WORK / 'running.log'
        running_log.write_text(RUNNING_LOG, encoding='utf-8')
        proc = _spawn_sleep(60)
        _write_json(
            WORK / 'active_train_job.json',
            {
                'pid': proc.pid,
                'log_file': str(running_log),
                'started_at': time.time() - 5,
                'device': '1',
                'requested_device': 'auto',
                'command': [sys.executable, '-c', 'import time; time.sleep(60)'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 5, 'device': '1'},
                'running': True,
                'updated_at': time.time() - 1,
            },
        )
        running_service = TrainService(state_dir=WORK)
        running = running_service.summarize_run()
        assert running['run_state'] == 'running'
        assert running['analysis_ready'] is True
        assert 'training_running' in running['signals']
        assert running['progress']['progress_ratio'] == 0.2
        assert running['observation_stage'] == 'early'

        cleanup = TrainService(state_dir=WORK)
        cleanup.stop()
        proc.wait(timeout=10)
        proc = None

        empty_work = WORK / 'empty'
        empty_work.mkdir(parents=True, exist_ok=True)
        empty_service = TrainService(state_dir=empty_work)
        unavailable = empty_service.summarize_run()
        assert unavailable['run_state'] == 'unavailable'
        assert unavailable['analysis_ready'] is False
        assert unavailable['observation_stage'] == 'early'
        print('training run summary ok')
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
