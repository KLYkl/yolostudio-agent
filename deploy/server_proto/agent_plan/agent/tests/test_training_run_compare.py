from __future__ import annotations

import json
import shutil
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


WORK = Path(__file__).resolve().parent / '_tmp_training_run_compare'

BASELINE_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.19G      3.267      4.407      2.853          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.420        0.55     0.365    0.210
3 epochs completed in 0.001 hours.
'''

LATEST_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.11G      2.967      3.707      2.153          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.520        0.60     0.465    0.260
3 epochs completed in 0.001 hours.
'''


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        baseline_log = WORK / 'train_log_100.txt'
        baseline_log.write_text(BASELINE_LOG, encoding='utf-8')
        latest_log = WORK / 'train_log_200.txt'
        latest_log.write_text(LATEST_LOG, encoding='utf-8')

        now = time.time()
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 90000,
                'log_file': str(latest_log),
                'started_at': now - 60,
                'device': '1',
                'requested_device': 'auto',
                'command': ['yolo', 'train', 'model=yolov8n.pt', 'data=/tmp/data.yaml', 'epochs=3', 'device=1'],
                'resolved_args': {'model': 'yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 3, 'device': '1'},
                'running': False,
                'return_code': 0,
                'forced': False,
                'stop_reason': 'process_exit',
                'updated_at': now - 1,
                'stopped_at': now - 1,
            },
        )
        older = now - 300
        baseline_log.touch()
        import os
        os.utime(baseline_log, (older, older))

        service = TrainService(state_dir=WORK)
        result = service.compare_training_runs()
        assert result['ok'] is True
        assert result['left_run_id'] == 'train_log_200'
        assert result['right_run_id'] == 'train_log_100'
        assert result['metric_deltas']['precision']['delta'] == 0.1
        assert result['metric_deltas']['map50']['delta'] == 0.1
        assert any('precision提升' in item for item in result['highlights'])
        assert result['next_actions']

        explicit = service.compare_training_runs('train_log_100', 'train_log_200')
        assert explicit['ok'] is True
        assert explicit['left_run_id'] == 'train_log_100'
        assert explicit['right_run_id'] == 'train_log_200'
        assert explicit['metric_deltas']['precision']['delta'] == -0.1

        missing = service.compare_training_runs('train_log_missing', 'train_log_200')
        assert missing['ok'] is False
        assert '未找到可对比的训练记录' in missing['summary']
        print('training run compare ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
