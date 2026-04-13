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


WORK = Path(__file__).resolve().parent / '_tmp_training_run_best'

RUN_A_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.11G      2.967      3.707      2.153          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.520        0.60     0.465    0.260
3 epochs completed in 0.001 hours.
'''

RUN_B_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        3/3      2.19G      3.267      4.407      2.853          7        640: 100% ━━━━━━━━━━━━ 2/2 0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 26.9it/s 0.0s
                   all          5          5    0.420        0.55     0.365    0.210
3 epochs completed in 0.001 hours.
'''

RUN_C_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/10      2.13G      3.076      3.867      2.729         35        640: 100% ━━━━━━━━━━━━ 2/2 2.2s
'''


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        run_b = WORK / 'train_log_100.txt'
        run_b.write_text(RUN_B_LOG, encoding='utf-8')
        run_a = WORK / 'train_log_200.txt'
        run_a.write_text(RUN_A_LOG, encoding='utf-8')
        run_c = WORK / 'train_log_300.txt'
        run_c.write_text(RUN_C_LOG, encoding='utf-8')

        now = time.time()
        _write_json(
            WORK / 'last_train_job.json',
            {
                'pid': 90000,
                'log_file': str(run_a),
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
        import os
        os.utime(run_b, (older, older))

        service = TrainService(state_dir=WORK)
        result = service.select_best_training_run(limit=5)
        assert result['ok'] is True
        assert result['best_run_id'] == 'train_log_200'
        assert result['best_run']['run_state'] == 'completed'
        assert result['evaluated_count'] == 3
        assert 'mAP50=0.465' in result['ranking_basis']
        assert result['candidates'][0]['run_id'] == 'train_log_200'
        assert result['candidates'][1]['run_id'] == 'train_log_100'
        print('training run best ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()