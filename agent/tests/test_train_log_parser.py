from __future__ import annotations

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

from yolostudio_agent.agent.server.services.train_log_parser import parse_training_log


WORK = Path('/mnt/d/yolodo2.0/agent_plan/agent/tests/_tmp_train_log_parser')


RUNNING_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/10      2.17G      3.068      4.656      2.849          4        640: 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 1.3s/it 1.3s
                   all          5          5    0.820        0.36     0.330     0.120
'''

LOSS_ONLY_LOG = '''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        2/20      2.17G      3.257      3.989      2.842         10        640: 100% ━━━━━━━━━━━━ 2/2 0.4s
'''

FAILED_LOG = '''
Traceback (most recent call last):
  File "/home/kly/miniconda3/envs/yolodo/bin/yolo", line 6, in <module>
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
'''


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        running_log = WORK / 'running.log'
        running_log.write_text(RUNNING_LOG, encoding='utf-8')
        parsed = parse_training_log(running_log)
        assert parsed['ok'] is True
        assert parsed['has_eval_metrics'] is True
        assert parsed['metrics']['precision'] == 0.82
        assert parsed['metrics']['recall'] == 0.36
        assert 'early_training_observation' in parsed['signals']
        assert any('评估指标' in item for item in parsed['facts'])

        loss_log = WORK / 'loss_only.log'
        loss_log.write_text(LOSS_ONLY_LOG, encoding='utf-8')
        loss_only = parse_training_log(loss_log)
        assert loss_only['ok'] is True
        assert loss_only['has_loss_metrics'] is True
        assert loss_only['has_eval_metrics'] is False
        assert 'loss_only_metrics' in loss_only['signals']
        assert 'missing_eval_metrics' in loss_only['signals']

        failed_log = WORK / 'failed.log'
        failed_log.write_text(FAILED_LOG, encoding='utf-8')
        failed = parse_training_log(failed_log)
        assert failed['ok'] is True
        assert failed['run_state_hint'] == 'failed'
        assert 'training_log_failed' in failed['signals']
        assert failed['error_lines']

        missing = parse_training_log(WORK / 'missing.log')
        assert missing['ok'] is False
        print('train log parser ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
