from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.train_service import TrainService


WORK = Path(r'C:\workspace\yolodo2.0\agent_plan\agent\tests\_tmp_train_run_registry')


def _spawn_sleep(seconds: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'])


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        log_file = WORK / 'dummy_train.log'
        log_file.write_text('epoch 1\n', encoding='utf-8')

        proc = _spawn_sleep(60)
        service = TrainService(state_dir=WORK)
        service._process = proc
        service._active_pid = proc.pid
        service._log_file = log_file
        service._start_time = time.time() - 2
        service._resolved_device = '1'
        service._requested_device = 'auto'
        service._command = [sys.executable, '-c', 'import time; time.sleep(60)']
        service._yolo_executable = sys.executable
        service._argument_sources = {'device': 'auto_resolved'}
        service._resolved_args = {
            'model': 'yolov8n.pt',
            'data_yaml': '/tmp/data.yaml',
            'epochs': 3,
            'device': '1',
            'device_policy': 'single_idle_gpu',
        }
        service._write_active_registry()

        recovered = TrainService(state_dir=WORK)
        status = recovered.status()
        assert status['running'] is True
        assert status['pid'] == proc.pid
        assert status['reattached'] is True
        assert status['registry_path'].endswith('active_train_job.json')

        stopped = recovered.stop()
        assert stopped['ok'] is True
        proc.wait(timeout=10)

        active_registry = WORK / 'active_train_job.json'
        last_registry = WORK / 'last_train_job.json'
        assert not active_registry.exists()
        assert last_registry.exists()

        last = json.loads(last_registry.read_text(encoding='utf-8'))
        assert last['pid'] == proc.pid
        assert last['running'] is False
        assert last['stop_reason'] == 'manual_stop'

        final_status = recovered.status()
        assert final_status['running'] is False
        assert final_status['last_run']['pid'] == proc.pid
        assert final_status['return_code'] is not None
        print('train run registry smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
