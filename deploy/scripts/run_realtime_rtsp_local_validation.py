from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _patch_windows_env() -> None:
    os.environ.setdefault('SystemRoot', r'C:\Windows')
    os.environ.setdefault('windir', r'C:\Windows')
    os.environ.setdefault('COMSPEC', r'C:\Windows\System32\cmd.exe')


def main() -> None:
    _patch_windows_env()
    repo_root = Path(__file__).resolve().parents[2]
    if 'YOLO_CONFIG_DIR' not in os.environ:
        cfg_root = repo_root / '.tmp_realtime_rtsp_local_config'
        cfg_root.mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(cfg_root)

    target = repo_root / 'agent' / 'tests' / 'test_realtime_rtsp_external_validation.py'
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
