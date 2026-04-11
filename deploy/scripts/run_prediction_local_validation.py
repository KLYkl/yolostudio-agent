from __future__ import annotations

import inspect
import os
import runpy
import sys
import types
from pathlib import Path


def _patch_windows_runtime() -> None:
    """
    Work around local Windows runtime issues seen in the yolo / yolodo conda envs:
    - WinError 10106 when typing_extensions imports asyncio.coroutines
    - provider initialization errors triggered by os.urandom() in multiprocessing import path
    """
    asyncio_mod = types.ModuleType("asyncio")
    coroutines_mod = types.ModuleType("asyncio.coroutines")
    coroutines_mod.iscoroutinefunction = inspect.iscoroutinefunction
    asyncio_mod.coroutines = coroutines_mod
    asyncio_mod.iscoroutinefunction = inspect.iscoroutinefunction
    sys.modules["asyncio"] = asyncio_mod
    sys.modules["asyncio.coroutines"] = coroutines_mod
    os.urandom = lambda n: b"0" * n  # type: ignore[assignment]


def main() -> None:
    _patch_windows_runtime()
    if "YOLO_CONFIG_DIR" not in os.environ:
        repo_root = Path(__file__).resolve().parents[3]
        cfg_root = repo_root / "agent_plan" / ".tmp_prediction_local_config"
        cfg_root.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(cfg_root)

    target = Path(__file__).resolve().parents[2] / "agent" / "tests" / "test_prediction_remote_real_media.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
