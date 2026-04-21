from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services import gpu_utils
from yolostudio_agent.agent.server.services.gpu_utils import (
    GpuAllocationPolicy,
    GpuInfo,
    query_gpu_status,
    resolve_auto_device,
)
from yolostudio_agent.agent.server.services.train_service import TrainService


def _fake_gpus() -> list[GpuInfo]:
    return [
        GpuInfo(index="0", uuid="gpu-0", free_mb=12000, busy=True),
        GpuInfo(index="1", uuid="gpu-1", free_mb=10000, busy=False),
        GpuInfo(index="2", uuid="gpu-2", free_mb=8000, busy=False),
    ]


def _fake_completed(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def _fake_subprocess_run(args, capture_output=True, text=True, timeout=10):  # noqa: ARG001
    command = ' '.join(args)
    if '--query-gpu=index,gpu_uuid,memory.free,memory.total,utilization.gpu' in command:
        return _fake_completed('0, GPU-0, 11935, 12288, 0\n1, GPU-1, 1522, 12288, 89\n')
    if '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' in command:
        return _fake_completed(
            'GPU-0, 235250, /home/kly/ollama/bin/ollama, 102 MiB\n'
            'GPU-1, 235250, /home/kly/ollama/bin/ollama, 10681 MiB\n'
        )
    raise AssertionError(f'unexpected subprocess args: {args}')


def _fake_subprocess_run_low_memory_but_high_util(args, capture_output=True, text=True, timeout=10):  # noqa: ARG001
    command = ' '.join(args)
    if '--query-gpu=index,gpu_uuid,memory.free,memory.total,utilization.gpu' in command:
        return _fake_completed('0, GPU-0, 11800, 12288, 35\n1, GPU-1, 9000, 12288, 0\n')
    if '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' in command:
        return _fake_completed(
            'GPU-0, 1001, /usr/bin/python, 128 MiB\n'
            'GPU-1, 1002, /usr/bin/python, 128 MiB\n'
        )
    raise AssertionError(f'unexpected subprocess args: {args}')


def main() -> None:
    gpus = _fake_gpus()

    assert resolve_auto_device(GpuAllocationPolicy.SINGLE_IDLE_GPU, gpus) == ("1", None)
    assert resolve_auto_device(GpuAllocationPolicy.ALL_IDLE_GPUS, gpus) == ("1,2", None)
    assert resolve_auto_device(GpuAllocationPolicy.MANUAL_ONLY, gpus) == ("", "当前策略为 manual_only，必须显式指定 device")

    globals_map = TrainService._resolve_device.__globals__
    original_query_with_error = globals_map["query_gpu_status_with_error"]
    original_policy = globals_map["get_effective_gpu_policy"]
    original_resolve_auto = globals_map["resolve_auto_device"]
    original_subprocess_run = gpu_utils.subprocess.run
    original_util_threshold = os.environ.get('YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT')
    original_memory_threshold = os.environ.get('YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB')
    try:
        globals_map["query_gpu_status_with_error"] = lambda: (gpus, None)
        globals_map["get_effective_gpu_policy"] = lambda: GpuAllocationPolicy.SINGLE_IDLE_GPU
        assert TrainService._resolve_device("1") == ("1", None)
        assert TrainService._resolve_device("1,2") == ("", "当前策略 single_idle_gpu 仅允许单卡；收到 device=1,2")
        assert TrainService._resolve_device("gpu1") == ("", "device 必须是 GPU 编号（如 0 / 1 / 0,1），或使用 auto")

        globals_map["get_effective_gpu_policy"] = lambda: GpuAllocationPolicy.ALL_IDLE_GPUS
        globals_map["resolve_auto_device"] = lambda policy=None: ("1,2", None)
        assert TrainService._resolve_device("auto") == ("1,2", None)
        assert TrainService._resolve_device("1,2") == ("1,2", None)
        assert TrainService._resolve_device("0") == ("", "GPU 0 上有进程在运行，不建议同时训练；可改用 device=auto 选择空闲 GPU")

        timeout = subprocess.TimeoutExpired(['nvidia-smi'], 10)
        globals_map["query_gpu_status_with_error"] = lambda: ([], f'nvidia-smi 调用超时（10s）: {timeout.cmd[0]}')
        assert TrainService._resolve_device("1") == ("1", None)
        globals_map["resolve_auto_device"] = lambda policy=None: ("", "当前无法自动选择 GPU：nvidia-smi 调用超时（10s）：nvidia-smi；请显式指定 device")
        assert TrainService._resolve_device("auto") == ("", "当前无法自动选择 GPU：nvidia-smi 调用超时（10s）：nvidia-smi；请显式指定 device")

        os.environ['YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT'] = '10'
        os.environ['YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB'] = '1024'
        gpu_utils.subprocess.run = _fake_subprocess_run
        observed, query_error = gpu_utils.query_gpu_status_with_error()
        assert query_error is None, query_error
        assert len(observed) == 2, observed
        assert observed[0].index == '0' and observed[0].busy is False, observed
        assert observed[1].index == '1' and observed[1].busy is True, observed
        assert observed[0].compute_used_mb == 102 and observed[0].utilization_pct == 0, observed
        assert observed[1].compute_used_mb == 10681 and observed[1].utilization_pct == 89, observed
        assert query_gpu_status() == observed
        assert resolve_auto_device(GpuAllocationPolicy.SINGLE_IDLE_GPU, observed) == ('0', None)

        gpu_utils.subprocess.run = _fake_subprocess_run_low_memory_but_high_util
        observed_high_util, query_error = gpu_utils.query_gpu_status_with_error()
        assert query_error is None, query_error
        assert observed_high_util[0].busy is True, observed_high_util
        assert observed_high_util[1].busy is False, observed_high_util
        assert observed_high_util[0].busy_reason.startswith('GPU 利用率 35%'), observed_high_util[0]
        assert resolve_auto_device(GpuAllocationPolicy.SINGLE_IDLE_GPU, observed_high_util) == ('1', None)
    finally:
        globals_map["query_gpu_status_with_error"] = original_query_with_error
        globals_map["get_effective_gpu_policy"] = original_policy
        globals_map["resolve_auto_device"] = original_resolve_auto
        gpu_utils.subprocess.run = original_subprocess_run
        if original_util_threshold is None:
            os.environ.pop('YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT', None)
        else:
            os.environ['YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT'] = original_util_threshold
        if original_memory_threshold is None:
            os.environ.pop('YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB', None)
        else:
            os.environ['YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB'] = original_memory_threshold

    print("gpu policy smoke ok")


if __name__ == "__main__":
    main()
