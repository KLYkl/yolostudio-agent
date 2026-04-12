from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.services.gpu_utils import (
    GpuAllocationPolicy,
    GpuInfo,
    resolve_auto_device,
)
from agent_plan.agent.server.services.train_service import TrainService


def _fake_gpus() -> list[GpuInfo]:
    return [
        GpuInfo(index="0", uuid="gpu-0", free_mb=12000, busy=True),
        GpuInfo(index="1", uuid="gpu-1", free_mb=10000, busy=False),
        GpuInfo(index="2", uuid="gpu-2", free_mb=8000, busy=False),
    ]


def main() -> None:
    gpus = _fake_gpus()

    print("auto:single", resolve_auto_device(GpuAllocationPolicy.SINGLE_IDLE_GPU, gpus))
    print("auto:all", resolve_auto_device(GpuAllocationPolicy.ALL_IDLE_GPUS, gpus))
    print("auto:manual", resolve_auto_device(GpuAllocationPolicy.MANUAL_ONLY, gpus))

    globals_map = TrainService._resolve_device.__globals__
    original_query = globals_map["query_gpu_status"]
    original_policy = globals_map["get_effective_gpu_policy"]
    original_resolve_auto = globals_map["resolve_auto_device"]
    try:
        globals_map["query_gpu_status"] = lambda: gpus
        globals_map["get_effective_gpu_policy"] = lambda: GpuAllocationPolicy.SINGLE_IDLE_GPU
        print("resolve:manual_single", TrainService._resolve_device("1"))
        print("resolve:manual_multi_blocked", TrainService._resolve_device("1,2"))

        globals_map["get_effective_gpu_policy"] = lambda: GpuAllocationPolicy.ALL_IDLE_GPUS
        globals_map["resolve_auto_device"] = lambda policy=None: ("1,2", None)
        print("resolve:auto_all", TrainService._resolve_device("auto"))
        print("resolve:manual_multi_allowed", TrainService._resolve_device("1,2"))
        print("resolve:busy_gpu_blocked", TrainService._resolve_device("0"))
    finally:
        globals_map["query_gpu_status"] = original_query
        globals_map["get_effective_gpu_policy"] = original_policy
        globals_map["resolve_auto_device"] = original_resolve_auto


if __name__ == "__main__":
    main()
