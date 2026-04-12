from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    """单张 GPU 的状态信息"""
    index: str
    uuid: str
    free_mb: int
    busy: bool


class GpuAllocationPolicy:
    SINGLE_IDLE_GPU = 'single_idle_gpu'
    ALL_IDLE_GPUS = 'all_idle_gpus'
    MANUAL_ONLY = 'manual_only'


def query_gpu_status() -> list[GpuInfo]:
    gpu_result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,gpu_uuid,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10,
    )
    if gpu_result.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in gpu_result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        gpus.append(GpuInfo(index=parts[0], uuid=parts[1], free_mb=int(parts[2]), busy=False))

    app_result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10,
    )
    if app_result.returncode == 0 and app_result.stdout.strip():
        busy_uuids: set[str] = set()
        for line in app_result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if parts:
                busy_uuids.add(parts[0])
        for gpu in gpus:
            if gpu.uuid in busy_uuids:
                gpu.busy = True

    return gpus


def get_idle_gpus(gpus: list[GpuInfo] | None = None) -> list[GpuInfo]:
    all_gpus = gpus if gpus is not None else query_gpu_status()
    return [gpu for gpu in all_gpus if not gpu.busy]


def get_effective_gpu_policy() -> str:
    policy = os.getenv('YOLOSTUDIO_TRAIN_DEVICE_POLICY', GpuAllocationPolicy.SINGLE_IDLE_GPU).strip().lower()
    valid = {
        GpuAllocationPolicy.SINGLE_IDLE_GPU,
        GpuAllocationPolicy.ALL_IDLE_GPUS,
        GpuAllocationPolicy.MANUAL_ONLY,
    }
    return policy if policy in valid else GpuAllocationPolicy.SINGLE_IDLE_GPU


def describe_gpu_policy(policy: str | None = None) -> str:
    effective = (policy or get_effective_gpu_policy()).strip().lower()
    mapping = {
        GpuAllocationPolicy.SINGLE_IDLE_GPU: 'auto 仅选择 1 张空闲 GPU',
        GpuAllocationPolicy.ALL_IDLE_GPUS: 'auto 会选择所有空闲 GPU，可形成多卡 device',
        GpuAllocationPolicy.MANUAL_ONLY: '不允许 auto，必须手动指定 device',
    }
    return mapping.get(effective, f'未知 GPU 策略: {effective}')


def resolve_auto_device(policy: str | None = None, gpus: list[GpuInfo] | None = None) -> tuple[str, str | None]:
    selected_policy = (policy or get_effective_gpu_policy()).strip().lower()
    idle_gpus = get_idle_gpus(gpus)

    if selected_policy == GpuAllocationPolicy.MANUAL_ONLY:
        return '', '当前策略为 manual_only，必须显式指定 device'
    if not idle_gpus:
        return '', '没有空闲 GPU（所有可见 GPU 都有进程占用，或当前不可见）'

    if selected_policy == GpuAllocationPolicy.ALL_IDLE_GPUS:
        ordered = sorted(idle_gpus, key=lambda gpu: int(gpu.index))
        return ','.join(gpu.index for gpu in ordered), None

    best = max(idle_gpus, key=lambda g: (g.free_mb, -int(g.index)))
    return best.index, None


def find_available_gpu() -> str | None:
    device, error = resolve_auto_device(policy=GpuAllocationPolicy.SINGLE_IDLE_GPU)
    return None if error else device


def get_gpu_status_summary() -> str:
    gpus = query_gpu_status()
    if not gpus:
        return '无法获取 GPU 信息（nvidia-smi 不可用）'
    lines = [f'GPU 策略: {describe_gpu_policy()}']
    for gpu in gpus:
        status = '忙碌（有进程占用）' if gpu.busy else '空闲'
        lines.append(f'GPU {gpu.index}: {status}, 空闲显存 {gpu.free_mb} MiB')
    return '\n'.join(lines)
