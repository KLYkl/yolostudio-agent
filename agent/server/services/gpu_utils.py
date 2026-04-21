from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class GpuInfo:
    """单张 GPU 的状态信息"""
    index: str
    uuid: str
    free_mb: int
    busy: bool
    total_mb: int = 0
    utilization_pct: int = 0
    compute_process_count: int = 0
    compute_used_mb: int = 0
    busy_reason: str = ''


class GpuAllocationPolicy:
    SINGLE_IDLE_GPU = 'single_idle_gpu'
    ALL_IDLE_GPUS = 'all_idle_gpus'
    MANUAL_ONLY = 'manual_only'


def _parse_int(raw: str | int | float | None) -> int:
    text = str(raw or '').strip().replace('MiB', '').replace('%', '').strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except Exception:
        return 0


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        return max(int(raw), minimum)
    except Exception:
        return max(default, minimum)


def get_busy_utilization_threshold_pct() -> int:
    return _env_int('YOLOSTUDIO_GPU_BUSY_UTILIZATION_THRESHOLD_PCT', 10, minimum=0)


def get_busy_compute_memory_threshold_mb() -> int:
    return _env_int('YOLOSTUDIO_GPU_BUSY_COMPUTE_MEMORY_THRESHOLD_MB', 1024, minimum=0)


def _classify_gpu_busy(gpu: GpuInfo) -> tuple[bool, str]:
    if gpu.compute_process_count <= 0:
        return False, ''

    util_threshold = get_busy_utilization_threshold_pct()
    memory_threshold_mb = get_busy_compute_memory_threshold_mb()

    reasons: list[str] = []
    if gpu.utilization_pct >= util_threshold > 0:
        reasons.append(f'GPU 利用率 {gpu.utilization_pct}% ≥ {util_threshold}%')
    if gpu.compute_used_mb >= memory_threshold_mb > 0:
        reasons.append(f'compute 显存占用 {gpu.compute_used_mb} MiB ≥ {memory_threshold_mb} MiB')

    if reasons:
        return True, '；'.join(reasons)

    return False, (
        f'仅检测到轻量 compute 占用（{gpu.compute_process_count} 个进程，'
        f'compute 显存 {gpu.compute_used_mb} MiB，GPU 利用率 {gpu.utilization_pct}%）'
    )


def _run_nvidia_smi(command: list[str]) -> tuple[Any | None, str | None]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None, f"nvidia-smi 调用超时（10s）: {' '.join(command)}"
    except FileNotFoundError:
        return None, 'nvidia-smi 不可用'
    except OSError as exc:
        return None, f'nvidia-smi 调用失败: {exc}'

    if getattr(result, 'returncode', 1) != 0:
        stderr = str(getattr(result, 'stderr', '') or '').strip()
        detail = f'：{stderr}' if stderr else ''
        return None, f"nvidia-smi 返回非零状态 {getattr(result, 'returncode', 1)}{detail}"
    return result, None


def query_gpu_status_with_error() -> tuple[list[GpuInfo], str | None]:
    gpu_result, gpu_error = _run_nvidia_smi(
        ['nvidia-smi', '--query-gpu=index,gpu_uuid,memory.free,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
    )
    if gpu_error:
        return [], gpu_error

    gpus: list[GpuInfo] = []
    for line in str(getattr(gpu_result, 'stdout', '') or '').strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        gpus.append(
            GpuInfo(
                index=parts[0],
                uuid=parts[1],
                free_mb=_parse_int(parts[2]),
                busy=False,
                total_mb=_parse_int(parts[3]),
                utilization_pct=_parse_int(parts[4]),
            )
        )

    if not gpus:
        return [], 'nvidia-smi 未返回可见 GPU'

    app_result, app_error = _run_nvidia_smi(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader']
    )
    if app_error:
        return gpus, f'GPU 占用明细暂不可用: {app_error}'

    if str(getattr(app_result, 'stdout', '') or '').strip():
        app_stats: dict[str, dict[str, int]] = {}
        for line in str(getattr(app_result, 'stdout', '') or '').strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 1:
                continue
            gpu_uuid = parts[0]
            stats = app_stats.setdefault(gpu_uuid, {'count': 0, 'used_mb': 0})
            stats['count'] += 1
            stats['used_mb'] += _parse_int(parts[3] if len(parts) > 3 else 0)
        for gpu in gpus:
            stats = app_stats.get(gpu.uuid)
            if not stats:
                continue
            gpu.compute_process_count = int(stats.get('count') or 0)
            gpu.compute_used_mb = int(stats.get('used_mb') or 0)
            gpu.busy, gpu.busy_reason = _classify_gpu_busy(gpu)

    return gpus, None


def query_gpu_status() -> list[GpuInfo]:
    gpus, _ = query_gpu_status_with_error()
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
    query_error: str | None = None
    observed_gpus = gpus
    if observed_gpus is None:
        observed_gpus, query_error = query_gpu_status_with_error()

    if selected_policy == GpuAllocationPolicy.MANUAL_ONLY:
        return '', '当前策略为 manual_only，必须显式指定 device'
    if query_error:
        return '', f'当前无法自动选择 GPU：{query_error}；请显式指定 device'

    idle_gpus = get_idle_gpus(observed_gpus)
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
    gpus, query_error = query_gpu_status_with_error()
    if not gpus:
        return f'无法获取 GPU 信息（{query_error or "nvidia-smi 不可用"}）'
    lines = [f'GPU 策略: {describe_gpu_policy()}']
    if query_error:
        lines.append(f'注意: {query_error}')
    for gpu in gpus:
        if gpu.busy:
            status = f'忙碌（{gpu.busy_reason or "资源占用较高"}）'
        elif gpu.compute_process_count > 0:
            status = f'可复用（{gpu.busy_reason or "仅轻量 compute 占用"}）'
        else:
            status = '空闲'
        lines.append(
            f'GPU {gpu.index}: {status}, 空闲显存 {gpu.free_mb} MiB'
            + (f', compute 占用 {gpu.compute_used_mb} MiB' if gpu.compute_process_count > 0 else '')
            + (f', 利用率 {gpu.utilization_pct}%' if gpu.total_mb > 0 or gpu.utilization_pct else '')
        )
    return '\n'.join(lines)
