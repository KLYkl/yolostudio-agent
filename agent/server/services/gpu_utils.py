"""
gpu_utils.py - GPU 状态检测工具
=======================================
用于训练前动态选择可用 GPU，避免与 Ollama 或其他 CUDA 进程冲突。

判断逻辑：
卡上无任何 compute 进程（不只查 Ollama，任何 CUDA 进程都算 busy）
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    """单张 GPU 的状态信息"""
    index: str          # "0", "1", ...
    uuid: str           # GPU-xxxx-xxxx
    free_mb: int        # 空闲显存 (MiB)
    busy: bool          # 是否有 compute 进程在跑


def query_gpu_status() -> list[GpuInfo]:
    """
    查询所有 GPU 的状态：设备号、UUID、空闲显存、是否有进程。

    实现方式：两条 nvidia-smi 命令
    1. --query-gpu: 拿 index, uuid, memory.free
    2. --query-compute-apps: 拿哪些 uuid 上有进程
    """
    # 查 GPU 列表 + 空闲显存
    gpu_result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,gpu_uuid,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if gpu_result.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in gpu_result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        gpus.append(GpuInfo(
            index=parts[0],
            uuid=parts[1],
            free_mb=int(parts[2]),
            busy=False,
        ))

    # 查哪些 GPU 上有 compute 进程
    app_result = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if app_result.returncode == 0 and app_result.stdout.strip():
        busy_uuids: set[str] = set()
        for line in app_result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts:
                busy_uuids.add(parts[0])
        # 标记 busy
        for gpu in gpus:
            if gpu.uuid in busy_uuids:
                gpu.busy = True

    return gpus


def find_available_gpu() -> str | None:
    """
    找一张 **无进程** 的 GPU，返回设备号。

    选择策略：
    - 只从无 compute 进程的卡里选
    - 多张空闲时，选空闲显存最大的（作为 tiebreaker）
    - 没有空闲卡返回 None

    Returns:
        GPU 设备号 (如 "0", "1") 或 None
    """
    gpus = query_gpu_status()
    candidates = [gpu for gpu in gpus if not gpu.busy]
    if not candidates:
        return None
    # 多张空闲时选显存最大的
    best = max(candidates, key=lambda g: g.free_mb)
    return best.index


def get_gpu_status_summary() -> str:
    """返回人类可读的 GPU 状态摘要（供 Agent 汇报用）"""
    gpus = query_gpu_status()
    if not gpus:
        return "无法获取 GPU 信息（nvidia-smi 不可用）"
    lines = []
    for gpu in gpus:
        status = "🔴 忙碌（有进程占用）" if gpu.busy else "🟢 空闲"
        lines.append(f"GPU {gpu.index}: {status}, 空闲显存 {gpu.free_mb} MiB")
    return "\n".join(lines)
