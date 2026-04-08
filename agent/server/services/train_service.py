"""
train_service.py - 训练服务（脱离 Qt 的 subprocess 封装）
=====================================================
用 subprocess.Popen 替代 QProcess，训练前动态检测可用 GPU。

设备选择逻辑：
- device="auto"（默认）：自动找无进程且显存足够的 GPU
- device="0"/"1"：手动指定，但会校验该卡是否空闲
- device="0,1"/多卡：拒绝
- device="cpu"：拒绝
- 不存在的卡号：拒绝
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent_plan.agent.server.services.gpu_utils import (
    find_available_gpu,
    query_gpu_status,
)
from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics

# 训练最低显存要求 (MiB)
MIN_TRAIN_FREE_MB = 6000


class TrainService:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None

    def start(
        self,
        model: str,
        data_yaml: str,
        epochs: int = 100,
        device: str = "auto",
    ) -> dict:
        """启动训练。device 默认 auto，自动选择空闲 GPU。"""
        # 检查是否已有训练在跑
        if self._process and self._process.poll() is None:
            return {"ok": False, "error": "已有训练任务在运行，请先停止或等待完成"}

        # ---- 设备校验 ----
        resolved_device, error = self._resolve_device(device)
        if error:
            return {"ok": False, "error": error}

        # ---- 启动训练 ----
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = runs_dir / f"train_log_{int(time.time())}.txt"

        cmd = [
            "yolo", "train",
            f"model={model}",
            f"data={data_yaml}",
            f"epochs={epochs}",
            f"device={resolved_device}",
        ]
        log_handle = self._log_file.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        return {
            "ok": True,
            "pid": self._process.pid,
            "device": resolved_device,
            "log_file": str(self._log_file),
        }

    def status(self) -> dict:
        """查看训练状态 + 最新指标"""
        running = bool(self._process and self._process.poll() is None)
        result: dict = {
            "running": running,
            "log_file": str(self._log_file) if self._log_file else None,
        }
        if self._process:
            result["pid"] = self._process.pid
            result["return_code"] = self._process.poll()
        if self._log_file:
            result["latest_metrics"] = parse_latest_metrics(self._log_file)
        return result

    def stop(self) -> dict:
        """停止当前训练"""
        if not self._process or self._process.poll() is not None:
            return {"ok": False, "error": "当前没有正在运行的训练任务"}
        self._process.terminate()
        return {"ok": True}

    @staticmethod
    def _resolve_device(device: str) -> tuple[str, str | None]:
        """
        校验并解析设备参数。

        Returns:
            (resolved_device, error)
            成功: ("0" 或 "1", None)
            失败: ("", "错误描述")
        """
        device = device.strip().lower()

        # 拒绝 cpu
        if device == "cpu":
            return "", "不支持 CPU 训练，请使用 GPU"

        # 拒绝多卡（包含逗号）
        if "," in device:
            return "", f"不支持多卡训练（收到 device={device}），请使用单张 GPU"

        # 自动选择
        if device == "auto":
            gpu_id = find_available_gpu(min_free_mb=MIN_TRAIN_FREE_MB)
            if gpu_id is None:
                return "", "没有可用的 GPU（所有卡都有进程在运行或显存不足）"
            return gpu_id, None

        # 手动指定设备号：校验是否存在且空闲
        gpus = query_gpu_status()
        gpu_map = {gpu.index: gpu for gpu in gpus}

        if device not in gpu_map:
            valid_ids = ", ".join(sorted(gpu_map.keys()))
            return "", f"GPU {device} 不存在（可用设备: {valid_ids}）"

        target = gpu_map[device]
        if target.busy:
            return "", (
                f"GPU {device} 上有进程在运行，不建议同时训练。"
                f"可使用 device=auto 自动选择空闲 GPU"
            )
        if target.free_mb < MIN_TRAIN_FREE_MB:
            return "", (
                f"GPU {device} 空闲显存不足（{target.free_mb} MiB < {MIN_TRAIN_FREE_MB} MiB）"
            )

        return device, None
