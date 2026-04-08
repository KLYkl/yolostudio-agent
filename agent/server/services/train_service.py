"""
train_service.py - 训练服务（脱离 Qt 的 subprocess 封装）
=====================================================
用 subprocess.Popen 替代 QProcess，训练前动态检测可用 GPU。
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from agent_plan.agent.server.services.gpu_utils import (
    find_available_gpu,
    query_gpu_status,
)
from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics


class TrainService:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None
        self._start_time: float | None = None
        self._resolved_device: str | None = None
        self._command: list[str] | None = None

    def start(
        self,
        model: str,
        data_yaml: str,
        epochs: int = 100,
        device: str = "auto",
    ) -> dict:
        """启动训练。device 默认 auto，自动选择空闲 GPU。"""
        if self._process and self._process.poll() is None:
            return {"ok": False, "error": "已有训练任务在运行，请先停止或等待完成"}

        validation_error = self._validate_inputs(model=model, data_yaml=data_yaml, epochs=epochs)
        if validation_error:
            return {"ok": False, "error": validation_error}

        resolved_device, error = self._resolve_device(device)
        if error:
            return {"ok": False, "error": error}

        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = runs_dir / f"train_log_{int(time.time())}.txt"
        self._command = [
            "yolo", "train",
            f"model={model}",
            f"data={data_yaml}",
            f"epochs={epochs}",
            f"device={resolved_device}",
        ]

        log_handle = self._log_file.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            log_handle.close()
            self._process = None
            return {"ok": False, "error": f"启动训练子进程失败: {exc}"}

        self._start_time = time.time()
        self._resolved_device = resolved_device
        return {
            "ok": True,
            "pid": self._process.pid,
            "device": resolved_device,
            "log_file": str(self._log_file),
            "command": self._command,
        }

    def status(self) -> dict:
        """查看训练状态 + 最新指标"""
        running = bool(self._process and self._process.poll() is None)
        result: dict = {
            "running": running,
            "log_file": str(self._log_file) if self._log_file else None,
            "device": self._resolved_device,
            "command": self._command,
            "started_at": self._start_time,
        }
        if self._process:
            result["pid"] = self._process.pid
            result["return_code"] = self._process.poll()
        if self._start_time:
            result["elapsed_seconds"] = round(max(0.0, time.time() - self._start_time), 2)
        if self._log_file:
            result["latest_metrics"] = parse_latest_metrics(self._log_file)
        return result

    def stop(self) -> dict:
        """停止当前训练，必要时强制 kill。"""
        if not self._process or self._process.poll() is not None:
            return {"ok": False, "error": "当前没有正在运行的训练任务"}

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
            forced = False
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
            forced = True

        return {
            "ok": True,
            "forced": forced,
            "return_code": self._process.returncode,
        }

    @staticmethod
    def _validate_inputs(model: str, data_yaml: str, epochs: int) -> str | None:
        if not str(model).strip():
            return "model 不能为空"
        if not str(data_yaml).strip():
            return "data_yaml 不能为空"
        if int(epochs) <= 0:
            return "epochs 必须大于 0"
        if not Path(data_yaml).exists():
            return f"数据配置文件不存在: {data_yaml}"
        if shutil.which("yolo") is None:
            return "未找到 yolo 命令，请先确认 Ultralytics CLI 已安装并在 PATH 中"
        model_path = Path(model)
        if model_path.suffix and model_path.suffix in {'.pt', '.onnx', '.yaml'} and not model_path.exists() and not model.startswith('yolo'):
            return f"模型文件不存在: {model}"
        return None

    @staticmethod
    def _resolve_device(device: str) -> tuple[str, str | None]:
        """校验并解析设备参数。"""
        device = device.strip().lower()

        if device == "cpu":
            return "", "不支持 CPU 训练，请使用 GPU"
        if "," in device:
            return "", f"不支持多卡训练（收到 device={device}），请使用单张 GPU"
        if device == "auto":
            gpu_id = find_available_gpu()
            if gpu_id is None:
                return "", "没有可用于训练的 GPU（要么已有进程占用，要么当前不可见）"
            return gpu_id, None

        gpus = query_gpu_status()
        gpu_map = {gpu.index: gpu for gpu in gpus}
        if device not in gpu_map:
            valid_ids = ", ".join(sorted(gpu_map.keys())) or "无"
            return "", f"GPU {device} 不存在（可用设备: {valid_ids}）"

        target = gpu_map[device]
        if target.busy:
            return "", (
                f"GPU {device} 上有进程在运行，不建议同时训练。"
                f"可使用 device=auto 自动选择空闲 GPU"
            )
        return device, None
