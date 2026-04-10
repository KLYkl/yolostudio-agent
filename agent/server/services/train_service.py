"""
train_service.py - 训练服务（脱离 Qt 的 subprocess 封装）
=====================================================
用 subprocess.Popen 替代 QProcess，训练前动态检测可用 GPU。
自动搜索 conda 环境中的 yolo 可执行文件，无需 yolo 在当前 PATH 中。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent_plan.agent.server.services.gpu_utils import (
    GpuAllocationPolicy,
    get_effective_gpu_policy,
    query_gpu_status,
    resolve_auto_device,
)
from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics


class TrainService:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None
        self._start_time: float | None = None
        self._resolved_device: str | None = None
        self._command: list[str] | None = None
        self._yolo_executable: str | None = None

    def start(
        self,
        model: str,
        data_yaml: str = "",
        epochs: int = 100,
        device: str = "auto",
    ) -> dict:
        if self._process and self._process.poll() is None:
            return {"ok": False, "error": "已有训练任务在运行，请先停止或等待完成"}

        validation_error = self._validate_inputs(model=model, data_yaml=data_yaml, epochs=epochs)
        if validation_error:
            return {"ok": False, "error": validation_error}

        resolved_device, error = self._resolve_device(device)
        if error:
            return {"ok": False, "error": error}

        yolo_exe = self._find_yolo_executable()
        if not yolo_exe:
            return {"ok": False, "error": "未找到 yolo 命令。请确认某个 conda 环境中已安装 ultralytics"}

        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = runs_dir / f"train_log_{int(time.time())}.txt"
        self._command = [
            yolo_exe, "train",
            f"model={model}",
            f"data={data_yaml}",
            f"epochs={epochs}",
            f"device={resolved_device}",
        ]
        self._yolo_executable = yolo_exe

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
        device_policy = get_effective_gpu_policy()
        argument_sources = {
            'model': 'request_or_agent_input',
            'data_yaml': 'request_or_tool_output',
            'epochs': 'request_or_default',
            'device': 'auto_resolved' if device.strip().lower() == 'auto' else 'manual_request',
        }
        return {
            "ok": True,
            "message": f"训练已启动：model={model}, data={data_yaml}, epochs={epochs}, device={resolved_device}, policy={device_policy}",
            "pid": self._process.pid,
            "device": resolved_device,
            "requested_device": device,
            "log_file": str(self._log_file),
            "argument_sources": argument_sources,
            "command": self._command,
            "resolved_args": {
                "model": model,
                "data_yaml": data_yaml,
                "epochs": epochs,
                "device": resolved_device,
                "device_policy": device_policy,
            },
            "yolo_executable": yolo_exe,
            "started_at": self._start_time,
        }

    def status(self) -> dict:
        running = bool(self._process and self._process.poll() is None)
        result: dict = {
            "ok": True,
            "running": running,
            "log_file": str(self._log_file) if self._log_file else None,
            "device": self._resolved_device,
            "command": self._command,
            "started_at": self._start_time,
            "yolo_executable": self._yolo_executable,
            "device_policy": get_effective_gpu_policy(),
        }
        if self._process:
            result["pid"] = self._process.pid
            result["return_code"] = self._process.poll()
        if self._start_time:
            result["elapsed_seconds"] = round(max(0.0, time.time() - self._start_time), 2)
        if self._log_file:
            result["latest_metrics"] = parse_latest_metrics(self._log_file)
        result["summary"] = self._build_status_summary(result)
        return result

    def stop(self) -> dict:
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
            "message": "训练任务已停止" if not forced else "训练任务已强制停止",
            "forced": forced,
            "return_code": self._process.returncode,
        }

    @staticmethod
    def _validate_inputs(model: str, data_yaml: str, epochs: int) -> str | None:
        if not str(model).strip():
            return "model 不能为空"
        if not str(data_yaml).strip():
            return "data_yaml 不能为空；请先提供 YAML 路径，或先完成数据集准备后再训练"
        if int(epochs) <= 0:
            return "epochs 必须大于 0"
        if not Path(data_yaml).exists():
            return f"数据配置文件不存在: {data_yaml}"
        model_path = Path(model)
        if model_path.suffix and model_path.suffix in {'.pt', '.onnx', '.yaml'} and not model_path.exists() and not model.startswith('yolo'):
            return f"模型文件不存在: {model}"
        return None

    @staticmethod
    def _find_yolo_executable() -> str | None:
        yolo_in_path = shutil.which("yolo")
        if yolo_in_path:
            return yolo_in_path

        search_roots: list[Path] = []

        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True, text=True, timeout=10,
                shell=(sys.platform == "win32"),
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = Path(parts[-1])
                        if env_path.is_dir():
                            yolo = _resolve_yolo_in_env(env_path)
                            if yolo:
                                return yolo
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        if sys.platform == "win32":
            search_roots = [
                Path.home() / "anaconda3" / "envs",
                Path.home() / "miniconda3" / "envs",
                Path.home() / ".conda" / "envs",
            ]
        else:
            search_roots = [
                Path.home() / "anaconda3" / "envs",
                Path.home() / "miniconda3" / "envs",
                Path.home() / ".conda" / "envs",
                Path("/opt/anaconda3/envs"),
                Path("/opt/miniconda3/envs"),
            ]

        for root in search_roots:
            if not root.exists():
                continue
            try:
                for env_dir in root.iterdir():
                    if not env_dir.is_dir():
                        continue
                    yolo = _resolve_yolo_in_env(env_dir)
                    if yolo:
                        return yolo
            except PermissionError:
                continue

        return None

    @staticmethod
    def _resolve_device(device: str) -> tuple[str, str | None]:
        device = device.strip().lower()
        policy = get_effective_gpu_policy()

        if device == "cpu":
            return "", "不支持 CPU 训练，请使用 GPU"
        if device == "auto":
            return resolve_auto_device(policy=policy)

        gpus = query_gpu_status()
        gpu_map = {gpu.index: gpu for gpu in gpus}
        requested_ids = [part.strip() for part in device.split(',') if part.strip()]
        if not requested_ids:
            return "", "device 不能为空"

        if len(requested_ids) > 1 and policy == GpuAllocationPolicy.SINGLE_IDLE_GPU:
            return "", f"当前策略 {policy} 仅允许单卡；收到 device={device}"

        missing = [gpu_id for gpu_id in requested_ids if gpu_id not in gpu_map]
        if missing:
            valid_ids = ", ".join(sorted(gpu_map.keys())) or "无"
            return "", f"GPU {', '.join(missing)} 不存在（可用设备: {valid_ids}）"

        busy = [gpu_id for gpu_id in requested_ids if gpu_map[gpu_id].busy]
        if busy:
            return "", f"GPU {', '.join(busy)} 上有进程在运行，不建议同时训练；可改用 device=auto 选择空闲 GPU"

        return ','.join(requested_ids), None

    @staticmethod
    def _build_status_summary(result: dict) -> str:
        if result.get("running"):
            elapsed = result.get("elapsed_seconds")
            elapsed_text = f", 已运行 {elapsed}s" if elapsed is not None else ""
            return f"训练进行中 (device={result.get('device')}, pid={result.get('pid')}{elapsed_text})"
        if result.get("return_code") is None and not result.get("log_file"):
            return "当前没有训练任务"
        return f"当前无训练在跑，最近 return_code={result.get('return_code')}"


def _resolve_yolo_in_env(env_path: Path) -> str | None:
    if sys.platform == "win32":
        yolo_exe = env_path / "Scripts" / "yolo.exe"
    else:
        yolo_exe = env_path / "bin" / "yolo"
    return str(yolo_exe) if yolo_exe.exists() else None
