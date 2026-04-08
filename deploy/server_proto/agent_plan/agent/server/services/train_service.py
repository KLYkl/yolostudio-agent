from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent_plan.agent.server.services.train_log_parser import parse_latest_metrics


class TrainService:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None

    def start(self, model: str, data_yaml: str, epochs: int = 100, device: str = "1") -> dict:
        if self._process and self._process.poll() is None:
            return {"ok": False, "error": "training already running"}
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = runs_dir / f"train_log_{int(time.time())}.txt"
        cmd = ["yolo", "train", f"model={model}", f"data={data_yaml}", f"epochs={epochs}", f"device={device}"]
        log_handle = self._log_file.open("w", encoding="utf-8")
        self._process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        return {"ok": True, "pid": self._process.pid, "log_file": str(self._log_file)}

    def status(self) -> dict:
        running = bool(self._process and self._process.poll() is None)
        status = {
            "running": running,
            "log_file": str(self._log_file) if self._log_file else None,
        }
        if self._process:
            status["pid"] = self._process.pid
            status["return_code"] = self._process.poll()
        if self._log_file:
            status["latest_metrics"] = parse_latest_metrics(self._log_file)
        return status

    def stop(self) -> dict:
        if not self._process or self._process.poll() is not None:
            return {"ok": False, "error": "no active training"}
        self._process.terminate()
        return {"ok": True}
