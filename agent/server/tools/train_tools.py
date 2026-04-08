from __future__ import annotations

from agent_plan.agent.server.services.gpu_utils import get_gpu_status_summary
from agent_plan.agent.server.services.train_service import TrainService

service = TrainService()


def start_training(model: str, data_yaml: str, epochs: int = 100, device: str = "auto") -> dict:
    """启动一次 YOLO 训练任务。device 默认 auto，自动选择空闲 GPU。"""
    return service.start(model=model, data_yaml=data_yaml, epochs=epochs, device=device)


def check_training_status() -> dict:
    """获取当前训练任务状态与最近日志指标。"""
    return service.status()


def stop_training() -> dict:
    """停止当前训练任务。"""
    return service.stop()


def check_gpu_status() -> str:
    """查询所有 GPU 的状态：是否有进程占用、空闲显存。"""
    return get_gpu_status_summary()
