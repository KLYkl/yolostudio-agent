from __future__ import annotations

from agent_plan.agent.server.services.train_service import TrainService

service = TrainService()


def start_training(model: str, data_yaml: str, epochs: int = 100, device: str = "1") -> dict:
    """启动一次 YOLO 训练任务。"""
    return service.start(model=model, data_yaml=data_yaml, epochs=epochs, device=device)


def check_training_status() -> dict:
    """获取当前训练任务状态与最近日志指标。"""
    return service.status()


def stop_training() -> dict:
    """停止当前训练任务。"""
    return service.stop()
