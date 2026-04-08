from __future__ import annotations

from typing import Any, Callable

from agent_plan.agent.server.services.gpu_utils import get_gpu_status_summary, query_gpu_status
from agent_plan.agent.server.services.train_service import TrainService

service = TrainService()


def _wrap(action: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict):
            return result
        return {"ok": True, "result": result}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{action}失败: {exc}",
            "error_type": exc.__class__.__name__,
        }


def start_training(model: str, data_yaml: str = "", epochs: int = 100, device: str = "auto") -> dict:
    """启动一次 YOLO 训练任务。优先传入 model / data_yaml / epochs，device 默认 auto。"""
    return _wrap("启动训练", service.start, model=model, data_yaml=data_yaml, epochs=epochs, device=device)


def check_training_status() -> dict:
    """获取当前训练任务状态与最近日志指标。"""
    return _wrap("查询训练状态", service.status)


def stop_training() -> dict:
    """停止当前训练任务。"""
    return _wrap("停止训练", service.stop)


def check_gpu_status() -> dict[str, Any]:
    """查询所有 GPU 的状态：是否有进程占用、空闲显存。"""
    def _impl() -> dict[str, Any]:
        gpus = query_gpu_status()
        return {
            "ok": True,
            "summary": get_gpu_status_summary(),
            "gpus": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "free_mb": gpu.free_mb,
                    "busy": gpu.busy,
                }
                for gpu in gpus
            ],
            "available_gpu_indexes": [gpu.index for gpu in gpus if not gpu.busy],
        }
    return _wrap("查询 GPU 状态", _impl)
