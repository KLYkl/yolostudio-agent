from __future__ import annotations

from typing import Any, Callable

from agent_plan.agent.server.services.gpu_utils import get_effective_gpu_policy, get_gpu_status_summary, query_gpu_status, resolve_auto_device
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
            "summary": f"{action}失败",
            "next_actions": ["请查看错误信息并调整参数后重试"],
        }


def start_training(model: str, data_yaml: str = "", epochs: int = 100, device: str = "auto") -> dict:
    """启动一次 YOLO 训练任务。优先传 model/data_yaml/epochs；device 默认 auto，由服务端按当前 GPU 策略解析。"""
    result = _wrap("启动训练", service.start, model=model, data_yaml=data_yaml, epochs=epochs, device=device)
    if result.get("ok"):
        requested_device = result.get('requested_device', device)
        device_note = f"{result.get('device')} (auto 解析)" if str(requested_device).strip().lower() == 'auto' else result.get('device')
        result.setdefault("summary", f"训练已启动: model={result.get('resolved_args', {}).get('model', model)}, data={result.get('resolved_args', {}).get('data_yaml', data_yaml)}, device={device_note}")
        result.setdefault("next_actions", [
            "可调用 check_training_status 查看训练进度",
            "如需中止，可调用 stop_training",
        ])
    else:
        result.setdefault("summary", "训练未启动")
        result.setdefault("next_actions", [
            "检查 data_yaml 路径、模型名称和当前 GPU 状态",
            "如不确定，可先调用 training_readiness 或 check_gpu_status",
        ])
    return result


def check_training_status() -> dict:
    """获取当前训练任务状态与最近日志指标。"""
    result = _wrap("查询训练状态", service.status)
    if result.get("ok"):
        result.setdefault("next_actions", [
            "训练进行中时可继续轮询 check_training_status",
            "如需停止训练，调用 stop_training",
        ] if result.get("running") else ["当前无训练在跑，可直接 start_training 启动新训练"])
    return result


def stop_training() -> dict:
    """停止当前训练任务。"""
    result = _wrap("停止训练", service.stop)
    if result.get("ok"):
        result.setdefault("summary", result.get("message") or "训练任务已停止")
        result.setdefault("next_actions", ["如需重新开始训练，可再次调用 start_training"])
    else:
        result.setdefault("summary", "当前没有可停止的训练任务")
        result.setdefault("next_actions", ["可先调用 check_training_status 确认当前状态"])
    return result


def check_gpu_status() -> dict[str, Any]:
    """查询所有 GPU 的状态：是否有进程占用、空闲显存，以及 auto 设备会如何解析。"""
    def _impl() -> dict[str, Any]:
        gpus = query_gpu_status()
        policy = get_effective_gpu_policy()
        auto_device, auto_error = resolve_auto_device(policy=policy, gpus=gpus)
        return {
            "ok": True,
            "summary": get_gpu_status_summary(),
            "device_policy": policy,
            "auto_device": auto_device,
            "auto_error": auto_error,
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
            "next_actions": [
                "如需自动选卡训练，start_training 时保持 device=auto",
                "如需手动指定 device，请确认所选 GPU 当前不 busy，且符合 device_policy",
            ],
        }
    return _wrap("查询 GPU 状态", _impl)
