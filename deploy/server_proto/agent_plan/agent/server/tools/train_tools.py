from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.server.services.gpu_utils import get_effective_gpu_policy, get_gpu_status_summary, query_gpu_status, resolve_auto_device
from yolostudio_agent.agent.server.services.train_service import TrainService

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


def start_training(
    model: str,
    data_yaml: str = "",
    epochs: int = 100,
    device: str = "auto",
    training_environment: str = "",
    batch: int | None = None,
    imgsz: int | None = None,
    optimizer: str = "",
    freeze: int | None = None,
    resume: bool | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    workers: int | None = None,
    amp: bool | None = None,
) -> dict:
    """启动一次 YOLO 训练任务。优先传 model/data_yaml/epochs；device 默认 auto，由服务端按当前 GPU 策略解析。"""
    result = _wrap(
        "启动训练",
        service.start,
        model=model,
        data_yaml=data_yaml,
        epochs=epochs,
        device=device,
        training_environment=training_environment,
        batch=batch,
        imgsz=imgsz,
        optimizer=optimizer,
        freeze=freeze,
        resume=resume,
        lr0=lr0,
        patience=patience,
        workers=workers,
        amp=amp,
    )
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


def list_training_environments() -> dict[str, Any]:
    """列出当前服务端可用的训练环境，并标出 start_training 默认将使用哪个环境。"""
    result = _wrap("查询训练环境", service.list_training_environments)
    if result.get('ok'):
        environments = result.get('environments') or []
        if environments:
            default_environment = result.get('default_environment') or environments[0]
            result.setdefault('summary', f"发现 {len(environments)} 个可用训练环境，默认将使用 {default_environment.get('display_name') or default_environment.get('name')}")
            result.setdefault('next_actions', [
                '如需确认这次训练会怎么启动，可继续调用 training_preflight',
                '如需真正开始训练，继续调用 start_training',
            ])
        else:
            result.setdefault('summary', '当前未发现可用训练环境')
            result.setdefault('next_actions', [
                '请先确认目标 conda 环境中已安装 ultralytics / yolo',
                '如环境本来已存在，请检查 PATH 或 conda env list 是否可见',
            ])
    return result


def training_preflight(
    model: str,
    data_yaml: str = "",
    epochs: int = 100,
    device: str = "auto",
    training_environment: str = "",
    batch: int | None = None,
    imgsz: int | None = None,
    optimizer: str = "",
    freeze: int | None = None,
    resume: bool | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    workers: int | None = None,
    amp: bool | None = None,
) -> dict[str, Any]:
    """只做训练预检，不真正启动训练；用于确认参数、训练环境和命令预览。"""
    result = _wrap(
        "训练预检",
        service.training_preflight,
        model=model,
        data_yaml=data_yaml,
        epochs=epochs,
        device=device,
        training_environment=training_environment,
        batch=batch,
        imgsz=imgsz,
        optimizer=optimizer,
        freeze=freeze,
        resume=resume,
        lr0=lr0,
        patience=patience,
        workers=workers,
        amp=amp,
    )
    if result.get('ok'):
        if result.get('ready_to_start'):
            result.setdefault('next_actions', [
                '当前参数和训练环境已可启动；如需真正开始训练，请调用 start_training',
                '如需再次确认 GPU 分配，可调用 check_gpu_status',
            ])
        else:
            result.setdefault('next_actions', [
                '请先修正预检中的阻塞项，再重新调用 training_preflight 或 start_training',
            ])
    return result


def list_training_runs(limit: int = 5) -> dict[str, Any]:
    """列出最近训练记录，便于查看最近一次训练、手动停止的训练或仅有日志的历史训练。"""
    result = _wrap("查询训练历史", service.list_training_runs, limit=limit)
    if result.get('ok'):
        runs = result.get('runs') or []
        if runs:
            first = runs[0]
            result.setdefault('summary', f"找到 {len(runs)} 条最近训练记录，最新状态为 {first.get('run_state')}")
            result.setdefault('next_actions', [
                '如需看最近一次训练效果，可继续调用 summarize_training_run',
                '如需判断下一步怎么做，可继续调用 analyze_training_outcome 或 recommend_next_training_step',
            ])
        else:
            result.setdefault('summary', '当前没有可读训练记录')
            result.setdefault('next_actions', [
                '可先调用 training_preflight 检查训练参数和环境',
                '准备好后可直接 start_training',
            ])
    return result


def inspect_training_run(run_id: str = '') -> dict[str, Any]:
    """查看某次训练记录详情；默认查看最近一次，也可传 run_id 或日志路径。"""
    result = _wrap("查看训练记录详情", service.inspect_training_run, run_id=run_id)
    if result.get('ok'):
        run_state = str(result.get('run_state') or '').strip().lower()
        if run_state == 'running':
            result.setdefault('next_actions', [
                '如需继续观察训练进度，可继续调用 check_training_status',
                '如需停止训练，可调用 stop_training',
            ])
        elif result.get('analysis_ready'):
            result.setdefault('next_actions', [
                '如需解释这次训练效果，可继续调用 analyze_training_outcome',
                '如需判断下一步动作，可继续调用 recommend_next_training_step',
            ])
        else:
            result.setdefault('next_actions', [
                '如需先看最近训练列表，可继续调用 list_training_runs',
                '如需汇总当前/最近训练事实，可继续调用 summarize_training_run',
            ])
    else:
        result.setdefault('next_actions', [
            '可先调用 list_training_runs 查看最近训练记录',
            '如果当前没有训练记录，可先调用 training_preflight 或 start_training',
        ])
    return result


def check_training_status() -> dict:
    """获取当前训练任务状态与最近日志指标。"""
    result = _wrap("查询训练状态", service.status)
    if result.get("ok"):
        if result.get('analysis_ready'):
            default_actions = [
                '可继续调用 summarize_training_run 汇总训练结果',
                '如需解释效果，可调用 analyze_training_outcome',
            ]
        elif result.get('minimum_facts_ready'):
            default_actions = [
                '可继续调用 summarize_training_run 汇总当前训练事实',
                '如缺少评估指标，先确认验证集、日志和结果目录是否完整',
            ]
        elif result.get("running"):
            default_actions = [
                "训练进行中时可继续轮询 check_training_status",
                "如需停止训练，调用 stop_training",
            ]
        else:
            default_actions = ["当前无训练在跑，可直接 start_training 启动新训练"]
        result.setdefault("next_actions", default_actions)
    return result


def summarize_training_run() -> dict[str, Any]:
    """汇总当前或最近一次训练运行的结构化事实，供知识工具消费。"""
    result = _wrap('汇总训练结果', service.summarize_run)
    if result.get('ok'):
        result.setdefault('next_actions', result.get('next_actions') or [
            '可继续调用 analyze_training_outcome 解释训练效果',
            '如需下一步动作建议，可调用 recommend_next_training_step',
        ])
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
