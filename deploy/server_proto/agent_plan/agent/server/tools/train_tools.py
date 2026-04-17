from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.server.services.gpu_utils import get_effective_gpu_policy, get_gpu_status_summary, query_gpu_status, resolve_auto_device
from yolostudio_agent.agent.server.services.train_service import TrainService

service = TrainService()


def _tool_candidate(
    *,
    tool: str,
    reason: str,
    args: dict[str, Any] | None = None,
    kind: str = 'tool_call',
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        'kind': kind,
        'tool': tool,
        'reason': reason,
    }
    if args:
        candidate['args'] = args
    return candidate


def _status_overview(result: dict[str, Any]) -> dict[str, Any]:
    return {
        'running': bool(result.get('running')),
        'run_state': result.get('run_state'),
        'observation_stage': result.get('observation_stage'),
        'analysis_ready': bool(result.get('analysis_ready')),
        'minimum_facts_ready': bool(result.get('minimum_facts_ready')),
        'has_progress': isinstance(result.get('progress'), dict),
        'has_metrics': isinstance(result.get('latest_metrics'), dict),
    }


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
    project: str = "",
    name: str = "",
    batch: int | None = None,
    imgsz: int | None = None,
    fraction: float | None = None,
    classes: list[int] | str | None = None,
    single_cls: bool | None = None,
    optimizer: str = "",
    freeze: int | None = None,
    resume: bool | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    workers: int | None = None,
    amp: bool | None = None,
) -> dict[str, Any]:
    """启动一次 YOLO 训练任务。优先传 model/data_yaml/epochs；device 默认 auto，由服务端按当前 GPU 策略解析。"""
    result = _wrap(
        "启动训练",
        service.start,
        model=model,
        data_yaml=data_yaml,
        epochs=epochs,
        device=device,
        training_environment=training_environment,
        project=project,
        name=name,
        batch=batch,
        imgsz=imgsz,
        fraction=fraction,
        classes=classes,
        single_cls=single_cls,
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
        result.setdefault('launch_overview', {
            'pid': result.get('pid'),
            'device': result.get('device'),
            'requested_device': requested_device,
            'environment_name': (result.get('training_environment') or {}).get('display_name') or (result.get('training_environment') or {}).get('name') or result.get('resolved_args', {}).get('training_environment'),
            'log_file': result.get('log_file'),
        })
        result.setdefault('action_candidates', [
            _tool_candidate(tool='check_training_status', reason='训练已启动，下一步通常先查询当前训练状态'),
            _tool_candidate(tool='stop_training', reason='如果需要立即终止当前训练，可以调用停止训练'),
        ])
        result.setdefault("next_actions", [
            "可调用 check_training_status 查看训练进度",
            "如需中止，可调用 stop_training",
        ])
    else:
        result.setdefault("summary", "训练未启动")
        result.setdefault('launch_overview', {
            'pid': result.get('pid'),
            'device': result.get('device'),
            'requested_device': device,
            'environment_name': (result.get('training_environment') or {}).get('display_name') or (result.get('training_environment') or {}).get('name') or result.get('resolved_args', {}).get('training_environment'),
            'log_file': result.get('log_file'),
        })
        result.setdefault('action_candidates', [
            _tool_candidate(tool='training_readiness', reason='启动失败时，先确认数据是否具备训练前提'),
            _tool_candidate(tool='check_gpu_status', reason='若怀疑是设备资源问题，可先检查 GPU 状态'),
        ])
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
        result.setdefault('environment_overview', {
            'environment_count': len(environments),
            'default_environment': result.get('default_environment') or {},
            'gpu_ready_count': len([item for item in environments if item.get('gpu_available') or item.get('torch_cuda_available')]),
        })
        if environments:
            default_environment = result.get('default_environment') or environments[0]
            result.setdefault('summary', f"发现 {len(environments)} 个可用训练环境，默认将使用 {default_environment.get('display_name') or default_environment.get('name')}")
            result.setdefault('action_candidates', [
                _tool_candidate(tool='training_preflight', reason='先确认参数和环境解析结果，再决定是否真正启动训练'),
                _tool_candidate(tool='start_training', reason='如参数和环境已明确，可以直接启动训练'),
            ])
            result.setdefault('next_actions', [
                '如需确认这次训练会怎么启动，可继续调用 training_preflight',
                '如需真正开始训练，继续调用 start_training',
            ])
        else:
            result.setdefault('summary', '当前未发现可用训练环境')
            result.setdefault('action_candidates', [])
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
    project: str = "",
    name: str = "",
    batch: int | None = None,
    imgsz: int | None = None,
    fraction: float | None = None,
    classes: list[int] | str | None = None,
    single_cls: bool | None = None,
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
        project=project,
        name=name,
        batch=batch,
        imgsz=imgsz,
        fraction=fraction,
        classes=classes,
        single_cls=single_cls,
        optimizer=optimizer,
        freeze=freeze,
        resume=resume,
        lr0=lr0,
        patience=patience,
        workers=workers,
        amp=amp,
    )
    if result.get('ok'):
        result.setdefault('preflight_overview', {
            'ready_to_start': bool(result.get('ready_to_start')),
            'resolved_device': result.get('resolved_device'),
            'environment_name': (result.get('training_environment') or {}).get('display_name') or (result.get('training_environment') or {}).get('name') or result.get('resolved_args', {}).get('training_environment'),
            'available_gpu_indexes': list(result.get('available_gpu_indexes') or []),
            'blocker_count': len(result.get('blockers') or []),
            'warning_count': len(result.get('warnings') or []),
        })
        if result.get('ready_to_start'):
            result.setdefault('action_candidates', [
                _tool_candidate(tool='start_training', reason='预检已通过，可以直接按当前参数启动训练'),
                _tool_candidate(tool='check_gpu_status', reason='如果还想复核自动选卡结果，可以先检查 GPU 状态'),
            ])
            result.setdefault('next_actions', [
                '当前参数和训练环境已可启动；如需真正开始训练，请调用 start_training',
                '如需再次确认 GPU 分配，可调用 check_gpu_status',
            ])
        else:
            result.setdefault('action_candidates', [])
            result.setdefault('next_actions', [
                '请先修正预检中的阻塞项，再重新调用 training_preflight 或 start_training',
            ])
    return result


def list_training_runs(
    limit: int = 5,
    run_state: str = '',
    analysis_ready: bool | None = None,
    model_keyword: str = '',
    data_keyword: str = '',
) -> dict[str, Any]:
    """列出训练记录列表。

    适用: “最近训练记录有哪些”“最近失败的训练”“把可分析的训练列出来”。
    不适用: 只看单条 run 详情时请用 inspect_training_run；比较两条 run 时请用 compare_training_runs；问哪次最好时请用 select_best_training_run。
    """
    result = _wrap(
        "查询训练历史",
        service.list_training_runs,
        limit=limit,
        run_state=run_state,
        analysis_ready=analysis_ready,
        model_keyword=model_keyword,
        data_keyword=data_keyword,
    )
    if result.get('ok'):
        runs = result.get('runs') or []
        result.setdefault('run_list_overview', {
            'run_count': len(runs),
            'first_run_id': (runs[0] if runs else {}).get('run_id'),
            'first_run_state': (runs[0] if runs else {}).get('run_state'),
        })
        if runs:
            first = runs[0]
            result.setdefault('summary', f"找到 {len(runs)} 条最近训练记录，最新状态为 {first.get('run_state')}")
            result.setdefault('action_candidates', [
                _tool_candidate(tool='summarize_training_run', reason='可先汇总最近一次训练的关键事实'),
                _tool_candidate(tool='analyze_training_outcome', reason='如果想解释训练结果，可进入知识分析链'),
            ])
            result.setdefault('next_actions', [
                '如需看最近一次训练效果，可继续调用 summarize_training_run',
                '如需判断下一步怎么做，可继续调用 analyze_training_outcome 或 recommend_next_training_step',
            ])
        else:
            result.setdefault('summary', '当前没有可读训练记录')
            result.setdefault('action_candidates', [
                _tool_candidate(tool='training_preflight', reason='当前没有历史训练时，可先做训练预检'),
                _tool_candidate(tool='start_training', reason='准备好参数后可直接启动一次新训练'),
            ])
            result.setdefault('next_actions', [
                '可先调用 training_preflight 检查训练参数和环境',
                '准备好后可直接 start_training',
            ])
    return result


def inspect_training_run(run_id: str = '') -> dict[str, Any]:
    """查看单条训练记录详情。

    适用: “看看 train_log_200 的详情”“最近一次训练具体情况”。
    不适用: 列表查询请用 list_training_runs；两次对比请用 compare_training_runs。
    """
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


def compare_training_runs(left_run_id: str = '', right_run_id: str = '') -> dict[str, Any]:
    """对比两次训练记录的状态、关键指标和差异摘要。

    适用: “对比最近两次训练”“比较 train_log_200 和 train_log_100”。
    默认比较最近两次可读训练；如果只想看哪次最好，请改用 select_best_training_run。
    """
    result = _wrap("对比训练记录", service.compare_training_runs, left_run_id=left_run_id, right_run_id=right_run_id)
    if result.get('ok'):
        result.setdefault('next_actions', [
            '如需查看其中某次训练详情，可继续调用 inspect_training_run',
            '如需解释这次差异意味着什么，可继续调用 analyze_training_outcome',
        ])
    else:
        result.setdefault('next_actions', [
            '可先调用 list_training_runs 查看最近训练记录',
            '如当前记录不足，可先完成一次训练后再比较',
        ])
    return result


def select_best_training_run(limit: int = 5) -> dict[str, Any]:
    """从最近若干条训练记录里选出最值得参考的一次。

    适用: “最近哪次训练最好”“最值得参考的训练记录是哪次”。
    不适用: 需要具体对比两次训练时请用 compare_training_runs。
    """
    result = _wrap("选择最佳训练记录", service.select_best_training_run, limit=limit)
    if result.get('ok'):
        result.setdefault('next_actions', [
            '如需查看最佳训练详情，可继续调用 inspect_training_run',
            '如需和最近一次训练做对比，可继续调用 compare_training_runs',
        ])
    else:
        result.setdefault('next_actions', [
            '可先调用 list_training_runs 查看最近训练记录',
            '如当前还没有可评估记录，可先完成一次训练',
        ])
    return result


def check_training_status() -> dict[str, Any]:
    """获取当前训练任务状态与最近日志指标。"""
    result = _wrap("查询训练状态", service.status)
    if result.get("ok"):
        result.setdefault('status_overview', _status_overview(result))
        if result.get('analysis_ready'):
            result.setdefault('action_candidates', [
                _tool_candidate(tool='summarize_training_run', reason='当前已有足够事实，可先汇总训练结果'),
                _tool_candidate(tool='analyze_training_outcome', reason='当前已有可分析信号，可继续解释训练效果'),
            ])
            default_actions = [
                '可继续调用 summarize_training_run 汇总训练结果',
                '如需解释效果，可调用 analyze_training_outcome',
            ]
        elif result.get('minimum_facts_ready'):
            result.setdefault('action_candidates', [
                _tool_candidate(tool='summarize_training_run', reason='当前已具备最小事实集，可先做结构化汇总'),
            ])
            default_actions = [
                '可继续调用 summarize_training_run 汇总当前训练事实',
                '如缺少评估指标，先确认验证集、日志和结果目录是否完整',
            ]
        elif result.get("running"):
            result.setdefault('action_candidates', [
                _tool_candidate(tool='check_training_status', reason='训练仍在运行，可继续轮询观察进度'),
                _tool_candidate(tool='stop_training', reason='如果需要终止当前训练，可以停止训练'),
            ])
            default_actions = [
                "训练进行中时可继续轮询 check_training_status",
                "如需停止训练，调用 stop_training",
            ]
        else:
            result.setdefault('action_candidates', [
                _tool_candidate(tool='start_training', reason='当前无训练在跑，可直接启动新训练'),
            ])
            default_actions = ["当前无训练在跑，可直接 start_training 启动新训练"]
        result.setdefault("next_actions", default_actions)
    return result


def summarize_training_run() -> dict[str, Any]:
    """汇总当前或最近一次训练运行的结构化事实，供知识工具消费。"""
    result = _wrap('汇总训练结果', service.summarize_run)
    if result.get('ok'):
        result.setdefault('summary_overview', {
            'run_state': result.get('run_state'),
            'observation_stage': result.get('observation_stage'),
            'analysis_ready': bool(result.get('analysis_ready')),
            'minimum_facts_ready': bool(result.get('minimum_facts_ready')),
            'fact_count': len(result.get('facts') or []),
            'signal_count': len(result.get('signals') or []),
        })
        result.setdefault('action_candidates', [
            _tool_candidate(tool='analyze_training_outcome', reason='在已有训练事实基础上解释当前效果'),
            _tool_candidate(tool='recommend_next_training_step', reason='基于当前事实生成下一步建议'),
        ])
        result.setdefault('next_actions', result.get('next_actions') or [
            '可继续调用 analyze_training_outcome 解释训练效果',
            '如需下一步动作建议，可调用 recommend_next_training_step',
        ])
    return result


def stop_training() -> dict[str, Any]:
    """停止当前普通训练任务。

    适用: “停止训练”“停掉当前训练”“终止当前训练进程”。
    不适用: 停止循环训练请用 stop_training_loop。
    """
    result = _wrap("停止训练", service.stop)
    if result.get("ok"):
        result.setdefault("summary", result.get("message") or "训练任务已停止")
        result.setdefault('status_overview', _status_overview(result))
        result.setdefault('action_candidates', [
            _tool_candidate(tool='start_training', reason='如果要重新开始训练，可再次启动训练'),
        ])
        result.setdefault("next_actions", ["如需重新开始训练，可再次调用 start_training"])
    else:
        result.setdefault("summary", "当前没有可停止的训练任务")
        result.setdefault('status_overview', _status_overview(result))
        result.setdefault('action_candidates', [
            _tool_candidate(tool='check_training_status', reason='可先确认当前到底是否存在活动训练'),
        ])
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
            "gpu_overview": {
                "gpu_count": len(gpus),
                "available_gpu_indexes": [gpu.index for gpu in gpus if not gpu.busy],
                "busy_gpu_indexes": [gpu.index for gpu in gpus if gpu.busy],
                "resolved_auto_device": auto_device,
                "policy": policy,
            },
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
            "action_candidates": [
                _tool_candidate(tool='start_training', reason='如需自动选卡训练，保持 device=auto 即可'),
            ],
            "next_actions": [
                "如需自动选卡训练，start_training 时保持 device=auto",
                "如需手动指定 device，请确认所选 GPU 当前不 busy，且符合 device_policy",
            ],
        }
    return _wrap("查询 GPU 状态", _impl)
