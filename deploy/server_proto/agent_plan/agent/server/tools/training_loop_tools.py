from __future__ import annotations

import os
import re
from typing import Annotated, Any, Callable, Literal

from pydantic import Field

import yolostudio_agent.agent.server.tools.knowledge_tools as knowledge_tools
import yolostudio_agent.agent.server.tools.train_tools as train_tools
from yolostudio_agent.agent.server.services.training_loop_service import TrainingLoopService

_DEFAULT_LOOP_EPOCHS = max(1, int(str(os.getenv('YOLOSTUDIO_LOOP_DEFAULT_EPOCHS', '10')).strip() or '10'))
_ALLOWED_TUNING_PARAM = Literal['lr0', 'batch', 'imgsz', 'epochs', 'optimizer']
_ALLOWED_TUNING_PARAMS_PARAM = Annotated[
    list[_ALLOWED_TUNING_PARAM] | None,
    Field(
        description='允许环训练自动调整的参数白名单。优先传数组，而不是自然语言字符串。',
        examples=[['batch', 'imgsz'], ['epochs']],
    ),
]


service = TrainingLoopService(
    train_service=train_tools.service,
    knowledge_service=knowledge_tools.service,
)


def configure_loop_planner_llm(loop_llm: Any | None) -> None:
    """由 host/runtime 显式注入 loop planner LLM；tools/service 层不自行构建。"""
    service.loop_llm = None if loop_llm is False else loop_llm


def _wrap(action: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict):
            return result
        return {'ok': True, 'result': result}
    except Exception as exc:
        return {
            'ok': False,
            'error': f'{action}失败: {exc}',
            'error_type': exc.__class__.__name__,
            'summary': f'{action}失败',
            'next_actions': ['请检查当前环训练状态和输入参数后重试'],
        }


def _coerce_allowed_tuning_params(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value or '').strip()
        if not text:
            return None
        items = [
            token.strip().lower()
            for token in re.split(r'[^a-zA-Z0-9_]+', text)
            if token.strip()
        ]
    supported = {'lr0', 'batch', 'imgsz', 'epochs', 'optimizer'}
    normalized: list[str] = []
    for item in items:
        token = str(item or '').strip().lower()
        if token in supported and token not in normalized:
            normalized.append(token)
    return normalized


def start_training_loop(
    model: str,
    data_yaml: str = '',
    epochs: int = _DEFAULT_LOOP_EPOCHS,
    device: str = 'auto',
    training_environment: str = '',
    project: str = '',
    name: str = '',
    batch: int | None = None,
    imgsz: int | None = None,
    fraction: float | None = None,
    classes: list[int] | str | None = None,
    single_cls: bool | None = None,
    optimizer: str = '',
    freeze: int | None = None,
    resume: bool | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    workers: int | None = None,
    amp: bool | None = None,
    loop_name: str = '',
    managed_level: str = 'conservative_auto',
    max_rounds: int = 5,
    target_metric: str = 'map50',
    target_metric_value: float | None = None,
    min_improvement: float = 0.005,
    no_improvement_rounds: int = 2,
    max_failures: int = 2,
    allowed_tuning_params: _ALLOWED_TUNING_PARAMS_PARAM = None,
    auto_handle_oom: bool = True,
) -> dict[str, Any]:
    """启动一个服务端持久化的 Agent 环训练任务。

    示例 allowed_tuning_params: ["batch", "imgsz"]
    """
    result = _wrap(
        '启动环训练',
        service.start_loop,
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
        loop_name=loop_name,
        managed_level=managed_level,
        max_rounds=max_rounds,
        target_metric=target_metric,
        target_metric_value=target_metric_value,
        min_improvement=min_improvement,
        no_improvement_rounds=no_improvement_rounds,
        max_failures=max_failures,
        allowed_tuning_params=_coerce_allowed_tuning_params(allowed_tuning_params),
        auto_handle_oom=auto_handle_oom,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [
            '可调用 check_training_loop_status 查看当前环训练状态',
            '如需训练完当前轮后停住，可调用 pause_training_loop',
            '如需立即终止整个环训练，可调用 stop_training_loop',
        ])
    return result


def list_training_loops(limit: int = 5) -> dict[str, Any]:
    """列出最近的环训练记录。"""
    return _wrap('查询环训练列表', service.list_loops, limit=limit)


def check_training_loop_status(loop_id: str = '') -> dict[str, Any]:
    """查询当前或指定环训练的状态。默认查看 active，没有 active 时回落到 latest。"""
    return _wrap('查询环训练状态', service.check_loop_status, loop_id=loop_id)


def inspect_training_loop(loop_id: str = '') -> dict[str, Any]:
    """查看当前或指定环训练的完整轮次详情。默认查看 active，没有 active 时回落到 latest。"""
    return _wrap('查看环训练详情', service.inspect_loop, loop_id=loop_id)


def pause_training_loop(loop_id: str = '') -> dict[str, Any]:
    """请求在当前轮结束后停住；如果还没进入下一轮，则直接进入 paused。"""
    return _wrap('暂停环训练', service.pause_loop, loop_id=loop_id)


def resume_training_loop(loop_id: str = '') -> dict[str, Any]:
    """恢复 paused / awaiting_review 的环训练，让其继续下一轮。"""
    return _wrap('恢复环训练', service.resume_loop, loop_id=loop_id)


def stop_training_loop(loop_id: str = '') -> dict[str, Any]:
    """立即终止整个环训练；若当前轮正在训练，会先停止该训练进程。"""
    return _wrap('停止环训练', service.stop_loop, loop_id=loop_id)
