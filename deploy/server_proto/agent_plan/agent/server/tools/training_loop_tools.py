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
_BATCH_PARAM = Annotated[
    int | None,
    Field(description='训练批大小；留空时沿用默认或交给自动调优。', examples=[8, 16]),
]
_IMGSZ_PARAM = Annotated[
    int | None,
    Field(description='训练输入尺寸；留空时沿用默认或交给自动调优。', examples=[640, 960]),
]
_FRACTION_PARAM = Annotated[
    float | None,
    Field(description='训练样本抽样比例；留空时使用全量数据。', examples=[0.25, 0.5]),
]
_SINGLE_CLS_PARAM = Annotated[
    bool | None,
    Field(description='是否将所有类别视为单类任务。', examples=[True, False]),
]
_OPTIMIZER_PARAM = Annotated[
    str,
    Field(description='优化器名称；留空时使用默认配置。', examples=['SGD', 'AdamW']),
]
_FREEZE_PARAM = Annotated[
    int | None,
    Field(description='冻结前多少层；留空表示不额外冻结。', examples=[0, 10]),
]
_RESUME_PARAM = Annotated[
    bool | None,
    Field(description='是否基于历史中断训练继续。', examples=[True, False]),
]
_LR0_PARAM = Annotated[
    float | None,
    Field(description='初始学习率；留空时使用模型默认值。', examples=[0.01, 0.001]),
]
_PATIENCE_PARAM = Annotated[
    int | None,
    Field(description='早停耐心轮数；留空时使用默认值。', examples=[20, 50]),
]
_WORKERS_PARAM = Annotated[
    int | None,
    Field(description='数据加载 worker 数；留空时自动决定。', examples=[4, 8]),
]
_AMP_PARAM = Annotated[
    bool | None,
    Field(description='是否启用 AMP 混合精度。', examples=[True, False]),
]
_LOOP_NAME_PARAM = Annotated[
    str,
    Field(description='环训练任务名；用于区分不同自动调优实验。', examples=['nightly_loop', 'edge_device_recovery']),
]
_MANAGED_LEVEL_PARAM = Annotated[
    str,
    Field(description='环训练自动管理级别。', examples=['conservative_auto', 'aggressive_auto']),
]
_MAX_ROUNDS_PARAM = Annotated[
    int,
    Field(description='最多允许自动迭代多少轮训练。', examples=[3, 5]),
]
_TARGET_METRIC_PARAM = Annotated[
    str,
    Field(description='环训练主要优化目标。', examples=['map50', 'map50-95', 'recall']),
]
_TARGET_METRIC_VALUE_PARAM = Annotated[
    float | None,
    Field(description='目标指标阈值；达到后可提前结束。', examples=[0.85, 0.92]),
]
_MIN_IMPROVEMENT_PARAM = Annotated[
    float,
    Field(description='两轮之间至少需要达到的最小提升幅度。', examples=[0.005, 0.01]),
]
_NO_IMPROVEMENT_ROUNDS_PARAM = Annotated[
    int,
    Field(description='连续多少轮无明显提升后停止环训练。', examples=[2, 3]),
]
_MAX_FAILURES_PARAM = Annotated[
    int,
    Field(description='允许训练失败的最大轮数。', examples=[1, 2]),
]
_AUTO_HANDLE_OOM_PARAM = Annotated[
    bool,
    Field(description='遇到显存不足时是否允许自动降级参数后重试。', examples=[True, False]),
]
_LOOP_LIST_LIMIT_PARAM = Annotated[
    int,
    Field(description='返回最近多少条环训练记录。', examples=[5, 10]),
]
_LOOP_ID_PARAM = Annotated[
    str,
    Field(description='环训练 ID；留空时默认 active，没有 active 则 latest。', examples=['loop_20260418_101530', '']),
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
    model: train_tools._MODEL_PARAM,
    data_yaml: train_tools._DATA_YAML_PARAM = '',
    epochs: train_tools._EPOCHS_PARAM = _DEFAULT_LOOP_EPOCHS,
    device: train_tools._DEVICE_PARAM = 'auto',
    training_environment: train_tools._TRAINING_ENVIRONMENT_PARAM = '',
    project: train_tools._PROJECT_PARAM = '',
    name: train_tools._RUN_NAME_PARAM = '',
    batch: _BATCH_PARAM = None,
    imgsz: _IMGSZ_PARAM = None,
    fraction: _FRACTION_PARAM = None,
    classes: train_tools._TRAIN_CLASSES_PARAM = None,
    single_cls: _SINGLE_CLS_PARAM = None,
    optimizer: _OPTIMIZER_PARAM = '',
    freeze: _FREEZE_PARAM = None,
    resume: _RESUME_PARAM = None,
    lr0: _LR0_PARAM = None,
    patience: _PATIENCE_PARAM = None,
    workers: _WORKERS_PARAM = None,
    amp: _AMP_PARAM = None,
    loop_name: _LOOP_NAME_PARAM = '',
    managed_level: _MANAGED_LEVEL_PARAM = 'conservative_auto',
    max_rounds: _MAX_ROUNDS_PARAM = 5,
    target_metric: _TARGET_METRIC_PARAM = 'map50',
    target_metric_value: _TARGET_METRIC_VALUE_PARAM = None,
    min_improvement: _MIN_IMPROVEMENT_PARAM = 0.005,
    no_improvement_rounds: _NO_IMPROVEMENT_ROUNDS_PARAM = 2,
    max_failures: _MAX_FAILURES_PARAM = 2,
    allowed_tuning_params: _ALLOWED_TUNING_PARAMS_PARAM = None,
    auto_handle_oom: _AUTO_HANDLE_OOM_PARAM = True,
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


def list_training_loops(limit: _LOOP_LIST_LIMIT_PARAM = 5) -> dict[str, Any]:
    """列出最近的环训练记录。"""
    return _wrap('查询环训练列表', service.list_loops, limit=limit)


def check_training_loop_status(loop_id: _LOOP_ID_PARAM = '') -> dict[str, Any]:
    """查询当前或指定环训练的状态。默认查看 active，没有 active 时回落到 latest。"""
    return _wrap('查询环训练状态', service.check_loop_status, loop_id=loop_id)


def inspect_training_loop(loop_id: _LOOP_ID_PARAM = '') -> dict[str, Any]:
    """查看当前或指定环训练的完整轮次详情。默认查看 active，没有 active 时回落到 latest。"""
    return _wrap('查看环训练详情', service.inspect_loop, loop_id=loop_id)


def pause_training_loop(loop_id: _LOOP_ID_PARAM = '') -> dict[str, Any]:
    """请求在当前轮结束后停住；如果还没进入下一轮，则直接进入 paused。"""
    return _wrap('暂停环训练', service.pause_loop, loop_id=loop_id)


def resume_training_loop(loop_id: _LOOP_ID_PARAM = '') -> dict[str, Any]:
    """恢复 paused / awaiting_review 的环训练，让其继续下一轮。"""
    return _wrap('恢复环训练', service.resume_loop, loop_id=loop_id)


def stop_training_loop(loop_id: _LOOP_ID_PARAM = '') -> dict[str, Any]:
    """立即终止整个环训练；若当前轮正在训练，会先停止该训练进程。"""
    return _wrap('停止环训练', service.stop_loop, loop_id=loop_id)
