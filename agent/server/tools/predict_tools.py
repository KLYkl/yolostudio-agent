from __future__ import annotations

from typing import Any, Callable

from agent_plan.agent.server.services.predict_service import PredictService

service = PredictService()


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
            'next_actions': ['请查看错误信息并调整参数后重试'],
        }


def predict_images(
    source_path: str,
    model: str,
    conf: float = 0.25,
    iou: float = 0.45,
    output_dir: str = '',
    save_annotated: bool = True,
    save_labels: bool = False,
    save_original: bool = False,
    generate_report: bool = True,
    max_images: int = 0,
) -> dict[str, Any]:
    """对单张图片或图片目录执行 YOLO 预测。优先传 source_path 和 model；默认保存标注图与 JSON 报告，不修改原始数据。"""
    result = _wrap(
        '图片预测',
        service.predict_images,
        source_path=source_path,
        model=model,
        conf=conf,
        iou=iou,
        output_dir=output_dir,
        save_annotated=save_annotated,
        save_labels=save_labels,
        save_original=save_original,
        generate_report=generate_report,
        max_images=max_images,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('detected_images', 0) > 0 and result.get('annotated_dir'):
            if f"可查看标注结果目录: {result.get('annotated_dir')}" not in result['next_actions']:
                result['next_actions'].insert(0, f"可查看标注结果目录: {result.get('annotated_dir')}")
    else:
        result.setdefault('summary', '预测未完成')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取图片，并检查模型路径'])
    return result
