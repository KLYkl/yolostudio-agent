from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.server.services.predict_service import PredictService

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
            suggestion = f"可查看标注结果目录: {result.get('annotated_dir')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测未完成')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取图片，并检查模型路径'])
    return result


def summarize_prediction_results(report_path: str = '', output_dir: str = '') -> dict[str, Any]:
    """读取预测 JSON 报告或预测输出目录，汇总当前预测结果，用于 grounded 总结与后续分析。"""
    result = _wrap(
        '预测结果汇总',
        service.summarize_prediction_results,
        report_path=report_path,
        output_dir=output_dir,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('annotated_dir'):
            suggestion = f"可查看标注结果目录: {result.get('annotated_dir')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测结果汇总未完成')
        result.setdefault('next_actions', ['请提供 report_path，或传入包含 prediction_report.json 的 output_dir'])
    return result


def inspect_prediction_outputs(report_path: str = '', output_dir: str = '') -> dict[str, Any]:
    """检查最近一次 prediction 的输出目录、报告路径和已生成产物，便于多轮 follow-up 继续复用。"""
    result = _wrap(
        '预测输出检查',
        service.inspect_prediction_outputs,
        report_path=report_path,
        output_dir=output_dir,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = f"可继续复用的输出目录: {result.get('output_dir')}"
        if result.get('output_dir') and suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测输出检查未完成')
        result.setdefault('next_actions', ['请提供 report_path，或传入包含 prediction_report.json / video_prediction_report.json 的 output_dir'])
    return result


def export_prediction_report(
    report_path: str = '',
    output_dir: str = '',
    export_path: str = '',
    export_format: str = 'markdown',
) -> dict[str, Any]:
    """把 prediction 结果导出成可复查报告；默认导出为 markdown，不会改写原始 prediction_report.json。"""
    result = _wrap(
        '预测报告导出',
        service.export_prediction_report,
        report_path=report_path,
        output_dir=output_dir,
        export_path=export_path,
        export_format=export_format,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = f"可查看导出报告: {result.get('export_path')}"
        if result.get('export_path') and suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测报告导出未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
    return result


def export_prediction_path_lists(report_path: str = '', output_dir: str = '', export_dir: str = '') -> dict[str, Any]:
    """导出命中 / 无命中 / 失败样本的路径清单，方便继续筛选、复查和批量处理。"""
    result = _wrap(
        '预测路径清单导出',
        service.export_prediction_path_lists,
        report_path=report_path,
        output_dir=output_dir,
        export_dir=export_dir,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = f"可查看路径清单目录: {result.get('export_dir')}"
        if result.get('export_dir') and suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测路径清单导出未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
    return result


def organize_prediction_results(
    report_path: str = '',
    output_dir: str = '',
    destination_dir: str = '',
    organize_by: str = 'detected_only',
    include_empty: bool = False,
    artifact_preference: str = 'auto',
) -> dict[str, Any]:
    """把 prediction 产物复制整理到新目录；支持只收集命中结果，或按类别分桶，不改写原始输出。"""
    result = _wrap(
        '预测结果整理',
        service.organize_prediction_results,
        report_path=report_path,
        output_dir=output_dir,
        destination_dir=destination_dir,
        organize_by=organize_by,
        include_empty=include_empty,
        artifact_preference=artifact_preference,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = f"可查看整理结果目录: {result.get('destination_dir')}"
        if result.get('destination_dir') and suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '预测结果整理未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
    return result


def predict_videos(
    source_path: str,
    model: str,
    conf: float = 0.25,
    iou: float = 0.45,
    output_dir: str = '',
    save_video: bool = True,
    save_keyframes_annotated: bool = True,
    save_keyframes_raw: bool = False,
    generate_report: bool = True,
    max_videos: int = 0,
    max_frames: int = 0,
) -> dict[str, Any]:
    """对单个视频或视频目录执行 YOLO 预测。默认保存结果视频与关键帧报告，不修改原始视频。"""
    result = _wrap(
        '视频预测',
        service.predict_videos,
        source_path=source_path,
        model=model,
        conf=conf,
        iou=iou,
        output_dir=output_dir,
        save_video=save_video,
        save_keyframes_annotated=save_keyframes_annotated,
        save_keyframes_raw=save_keyframes_raw,
        generate_report=generate_report,
        max_videos=max_videos,
        max_frames=max_frames,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = f"可查看视频预测输出目录: {result.get('output_dir')}"
        if result.get('output_dir') and suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
    else:
        result.setdefault('summary', '视频预测未完成')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取视频，并检查模型路径'])
    return result
