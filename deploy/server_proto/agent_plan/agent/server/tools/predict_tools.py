from __future__ import annotations

from typing import Annotated, Any, Callable

from pydantic import Field

from yolostudio_agent.agent.server.services.predict_service import PredictService

service = PredictService()

_SOURCE_PATH_PARAM = Annotated[
    str,
    Field(
        description='待预测的图片或视频文件路径，也可以是待批量处理的目录路径。',
        examples=['/data/images', '/data/videos/demo.mp4'],
    ),
]
_MODEL_PARAM = Annotated[
    str,
    Field(
        description='用于预测的模型或权重路径。可以是 yolov8n.pt 这类模型名，或本地可访问的权重文件路径。',
        examples=['yolov8n.pt', '/models/best.pt'],
    ),
]
_CONF_PARAM = Annotated[
    float,
    Field(
        description='预测置信度阈值，范围 0 到 1。',
        ge=0.0,
        le=1.0,
        examples=[0.25, 0.5],
    ),
]
_IOU_PARAM = Annotated[
    float,
    Field(
        description='NMS IOU 阈值，范围 0 到 1。',
        ge=0.0,
        le=1.0,
        examples=[0.45, 0.6],
    ),
]
_OUTPUT_DIR_PARAM = Annotated[
    str,
    Field(
        description='预测输出目录。留空时由服务端自动创建默认输出目录。',
        examples=['/tmp/predict_out', 'runs/predict'],
    ),
]
_SESSION_ID_PARAM = Annotated[
    str,
    Field(
        description='后台或实时预测会话 ID。留空时默认使用当前活动会话。',
        examples=['image-predict-1234abcd', 'realtime-camera-12345678'],
    ),
]
_REPORT_PATH_PARAM = Annotated[
    str,
    Field(
        description='prediction_report.json 或 video_prediction_report.json 路径。',
        examples=['/tmp/predict/prediction_report.json', '/tmp/predict/video_prediction_report.json'],
    ),
]
_EXPORT_PATH_PARAM = Annotated[
    str,
    Field(
        description='导出报告文件路径。留空时按默认命名写入输出目录。',
        examples=['/tmp/predict/report.md', '/tmp/predict/report.json'],
    ),
]
_EXPORT_DIR_PARAM = Annotated[
    str,
    Field(
        description='导出路径清单目录。留空时按默认命名写入输出目录。',
        examples=['/tmp/predict/lists'],
    ),
]
_EXPORT_FORMAT_PARAM = Annotated[
    str,
    Field(
        description='报告导出格式。常用 markdown 或 json。',
        examples=['markdown', 'json'],
    ),
]
_DESTINATION_DIR_PARAM = Annotated[
    str,
    Field(
        description='整理预测结果时复制到的新目录。',
        examples=['/tmp/predict/organized'],
    ),
]
_ORGANIZE_BY_PARAM = Annotated[
    str,
    Field(
        description='整理策略。常用 detected_only 或 class。',
        examples=['detected_only', 'class'],
    ),
]
_ARTIFACT_PREFERENCE_PARAM = Annotated[
    str,
    Field(
        description='整理时优先复制哪类产物。常用 auto、annotated 或 original。',
        examples=['auto', 'annotated', 'original'],
    ),
]
_MAX_IMAGES_PARAM = Annotated[
    int,
    Field(
        description='最多处理的图片数量；0 表示不限制。',
        ge=0,
        examples=[0, 100],
    ),
]
_MAX_VIDEOS_PARAM = Annotated[
    int,
    Field(
        description='最多处理的视频数量；0 表示不限制。',
        ge=0,
        examples=[0, 10],
    ),
]
_MAX_FRAMES_PARAM = Annotated[
    int,
    Field(
        description='最多处理的帧数；0 表示不限制。',
        ge=0,
        examples=[0, 300],
    ),
]
_CAMERA_ID_PARAM = Annotated[
    int,
    Field(
        description='摄像头设备编号。',
        ge=0,
        examples=[0, 1],
    ),
]
_SCREEN_ID_PARAM = Annotated[
    int,
    Field(
        description='屏幕编号。',
        ge=0,
        examples=[1, 2],
    ),
]
_RTSP_URL_PARAM = Annotated[
    str,
    Field(
        description='RTSP 视频流地址。',
        examples=['rtsp://demo/live'],
    ),
]
_TIMEOUT_MS_PARAM = Annotated[
    int,
    Field(
        description='RTSP 地址探测超时时间，单位毫秒。',
        ge=1,
        examples=[3000, 5000],
    ),
]
_FRAME_INTERVAL_MS_PARAM = Annotated[
    int,
    Field(
        description='实时预测采样间隔，单位毫秒。',
        ge=0,
        examples=[100, 500],
    ),
]
_MAX_DEVICES_PARAM = Annotated[
    int,
    Field(
        description='扫描摄像头时最多探测的设备数量。',
        ge=1,
        examples=[5, 10],
    ),
]
_SAVE_ANNOTATED_PARAM = Annotated[
    bool,
    Field(
        description='是否保存带框标注图。',
        examples=[True, False],
    ),
]
_SAVE_LABELS_PARAM = Annotated[
    bool,
    Field(
        description='是否额外导出 YOLO txt 标签文件。',
        examples=[False, True],
    ),
]
_SAVE_ORIGINAL_PARAM = Annotated[
    bool,
    Field(
        description='是否在输出目录保留原始输入图片副本。',
        examples=[False, True],
    ),
]
_GENERATE_REPORT_PARAM = Annotated[
    bool,
    Field(
        description='是否生成结构化 prediction_report.json 报告。',
        examples=[True, False],
    ),
]
_INCLUDE_EMPTY_PARAM = Annotated[
    bool,
    Field(
        description='整理结果时是否把未检出的空样本也复制进去。',
        examples=[False, True],
    ),
]
_SAVE_VIDEO_PARAM = Annotated[
    bool,
    Field(
        description='视频预测时是否导出带框结果视频。',
        examples=[True, False],
    ),
]
_SAVE_KEYFRAMES_ANNOTATED_PARAM = Annotated[
    bool,
    Field(
        description='视频预测时是否导出带框关键帧。',
        examples=[True, False],
    ),
]
_SAVE_KEYFRAMES_RAW_PARAM = Annotated[
    bool,
    Field(
        description='视频预测时是否额外导出原始关键帧。',
        examples=[False, True],
    ),
]


def _action_candidates_from_next_actions(next_actions: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(next_actions, list):
        return candidates
    for item in next_actions:
        if isinstance(item, dict):
            compact = {
                'action': item.get('action'),
                'tool': item.get('tool'),
                'description': item.get('description') or item.get('reason') or item.get('summary'),
            }
            compact = {key: value for key, value in compact.items() if value not in (None, '', [], {})}
            if compact:
                candidates.append(compact)
        else:
            text = str(item or '').strip()
            if text:
                candidates.append({'description': text})
    return candidates


def _prediction_overview(result: dict[str, Any], *, mode: str) -> dict[str, Any]:
    overview = {
        'mode': mode,
        'processed_images': result.get('processed_images'),
        'processed_videos': result.get('processed_videos'),
        'detected_images': result.get('detected_images'),
        'empty_images': result.get('empty_images'),
        'total_frames': result.get('total_frames'),
        'detected_frames': result.get('detected_frames'),
        'total_detections': result.get('total_detections'),
        'annotated_dir': result.get('annotated_dir'),
        'output_dir': result.get('output_dir'),
        'report_path': result.get('report_path'),
        'warning_count': len(result.get('warnings') or []),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _prediction_session_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'session_id': result.get('session_id'),
        'status': result.get('status'),
        'run_state': result.get('run_state'),
        'running': result.get('running'),
        'total_images': result.get('total_images'),
        'processed_images': result.get('processed_images'),
        'detected_images': result.get('detected_images'),
        'output_dir': result.get('output_dir'),
        'report_path': result.get('report_path'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _prediction_output_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'output_dir': result.get('output_dir'),
        'report_path': result.get('report_path'),
        'artifact_root_count': result.get('artifact_root_count'),
        'path_list_count': result.get('path_list_count'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _export_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'export_format': result.get('export_format'),
        'export_path': result.get('export_path'),
        'export_dir': result.get('export_dir'),
        'detected_count': result.get('detected_count'),
        'empty_count': result.get('empty_count'),
        'failed_count': result.get('failed_count'),
        'detected_items_path': result.get('detected_items_path'),
        'empty_items_path': result.get('empty_items_path'),
        'failed_items_path': result.get('failed_items_path'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _organization_overview(result: dict[str, Any]) -> dict[str, Any]:
    overview = {
        'destination_dir': result.get('destination_dir'),
        'organize_by': result.get('organize_by'),
        'copied_items': result.get('copied_items'),
        'bucket_count': len(result.get('bucket_stats') or {}),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _source_overview(result: dict[str, Any], *, source_type: str) -> dict[str, Any]:
    overview = {
        'source_type': source_type,
        'camera_count': result.get('camera_count'),
        'screen_count': result.get('screen_count'),
        'camera_id': result.get('camera_id'),
        'screen_id': result.get('screen_id'),
        'rtsp_url': result.get('rtsp_url'),
        'source_label': result.get('source_label'),
        'session_id': result.get('session_id'),
        'status': result.get('status'),
        'output_dir': result.get('output_dir'),
        'report_path': result.get('report_path'),
    }
    return {key: value for key, value in overview.items() if value not in (None, '', [], {})}


def _apply_structured_defaults(result: dict[str, Any], *, overview_key: str, overview_value: dict[str, Any]) -> None:
    result.setdefault(overview_key, overview_value)
    result.setdefault('action_candidates', _action_candidates_from_next_actions(result.get('next_actions')))


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
    source_path: _SOURCE_PATH_PARAM,
    model: _MODEL_PARAM,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    save_annotated: _SAVE_ANNOTATED_PARAM = True,
    save_labels: _SAVE_LABELS_PARAM = False,
    save_original: _SAVE_ORIGINAL_PARAM = False,
    generate_report: _GENERATE_REPORT_PARAM = True,
    max_images: _MAX_IMAGES_PARAM = 0,
) -> dict[str, Any]:
    """对单张图片或图片目录执行 YOLO 预测。优先传 source_path 和 model；默认保存标注图与 JSON 报告，不修改原始数据。目录很大时会自动转成后台图片预测会话。"""
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
        if result.get('started_in_background'):
            suggestion = '可继续调用 check_image_prediction_status 查看后台进度'
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
            _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
        elif result.get('detected_images', 0) > 0 and result.get('annotated_dir'):
            suggestion = f"可查看标注结果目录: {result.get('annotated_dir')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
            _apply_structured_defaults(result, overview_key='prediction_overview', overview_value=_prediction_overview(result, mode='images'))
        else:
            _apply_structured_defaults(result, overview_key='prediction_overview', overview_value=_prediction_overview(result, mode='images'))
    else:
        result.setdefault('summary', '预测未完成')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取图片，并检查模型路径'])
        _apply_structured_defaults(result, overview_key='prediction_overview', overview_value=_prediction_overview(result, mode='images'))
    return result


def start_image_prediction(
    source_path: _SOURCE_PATH_PARAM,
    model: _MODEL_PARAM,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    save_annotated: _SAVE_ANNOTATED_PARAM = True,
    save_labels: _SAVE_LABELS_PARAM = False,
    save_original: _SAVE_ORIGINAL_PARAM = False,
    generate_report: _GENERATE_REPORT_PARAM = True,
    max_images: _MAX_IMAGES_PARAM = 0,
) -> dict[str, Any]:
    """显式启动后台图片预测会话。适用于大目录或预计会持续较久的图片预测。"""
    result = _wrap(
        '后台图片预测启动',
        service.start_image_prediction,
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
        suggestion = '可继续调用 check_image_prediction_status 查看后台进度'
        if suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    else:
        result.setdefault('summary', '后台图片预测未启动')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取图片，并检查模型路径'])
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    return result


def check_image_prediction_status(session_id: _SESSION_ID_PARAM = '') -> dict[str, Any]:
    """查看当前或指定后台图片预测会话的运行状态、已处理图片数和检测统计。"""
    result = _wrap(
        '后台图片预测状态查询',
        service.check_image_prediction_status,
        session_id=session_id,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('running'):
            suggestion = '如需结束，可继续调用 stop_image_prediction'
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    else:
        result.setdefault('summary', '后台图片预测状态查询未完成')
        result.setdefault('next_actions', ['请先启动后台图片预测'])
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    return result


def stop_image_prediction(session_id: _SESSION_ID_PARAM = '') -> dict[str, Any]:
    """停止当前或指定后台图片预测会话，并返回已完成的部分统计。"""
    result = _wrap(
        '后台图片预测停止',
        service.stop_image_prediction,
        session_id=session_id,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('report_path'):
            suggestion = f"可查看预测报告: {result.get('report_path')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    else:
        result.setdefault('summary', '停止后台图片预测未完成')
        result.setdefault('next_actions', ['当前没有运行中的后台图片预测会话'])
        _apply_structured_defaults(result, overview_key='prediction_session_overview', overview_value=_prediction_session_overview(result))
    return result


def summarize_prediction_results(report_path: _REPORT_PATH_PARAM = '', output_dir: _OUTPUT_DIR_PARAM = '') -> dict[str, Any]:
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
        overview_mode = 'videos' if str(result.get('mode') or '').strip() == 'videos' else 'images'
        _apply_structured_defaults(result, overview_key='prediction_summary_overview', overview_value=_prediction_overview(result, mode=overview_mode))
    else:
        result.setdefault('summary', '预测结果汇总未完成')
        result.setdefault('next_actions', ['请提供 report_path，或传入包含 prediction_report.json 的 output_dir'])
        _apply_structured_defaults(result, overview_key='prediction_summary_overview', overview_value=_prediction_overview(result, mode='summary'))
    return result


def inspect_prediction_outputs(report_path: _REPORT_PATH_PARAM = '', output_dir: _OUTPUT_DIR_PARAM = '') -> dict[str, Any]:
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
        _apply_structured_defaults(result, overview_key='prediction_output_overview', overview_value=_prediction_output_overview(result))
    else:
        result.setdefault('summary', '预测输出检查未完成')
        result.setdefault('next_actions', ['请提供 report_path，或传入包含 prediction_report.json / video_prediction_report.json 的 output_dir'])
        _apply_structured_defaults(result, overview_key='prediction_output_overview', overview_value=_prediction_output_overview(result))
    return result


def export_prediction_report(
    report_path: _REPORT_PATH_PARAM = '',
    output_dir: _OUTPUT_DIR_PARAM = '',
    export_path: _EXPORT_PATH_PARAM = '',
    export_format: _EXPORT_FORMAT_PARAM = 'markdown',
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
        _apply_structured_defaults(result, overview_key='export_overview', overview_value=_export_overview(result))
    else:
        result.setdefault('summary', '预测报告导出未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
        _apply_structured_defaults(result, overview_key='export_overview', overview_value=_export_overview(result))
    return result


def export_prediction_path_lists(
    report_path: _REPORT_PATH_PARAM = '',
    output_dir: _OUTPUT_DIR_PARAM = '',
    export_dir: _EXPORT_DIR_PARAM = '',
) -> dict[str, Any]:
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
        _apply_structured_defaults(result, overview_key='path_list_overview', overview_value=_export_overview(result))
    else:
        result.setdefault('summary', '预测路径清单导出未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
        _apply_structured_defaults(result, overview_key='path_list_overview', overview_value=_export_overview(result))
    return result


def organize_prediction_results(
    report_path: _REPORT_PATH_PARAM = '',
    output_dir: _OUTPUT_DIR_PARAM = '',
    destination_dir: _DESTINATION_DIR_PARAM = '',
    organize_by: _ORGANIZE_BY_PARAM = 'detected_only',
    include_empty: _INCLUDE_EMPTY_PARAM = False,
    artifact_preference: _ARTIFACT_PREFERENCE_PARAM = 'auto',
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
        _apply_structured_defaults(result, overview_key='organization_overview', overview_value=_organization_overview(result))
    else:
        result.setdefault('summary', '预测结果整理未完成')
        result.setdefault('next_actions', ['请提供 report_path，或先对 prediction 输出目录执行 inspect / summarize'])
        _apply_structured_defaults(result, overview_key='organization_overview', overview_value=_organization_overview(result))
    return result


def scan_cameras(max_devices: _MAX_DEVICES_PARAM = 5) -> dict[str, Any]:
    """扫描当前环境可用的本地摄像头设备，返回 camera_id 列表，便于后续启动实时预测。"""
    result = _wrap(
        '摄像头扫描',
        service.scan_cameras,
        max_devices=max_devices,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('camera_count', 0) > 0:
            suggestion = '如需开始实时预测，可继续调用 start_camera_prediction'
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='camera_overview', overview_value=_source_overview(result, source_type='camera'))
    else:
        result.setdefault('summary', '摄像头扫描未完成')
        result.setdefault('next_actions', ['请确认当前环境已安装 opencv-python，且允许访问摄像头'])
        _apply_structured_defaults(result, overview_key='camera_overview', overview_value=_source_overview(result, source_type='camera'))
    return result


def scan_screens() -> dict[str, Any]:
    """扫描当前环境可用的屏幕/显示器，返回 screen_id 列表，便于后续启动屏幕预测。"""
    result = _wrap(
        '屏幕扫描',
        service.scan_screens,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('screen_count', 0) > 0:
            suggestion = '如需开始屏幕预测，可继续调用 start_screen_prediction'
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='screen_overview', overview_value=_source_overview(result, source_type='screen'))
    else:
        result.setdefault('summary', '屏幕扫描未完成')
        result.setdefault('next_actions', ['请确认当前环境已安装 mss，且允许运行屏幕采集'])
        _apply_structured_defaults(result, overview_key='screen_overview', overview_value=_source_overview(result, source_type='screen'))
    return result


def test_rtsp_stream(rtsp_url: _RTSP_URL_PARAM, timeout_ms: _TIMEOUT_MS_PARAM = 5000) -> dict[str, Any]:
    """测试 RTSP 地址是否可连通并能读取视频帧，不启动长时间预测。"""
    result = _wrap(
        'RTSP 流测试',
        service.test_rtsp_stream,
        rtsp_url=rtsp_url,
        timeout_ms=timeout_ms,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = '如需开始实时预测，可继续调用 start_rtsp_prediction'
        if suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='stream_test_overview', overview_value=_source_overview(result, source_type='rtsp'))
    else:
        result.setdefault('summary', 'RTSP 流测试未完成')
        result.setdefault('next_actions', ['请确认 rtsp_url 是否可用，或先在当前环境做网络连通性检查'])
        _apply_structured_defaults(result, overview_key='stream_test_overview', overview_value=_source_overview(result, source_type='rtsp'))
    return result


def start_camera_prediction(
    model: _MODEL_PARAM,
    camera_id: _CAMERA_ID_PARAM = 0,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    frame_interval_ms: _FRAME_INTERVAL_MS_PARAM = 100,
    max_frames: _MAX_FRAMES_PARAM = 0,
) -> dict[str, Any]:
    """启动摄像头实时预测。默认只记录统计和 report，不改原始输入；需要结束时调用 stop_realtime_prediction。"""
    result = _wrap(
        '摄像头实时预测启动',
        service.start_camera_prediction,
        model=model,
        camera_id=camera_id,
        conf=conf,
        iou=iou,
        output_dir=output_dir,
        frame_interval_ms=frame_interval_ms,
        max_frames=max_frames,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = '可继续调用 check_realtime_prediction_status 查看实时进度'
        if suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='camera'))
    else:
        result.setdefault('summary', '摄像头实时预测未启动')
        result.setdefault('next_actions', ['请先 scan_cameras 确认可用 camera_id，并检查模型路径'])
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='camera'))
    return result


def start_rtsp_prediction(
    model: _MODEL_PARAM,
    rtsp_url: _RTSP_URL_PARAM,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    frame_interval_ms: _FRAME_INTERVAL_MS_PARAM = 100,
    max_frames: _MAX_FRAMES_PARAM = 0,
) -> dict[str, Any]:
    """启动 RTSP 实时预测。建议先用 test_rtsp_stream 确认地址可用。"""
    result = _wrap(
        'RTSP 实时预测启动',
        service.start_rtsp_prediction,
        model=model,
        rtsp_url=rtsp_url,
        conf=conf,
        iou=iou,
        output_dir=output_dir,
        frame_interval_ms=frame_interval_ms,
        max_frames=max_frames,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = '可继续调用 check_realtime_prediction_status 查看实时进度'
        if suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='rtsp'))
    else:
        result.setdefault('summary', 'RTSP 实时预测未启动')
        result.setdefault('next_actions', ['建议先调用 test_rtsp_stream 验证地址可用，再启动 RTSP 预测'])
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='rtsp'))
    return result


def start_screen_prediction(
    model: _MODEL_PARAM,
    screen_id: _SCREEN_ID_PARAM = 1,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    frame_interval_ms: _FRAME_INTERVAL_MS_PARAM = 100,
    max_frames: _MAX_FRAMES_PARAM = 0,
) -> dict[str, Any]:
    """启动屏幕实时预测。默认按 screen_id 选择显示器并持续采集，直到 stop_realtime_prediction。"""
    result = _wrap(
        '屏幕实时预测启动',
        service.start_screen_prediction,
        model=model,
        screen_id=screen_id,
        conf=conf,
        iou=iou,
        output_dir=output_dir,
        frame_interval_ms=frame_interval_ms,
        max_frames=max_frames,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        suggestion = '可继续调用 check_realtime_prediction_status 查看实时进度'
        if suggestion not in result['next_actions']:
            result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='screen'))
    else:
        result.setdefault('summary', '屏幕实时预测未启动')
        result.setdefault('next_actions', ['建议先调用 scan_screens 确认可用 screen_id，再启动屏幕预测'])
        _apply_structured_defaults(result, overview_key='realtime_session_overview', overview_value=_source_overview(result, source_type='screen'))
    return result


def check_realtime_prediction_status(session_id: _SESSION_ID_PARAM = '') -> dict[str, Any]:
    """查看当前或指定实时预测会话的运行状态、已处理帧数和检测统计。"""
    result = _wrap(
        '实时预测状态查询',
        service.check_realtime_prediction_status,
        session_id=session_id,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('running'):
            suggestion = '如需结束，可继续调用 stop_realtime_prediction'
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='realtime_status_overview', overview_value=_source_overview(result, source_type=str(result.get('source_type') or 'realtime')))
    else:
        result.setdefault('summary', '实时预测状态查询未完成')
        result.setdefault('next_actions', ['请先启动摄像头 / RTSP / 屏幕实时预测'])
        _apply_structured_defaults(result, overview_key='realtime_status_overview', overview_value=_source_overview(result, source_type=str(result.get('source_type') or 'realtime')))
    return result


def stop_realtime_prediction(session_id: _SESSION_ID_PARAM = '') -> dict[str, Any]:
    """停止当前或指定实时预测会话，并返回最终统计。

    适用: “停止实时预测”“停掉摄像头预测”“终止 RTSP 实时检测”。
    不适用: 查询进度请用 check_realtime_prediction_status；停止普通训练请用 stop_training。
    """
    result = _wrap(
        '实时预测停止',
        service.stop_realtime_prediction,
        session_id=session_id,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        if result.get('report_path'):
            suggestion = f"可查看实时预测报告: {result.get('report_path')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].insert(0, suggestion)
        _apply_structured_defaults(result, overview_key='realtime_status_overview', overview_value=_source_overview(result, source_type=str(result.get('source_type') or 'realtime')))
    else:
        result.setdefault('summary', '停止实时预测未完成')
        result.setdefault('next_actions', ['当前没有运行中的实时预测会话'])
        _apply_structured_defaults(result, overview_key='realtime_status_overview', overview_value=_source_overview(result, source_type=str(result.get('source_type') or 'realtime')))
    return result


def predict_videos(
    source_path: _SOURCE_PATH_PARAM,
    model: _MODEL_PARAM,
    conf: _CONF_PARAM = 0.25,
    iou: _IOU_PARAM = 0.45,
    output_dir: _OUTPUT_DIR_PARAM = '',
    save_video: _SAVE_VIDEO_PARAM = True,
    save_keyframes_annotated: _SAVE_KEYFRAMES_ANNOTATED_PARAM = True,
    save_keyframes_raw: _SAVE_KEYFRAMES_RAW_PARAM = False,
    generate_report: _GENERATE_REPORT_PARAM = True,
    max_videos: _MAX_VIDEOS_PARAM = 0,
    max_frames: _MAX_FRAMES_PARAM = 0,
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
        _apply_structured_defaults(result, overview_key='prediction_overview', overview_value=_prediction_overview(result, mode='videos'))
    else:
        result.setdefault('summary', '视频预测未完成')
        result.setdefault('next_actions', ['请确认 source_path 是否包含可读取视频，并检查模型路径'])
        _apply_structured_defaults(result, overview_key='prediction_overview', overview_value=_prediction_overview(result, mode='videos'))
    return result
